//! Exact resident binding for the OODS -> numerator -> quotient seam.
//!
//! The arena planner has already fixed the four-tree opening order and aliased
//! every transcript hand-off. This module performs the one remaining checked
//! binding pass and constructs the three allocation-free launch objects in
//! dependency order. No caller may recreate quotient constants on the host.

use stwo_backend_cuda::{
    ArenaError, ArenaSlice, OodsCoefficientColumn, PreparedOodsError, PreparedOodsGraph,
    PreparedQuotientError, PreparedQuotientGraph, PreparedQuotientNumeratorError,
    PreparedQuotientNumeratorGraph, QuotientNumeratorColumn, QuotientNumeratorColumnSource,
    QuotientNumeratorDestination,
};

use crate::arena_plan::{ArenaBinding, OpenedColumnSource};
use crate::graphs::GraphWorkspace;

#[derive(Debug)]
pub enum ResidentOodsError {
    ColumnCountMismatch {
        oods: usize,
        numerator: usize,
    },
    ColumnIdentityMismatch {
        index: usize,
        oods: OpenedColumnSource,
        numerator: OpenedColumnSource,
    },
    ColumnBindingMismatch {
        index: usize,
    },
    LogicalSizeMismatch {
        role: &'static str,
        expected_words: usize,
        actual_words: usize,
    },
    DestinationMismatch(&'static str),
    SizeOverflow,
    Arena(ArenaError),
    Oods(PreparedOodsError),
    Numerator(PreparedQuotientNumeratorError),
    Quotient(PreparedQuotientError),
}

impl core::fmt::Display for ResidentOodsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "resident OODS pipeline binding rejected: {self:?}")
    }
}

impl std::error::Error for ResidentOodsError {}

impl From<ArenaError> for ResidentOodsError {
    fn from(value: ArenaError) -> Self {
        Self::Arena(value)
    }
}

impl From<PreparedOodsError> for ResidentOodsError {
    fn from(value: PreparedOodsError) -> Self {
        Self::Oods(value)
    }
}

impl From<PreparedQuotientNumeratorError> for ResidentOodsError {
    fn from(value: PreparedQuotientNumeratorError) -> Self {
        Self::Numerator(value)
    }
}

impl From<PreparedQuotientError> for ResidentOodsError {
    fn from(value: PreparedQuotientError) -> Self {
        Self::Quotient(value)
    }
}

/// Prepared device-only opening and quotient path. Field order is deliberate:
/// quotient is dropped before the numerator sources it references, and the
/// numerator is dropped before the OODS destinations it consumes.
pub(crate) struct ResidentOodsPipeline<'a> {
    quotient: PreparedQuotientGraph<'a>,
    numerator: PreparedQuotientNumeratorGraph<'a>,
    oods: PreparedOodsGraph<'a>,
}

impl<'a> ResidentOodsPipeline<'a> {
    pub(crate) fn prepare(workspace: &'a GraphWorkspace) -> Result<Self, ResidentOodsError> {
        let arena = workspace.arena();
        let oods_plan = workspace.plan().oods();
        let numerator_plan = workspace.plan().quotient_numerator();
        if oods_plan.columns.len() != numerator_plan.columns.len() {
            return Err(ResidentOodsError::ColumnCountMismatch {
                oods: oods_plan.columns.len(),
                numerator: numerator_plan.columns.len(),
            });
        }

        let mut oods_columns = Vec::with_capacity(oods_plan.columns.len());
        for column in &oods_plan.columns {
            let coefficients = bind_exact(
                workspace,
                column.coefficients,
                words_for_log(column.coefficient_log_size)?,
                "OODS coefficient column",
            )?;
            oods_columns.push(OodsCoefficientColumn {
                coefficients,
                topology: column.topology(),
            });
        }
        let oods = PreparedOodsGraph::prepare(
            arena,
            oods_plan.config,
            &oods_columns,
            bind_exact(workspace, oods_plan.oods_point_parameter, 4, "OODS point")?,
            &oods_plan.slots,
        )?;
        require_same_slice(
            oods.sample_points_destination(),
            bind_logical(workspace, oods_plan.sample_points)?,
            "OODS sample-point destination",
        )?;
        require_same_slice(
            oods.sampled_values_destination(),
            bind_logical(workspace, oods_plan.sampled_values)?,
            "OODS sampled-value destination",
        )?;

        let mut numerator_columns = Vec::with_capacity(numerator_plan.columns.len());
        for (index, (oods_column, column)) in oods_plan
            .columns
            .iter()
            .zip(&numerator_plan.columns)
            .enumerate()
        {
            if oods_column.source != column.source {
                return Err(ResidentOodsError::ColumnIdentityMismatch {
                    index,
                    oods: oods_column.source,
                    numerator: column.source,
                });
            }
            if oods_column.coefficients != column.coefficients {
                return Err(ResidentOodsError::ColumnBindingMismatch { index });
            }
            let coefficients = bind_exact(
                workspace,
                column.coefficients,
                words_for_log(column.topology.coefficient_log_size)?,
                "quotient numerator coefficient column",
            )?;
            numerator_columns.push(QuotientNumeratorColumn {
                coefficient_log_size: column.topology.coefficient_log_size,
                source: QuotientNumeratorColumnSource::Coefficients(coefficients),
                samples: column.topology.samples.clone(),
            });
        }
        let destinations = numerator_plan
            .destinations
            .iter()
            .map(|destination| {
                Ok(QuotientNumeratorDestination {
                    log_size: destination.log_size,
                    coordinates: destination
                        .coordinates
                        .map(|binding| {
                            bind_exact(
                                workspace,
                                binding,
                                words_for_log(destination.log_size)?,
                                "quotient numerator destination",
                            )
                        })
                        .into_iter()
                        .collect::<Result<Vec<_>, ResidentOodsError>>()?
                        .try_into()
                        .expect("quotient numerator has four coordinates"),
                })
            })
            .collect::<Result<Vec<_>, ResidentOodsError>>()?;
        let numerator = PreparedQuotientNumeratorGraph::prepare(
            arena,
            numerator_plan.config,
            &numerator_columns,
            bind_logical(workspace, numerator_plan.oods_sample_points)?,
            bind_logical(workspace, numerator_plan.oods_sampled_values)?,
            bind_exact(
                workspace,
                numerator_plan.random_coefficient,
                4,
                "quotient random coefficient",
            )?,
            bind_logical(workspace, numerator_plan.sample_points_destination)?,
            bind_logical(workspace, numerator_plan.first_linear_terms_destination)?,
            &destinations,
            bind_logical(workspace, numerator_plan.forward_twiddles)?,
            &numerator_plan.slots,
        )?;

        let quotient_plan = workspace.plan().quotient();
        let quotient_sources = numerator.quotient_sources();
        let quotient = PreparedQuotientGraph::prepare(
            arena,
            quotient_plan.config,
            &quotient_sources,
            bind_logical(workspace, quotient_plan.forward_twiddles)?,
            bind_logical(workspace, quotient_plan.inverse_subdomain_twiddles)?,
            &quotient_plan.slots,
        )?;
        require_same_slice(
            quotient.sample_points_destination(),
            bind_logical(workspace, numerator_plan.sample_points_destination)?,
            "numerator to quotient sample points",
        )?;
        require_same_slice(
            quotient.first_linear_terms_destination(),
            bind_logical(workspace, numerator_plan.first_linear_terms_destination)?,
            "numerator to quotient linear terms",
        )?;
        require_same_slice(
            quotient.output_evaluation(),
            bind_logical(workspace, workspace.plan().fri().input_values)?,
            "quotient to FRI input",
        )?;

        Ok(Self {
            quotient,
            numerator,
            oods,
        })
    }

    pub(crate) fn launch_oods(&self) -> Result<(), ResidentOodsError> {
        self.oods.launch()?;
        Ok(())
    }

    pub(crate) fn launch_numerator_and_quotient(&self) -> Result<(), ResidentOodsError> {
        self.numerator.launch()?;
        self.quotient.launch()?;
        Ok(())
    }

    pub(crate) const fn quotient(&self) -> &PreparedQuotientGraph<'a> {
        &self.quotient
    }
}

fn words_for_log(log_size: u32) -> Result<usize, ResidentOodsError> {
    1usize
        .checked_shl(log_size)
        .ok_or(ResidentOodsError::SizeOverflow)
}

fn bind_exact(
    workspace: &GraphWorkspace,
    binding: ArenaBinding,
    expected_words: usize,
    role: &'static str,
) -> Result<ArenaSlice, ResidentOodsError> {
    if binding.len_words != expected_words {
        return Err(ResidentOodsError::LogicalSizeMismatch {
            role,
            expected_words,
            actual_words: binding.len_words,
        });
    }
    bind_logical(workspace, binding)
}

fn bind_logical(
    workspace: &GraphWorkspace,
    binding: ArenaBinding,
) -> Result<ArenaSlice, ResidentOodsError> {
    let slice = workspace.arena().bind(binding.physical)?;
    if slice.len_words() < binding.len_words {
        return Err(ResidentOodsError::LogicalSizeMismatch {
            role: "physical arena binding",
            expected_words: binding.len_words,
            actual_words: slice.len_words(),
        });
    }
    Ok(slice)
}

fn require_same_slice(
    actual: ArenaSlice,
    expected: ArenaSlice,
    role: &'static str,
) -> Result<(), ResidentOodsError> {
    if actual.id() != expected.id() || actual.len_words() != expected.len_words() {
        return Err(ResidentOodsError::DestinationMismatch(role));
    }
    Ok(())
}
