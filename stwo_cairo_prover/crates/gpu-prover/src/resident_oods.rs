//! Exact resident binding for the OODS -> numerator -> quotient seam.
//!
//! The arena planner has already fixed the four-tree opening order and aliased
//! every transcript hand-off. This module performs the one remaining checked
//! binding pass and constructs the three allocation-free launch objects in
//! dependency order. No caller may recreate quotient constants on the host.

use stwo_backend_cuda::{
    quotient_numerator_staged_single_write_plan_with_overflow_capacities, ArenaError, ArenaSlice,
    OodsColumnSource, OodsPolynomialColumn, OodsSourceKind, PreparedNumeratorSchedule,
    PreparedOodsError, PreparedOodsGraph, PreparedQuotientError, PreparedQuotientGraph,
    PreparedQuotientNumeratorError, PreparedQuotientNumeratorGraph, QuotientNumeratorColumn,
    QuotientNumeratorColumnSource, QuotientNumeratorDestination, QuotientNumeratorSingleWriteError,
    QuotientNumeratorSourceKind, QuotientNumeratorStagedSingleWriteError,
};

use crate::arena_plan::{ArenaBinding, OpenedColumnSource, QuotientNumeratorSchedule};
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
    ColumnGeometryMismatch {
        index: usize,
    },
    LogicalSizeMismatch {
        role: &'static str,
        expected_words: usize,
        actual_words: usize,
    },
    DestinationMismatch(&'static str),
    NumeratorScheduleMismatch {
        planned: QuotientNumeratorSchedule,
        actual: PreparedNumeratorSchedule,
    },
    SizeOverflow,
    Arena(ArenaError),
    Oods(PreparedOodsError),
    Numerator(PreparedQuotientNumeratorError),
    NumeratorSchedule(QuotientNumeratorSingleWriteError),
    NumeratorStagedSchedule(QuotientNumeratorStagedSingleWriteError),
    StagedNumeratorBinding(&'static str),
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

impl From<QuotientNumeratorSingleWriteError> for ResidentOodsError {
    fn from(value: QuotientNumeratorSingleWriteError) -> Self {
        Self::NumeratorSchedule(value)
    }
}

impl From<QuotientNumeratorStagedSingleWriteError> for ResidentOodsError {
    fn from(value: QuotientNumeratorStagedSingleWriteError) -> Self {
        Self::NumeratorStagedSchedule(value)
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
            let topology = column.topology();
            let source = bind_exact(
                workspace,
                column.source_binding,
                words_for_log(topology.log_size)?,
                "OODS polynomial source",
            )?;
            let source = match column.source_kind {
                OodsSourceKind::Coefficients => OodsColumnSource::Coefficients(source),
                OodsSourceKind::Evaluations => OodsColumnSource::Evaluations(source),
            };
            oods_columns.push(OodsPolynomialColumn { source, topology });
        }
        let oods = PreparedOodsGraph::prepare_mixed(
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
            let expected_evaluation_log = column
                .topology
                .coefficient_log_size
                .checked_add(numerator_plan.config.log_blowup_factor)
                .ok_or(ResidentOodsError::SizeOverflow)?;
            if oods_column.coefficient_log_size != column.topology.coefficient_log_size
                || oods_column.evaluation_log_size != expected_evaluation_log
            {
                return Err(ResidentOodsError::ColumnGeometryMismatch { index });
            }
            if (oods_column.source_kind == OodsSourceKind::Coefficients
                && oods_column.source_binding != column.coefficients)
                || (oods_column.source_kind == OodsSourceKind::Evaluations
                    && column.topology.source_kind == QuotientNumeratorSourceKind::Evaluation
                    && oods_column.source_binding != column.numerator_source)
            {
                return Err(ResidentOodsError::ColumnBindingMismatch { index });
            }
            let source = match column.topology.source_kind {
                QuotientNumeratorSourceKind::Coefficients => {
                    if column.numerator_source != column.coefficients {
                        return Err(ResidentOodsError::ColumnBindingMismatch { index });
                    }
                    QuotientNumeratorColumnSource::Coefficients(bind_exact(
                        workspace,
                        column.coefficients,
                        words_for_log(column.topology.coefficient_log_size)?,
                        "quotient numerator coefficient column",
                    )?)
                }
                QuotientNumeratorSourceKind::Evaluation => {
                    QuotientNumeratorColumnSource::Evaluation(bind_exact(
                        workspace,
                        column.numerator_source,
                        words_for_log(expected_evaluation_log)?,
                        "quotient numerator retained evaluation column",
                    )?)
                }
            };
            numerator_columns.push(QuotientNumeratorColumn {
                coefficient_log_size: column.topology.coefficient_log_size,
                source,
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
        let oods_sample_points = bind_logical(workspace, numerator_plan.oods_sample_points)?;
        let oods_sampled_values = bind_logical(workspace, numerator_plan.oods_sampled_values)?;
        let random_coefficient = bind_exact(
            workspace,
            numerator_plan.random_coefficient,
            4,
            "quotient random coefficient",
        )?;
        let sample_points_destination =
            bind_logical(workspace, numerator_plan.sample_points_destination)?;
        let first_linear_terms_destination =
            bind_logical(workspace, numerator_plan.first_linear_terms_destination)?;
        let forward_twiddles = bind_logical(workspace, numerator_plan.forward_twiddles)?;
        if (numerator_plan.schedule == QuotientNumeratorSchedule::StagedPackedSingleWrite)
            != numerator_plan.staged_single_write.is_some()
        {
            return Err(ResidentOodsError::StagedNumeratorBinding(
                "planned schedule and staged quotient manifest presence differ",
            ));
        }
        let numerator = match numerator_plan.schedule {
            QuotientNumeratorSchedule::LegacyBatches => PreparedQuotientNumeratorGraph::prepare(
                arena,
                numerator_plan.config,
                &numerator_columns,
                oods_sample_points,
                oods_sampled_values,
                random_coefficient,
                sample_points_destination,
                first_linear_terms_destination,
                &destinations,
                forward_twiddles,
                &numerator_plan.slots,
            )?,
            QuotientNumeratorSchedule::HybridSingleWrite => {
                PreparedQuotientNumeratorGraph::prepare_hybrid_candidate(
                    arena,
                    numerator_plan.config,
                    &numerator_columns,
                    oods_sample_points,
                    oods_sampled_values,
                    random_coefficient,
                    sample_points_destination,
                    first_linear_terms_destination,
                    &destinations,
                    forward_twiddles,
                    &numerator_plan.slots,
                )?
            }
            QuotientNumeratorSchedule::StagedPackedSingleWrite => {
                let staged = numerator_plan.staged_single_write.as_ref().ok_or(
                    ResidentOodsError::StagedNumeratorBinding(
                        "replacement schedule has no staged quotient manifest",
                    ),
                )?;
                if staged.requirements() != &numerator_plan.requirements {
                    return Err(ResidentOodsError::StagedNumeratorBinding(
                        "staged quotient manifest differs from the arena requirements",
                    ));
                }
                let rebound_topology = numerator_plan
                    .columns
                    .iter()
                    .map(|column| column.topology.clone())
                    .collect::<Vec<_>>();
                let rebound_capacities = numerator_plan
                    .staged_overflows
                    .iter()
                    .map(|role| role.staging.len_words)
                    .collect::<Vec<_>>();
                let rebound = quotient_numerator_staged_single_write_plan_with_overflow_capacities(
                    numerator_plan.config,
                    &rebound_topology,
                    &rebound_capacities,
                )?;
                if &rebound != staged {
                    return Err(ResidentOodsError::StagedNumeratorBinding(
                        "runtime arena roles recompile to a different staged quotient program",
                    ));
                }
                let role_words = staged.overflow_role_words();
                if role_words.len() != numerator_plan.staged_overflows.len() {
                    return Err(ResidentOodsError::StagedNumeratorBinding(
                        "staged quotient manifest and arena role counts differ",
                    ));
                }
                let overflow_roles = numerator_plan
                    .staged_overflows
                    .iter()
                    .zip(role_words)
                    .map(|(role, required_words)| {
                        if role.used_words != required_words
                            || role.used_words > role.staging.len_words
                            || role.released_slab.physical != role.staging.physical
                            || role.released_slab.len_words != role.staging.len_words
                        {
                            return Err(ResidentOodsError::StagedNumeratorBinding(
                                "staged quotient role is not its exact released arena slab",
                            ));
                        }
                        bind_exact(
                            workspace,
                            role.staging,
                            role.staging.len_words,
                            "staged quotient overflow role",
                        )
                    })
                    .collect::<Result<Vec<_>, ResidentOodsError>>()?;
                PreparedQuotientNumeratorGraph::prepare_staged_packed_single_write(
                    arena,
                    numerator_plan.config,
                    &numerator_columns,
                    oods_sample_points,
                    oods_sampled_values,
                    random_coefficient,
                    sample_points_destination,
                    first_linear_terms_destination,
                    &destinations,
                    forward_twiddles,
                    &numerator_plan.slots,
                    &overflow_roles,
                )?
            }
        };
        let schedule_matches = matches!(
            (numerator_plan.schedule, numerator.schedule()),
            (
                QuotientNumeratorSchedule::LegacyBatches,
                PreparedNumeratorSchedule::LegacyBatches
            ) | (
                QuotientNumeratorSchedule::HybridSingleWrite,
                PreparedNumeratorSchedule::HybridCandidate { .. }
            ) | (
                QuotientNumeratorSchedule::StagedPackedSingleWrite,
                PreparedNumeratorSchedule::StagedPackedSingleWrite { .. }
            )
        );
        if let (
            Some(staged),
            PreparedNumeratorSchedule::StagedPackedSingleWrite { packed_output_rows },
        ) = (&numerator_plan.staged_single_write, numerator.schedule())
        {
            if packed_output_rows != staged.packed_output_rows() {
                return Err(ResidentOodsError::StagedNumeratorBinding(
                    "prepared packed row count differs from the sealed arena manifest",
                ));
            }
        }
        if !schedule_matches {
            return Err(ResidentOodsError::NumeratorScheduleMismatch {
                planned: numerator_plan.schedule,
                actual: numerator.schedule(),
            });
        }

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

    pub(crate) fn numerator_schedule(&self) -> PreparedNumeratorSchedule {
        self.numerator.schedule()
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
    // The physical slot may be pooled larger than this logical buffer; expose
    // only the logical extent so downstream sizes never see the surplus.
    Ok(slice.truncated(binding.len_words))
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
