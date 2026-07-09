//! Typed relation-lookup inputs captured after base witness generation.
//!
//! The interaction layer consumes these sources in logical word-major order.
//! Device producers keep ownership on the device: a source is either one
//! contiguous word-major buffer or a projection of existing columns/constants.

use std::collections::HashSet;
use std::fmt;

use stwo::prover::backend::simd::m31::{PackedM31, N_LANES};
use stwo_backend_cuda::BaseFieldVec;

use super::exec_context::{FinalShapeError, WitnessExecContext};
use super::proof_shape::{ComponentId, TracePartId, TracePartShape};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RelationSourceId {
    pub component: ComponentId,
    pub part: TracePartId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelationSourceEncoding {
    /// Every word is already present, including relation ids and multiplicities.
    LookupWords,
    /// Interleaved id/multiplicity columns; addresses are the implicit sequence.
    MemoryAddressToId { chunks: u32 },
    /// Limb columns followed by multiplicity; ids start at `id_offset`.
    MemoryIdToBig { id_offset: u32 },
    /// Small-memory limbs followed by multiplicity.
    MemoryIdToSmall,
    /// Expanded xor-12 multiplicity columns; enumeration values are implicit.
    BitwiseXor12 { columns: u32 },
}

/// One logical device-resident word. `Column` owns the existing allocation;
/// `Constant` is materialized or fused by the eventual device consumer.
#[derive(Debug)]
pub enum DeviceRelationWord {
    Column(BaseFieldVec),
    Constant(u32),
}

/// Transport is explicit so a consumer cannot accidentally hide a D2H copy.
#[derive(Debug)]
pub enum RelationLookupTransfer {
    HostWordMajor(Vec<u32>),
    DeviceWordMajor(BaseFieldVec),
    DeviceProjectedWords(Vec<DeviceRelationWord>),
}

#[derive(Debug)]
pub struct RelationLookupSource {
    pub id: RelationSourceId,
    pub encoding: RelationSourceEncoding,
    pub n_real_rows: u64,
    pub padded_rows: u64,
    pub words_per_row: usize,
    pub transfer: RelationLookupTransfer,
}

impl RelationLookupSource {
    pub fn new(
        id: RelationSourceId,
        encoding: RelationSourceEncoding,
        shape: TracePartShape,
        words_per_row: usize,
        transfer: RelationLookupTransfer,
    ) -> Result<Self, RelationSourceError> {
        if shape.part != id.part {
            return Err(RelationSourceError::PartMismatch {
                id,
                shape_part: shape.part,
            });
        }
        if words_per_row == 0 {
            return Err(RelationSourceError::ZeroWords(id));
        }
        if shape.n_real_rows > shape.padded_rows {
            return Err(RelationSourceError::RowsExceedPadding {
                id,
                n_real_rows: shape.n_real_rows,
                padded_rows: shape.padded_rows,
            });
        }
        let padded_rows = usize::try_from(shape.padded_rows)
            .map_err(|_| RelationSourceError::RowCountOverflow(id))?;
        let expected_len = padded_rows
            .checked_mul(words_per_row)
            .ok_or(RelationSourceError::WordCountOverflow(id))?;
        match &transfer {
            RelationLookupTransfer::HostWordMajor(words) => {
                if words.len() != expected_len {
                    return Err(RelationSourceError::FlatLength {
                        id,
                        expected: expected_len,
                        actual: words.len(),
                    });
                }
            }
            RelationLookupTransfer::DeviceWordMajor(words) => {
                if words.size != expected_len {
                    return Err(RelationSourceError::FlatLength {
                        id,
                        expected: expected_len,
                        actual: words.size,
                    });
                }
            }
            RelationLookupTransfer::DeviceProjectedWords(words) => {
                if words.len() != words_per_row {
                    return Err(RelationSourceError::ProjectedWordCount {
                        id,
                        expected: words_per_row,
                        actual: words.len(),
                    });
                }
                for (word, source) in words.iter().enumerate() {
                    if let DeviceRelationWord::Column(column) = source {
                        if column.size != padded_rows {
                            return Err(RelationSourceError::ProjectedColumnLength {
                                id,
                                word,
                                expected: padded_rows,
                                actual: column.size,
                            });
                        }
                    }
                }
            }
        }
        Ok(Self {
            id,
            encoding,
            n_real_rows: shape.n_real_rows,
            padded_rows: shape.padded_rows,
            words_per_row,
            transfer,
        })
    }
}

/// Validated aggregate returned by the generated Cairo interaction state.
#[derive(Debug)]
pub struct CairoRelationSourceSet {
    sources: Vec<RelationLookupSource>,
}

impl CairoRelationSourceSet {
    pub fn new(sources: Vec<RelationLookupSource>) -> Result<Self, RelationSourceError> {
        let mut ids = HashSet::with_capacity(sources.len());
        for source in &sources {
            if !ids.insert(source.id) {
                return Err(RelationSourceError::DuplicateSource(source.id));
            }
        }
        Ok(Self { sources })
    }

    pub fn as_slice(&self) -> &[RelationLookupSource] {
        &self.sources
    }

    pub fn into_vec(self) -> Vec<RelationLookupSource> {
        self.sources
    }

    pub fn get(&self, component: &str, part: TracePartId) -> Option<&RelationLookupSource> {
        self.sources
            .iter()
            .find(|source| source.id.component == component && source.id.part == part)
    }
}

/// Implemented by every generated component interaction state, including the
/// two backend-specific states. The generated aggregate invokes only this API.
pub trait RelationLookupSourceExport: Send {
    fn export_relation_lookup_sources(
        self,
        component: ComponentId,
        exec_context: &WitnessExecContext,
    ) -> Result<Vec<RelationLookupSource>, RelationSourceError>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RelationSourceError {
    FinalShape(FinalShapeError),
    DuplicateSource(RelationSourceId),
    MissingMemoryPart {
        component: ComponentId,
        expected_parts: usize,
        actual_parts: usize,
    },
    PartMismatch {
        id: RelationSourceId,
        shape_part: TracePartId,
    },
    RowCountOverflow(RelationSourceId),
    WordCountOverflow(RelationSourceId),
    ZeroWords(RelationSourceId),
    RowsExceedPadding {
        id: RelationSourceId,
        n_real_rows: u64,
        padded_rows: u64,
    },
    FieldWidth {
        id: RelationSourceId,
        expected: usize,
        actual: usize,
    },
    FieldRows {
        id: RelationSourceId,
        expected: usize,
        actual: usize,
    },
    FlatLength {
        id: RelationSourceId,
        expected: usize,
        actual: usize,
    },
    ProjectedWordCount {
        id: RelationSourceId,
        expected: usize,
        actual: usize,
    },
    DeviceColumnCount {
        id: RelationSourceId,
        expected: usize,
        actual: usize,
    },
    ProjectedColumnLength {
        id: RelationSourceId,
        word: usize,
        expected: usize,
        actual: usize,
    },
}

impl From<FinalShapeError> for RelationSourceError {
    fn from(value: FinalShapeError) -> Self {
        Self::FinalShape(value)
    }
}

impl fmt::Display for RelationSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RelationSourceError {}

pub(crate) trait PackedRelationField {
    fn width(&self) -> usize;
    fn packed_rows(&self) -> usize;
    fn append_word_major(&self, words: &mut Vec<u32>);
}

impl PackedRelationField for Vec<PackedM31> {
    fn width(&self) -> usize {
        1
    }

    fn packed_rows(&self) -> usize {
        self.len()
    }

    fn append_word_major(&self, words: &mut Vec<u32>) {
        for packed in self {
            words.extend(packed.to_array().map(|value| value.0));
        }
    }
}

impl<const N: usize> PackedRelationField for Vec<[PackedM31; N]> {
    fn width(&self) -> usize {
        N
    }

    fn packed_rows(&self) -> usize {
        self.len()
    }

    fn append_word_major(&self, words: &mut Vec<u32>) {
        for word in 0..N {
            for packed_row in self {
                words.extend(packed_row[word].to_array().map(|value| value.0));
            }
        }
    }
}

pub(crate) struct HostWordMajorBuilder {
    id: RelationSourceId,
    shape: TracePartShape,
    padded_rows: usize,
    words_per_row: usize,
    words: Vec<u32>,
}

impl HostWordMajorBuilder {
    pub(crate) fn new(
        component: ComponentId,
        shape: TracePartShape,
    ) -> Result<Self, RelationSourceError> {
        let id = RelationSourceId {
            component,
            part: shape.part,
        };
        let padded_rows = usize::try_from(shape.padded_rows)
            .map_err(|_| RelationSourceError::RowCountOverflow(id))?;
        Ok(Self {
            id,
            shape,
            padded_rows,
            words_per_row: 0,
            words: Vec::new(),
        })
    }

    pub(crate) fn push<T: PackedRelationField>(
        &mut self,
        field: &T,
        expected_width: usize,
    ) -> Result<(), RelationSourceError> {
        if field.width() != expected_width {
            return Err(RelationSourceError::FieldWidth {
                id: self.id,
                expected: expected_width,
                actual: field.width(),
            });
        }
        let actual_rows = field
            .packed_rows()
            .checked_mul(N_LANES)
            .ok_or(RelationSourceError::RowCountOverflow(self.id))?;
        if actual_rows != self.padded_rows {
            return Err(RelationSourceError::FieldRows {
                id: self.id,
                expected: self.padded_rows,
                actual: actual_rows,
            });
        }
        let additional = self
            .padded_rows
            .checked_mul(expected_width)
            .ok_or(RelationSourceError::WordCountOverflow(self.id))?;
        self.words.reserve(additional);
        field.append_word_major(&mut self.words);
        self.words_per_row += expected_width;
        Ok(())
    }

    pub(crate) fn finish(
        self,
        encoding: RelationSourceEncoding,
    ) -> Result<RelationLookupSource, RelationSourceError> {
        RelationLookupSource::new(
            self.id,
            encoding,
            self.shape,
            self.words_per_row,
            RelationLookupTransfer::HostWordMajor(self.words),
        )
    }
}

/// Standard AIR-generated `LookupData` exporter. The post-processor emits one
/// invocation per component and derives widths from the generated field types.
#[macro_export]
macro_rules! relation_lookup_source {
    ($( $field:ident : $width:tt ),+ $(,)?) => {
        impl $crate::witness::relation_sources::RelationLookupSourceExport
            for InteractionClaimGenerator
        {
            fn export_relation_lookup_sources(
                self,
                component: $crate::witness::proof_shape::ComponentId,
                exec_context: &$crate::witness::exec_context::WitnessExecContext,
            ) -> Result<
                Vec<$crate::witness::relation_sources::RelationLookupSource>,
                $crate::witness::relation_sources::RelationSourceError,
            > {
                use $crate::witness::relation_sources::{
                    HostWordMajorBuilder, RelationLookupSource, RelationLookupTransfer,
                    RelationSourceEncoding, RelationSourceId,
                };
                let shape = exec_context.exact_relation_part(
                    component,
                    $crate::witness::proof_shape::TracePartId::Main,
                )?;
                let words_per_row = 0usize $(
                    + $crate::relation_lookup_source!(@width $width)
                )+;
                if let Some(device) = exec_context.take_device_lookup(component) {
                    let id = RelationSourceId {
                        component,
                        part: shape.part,
                    };
                    let expected_rows = usize::try_from(shape.padded_rows).map_err(|_| {
                        $crate::witness::relation_sources::RelationSourceError::RowCountOverflow(id)
                    })?;
                    if device.n_rows != expected_rows {
                        return Err(
                            $crate::witness::relation_sources::RelationSourceError::FieldRows {
                                id,
                                expected: expected_rows,
                                actual: device.n_rows,
                            },
                        );
                    }
                    let _device_reported_real_rows = device.n_real;
                    let source = RelationLookupSource::new(
                        id,
                        RelationSourceEncoding::LookupWords,
                        shape,
                        words_per_row,
                        RelationLookupTransfer::DeviceWordMajor(device.buffer),
                    )?;
                    return Ok(vec![source]);
                }

                let mut builder = HostWordMajorBuilder::new(component, shape)?;
                $(
                    builder.push(
                        &self.lookup_data.$field,
                        $crate::relation_lookup_source!(@width $width),
                    )?;
                )+
                Ok(vec![builder.finish(RelationSourceEncoding::LookupWords)?])
            }
        }
    };
    (@width scalar) => { 1usize };
    (@width $width:literal) => { $width as usize };
}

#[macro_export]
macro_rules! relation_lookup_source_memory_address_to_id {
    () => {
        impl $crate::witness::relation_sources::RelationLookupSourceExport
            for InteractionClaimGenerator
        {
            fn export_relation_lookup_sources(
                self,
                component: $crate::witness::proof_shape::ComponentId,
                exec_context: &$crate::witness::exec_context::WitnessExecContext,
            ) -> Result<
                Vec<$crate::witness::relation_sources::RelationLookupSource>,
                $crate::witness::relation_sources::RelationSourceError,
            > {
                use $crate::witness::relation_sources::{
                    HostWordMajorBuilder, RelationSourceEncoding,
                };
                let shape = exec_context.exact_relation_part(
                    component,
                    $crate::witness::proof_shape::TracePartId::Main,
                )?;
                let mut builder = HostWordMajorBuilder::new(component, shape)?;
                for (ids, multiplicities) in self.ids.iter().zip(&self.multiplicities) {
                    builder.push(ids, 1)?;
                    builder.push(multiplicities, 1)?;
                }
                Ok(vec![builder.finish(
                    RelationSourceEncoding::MemoryAddressToId {
                        chunks: self.ids.len() as u32,
                    },
                )?])
            }
        }
    };
}

#[macro_export]
macro_rules! relation_lookup_source_memory_id_to_big {
    () => {
        impl $crate::witness::relation_sources::RelationLookupSourceExport
            for InteractionClaimGenerator
        {
            fn export_relation_lookup_sources(
                self,
                component: $crate::witness::proof_shape::ComponentId,
                exec_context: &$crate::witness::exec_context::WitnessExecContext,
            ) -> Result<
                Vec<$crate::witness::relation_sources::RelationLookupSource>,
                $crate::witness::relation_sources::RelationSourceError,
            > {
                use $crate::witness::relation_sources::{
                    HostWordMajorBuilder, RelationSourceEncoding, RelationSourceError,
                };
                let InteractionClaimGenerator {
                    big_components_values,
                    big_multiplicities,
                    small_values,
                    small_multiplicities,
                } = self;
                if big_components_values.len() != big_multiplicities.len() {
                    return Err(RelationSourceError::MissingMemoryPart {
                        component,
                        expected_parts: big_components_values.len(),
                        actual_parts: big_multiplicities.len(),
                    });
                }
                let mut sources = Vec::with_capacity(big_components_values.len() + 1);
                let mut id_offset = 0u32;
                for (index, (values, multiplicities)) in big_components_values
                    .into_iter()
                    .zip(big_multiplicities)
                    .enumerate()
                {
                    let shape = exec_context.exact_relation_part(
                        component,
                        $crate::witness::proof_shape::TracePartId::MemoryBig(index as u32),
                    )?;
                    let mut builder = HostWordMajorBuilder::new(component, shape)?;
                    for values in &values {
                        builder.push(values, 1)?;
                    }
                    builder.push(&multiplicities, 1)?;
                    sources
                        .push(builder.finish(RelationSourceEncoding::MemoryIdToBig { id_offset })?);
                    id_offset = id_offset
                        .checked_add(u32::try_from(shape.padded_rows).map_err(|_| {
                            RelationSourceError::RowCountOverflow(
                                $crate::witness::relation_sources::RelationSourceId {
                                    component,
                                    part: shape.part,
                                },
                            )
                        })?)
                        .ok_or(RelationSourceError::RowCountOverflow(
                            $crate::witness::relation_sources::RelationSourceId {
                                component,
                                part: shape.part,
                            },
                        ))?;
                }
                let shape = exec_context.exact_relation_part(
                    component,
                    $crate::witness::proof_shape::TracePartId::MemorySmall,
                )?;
                let mut builder = HostWordMajorBuilder::new(component, shape)?;
                for values in &small_values {
                    builder.push(values, 1)?;
                }
                builder.push(&small_multiplicities, 1)?;
                sources.push(builder.finish(RelationSourceEncoding::MemoryIdToSmall)?);
                Ok(sources)
            }
        }
    };
}

#[macro_export]
macro_rules! relation_lookup_source_xor12 {
    () => {
        impl $crate::witness::relation_sources::RelationLookupSourceExport
            for InteractionClaimGenerator
        {
            fn export_relation_lookup_sources(
                self,
                component: $crate::witness::proof_shape::ComponentId,
                exec_context: &$crate::witness::exec_context::WitnessExecContext,
            ) -> Result<
                Vec<$crate::witness::relation_sources::RelationLookupSource>,
                $crate::witness::relation_sources::RelationSourceError,
            > {
                use $crate::witness::relation_sources::{
                    HostWordMajorBuilder, RelationSourceEncoding,
                };
                let shape = exec_context.exact_relation_part(
                    component,
                    $crate::witness::proof_shape::TracePartId::Main,
                )?;
                let mut builder = HostWordMajorBuilder::new(component, shape)?;
                for multiplicities in &self.lookup_data.mults {
                    builder.push(multiplicities, 1)?;
                }
                Ok(vec![builder.finish(
                    RelationSourceEncoding::BitwiseXor12 {
                        columns: self.lookup_data.mults.len() as u32,
                    },
                )?])
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use stwo::core::fields::m31::M31;

    use super::*;

    #[test]
    fn host_builder_transposes_packed_rows_to_word_major() {
        let field: Vec<[PackedM31; 2]> = (0..2)
            .map(|packed_row| {
                std::array::from_fn(|word| {
                    PackedM31::from_array(std::array::from_fn(|lane| {
                        M31::from_u32_unchecked((word * 1_000 + packed_row * N_LANES + lane) as u32)
                    }))
                })
            })
            .collect();
        let shape = TracePartShape {
            part: TracePartId::Main,
            n_real_rows: 19,
            padded_rows: 2 * N_LANES as u64,
        };
        let mut builder = HostWordMajorBuilder::new("component", shape).unwrap();
        builder.push(&field, 2).unwrap();
        let source = builder.finish(RelationSourceEncoding::LookupWords).unwrap();
        let RelationLookupTransfer::HostWordMajor(words) = source.transfer else {
            panic!("expected host words")
        };
        assert_eq!(
            words[..2 * N_LANES],
            (0..2 * N_LANES as u32).collect::<Vec<_>>()
        );
        assert_eq!(
            words[2 * N_LANES..],
            (1_000..1_000 + 2 * N_LANES as u32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn device_source_validation_does_not_copy_to_host() {
        let shape = TracePartShape {
            part: TracePartId::Main,
            n_real_rows: 9,
            padded_rows: 16,
        };
        let ptr = std::ptr::without_provenance::<u32>(0x1000);
        let source = RelationLookupSource::new(
            RelationSourceId {
                component: "component",
                part: TracePartId::Main,
            },
            RelationSourceEncoding::LookupWords,
            shape,
            3,
            RelationLookupTransfer::DeviceWordMajor(BaseFieldVec::from_borrowed_ptr(ptr, 48)),
        )
        .unwrap();
        let RelationLookupTransfer::DeviceWordMajor(buffer) = source.transfer else {
            panic!("expected device words")
        };
        assert_eq!(buffer.device_ptr, ptr);
        assert!(!buffer.owns_memory);
    }

    #[test]
    fn aggregate_rejects_duplicate_component_parts() {
        let shape = TracePartShape {
            part: TracePartId::Main,
            n_real_rows: 16,
            padded_rows: 16,
        };
        let make = || {
            RelationLookupSource::new(
                RelationSourceId {
                    component: "component",
                    part: TracePartId::Main,
                },
                RelationSourceEncoding::LookupWords,
                shape,
                1,
                RelationLookupTransfer::HostWordMajor(vec![0; 16]),
            )
            .unwrap()
        };
        assert!(matches!(
            CairoRelationSourceSet::new(vec![make(), make()]),
            Err(RelationSourceError::DuplicateSource(_))
        ));
    }
}
