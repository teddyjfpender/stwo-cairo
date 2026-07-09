//! Per-proof component row geometry, captured before witness generators are consumed.
//!
//! This is deliberately independent of CUDA. It is the input to later GPU memory and
//! graph planning, and therefore distinguishes exact row counts from components whose
//! final rows are still produced by witness-side relation feeds. An unresolved shape is
//! valid to carry through the legacy pipeline, but [`ProofShape::require_arena_ready`]
//! fails closed rather than turning an observed pre-witness count into an allocation fact.

use std::fmt;

pub type ComponentId = &'static str;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TracePartId {
    Main,
    MemoryBig(u32),
    MemorySmall,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TracePartShape {
    pub part: TracePartId,
    pub n_real_rows: u64,
    pub padded_rows: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PendingRowsReason {
    /// The generator is a relation target. Its inputs/multiplicity keys are added
    /// while upstream witness components run, so its current length is not final.
    WitnessRelationFeeds,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CapacityBound {
    /// Rows already present when ingest snapshots the generator. Diagnostic only.
    pub observed_rows: u64,
    /// Checked upper bound from generated producer/feed multiplicities.
    pub max_rows: u64,
    /// Power-of-two allocation capacity used by the witness writer.
    pub padded_capacity: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RowResolution {
    Absent,
    Resolved(Vec<TracePartShape>),
    Pending {
        reason: PendingRowsReason,
        /// Diagnostic only. This is never accepted as an arena capacity.
        observed_n_real_rows: u64,
    },
    Bounded {
        reason: PendingRowsReason,
        bound: CapacityBound,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RuntimeComponentShape {
    pub id: ComponentId,
    pub rows: RowResolution,
}

impl RuntimeComponentShape {
    pub const fn absent(id: ComponentId) -> Self {
        Self {
            id,
            rows: RowResolution::Absent,
        }
    }

    pub fn uniform(
        id: ComponentId,
        n_real_rows: u64,
        padded_rows: u64,
    ) -> Result<Self, ProofShapeError> {
        Self::parts(
            id,
            vec![TracePartShape {
                part: TracePartId::Main,
                n_real_rows,
                padded_rows,
            }],
        )
    }

    pub fn parts(id: ComponentId, parts: Vec<TracePartShape>) -> Result<Self, ProofShapeError> {
        validate_parts(id, &parts)?;
        Ok(Self {
            id,
            rows: RowResolution::Resolved(parts),
        })
    }

    pub const fn pending(
        id: ComponentId,
        reason: PendingRowsReason,
        observed_n_real_rows: u64,
    ) -> Self {
        Self {
            id,
            rows: RowResolution::Pending {
                reason,
                observed_n_real_rows,
            },
        }
    }

    pub const fn bounded(id: ComponentId, reason: PendingRowsReason, bound: CapacityBound) -> Self {
        Self {
            id,
            rows: RowResolution::Bounded { reason, bound },
        }
    }

    pub fn is_present(&self) -> bool {
        !matches!(self.rows, RowResolution::Absent)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProofShapeKey(pub u64);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofShape {
    components: Vec<RuntimeComponentShape>,
    key: ProofShapeKey,
}

impl ProofShape {
    pub fn new(mut components: Vec<RuntimeComponentShape>) -> Result<Self, ProofShapeError> {
        components.sort_unstable_by_key(|component| component.id);
        for pair in components.windows(2) {
            if pair[0].id == pair[1].id {
                return Err(ProofShapeError::DuplicateComponent(pair[0].id));
            }
        }
        for component in &components {
            match &component.rows {
                RowResolution::Resolved(parts) => validate_parts(component.id, parts)?,
                RowResolution::Bounded { bound, .. } => {
                    validate_capacity_bound(component.id, *bound)?
                }
                RowResolution::Absent | RowResolution::Pending { .. } => {}
            }
        }
        let key = stable_shape_key(&components);
        Ok(Self { components, key })
    }

    pub fn components(&self) -> &[RuntimeComponentShape] {
        &self.components
    }

    pub const fn key(&self) -> ProofShapeKey {
        self.key
    }

    pub fn component(&self, id: &str) -> Option<&RuntimeComponentShape> {
        self.components
            .binary_search_by_key(&id, |component| component.id)
            .ok()
            .map(|index| &self.components[index])
    }

    /// Arena/graph planning may only consume exact final row counts.
    pub fn require_arena_ready(&self) -> Result<(), ProofShapeError> {
        if let Some(component) = self
            .components
            .iter()
            .find(|component| matches!(component.rows, RowResolution::Pending { .. }))
        {
            return Err(ProofShapeError::PendingComponentRows(component.id));
        }
        Ok(())
    }

    /// CUDA graph capture requires exact final row counts, not only safe capacity.
    pub fn require_capture_ready(&self) -> Result<(), ProofShapeError> {
        if let Some(component) = self.components.iter().find(|component| {
            matches!(
                component.rows,
                RowResolution::Pending { .. } | RowResolution::Bounded { .. }
            )
        }) {
            return match component.rows {
                RowResolution::Pending { .. } => {
                    Err(ProofShapeError::PendingComponentRows(component.id))
                }
                RowResolution::Bounded { .. } => {
                    Err(ProofShapeError::BoundedComponentRows(component.id))
                }
                _ => unreachable!(),
            };
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProofShapeError {
    DuplicateComponent(ComponentId),
    EmptyResolvedComponent(ComponentId),
    DuplicateTracePart {
        component: ComponentId,
        part: TracePartId,
    },
    ZeroRows(ComponentId),
    RowsExceedPadding {
        component: ComponentId,
        n_real_rows: u64,
        padded_rows: u64,
    },
    ObservedRowsExceedBound {
        component: ComponentId,
        observed_rows: u64,
        max_rows: u64,
    },
    CapacityExceedsPadding {
        component: ComponentId,
        max_rows: u64,
        padded_capacity: u64,
    },
    NonPowerOfTwoPadding {
        component: ComponentId,
        padded_rows: u64,
    },
    LogSizeOverflow {
        component: ComponentId,
        log_size: u32,
    },
    RowCountOverflow(ComponentId),
    PendingComponentRows(ComponentId),
    BoundedComponentRows(ComponentId),
    InvalidMemoryComponentCount {
        requested: usize,
        required: usize,
    },
}

impl fmt::Display for ProofShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ProofShapeError {}

pub fn rows_from_log_size(component: ComponentId, log_size: u32) -> Result<u64, ProofShapeError> {
    1u64.checked_shl(log_size)
        .ok_or(ProofShapeError::LogSizeOverflow {
            component,
            log_size,
        })
}

pub fn padded_rows(
    component: ComponentId,
    n_real_rows: u64,
    min_rows: u64,
) -> Result<u64, ProofShapeError> {
    if n_real_rows == 0 {
        return Err(ProofShapeError::ZeroRows(component));
    }
    Ok(n_real_rows.next_power_of_two().max(min_rows))
}

fn validate_parts(component: ComponentId, parts: &[TracePartShape]) -> Result<(), ProofShapeError> {
    if parts.is_empty() {
        return Err(ProofShapeError::EmptyResolvedComponent(component));
    }
    for (index, part) in parts.iter().enumerate() {
        if parts[..index].iter().any(|other| other.part == part.part) {
            return Err(ProofShapeError::DuplicateTracePart {
                component,
                part: part.part,
            });
        }
        if part.n_real_rows == 0 || part.padded_rows == 0 {
            return Err(ProofShapeError::ZeroRows(component));
        }
        if part.n_real_rows > part.padded_rows {
            return Err(ProofShapeError::RowsExceedPadding {
                component,
                n_real_rows: part.n_real_rows,
                padded_rows: part.padded_rows,
            });
        }
        if !part.padded_rows.is_power_of_two() {
            return Err(ProofShapeError::NonPowerOfTwoPadding {
                component,
                padded_rows: part.padded_rows,
            });
        }
    }
    Ok(())
}

fn validate_capacity_bound(
    component: ComponentId,
    bound: CapacityBound,
) -> Result<(), ProofShapeError> {
    if bound.max_rows == 0 || bound.padded_capacity == 0 {
        return Err(ProofShapeError::ZeroRows(component));
    }
    if bound.observed_rows > bound.max_rows {
        return Err(ProofShapeError::ObservedRowsExceedBound {
            component,
            observed_rows: bound.observed_rows,
            max_rows: bound.max_rows,
        });
    }
    if bound.max_rows > bound.padded_capacity {
        return Err(ProofShapeError::CapacityExceedsPadding {
            component,
            max_rows: bound.max_rows,
            padded_capacity: bound.padded_capacity,
        });
    }
    if !bound.padded_capacity.is_power_of_two() {
        return Err(ProofShapeError::NonPowerOfTwoPadding {
            component,
            padded_rows: bound.padded_capacity,
        });
    }
    Ok(())
}

fn stable_shape_key(components: &[RuntimeComponentShape]) -> ProofShapeKey {
    // FNV-1a is intentionally local and versionable: unlike DefaultHasher it has
    // stable bytes across processes/toolchains, which a graph-cache key requires.
    let mut hash = 0xcbf29ce484222325u64;
    let mut feed = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    };
    feed(b"stwo-cairo-proof-shape-v1\0");
    for component in components {
        feed(component.id.as_bytes());
        feed(&[0]);
        match &component.rows {
            RowResolution::Absent => feed(&[0]),
            RowResolution::Resolved(parts) => {
                feed(&[1]);
                feed(&(parts.len() as u64).to_le_bytes());
                for part in parts {
                    match part.part {
                        TracePartId::Main => feed(&[0]),
                        TracePartId::MemoryBig(index) => {
                            feed(&[1]);
                            feed(&index.to_le_bytes());
                        }
                        TracePartId::MemorySmall => feed(&[2]),
                    }
                    feed(&part.n_real_rows.to_le_bytes());
                    feed(&part.padded_rows.to_le_bytes());
                }
            }
            RowResolution::Pending {
                reason,
                observed_n_real_rows,
            } => {
                feed(&[2]);
                match reason {
                    PendingRowsReason::WitnessRelationFeeds => feed(&[0]),
                }
                feed(&observed_n_real_rows.to_le_bytes());
            }
            RowResolution::Bounded { reason, bound } => {
                feed(&[3]);
                match reason {
                    PendingRowsReason::WitnessRelationFeeds => feed(&[0]),
                }
                feed(&bound.observed_rows.to_le_bytes());
                feed(&bound.max_rows.to_le_bytes());
                feed(&bound.padded_capacity.to_le_bytes());
            }
        }
    }
    ProofShapeKey(hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::witness::cairo_claim_generator::CairoClaimGenerator;

    #[test]
    fn shape_key_is_order_independent_and_geometry_sensitive() {
        let a = RuntimeComponentShape::uniform("a", 9, 16).unwrap();
        let b = RuntimeComponentShape::absent("b");
        let first = ProofShape::new(vec![a.clone(), b.clone()]).unwrap();
        let reordered = ProofShape::new(vec![b, a]).unwrap();
        assert_eq!(first.key(), reordered.key());

        let changed = ProofShape::new(vec![
            RuntimeComponentShape::uniform("a", 9, 32).unwrap(),
            RuntimeComponentShape::absent("b"),
        ])
        .unwrap();
        assert_ne!(first.key(), changed.key());
    }

    #[test]
    fn pending_rows_fail_only_the_arena_readiness_gate() {
        let shape = ProofShape::new(vec![RuntimeComponentShape::pending(
            "consumer",
            PendingRowsReason::WitnessRelationFeeds,
            0,
        )])
        .unwrap();
        assert_eq!(
            shape.require_arena_ready(),
            Err(ProofShapeError::PendingComponentRows("consumer"))
        );
    }

    #[test]
    fn capacity_bound_is_arena_ready_but_not_capture_ready() {
        let shape = ProofShape::new(vec![RuntimeComponentShape::bounded(
            "consumer",
            PendingRowsReason::WitnessRelationFeeds,
            CapacityBound {
                observed_rows: 16,
                max_rows: 48,
                padded_capacity: 64,
            },
        )])
        .unwrap();
        shape.require_arena_ready().unwrap();
        assert_eq!(
            shape.require_capture_ready(),
            Err(ProofShapeError::BoundedComponentRows("consumer"))
        );
    }

    #[test]
    fn generated_default_shape_is_complete_and_deterministic() {
        let first = CairoClaimGenerator::default().proof_shape(None).unwrap();
        let second = CairoClaimGenerator::default().proof_shape(None).unwrap();
        assert_eq!(first.components().len(), 67);
        assert_eq!(first.key(), second.key());
        first.require_capture_ready().unwrap();
    }
}
