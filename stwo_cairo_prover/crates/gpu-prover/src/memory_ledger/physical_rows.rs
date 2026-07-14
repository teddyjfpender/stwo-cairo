use std::collections::BTreeMap;
use std::num::NonZeroUsize;

/// One disjoint non-arena physical accounting row.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum PhysicalAllocationId {
    PrimaryContextDriverBaseline,
    ModuleCodeAndGlobals,
    GraphAndEventMetadata,
    AllocatorPoolNetSlack,
    ProfilingOverhead,
    OperationalSafetyReserve,
}

impl PhysicalAllocationId {
    const ALL: [Self; 6] = [
        Self::PrimaryContextDriverBaseline,
        Self::ModuleCodeAndGlobals,
        Self::GraphAndEventMetadata,
        Self::AllocatorPoolNetSlack,
        Self::ProfilingOverhead,
        Self::OperationalSafetyReserve,
    ];

    const fn id(self) -> &'static str {
        match self {
            Self::PrimaryContextDriverBaseline => "primary_context_driver_baseline",
            Self::ModuleCodeAndGlobals => "module_code_and_globals",
            Self::GraphAndEventMetadata => "graph_and_event_metadata",
            Self::AllocatorPoolNetSlack => "allocator_pool_net_slack",
            Self::ProfilingOverhead => "profiling_overhead",
            Self::OperationalSafetyReserve => "operational_safety_reserve",
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::PrimaryContextDriverBaseline => "primary context and driver baseline",
            Self::ModuleCodeAndGlobals => "module code and globals",
            Self::GraphAndEventMetadata => "graph and event metadata",
            Self::AllocatorPoolNetSlack => "allocator pool slack",
            Self::ProfilingOverhead => "profiling overhead",
            Self::OperationalSafetyReserve => "operational safety reserve",
        }
    }

    const fn owner(self) -> PhysicalAllocationOwnerId {
        match self {
            Self::PrimaryContextDriverBaseline => PhysicalAllocationOwnerId::CudaPrimaryContext,
            Self::ModuleCodeAndGlobals => PhysicalAllocationOwnerId::ResidentModuleSet,
            Self::GraphAndEventMetadata => PhysicalAllocationOwnerId::ResidentGraphSet,
            Self::AllocatorPoolNetSlack => PhysicalAllocationOwnerId::CudaMemoryPools,
            Self::ProfilingOverhead => PhysicalAllocationOwnerId::Profiler,
            Self::OperationalSafetyReserve => PhysicalAllocationOwnerId::AdmissionPolicy,
        }
    }

    const fn permits_zero(self) -> bool {
        matches!(self, Self::AllocatorPoolNetSlack | Self::ProfilingOverhead)
    }
}

/// One synchronized native checkpoint over the two allocator pools that can
/// retain resident-prover allocations.
///
/// `attributed_bytes` are allocations already counted by another physical
/// ledger row. Net slack is therefore `reserved - attributed`, not
/// `reserved - used`: the latter would silently omit any live pool allocation
/// whose owner has not yet been classified.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocatorPoolCheckpoint {
    isolated_used_bytes: usize,
    isolated_reserved_bytes: usize,
    isolated_attributed_bytes: usize,
    default_used_bytes: usize,
    default_reserved_bytes: usize,
    default_attributed_bytes: usize,
    net_slack_bytes: usize,
}

impl AllocatorPoolCheckpoint {
    pub fn try_new(
        isolated_used_bytes: usize,
        isolated_reserved_bytes: usize,
        isolated_attributed_bytes: usize,
        default_used_bytes: usize,
        default_reserved_bytes: usize,
        default_attributed_bytes: usize,
    ) -> Result<Self, &'static str> {
        validate_pool(
            isolated_used_bytes,
            isolated_reserved_bytes,
            isolated_attributed_bytes,
        )?;
        validate_pool(
            default_used_bytes,
            default_reserved_bytes,
            default_attributed_bytes,
        )?;
        let net_slack_bytes = isolated_reserved_bytes
            .checked_sub(isolated_attributed_bytes)
            .and_then(|bytes| {
                default_reserved_bytes
                    .checked_sub(default_attributed_bytes)
                    .and_then(|default| bytes.checked_add(default))
            })
            .ok_or("allocator pool net slack overflow")?;
        Ok(Self {
            isolated_used_bytes,
            isolated_reserved_bytes,
            isolated_attributed_bytes,
            default_used_bytes,
            default_reserved_bytes,
            default_attributed_bytes,
            net_slack_bytes,
        })
    }

    pub const fn isolated_used_bytes(self) -> usize {
        self.isolated_used_bytes
    }

    pub const fn isolated_reserved_bytes(self) -> usize {
        self.isolated_reserved_bytes
    }

    pub const fn isolated_attributed_bytes(self) -> usize {
        self.isolated_attributed_bytes
    }

    pub const fn default_used_bytes(self) -> usize {
        self.default_used_bytes
    }

    pub const fn default_reserved_bytes(self) -> usize {
        self.default_reserved_bytes
    }

    pub const fn default_attributed_bytes(self) -> usize {
        self.default_attributed_bytes
    }

    pub const fn net_slack_bytes(self) -> usize {
        self.net_slack_bytes
    }
}

fn validate_pool(
    used_bytes: usize,
    reserved_bytes: usize,
    attributed_bytes: usize,
) -> Result<(), &'static str> {
    if used_bytes > reserved_bytes {
        return Err("allocator pool used bytes exceed reserved bytes");
    }
    if attributed_bytes > used_bytes {
        return Err("allocator pool attributed bytes exceed used bytes");
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhysicalAllocationOwnerId {
    CudaPrimaryContext,
    ResidentModuleSet,
    ResidentGraphSet,
    CudaMemoryPools,
    Profiler,
    AdmissionPolicy,
}

impl PhysicalAllocationOwnerId {
    const fn id(self) -> &'static str {
        match self {
            Self::CudaPrimaryContext => "cuda_primary_context",
            Self::ResidentModuleSet => "resident_module_set",
            Self::ResidentGraphSet => "resident_graph_set",
            Self::CudaMemoryPools => "cuda_memory_pools",
            Self::Profiler => "profiler",
            Self::AdmissionPolicy => "admission_policy",
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PhysicalMemoryInputs {
    rows: BTreeMap<PhysicalAllocationId, usize>,
}

pub(super) struct PhysicalAdmission {
    pub rows: Vec<serde_json::Value>,
    pub measured_bytes: usize,
    pub peak_bytes: Option<usize>,
    pub complete: bool,
    pub pass: bool,
    pub missing_ids: Vec<&'static str>,
    pub missing_rows: Vec<&'static str>,
}

impl PhysicalMemoryInputs {
    pub fn try_from_rows(
        rows: impl IntoIterator<Item = (PhysicalAllocationId, PhysicalAllocationOwnerId, usize)>,
    ) -> Result<Self, &'static str> {
        let mut validated = Self::default();
        for (allocation, owner, bytes) in rows {
            validated.insert(allocation, owner, bytes)?;
        }
        Ok(validated)
    }

    pub fn with_allocator_pool_checkpoint(
        mut self,
        checkpoint: AllocatorPoolCheckpoint,
    ) -> Result<Self, &'static str> {
        self.insert(
            PhysicalAllocationId::AllocatorPoolNetSlack,
            PhysicalAllocationOwnerId::CudaMemoryPools,
            checkpoint.net_slack_bytes(),
        )?;
        Ok(self)
    }

    pub fn with_operational_safety_reserve(
        mut self,
        bytes: NonZeroUsize,
    ) -> Result<Self, &'static str> {
        self.insert(
            PhysicalAllocationId::OperationalSafetyReserve,
            PhysicalAllocationOwnerId::AdmissionPolicy,
            bytes.get(),
        )?;
        Ok(self)
    }

    pub fn get(&self, id: PhysicalAllocationId) -> Option<usize> {
        self.rows.get(&id).copied()
    }

    pub fn rows_json(&self) -> Vec<serde_json::Value> {
        PhysicalAllocationId::ALL
            .into_iter()
            .map(|id| {
                let bytes = self.rows.get(&id).copied();
                let missing_status = match id {
                    PhysicalAllocationId::OperationalSafetyReserve => "missing_policy_value",
                    _ => "missing_native_measurement",
                };
                serde_json::json!({
                    "allocation_id": id.id(),
                    "owner_id": id.owner().id(),
                    "description": id.label(),
                    "bytes": bytes,
                    "status": if bytes.is_some() { "supplied" } else { missing_status },
                    "accounting_contract": match id {
                        PhysicalAllocationId::AllocatorPoolNetSlack =>
                            "reserved bytes minus allocations already attributed elsewhere in this ledger",
                        PhysicalAllocationId::OperationalSafetyReserve =>
                            "explicit deployment policy reserve; never inferred from remaining headroom",
                        _ => "disjoint native measurement",
                    },
                })
            })
            .collect()
    }

    pub fn missing_allocation_ids(&self) -> Vec<&'static str> {
        PhysicalAllocationId::ALL
            .into_iter()
            .filter(|id| !self.rows.contains_key(id))
            .map(PhysicalAllocationId::id)
            .collect()
    }

    fn insert(
        &mut self,
        allocation: PhysicalAllocationId,
        owner: PhysicalAllocationOwnerId,
        bytes: usize,
    ) -> Result<(), &'static str> {
        if owner != allocation.owner() {
            return Err("physical allocation row has the wrong owner identity");
        }
        if bytes == 0 && !allocation.permits_zero() {
            return Err("physical allocation row requires a non-zero byte count");
        }
        if self.rows.insert(allocation, bytes).is_some() {
            return Err("physical allocation input contains a duplicate allocation ID");
        }
        Ok(())
    }

    pub(super) fn admission(
        &self,
        known_peak_bytes: usize,
        ceiling_bytes: usize,
    ) -> Result<PhysicalAdmission, &'static str> {
        let rows = self.rows_json();
        let missing = PhysicalAllocationId::ALL
            .into_iter()
            .filter(|id| !self.rows.contains_key(id))
            .collect::<Vec<_>>();
        let measured_bytes = self.rows.values().try_fold(0usize, |sum, &bytes| {
            sum.checked_add(bytes)
                .ok_or("measured non-arena allocation subtotal overflow")
        })?;
        let peak_bytes = missing
            .is_empty()
            .then(|| {
                known_peak_bytes
                    .checked_add(measured_bytes)
                    .ok_or("physical peak byte size overflow")
            })
            .transpose()?;
        Ok(PhysicalAdmission {
            rows,
            measured_bytes,
            peak_bytes,
            complete: missing.is_empty(),
            pass: peak_bytes.is_some_and(|bytes| bytes <= ceiling_bytes),
            missing_ids: self.missing_allocation_ids(),
            missing_rows: missing.iter().map(|id| id.label()).collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_rows_keep_admission_fail_closed() {
        let admission = PhysicalMemoryInputs::default().admission(100, 200).unwrap();
        assert_eq!(admission.measured_bytes, 0);
        assert_eq!(admission.peak_bytes, None);
        assert!(!admission.complete);
        assert!(!admission.pass);
        assert_eq!(admission.missing_ids.len(), 6);
    }

    #[test]
    fn complete_rows_derive_peak_and_admission() {
        let inputs = PhysicalMemoryInputs::try_from_rows(
            PhysicalAllocationId::ALL.map(|id| (id, id.owner(), 1)),
        )
        .unwrap();
        let fits = inputs.admission(100, 106).unwrap();
        assert_eq!(fits.measured_bytes, 6);
        assert_eq!(fits.peak_bytes, Some(106));
        assert!(fits.complete && fits.pass);
        assert!(!inputs.admission(100, 105).unwrap().pass);
    }

    #[test]
    fn rows_reject_owner_aliases_duplicates_and_zero_reserve() {
        use PhysicalAllocationId::{AllocatorPoolNetSlack, OperationalSafetyReserve};
        use PhysicalAllocationOwnerId::{AdmissionPolicy, CudaMemoryPools, Profiler};
        let row = (OperationalSafetyReserve, AdmissionPolicy, 1);
        assert_eq!(
            PhysicalMemoryInputs::try_from_rows([row, row]),
            Err("physical allocation input contains a duplicate allocation ID")
        );
        assert_eq!(
            PhysicalMemoryInputs::try_from_rows([(OperationalSafetyReserve, Profiler, 1)]),
            Err("physical allocation row has the wrong owner identity")
        );
        assert_eq!(
            PhysicalMemoryInputs::try_from_rows([(OperationalSafetyReserve, AdmissionPolicy, 0)]),
            Err("physical allocation row requires a non-zero byte count")
        );
        assert!(PhysicalMemoryInputs::try_from_rows([
            (AllocatorPoolNetSlack, CudaMemoryPools, 0,)
        ])
        .is_ok());
    }

    #[test]
    fn native_pool_checkpoint_charges_every_unattributed_reserved_byte() {
        let checkpoint = AllocatorPoolCheckpoint::try_new(100, 128, 96, 20, 32, 16).unwrap();
        assert_eq!(checkpoint.net_slack_bytes(), 48);
        let inputs = PhysicalMemoryInputs::default()
            .with_allocator_pool_checkpoint(checkpoint)
            .unwrap()
            .with_operational_safety_reserve(NonZeroUsize::new(64).unwrap())
            .unwrap();
        assert_eq!(
            inputs.get(PhysicalAllocationId::AllocatorPoolNetSlack),
            Some(48)
        );
        assert_eq!(
            inputs.get(PhysicalAllocationId::OperationalSafetyReserve),
            Some(64)
        );
        assert_eq!(inputs.missing_allocation_ids().len(), 4);
        assert!(inputs
            .missing_allocation_ids()
            .contains(&"primary_context_driver_baseline"));
    }

    #[test]
    fn native_pool_checkpoint_rejects_impossible_or_overflowing_snapshots() {
        assert_eq!(
            AllocatorPoolCheckpoint::try_new(2, 1, 1, 0, 0, 0),
            Err("allocator pool used bytes exceed reserved bytes")
        );
        assert_eq!(
            AllocatorPoolCheckpoint::try_new(1, 1, 2, 0, 0, 0),
            Err("allocator pool attributed bytes exceed used bytes")
        );
        assert_eq!(
            AllocatorPoolCheckpoint::try_new(0, usize::MAX, 0, 0, 1, 0),
            Err("allocator pool net slack overflow")
        );
    }

    #[test]
    fn native_rows_cannot_overwrite_a_supplied_owner() {
        let inputs = PhysicalMemoryInputs::try_from_rows([(
            PhysicalAllocationId::AllocatorPoolNetSlack,
            PhysicalAllocationOwnerId::CudaMemoryPools,
            1,
        )])
        .unwrap();
        let checkpoint = AllocatorPoolCheckpoint::try_new(0, 0, 0, 0, 0, 0).unwrap();
        assert_eq!(
            inputs.with_allocator_pool_checkpoint(checkpoint),
            Err("physical allocation input contains a duplicate allocation ID")
        );
    }
}
