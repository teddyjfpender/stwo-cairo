use std::collections::BTreeMap;

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
        let mut validated = BTreeMap::new();
        for (allocation, owner, bytes) in rows {
            if owner != allocation.owner() {
                return Err("physical allocation row has the wrong owner identity");
            }
            if bytes == 0 && !allocation.permits_zero() {
                return Err("physical allocation row requires a non-zero byte count");
            }
            if validated.insert(allocation, bytes).is_some() {
                return Err("physical allocation input contains a duplicate allocation ID");
            }
        }
        Ok(Self { rows: validated })
    }

    pub(super) fn admission(
        &self,
        known_peak_bytes: usize,
        ceiling_bytes: usize,
    ) -> Result<PhysicalAdmission, &'static str> {
        let rows = PhysicalAllocationId::ALL
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
            .collect();
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
            missing_ids: missing.iter().map(|id| id.id()).collect(),
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
}
