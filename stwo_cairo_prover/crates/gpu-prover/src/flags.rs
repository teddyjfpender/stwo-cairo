//! The flag registry (design rule R4).
//!
//! Every env flag this crate reads is registered here with its default, purpose,
//! and DELETION MILESTONE — flags are migration scaffolding, not configuration
//! (design U3: most of this table deletes at M6). Reading an unregistered flag
//! from this crate is a bug; [`flag_on`] debug-asserts registration.

pub struct FlagDef {
    pub name: &'static str,
    /// What the flag gates while it exists.
    pub purpose: &'static str,
    /// The milestone at which the flag (and the lane it gates) is deleted or
    /// becomes the unconditional default.
    pub deletion_milestone: &'static str,
}

/// Flags consumed by the gpu-native pipeline. Shared legacy flags are listed here
/// too when this crate honors them — same name, same semantics, so a manifest
/// step's env applies identically to both engines during A/B.
pub const FLAGS: &[FlagDef] = &[
    FlagDef {
        name: "STWO_CAIRO_LOW_MEMORY",
        purpose: "legacy-shared: compact committed columns post-commit, regenerate at decommit",
        deletion_milestone: "M4 (superseded by the VRAM diet as default on 24GB cards)",
    },
    FlagDef {
        name: "STWO_VRAM_PHASES",
        purpose: "per-phase pool high-water attribution (R5): log + reset at phase boundaries",
        deletion_milestone: "never (the residency ledger instrument)",
    },
    FlagDef {
        name: "STWO_CAIRO_STREAM_LDE",
        purpose: "legacy-shared: release evaluations per tree at commit; later phases run from coefficients",
        deletion_milestone: "M4 (folds into the diet)",
    },
    FlagDef {
        name: "STWO_CUDA_STREAM_FANOUT",
        purpose: "legacy-shared: witness lanes launch on pool streams (fork/join bridged) so concurrent lanes overlap on-device",
        deletion_milestone: "M6 (fanout becomes the unconditional lane path)",
    },
    FlagDef {
        name: "STWO_CUDA_PIPELINED_COMMIT",
        purpose: "legacy-shared: warm proves interpolate finished lanes on a committer thread under the witness phase",
        deletion_milestone: "M6 (the committer becomes the unconditional path)",
    },
];

/// The gpu-native engine's DEFAULTS (design §3: the new pipeline IS the composed
/// fast configuration — device witness lanes, device interaction, device edges,
/// and a witness governor sized for the biggest recorded program). Applied as
/// process env at prover construction ONLY where the variable is unset, so a
/// manifest step's explicit value (including `=0` kill switches) always wins.
/// Migration scaffolding (R4): deleted at M6 when the lanes become the
/// unconditional single path.
pub const GPU_NATIVE_DEFAULTS: &[(&str, &str)] = &[
    ("STWO_CUDA_WITNESS_JIT_PROVE", "1"),
    ("STWO_CUDA_WITNESS_JIT_MAX_INSTRS", "20000"),
    ("STWO_CUDA_DEVICE_INTERACTION", "1"),
    ("STWO_CUDA_WITNESS_EDGES", "1"),
    ("STWO_CUDA_MEM_COUNT_FEEDS", "1"),
    // Stage B2 fanout: the concurrent (rayon) opcode lanes launch on pool
    // streams with per-lane fork/join bridges to legacy, so their kernels
    // overlap on-device. Post-Merkle ledger: Write Base trace 4.08s over
    // ~5.9s of sequential lane spans is the #1 remaining lever.
    ("STWO_CUDA_STREAM_FANOUT", "1"),
    // NOTE: STWO_CUDA_PIPELINED_COMMIT is intentionally NOT a default. M5b
    // hardware A/B (pod sk60d6jcg5p4lu, within-session variance 2%): the
    // per-lane committer helps the small PIE (SN2 ~0.3s) but is within noise
    // or slightly negative on the 14M-step PIEs (its iFFT contends with the
    // witness arms on a single stream). Overlap must PAY to default on — it
    // does not here. The flag + code stay (U3 scaffolding) for when true
    // multi-stream async makes it reliably pay.
];

/// Apply [`GPU_NATIVE_DEFAULTS`] (unset variables only). Called once at
/// `GpuCairoProver::new`; benign on SIMD (the flags gate CUDA-only seams).
pub fn apply_gpu_native_defaults() {
    for (name, value) in GPU_NATIVE_DEFAULTS {
        if std::env::var_os(name).is_none() {
            // SAFETY-ADJACENT NOTE: setenv concurrent with getenv is racy; the
            // prover is constructed before prove-time threads read these.
            std::env::set_var(name, value);
        }
    }
}

/// `true` iff `name` is registered in [`FLAGS`] and set to `1` in the environment.
pub fn flag_on(name: &str) -> bool {
    debug_assert!(
        FLAGS.iter().any(|f| f.name == name),
        "unregistered flag read from gpu-prover: {name} (register it in flags::FLAGS, R4)"
    );
    std::env::var(name).as_deref() == Ok("1")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_flags_are_unique() {
        let mut names: Vec<_> = FLAGS.iter().map(|f| f.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), FLAGS.len());
    }

    #[test]
    #[should_panic(expected = "unregistered flag")]
    #[cfg(debug_assertions)]
    fn unregistered_flag_panics_in_debug() {
        let _ = flag_on("STWO_DEFINITELY_NOT_REGISTERED");
    }
}
