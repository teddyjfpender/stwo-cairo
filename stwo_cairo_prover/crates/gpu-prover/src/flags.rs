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
        name: "STWO_CAIRO_STREAM_LDE",
        purpose: "legacy-shared: release evaluations per tree at commit; later phases run from coefficients",
        deletion_milestone: "M4 (folds into the diet)",
    },
];

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
