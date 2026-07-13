use std::collections::BTreeMap;
use std::fs::{self, hard_link};
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::fri_round6_provenance::{
    register_identity, require_executable, resolve_bundle_path, validate_relative_path,
    IDENTITY_PREFLIGHT_STATUS,
};

static SCRATCH_SERIAL: AtomicU64 = AtomicU64::new(0);

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let serial = SCRATCH_SERIAL.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "stwo-fri-provenance-test-{}-{serial}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn rejects_intermediate_symlink_even_when_it_stays_inside_bundle() {
    let scratch = Scratch::new();
    let real = scratch.path().join("real");
    fs::create_dir(&real).unwrap();
    fs::write(real.join("proof.bin"), b"proof").unwrap();
    symlink(&real, scratch.path().join("link")).unwrap();

    let error = resolve_bundle_path(scratch.path(), "link/proof.bin").unwrap_err();
    assert!(error.contains("symlink"), "{error}");
}

#[test]
fn rejects_hardlink_alias_with_manifest_identity() {
    let scratch = Scratch::new();
    let manifest = scratch.path().join("manifest.json");
    let artifact = scratch.path().join("proof.bin");
    fs::write(&manifest, b"same inode").unwrap();
    hard_link(&manifest, &artifact).unwrap();

    let mut identities = BTreeMap::new();
    register_identity(
        &mut identities,
        &fs::metadata(&manifest).unwrap(),
        "manifest",
    )
    .unwrap();
    let error =
        register_identity(&mut identities, &fs::metadata(&artifact).unwrap(), "proof").unwrap_err();
    assert!(
        error.contains("manifest") && error.contains("proof"),
        "{error}"
    );
}

#[test]
fn rejects_ambiguous_relative_path_spellings() {
    for path in ["a//b", "a\\b", "a/../b", "./a", "/a", "a/"] {
        assert!(
            validate_relative_path(path, "test path").is_err(),
            "accepted {path:?}"
        );
    }
    validate_relative_path("a/b", "test path").unwrap();
}

#[test]
fn adapter_tool_must_be_executable() {
    let scratch = Scratch::new();
    let tool = scratch.path().join("adapter");
    fs::write(&tool, b"#!/bin/sh\n").unwrap();
    fs::set_permissions(&tool, fs::Permissions::from_mode(0o600)).unwrap();
    assert!(require_executable(&fs::metadata(&tool).unwrap()).is_err());
    fs::set_permissions(&tool, fs::Permissions::from_mode(0o700)).unwrap();
    require_executable(&fs::metadata(&tool).unwrap()).unwrap();
}

#[test]
fn success_wording_cannot_claim_execution_or_proof_admission() {
    for pending in [
        "production_admissible=false",
        "proof_verification=pending",
        "adapter_execution_attestation=pending",
        "verifier_closure_match=pending",
    ] {
        assert!(IDENTITY_PREFLIGHT_STATUS.contains(pending));
    }
    assert!(IDENTITY_PREFLIGHT_STATUS.contains("IDENTITY_PREFLIGHT=PASS"));
}
