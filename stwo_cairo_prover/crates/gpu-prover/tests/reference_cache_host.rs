use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "common/reference_cache.rs"]
mod reference_cache;

use reference_cache::{
    decode, encode, git_identity, key_digest, runner_identity, source_identity, staging_path,
};

#[test]
fn cache_payload_rejects_corruption_and_truncation() {
    let bytes = encode(&[starknet_ff::FieldElement::ONE]);
    assert_eq!(decode(&bytes).unwrap().len(), 1);
    assert!(decode(&bytes[..bytes.len() - 1]).is_none());
    let mut corrupt = bytes;
    *corrupt.last_mut().unwrap() ^= 1;
    assert!(decode(&corrupt).is_none());
}

#[test]
fn cache_key_changes_for_every_semantic_input() {
    let base = [
        b"params".as_slice(),
        b"input",
        b"fixture",
        b"stwo",
        b"cairo",
    ];
    let baseline = key_digest(base[0], base[1], base[2], base[3], base[4]);
    for index in 0..base.len() {
        let mut changed = base;
        changed[index] = b"changed";
        assert_ne!(
            key_digest(changed[0], changed[1], changed[2], changed[3], changed[4]),
            baseline,
            "key input {index} did not perturb the digest"
        );
    }
}

#[test]
fn git_identity_ignores_results_but_includes_untracked_source() {
    let root =
        std::env::temp_dir().join(format!("stwo-reference-cache-git-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("gpu_benchmarks/loop/results/run")).unwrap();
    let git = |args: &[&str]| {
        let status = Command::new("git")
            .args(args)
            .current_dir(&root)
            .status()
            .unwrap();
        assert!(status.success(), "git {args:?} failed");
    };
    git(&["init", "-q"]);
    std::fs::write(root.join("tracked.txt"), b"tracked").unwrap();
    git(&["add", "tracked.txt"]);
    git(&[
        "-c",
        "user.name=cache test",
        "-c",
        "user.email=cache@test.invalid",
        "commit",
        "-qm",
        "fixture",
    ]);

    let baseline = git_identity(&root).unwrap();
    std::fs::write(
        root.join("gpu_benchmarks/loop/results/run/output.txt"),
        b"large result",
    )
    .unwrap();
    assert_eq!(git_identity(&root).unwrap(), baseline);
    std::fs::write(root.join("scratch.rs"), b"fn changed() {}").unwrap();
    assert_ne!(git_identity(&root).unwrap(), baseline);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn git_identity_is_prefix_free_and_binds_file_kind() {
    let root = std::env::temp_dir().join(format!(
        "stwo-reference-cache-framing-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let git = |args: &[&str]| {
        let status = Command::new("git")
            .args(args)
            .current_dir(&root)
            .status()
            .unwrap();
        assert!(status.success(), "git {args:?} failed");
    };
    git(&["init", "-q"]);
    git(&[
        "-c",
        "user.name=cache test",
        "-c",
        "user.email=cache@test.invalid",
        "commit",
        "--allow-empty",
        "-qm",
        "fixture",
    ]);

    // These two trees collided under the old path || raw-content encoding.
    std::fs::write(root.join("p.rs"), b"q.rsX").unwrap();
    let one_file = git_identity(&root).unwrap();
    std::fs::write(root.join("p.rs"), b"").unwrap();
    std::fs::write(root.join("q.rs"), b"X").unwrap();
    let two_files = git_identity(&root).unwrap();
    assert_ne!(one_file, two_files);

    #[cfg(unix)]
    {
        use std::os::unix::fs::{symlink, PermissionsExt};

        std::fs::remove_file(root.join("q.rs")).unwrap();
        std::fs::write(root.join("p.rs"), b"X").unwrap();
        let regular = git_identity(&root).unwrap();

        let mut permissions = std::fs::metadata(root.join("p.rs")).unwrap().permissions();
        permissions.set_mode(permissions.mode() | 0o111);
        std::fs::set_permissions(root.join("p.rs"), permissions).unwrap();
        let executable = git_identity(&root).unwrap();
        assert_ne!(regular, executable);

        std::fs::remove_file(root.join("p.rs")).unwrap();
        std::fs::write(root.join("q.rs"), b"X").unwrap();
        symlink("q.rs", root.join("p.rs")).unwrap();
        let linked = git_identity(&root).unwrap();
        assert_ne!(regular, linked);
        assert_ne!(executable, linked);
    }

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn git_failure_disables_identity() {
    assert!(git_identity(Path::new("/definitely/not/a/repository")).is_none());
}

#[test]
fn runner_identity_requires_both_valid_source_fields() {
    let head = "1".repeat(40);
    let hash = "2".repeat(64);
    assert!(runner_identity(Some(&head), Some(&hash)).is_some());
    assert!(runner_identity(Some(&head), None).is_none());
    assert!(runner_identity(None, Some(&hash)).is_none());
    assert!(runner_identity(Some("short"), Some(&hash)).is_none());
    assert!(runner_identity(Some(&head), Some(&"G".repeat(64))).is_none());
}

#[test]
fn runner_identity_is_used_only_without_git_metadata() {
    let root = std::env::temp_dir().join(format!(
        "stwo-reference-cache-source-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let runner = b"runner identity".to_vec();
    assert_eq!(
        source_identity(&root, Some(runner.clone())).unwrap(),
        runner
    );

    // Present-but-invalid metadata must disable caching, not fall back to a
    // runner identity that could mask a broken local checkout.
    std::fs::create_dir(root.join(".git")).unwrap();
    assert!(source_identity(&root, Some(b"fallback".to_vec())).is_none());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn concurrent_staging_paths_are_unique() {
    let path = PathBuf::from("reference.ref");
    let paths = (0..32)
        .map(|_| {
            let path = path.clone();
            std::thread::spawn(move || staging_path(&path))
        })
        .map(|thread| thread.join().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(paths.iter().collect::<HashSet<_>>().len(), paths.len());
}
