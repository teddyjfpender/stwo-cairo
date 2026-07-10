use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Native CUDA integration targets are compiled only when the sibling STWO
    // kernel archive is available. Declare the build-script cfg so CPU checks
    // do not hide a misspelled gate behind `unexpected_cfgs` warnings.
    println!("cargo:rustc-check-cfg=cfg(stwo_cuda_link)");
    println!("cargo:rerun-if-env-changed=STWO_CUDA_NVCC");
    let nvcc = std::env::var("STWO_CUDA_NVCC").unwrap_or_else(|_| "nvcc".to_string());
    if Command::new(nvcc)
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
    {
        println!("cargo:rustc-cfg=stwo_cuda_link");
    }

    // The bootloader JSON is only needed for the CairoPie ingestion path in
    // gpu_bench, which is compiled solely under the `pie-bench` feature. When the
    // feature is off, cairo-program-runner-lib is not in the dependency tree, so we
    // must not attempt to locate it (and BOOTLOADER_JSON_PATH is never referenced).
    if std::env::var_os("CARGO_FEATURE_PIE_BENCH").is_none() {
        // Keep the build script cheap and re-run triggers stable.
        println!("cargo:rerun-if-changed=Cargo.toml");
        println!("cargo:rerun-if-changed=Cargo.lock");
        return;
    }

    // Run `cargo metadata` to locate the cairo-program-runner-lib package. The dep is
    // optional (behind `pie-bench`); with feature resolver v2, plain `cargo metadata`
    // prunes it from `packages`, so we must select the feature to make it appear. We
    // only reach this branch when CARGO_FEATURE_PIE_BENCH is set, so this is consistent.
    let output = Command::new(env!("CARGO"))
        .args(["metadata", "--format-version=1", "--features", "pie-bench"])
        .output()
        .expect("Failed to run `cargo metadata`");
    assert!(
        output.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let metadata: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("Failed to parse cargo metadata JSON");

    let packages = metadata["packages"]
        .as_array()
        .expect("packages field missing");

    let manifest_path = packages
        .iter()
        .find(|p| p["name"].as_str() == Some("cairo-program-runner-lib"))
        .expect("cairo-program-runner-lib not found in cargo metadata")["manifest_path"]
        .as_str()
        .expect("manifest_path is not a string");

    let pkg_dir = PathBuf::from(manifest_path)
        .parent()
        .expect("manifest_path has no parent")
        .to_path_buf();

    let bootloader_path = std::env::var_os("STWO_BOOTLOADER_JSON")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            pkg_dir.join("resources/compiled_programs/bootloaders/simple_bootloader_compiled.json")
        });
    assert!(
        bootloader_path.is_file(),
        "bootloader JSON not found: {}",
        bootloader_path.display()
    );

    println!(
        "cargo:rustc-env=BOOTLOADER_JSON_PATH={}",
        bootloader_path.display()
    );

    // Re-run if the dependency tree changes.
    println!("cargo:rerun-if-env-changed=STWO_BOOTLOADER_JSON");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=Cargo.lock");
}
