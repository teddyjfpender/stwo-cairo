//! Compile native CUDA integration gates only when nvcc is available.

use std::process::Command;

fn main() {
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
}
