// Stages the sm_120 cubins. They are not committed; `FERRUM_KERNEL_DIR`
// points at a directory holding `kernel/` (the `kernels` flake output,
// nix/kernels.nix), and every artifact is copied into `OUT_DIR/kernel` so
// `kernel_abi.rs` can include it. The env var has no default: the cubins are
// only reachable through the Nix build, which is the point.

fn main() {
    println!("cargo:rerun-if-env-changed=FERRUM_KERNEL_DIR");
    stage_kernels();
}

fn stage_kernels() {
    let src = std::env::var("FERRUM_KERNEL_DIR").unwrap_or_else(|_| {
        panic!(
            "FERRUM_KERNEL_DIR must point at the directory holding kernel/ (the `kernels` flake \
             output); the cubins are built by nix/kernels.nix and are no longer committed"
        )
    });
    let src = std::path::PathBuf::from(src).join("kernel");
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("kernel");
    std::fs::create_dir_all(&out).unwrap();
    for entry in std::fs::read_dir(&src).unwrap_or_else(|e| {
        panic!(
            "cannot read FERRUM_KERNEL_DIR/kernel at {}: {e}",
            src.display()
        )
    }) {
        let entry = entry.unwrap();
        let to = out.join(entry.file_name());
        // The store files are mode 0444, and `copy` preserves that, so a second
        // build in the same OUT_DIR would fail to reopen its own output.
        if to.exists() {
            std::fs::remove_file(&to).unwrap();
        }
        std::fs::copy(entry.path(), &to).unwrap();
        println!("cargo:rerun-if-changed={}", entry.path().display());
    }
    println!("cargo:rerun-if-changed={}", src.display());
}
