// Compiles the tiny C++ bridge that exposes two torch CUDA facilities `tch`
// does not, and only when the `cuda` feature is on: the CPU-only build must not
// need a C++ compiler or the libtorch headers.
//
// `LIBTORCH_INCLUDE` is the libtorch -dev root (decision 19); the headers live
// under `<root>/include`. The devshell exports it next to `LIBTORCH_LIB`.

fn main() {
    println!("cargo:rerun-if-changed=shim/c10_graph_shim.cpp");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let root = std::env::var("LIBTORCH_INCLUDE").expect(
        "LIBTORCH_INCLUDE (the libtorch -dev root holding include/) must be set for the cuda feature",
    );
    let cuda = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .expect(
            "CUDA_HOME (the CUDA toolkit root holding include/) must be set for the cuda feature",
        );
    let lib = std::env::var("LIBTORCH_LIB")
        .expect("LIBTORCH_LIB (the libtorch root holding lib/) must be set for the cuda feature");
    // torch-sys links torch_cuda, torch_cpu and c10 but not c10_cuda, which is
    // where `at::cuda`/`c10::cuda` live; the shim references them directly.
    println!("cargo:rustc-link-search=native={lib}/lib");
    println!("cargo:rustc-link-lib=dylib=c10_cuda");
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .warnings(false)
        .pic(true)
        .include(format!("{root}/include"))
        .include(format!("{root}/include/torch/csrc/api/include"))
        .include(format!("{cuda}/include"))
        .file("shim/c10_graph_shim.cpp")
        .compile("ferrum_c10_graph_shim");
}
