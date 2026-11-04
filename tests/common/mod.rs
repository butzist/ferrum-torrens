// Shared plumbing for the gated device tests. Included with `mod common;`, so
// only the files that use it compile it.
//
// M8.1: the device gates are parameterized by the converted checkpoint, so the
// same gate can run against a second model. `FERRUM_FTW_DIR` points at an FTW
// directory; unset means the reference checkpoint. An explicit override also
// means "not the reference checkpoint", which the numeric pins use to decide
// whether their oracle applies.
#![allow(dead_code)]

/// The converted checkpoint under test.
pub fn ftw_dir() -> String {
    std::env::var("FERRUM_FTW_DIR")
        .unwrap_or_else(|_| ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw".to_string())
}

/// Whether this run is the reference checkpoint, whose exact token ids are
/// pinned against the transformers/FreeToken oracle. A second checkpoint has no
/// such oracle in this repo, so its gates assert semantics instead (the answer
/// names the capital, the description names the image) rather than ids.
pub fn is_reference_checkpoint() -> bool {
    std::env::var_os("FERRUM_FTW_DIR").is_none()
}
