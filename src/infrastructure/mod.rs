// Infrastructure adapters: implement the domain ports. May import IO, mmap,
// CUDA and HTTP types; never imported by the domain core.

pub mod ftw;
pub mod http;
pub mod tokenizers_hf;
pub mod torch;
