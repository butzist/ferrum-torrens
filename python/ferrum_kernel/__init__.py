"""ferrum_kernel: the hand-written GPU kernels the Rust engine calls.

Only the NVFP4 grouped-MoE GEMV lives here (TODO.md decision 24); everything
else goes through libtorch.
"""