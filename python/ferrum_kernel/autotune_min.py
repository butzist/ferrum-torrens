"""Torch-free timing proof for a Triton kernel (the autotune dependency question).

Triton's launcher converts any argument with a `.data_ptr()` method to a pointer
(`triton/runtime/jit.py`), and the CUDA driver API is reachable through `ctypes`,
so an autotune sweep does not actually need torch:

  * `cuMemAlloc` / `cuMemcpyHtoD` allocate and fill device buffers,
  * `cuEventCreate` / `cuEventRecord` / `cuEventSynchronize` / `cuEventElapsedTime`
    time a launch,
  * the Triton JIT compiles and launches the kernel from `DevPtr` shims.

This script times the dense NVFP4 GEMV (the lm_head shape) both ways and prints
both numbers, so the two paths can be compared on the same GPU.

    $FERRUM_PYTHON -m ferrum_kernel.autotune_min

Result on the RTX 5060: 1062 us, inside the 1007-1094 us the torch `do_bench`
reports for the same shape. So the *measurement path* (allocation, fill, timing,
launch) does not need torch.

The remaining coupling is elsewhere: every kernel wrapper module does
`import torch` at module scope (for type hints, `Tensor.view(torch.int32)` and
`require_native_e4m3`), so importing a kernel still pulls torch in. A fully
torch-free autotune needs those kernels moved into torch-free modules (wrappers
stay separate), the fp16 comparison replaced with a Triton fp16 kernel, and
`require_native_e4m3`/dtype constants expressed with `tl` types. The other
option is a Rust harness over the committed cubins.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import time

import triton
import triton.language as tl

from .linear import DENSE_BLOCK_KW, DENSE_BLOCK_N, DENSE_WARPS, _nvfp4_dense_gemv_kernel

LIB = ctypes.CDLL("libcuda.so.1")
CUdeviceptr = ctypes.c_uint64
CUstream = ctypes.c_void_p
CUevent = ctypes.c_void_p
CUcontext = ctypes.c_void_p

CU_EVENT_DEFAULT = 0
CU_MEMHOSTALLOC_PORTABLE = 0x01


from triton.runtime.jit import TensorWrapper


class DevPtr:
    """What Triton's launcher needs: an integer device address and a dtype."""

    __slots__ = ("ptr", "dtype")

    def __init__(self, ptr: int, dtype):
        self.ptr = ptr
        self.dtype = dtype

    def data_ptr(self) -> int:
        return self.ptr


def _alloc(nbytes: int) -> int:
    p = CUdeviceptr(0)
    rc = LIB.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(nbytes))
    if rc != 0:
        raise RuntimeError(f"cuMemAlloc({nbytes}) -> {rc}")
    return p.value


def _alloc_host(nbytes: int) -> int:
    p = ctypes.c_void_p(0)
    rc = LIB.cuMemHostAlloc(ctypes.byref(p), ctypes.c_size_t(nbytes), CU_MEMHOSTALLOC_PORTABLE)
    if rc != 0:
        raise RuntimeError(f"cuMemHostAlloc({nbytes}) -> {rc}")
    return p.value


def _free(p: int) -> None:
    LIB.cuMemFree_v2(CUdeviceptr(p))


def _free_host(p: int) -> None:
    LIB.cuMemFreeHost(ctypes.c_void_p(p))


def _upload(dst: int, data: bytes) -> None:
    rc = LIB.cuMemcpyHtoD_v2(CUdeviceptr(dst), ctypes.c_char_p(data), ctypes.c_size_t(len(data)))
    if rc != 0:
        raise RuntimeError(f"cuMemcpyHtoD -> {rc}")


def _event() -> CUevent:
    e = CUevent()
    rc = LIB.cuEventCreate(ctypes.byref(e), ctypes.c_uint(CU_EVENT_DEFAULT))
    if rc != 0:
        raise RuntimeError(f"cuEventCreate -> {rc}")
    return e


def _elapsed(a: CUevent, b: CUevent) -> float:
    ms = ctypes.c_float(0)
    LIB.cuEventElapsedTime(ctypes.byref(ms), a, b)
    return ms.value


def open_context() -> None:
    rc = LIB.cuInit(0)
    if rc != 0:
        raise RuntimeError(f"cuInit -> {rc}")
    dev = ctypes.c_int(0)
    LIB.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = CUcontext()
    rc = LIB.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
    if rc != 0:
        raise RuntimeError(f"cuDevicePrimaryCtxRetain -> {rc}")
    rc = LIB.cuCtxSetCurrent(ctx)
    if rc != 0:
        raise RuntimeError(f"cuCtxSetCurrent -> {rc}")


def stream0() -> int:
    # Torch-free Triton launches on the driver's default stream, so the timing
    # events go on the same stream.
    return 0


def timed(fn, reps: int = 100) -> float:
    """`fn` is a callable that launches once. Returns ms per launch (median-ish)."""
    fn()  # warm / compile
    LIB.cuCtxSynchronize()
    times = []
    for _ in range(5):
        a = _event()
        b = _event()
        LIB.cuEventRecord(a, CUstream(stream0()))
        fn()
        LIB.cuEventRecord(b, CUstream(stream0()))
        LIB.cuEventSynchronize(b)
        times.append(_elapsed(a, b))
        LIB.cuEventDestroy(a)
        LIB.cuEventDestroy(b)
    return sum(times) / len(times)


def run(shape, block_n, block_kw, warps) -> float:
    m, n, k = shape
    k_words = k // 8
    # Bytes: activations bf16 [m, k], packed int32 [k_words, n], scale u8
    # [k // 16, n], global u8 [n * 2], output f32 [m, n].
    a_bytes = m * k * 2
    p_bytes = k_words * n * 4
    s_bytes = (k // 16) * n
    g_bytes = n * 2
    c_bytes = m * n * 4
    a = _alloc(a_bytes)
    p = _alloc(p_bytes)
    s = _alloc(s_bytes)
    g = _alloc(g_bytes)
    c = _alloc(c_bytes)
    _upload(a, os.urandom(a_bytes))
    _upload(p, os.urandom(p_bytes))
    _upload(s, os.urandom(s_bytes))
    _upload(g, os.urandom(g_bytes))

    grid = (m, triton.cdiv(n, block_n))
    a_d, p_d, s_d, g_d, c_d = (
        DevPtr(a, tl.bfloat16),
        DevPtr(p, tl.int32),
        DevPtr(s, tl.uint8),
        DevPtr(g, tl.float16),
        DevPtr(c, tl.float32),
    )

    def launch():
        _nvfp4_dense_gemv_kernel[grid](
            a_d, p_d, s_d, g_d, c_d,
            m, n, k_words,
            k, 1,          # a [m, k] strides
            1, n,          # packed [k_words, n] K-major: N stride 1, word stride n
            1, n,          # scale [k // 16, n]
            n, 1,          # c [m, n]
            BLOCK_N=block_n,
            BLOCK_KW=block_kw,
            num_warps=warps,
        )

    ms = timed(launch)
    for buf in (a, p, s, g, c):
        _free(buf)
    return ms


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=248320, help="lm_head output dim")
    p.add_argument("--k", type=int, default=2048, help="hidden size")
    args = p.parse_args(argv)
    open_context()
    t0 = time.perf_counter()
    ms = run((1, args.n, args.k), DENSE_BLOCK_N, DENSE_BLOCK_KW, DENSE_WARPS)
    print(
        f"torch-free dense GEMV [1, {args.n}, {args.k}]: {ms * 1e3:.1f} us "
        f"(wall {time.perf_counter() - t0:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
