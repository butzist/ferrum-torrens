// Gated proof that the committed cubins load and compute through the hand-rolled
// CUDA driver FFI, with no Python in the process. Run with:
//   cargo test --release --features cuda --test kernel_cubin_smoke -- --ignored --nocapture
//
// This is decision 36 slice 3's gate. The two kernels here need no expert banks,
// so they isolate exactly what slice 3 adds: dlopen, module load, the packed
// argument array from `kernel_abi`, and a launch. The GEMV pair stays for the
// checkpoint gate (slice 5), which needs the real slabs to fill the banks.
//
// bf16 is the serving dtype the cubins were specialized for, so the harness
// converts f32 to bf16 and back; the tolerance is bf16's, not f32's.

#![cfg(feature = "cuda")]

use ferrum_torrens::infrastructure::cuda_driver::{CudaKernels, Driver};
use ferrum_torrens::infrastructure::kernel_abi::{
    Abi, GEMV_DOWN, GEMV_GATE_UP, MoEArgs, SILU_MUL, SUM_REDUCE, grid, silu_mul_args,
    sum_reduce_args,
};

/// bf16 round-to-nearest-even, the same rounding the conversion does.
fn to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits + 0x7fff + lsb) >> 16) as u16
}

fn from_bf16(v: u16) -> f32 {
    f32::from_bits(u32::from(v) << 16)
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| to_bf16(*v).to_le_bytes())
        .collect()
}

fn read_bf16(bytes: &[u8]) -> Vec<f32> {
    let (pairs, _) = bytes.as_chunks::<2>();
    pairs
        .iter()
        .map(|c| from_bf16(u16::from_le_bytes(*c)))
        .collect()
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// Deterministic inputs in a range both kernels handle without inf: the reduce
/// sums `top_k` of them, and bf16 has the exponent headroom for 8x.
fn values(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) % 2048) as f32;
            (x - 1024.0) / 1024.0
        })
        .collect()
}

/// bf16 has 8 mantissa bits, so one rounding is ~0.4% and the reduce's eight
/// additions can stack a little more. FreeToken compares against the term scale
/// with the same order of magnitude.
const TOL: f32 = 0.02;

fn check(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let mut worst = 0.0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs() / w.abs().max(1e-3);
        if err > worst {
            worst = err;
            assert!(
                err < TOL,
                "{name}[{i}]: got {g}, want {w} (rel {err}, worst so far {worst})"
            );
        }
    }
    eprintln!("{name}: {} values, worst rel {worst:.3e}", got.len());
}

#[test]
#[ignore = "needs a CUDA device"]
fn the_committed_cubins_match_a_cpu_reference_through_the_driver_api() {
    let driver = std::sync::Arc::new(Driver::open().expect("libcuda"));
    driver.init().expect("cuInit");
    driver.attach_primary_context(0).expect("primary context");
    let abi = Abi::committed().expect("committed manifest");
    let kernels =
        CudaKernels::load(driver.clone(), abi.clone()).expect("load the committed cubins");

    // Default stream: null, which the driver accepts for the legacy stream. The
    // real path uses libtorch's stream, which is what the checkpoint gate needs.
    let stream = std::ptr::null_mut();

    let hidden = abi.hidden;
    let inter = abi.inter;
    let top_k = abi.top_k;

    {
        // `sum_reduce`: out[t] = sum_j ic3[t * top_k + j], one token.
        let ic3 = values(top_k * hidden, 7);
        let want: Vec<f32> = (0..hidden)
            .map(|c| (0..top_k).map(|j| ic3[j * hidden + c]).sum())
            .collect();
        let d_ic3 = driver.alloc(ic3.len() * 2).expect("ic3");
        let d_out = driver.alloc(hidden * 2).expect("out");
        // SAFETY: the allocations are live for the whole block and the byte
        // lengths match what the kernels index.
        unsafe {
            driver
                .copy_h2d(d_ic3.ptr, &bf16_bytes(&ic3))
                .expect("fill ic3")
        };
        let args = MoEArgs {
            tokens: 1,
            routes: top_k,
            ic3: d_ic3.ptr,
            out: d_out.ptr,
            ..MoEArgs::default()
        };
        let pack = sum_reduce_args(&abi, &args).expect("sum_reduce args");
        let mut params = pack.kernel_params().expect("kernelParams");
        let kernel = abi.kernel(SUM_REDUCE).unwrap();
        // SAFETY: the pack is for this kernel, and the stream is the default one
        // of the context we just made current.
        unsafe {
            kernels
                .launch(kernel, grid(1, kernel), &mut params, stream)
                .expect("launch sum_reduce");
            kernels.synchronize(stream).expect("sync");
        }
        let mut out = vec![0u8; hidden * 2];
        // SAFETY: `out` is `hidden` bf16 values and `d_out` is that size.
        unsafe { driver.copy_d2h(&mut out, d_out.ptr).expect("read out") };
        check("sum_reduce", &read_bf16(&out), &want);
    }

    {
        // `silu_mul`: ic2[r, i] = silu(ic1[r, i]) * ic1[r, I + i].
        let routes = 2;
        let ic1 = values(routes * 2 * inter, 11);
        let mut want = Vec::with_capacity(routes * inter);
        for r in 0..routes {
            for i in 0..inter {
                want.push(silu(ic1[r * 2 * inter + i]) * ic1[r * 2 * inter + inter + i]);
            }
        }
        let d_ic1 = driver.alloc(ic1.len() * 2).expect("ic1");
        let d_ic2 = driver.alloc(routes * inter * 2).expect("ic2");
        // SAFETY: as above; the bytes are copied in before the launch.
        unsafe { driver.copy_h2d(d_ic1.ptr, &bf16_bytes(&ic1)).unwrap() };
        let args = MoEArgs {
            routes,
            ic1: d_ic1.ptr,
            ic2: d_ic2.ptr,
            ..MoEArgs::default()
        };
        let pack = silu_mul_args(&abi, &args).expect("silu args");
        let mut params = pack.kernel_params().expect("kernelParams");
        let kernel = abi.kernel(SILU_MUL).unwrap();
        // SAFETY: the pack is for this kernel on the current context's stream.
        unsafe {
            kernels
                .launch(kernel, grid(routes, kernel), &mut params, stream)
                .expect("launch silu_mul");
            kernels.synchronize(stream).expect("sync");
        }
        let mut out = vec![0u8; routes * inter * 2];
        // SAFETY: `out` is `routes * inter` bf16 values and `d_ic2` that size.
        unsafe { driver.copy_d2h(&mut out, d_ic2.ptr).expect("read ic2") };
        check("silu_mul", &read_bf16(&out), &want);
    }
}

/// The two GEMV cubins are compiled from one Triton entry, so they are only
/// distinguishable by their manifest *name*. An entry-keyed handle table hands
/// out the gate/up function for the down one, which still runs: it just reads
/// `K_BYTES = inter/2` out of a slab laid out for `hidden` and drops the route
/// weights. The numbers come out plausible, so this has to be an explicit
/// identity check rather than a tolerance.
#[test]
#[ignore = "needs a CUDA device"]
fn every_committed_kernel_resolves_to_its_own_function() {
    let driver = std::sync::Arc::new(Driver::open().expect("libcuda"));
    driver.init().expect("cuInit");
    driver.attach_primary_context(0).expect("primary context");
    let abi = Abi::committed().expect("committed manifest");
    let kernels =
        CudaKernels::load(driver.clone(), abi.clone()).expect("load the committed cubins");

    for a in Abi::kernel_names() {
        for b in Abi::kernel_names() {
            if a == b {
                continue;
            }
            let fa = kernels.function(abi.kernel(a).unwrap()).expect(a);
            let fb = kernels.function(abi.kernel(b).unwrap()).expect(b);
            assert_ne!(
                fa,
                fb,
                "{a} and {b} resolved to the same function; both cubins share the entry {}",
                abi.kernel(a).unwrap().entry
            );
        }
    }
    // The pair that started this: same entry, different constexprs.
    assert_eq!(
        abi.kernel(GEMV_GATE_UP).unwrap().entry,
        abi.kernel(GEMV_DOWN).unwrap().entry
    );
}
