//! M7.1a spike gates 1-3: the minimal `cudarc` tensor layer against the one
//! gate candle failed.
//!
//! 1. Direct cuBLAS BF16 GEMM parity and speed on the M7.0 projection shapes,
//!    including ATen's transposed-weight layout.
//! 2. Capture a GEMM into an engine-owned fixed output buffer, replay it, and
//!    match eager bit for bit.
//! 3. Run the committed `add_rmsnorm` cubin on cudarc storage and the cudarc
//!    side stream, and compare against the tch-fed launch.
//!
//! Run inside the devshell with:
//!
//!   FERRUM_KERNEL_DIR=<kernels store path> \
//!   cargo run --release --features cuda,cudarc --example m7_1a_gemm -- all
//!
//! This is a spike: it links libtorch and cudarc on purpose (gates 1 and 3
//! compare them).

use std::sync::Arc;
use std::time::{Duration, Instant};

use ferrum_torrens::infrastructure::cuda_driver::{CUstream, Driver};
use ferrum_torrens::infrastructure::cuda_gdn::GdnDecode;
use ferrum_torrens::infrastructure::cudarc as cud;
use ferrum_torrens::infrastructure::torch::{Device as TDevice, bf16_tensor, to_f32_vec};
use half::bf16;
use tch::Kind;

const H: usize = 2048;
const I: usize = 512;
const VOCAB: usize = 248320;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn bf16_bits(n: usize, seed: u64) -> Vec<u16> {
    // Exact bf16 values, so both backends see the same input bits without
    // depending on their f32->bf16 rounding being identical.
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut out = Vec::with_capacity(n);
    for _i in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (state >> 32) as u32;
        let v = bf16::from_f32((u as f32 / u32::MAX as f32) * 2.0 - 1.0);
        out.push(v.to_bits());
    }
    out
}

fn tch_bf16(bits: &[u16], shape: &[i64]) -> tch::Tensor {
    let vals: Vec<f32> = bits.iter().map(|b| bf16::from_bits(*b).to_f32()).collect();
    bf16_tensor(&vals, shape, TDevice::Cuda(0))
}

fn tch_stream() -> CUstream {
    // The thread-local torch stream; the default stream is the legacy null
    // stream, which CUDA events record against like any other.
    unsafe { ferrum_torrens::infrastructure::cuda_graph::current_stream(0) }
}

struct Timing {
    driver: Arc<Driver>,
}

impl Timing {
    /// One sample: `reps` back-to-back ops between a pair of events, so the
    /// event and host-sync overhead is amortised instead of dominating a small
    /// GEMM.
    fn batch<F>(&self, stream: CUstream, reps: usize, f: &mut F) -> Res<f64>
    where
        F: FnMut() -> Res<()>,
    {
        let start = self.driver.owned_timing_event()?;
        let end = self.driver.owned_timing_event()?;
        unsafe { start.record(stream)? };
        for _ in 0..reps {
            f()?;
        }
        unsafe { end.record(stream)? };
        unsafe { self.driver.synchronize(stream)? };
        Ok(start.elapsed(&end)? as f64 / reps as f64)
    }

    /// Order-balanced A/B: alternate which backend runs first each round and
    /// take the median of the per-round means.
    fn ab<F, G>(
        &self,
        tch_stream: CUstream,
        cud_stream: CUstream,
        rounds: usize,
        reps: usize,
        mut tch: F,
        mut cud: G,
    ) -> Res<(Vec<f64>, Vec<f64>)>
    where
        F: FnMut() -> Res<()>,
        G: FnMut() -> Res<()>,
    {
        for _ in 0..5 {
            tch()?;
            cud()?;
        }
        // This GPU drops to a lower clock a couple of seconds into a sustained
        // GEMM loop, and the two backends then measure in different phases
        // (M7.1a: ~0.52 -> ~0.72 ms on H->H at 2057). Warm both into the same
        // steady state before sampling so the comparison is not a phase race.
        let warm = Instant::now();
        while warm.elapsed() < Duration::from_millis(1500) {
            tch()?;
            cud()?;
        }
        unsafe {
            self.driver.synchronize(tch_stream)?;
            self.driver.synchronize(cud_stream)?;
        }
        let (mut a, mut b) = (Vec::with_capacity(rounds), Vec::with_capacity(rounds));
        for round in 0..rounds {
            if round % 2 == 0 {
                a.push(self.batch(tch_stream, reps, &mut tch)?);
                b.push(self.batch(cud_stream, reps, &mut cud)?);
            } else {
                b.push(self.batch(cud_stream, reps, &mut cud)?);
                a.push(self.batch(tch_stream, reps, &mut tch)?);
            }
        }
        Ok((a, b))
    }
}

fn stats(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = v[v.len() / 2];
    let best = v[0];
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    (best, med, mean)
}

fn argmax_rows(vals: &[f32], rows: usize, cols: usize) -> Vec<usize> {
    (0..rows)
        .map(|r| {
            let row = &vals[r * cols..(r + 1) * cols];
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        })
        .collect()
}

struct Shape {
    name: &'static str,
    in_dim: usize,
    out_dim: usize,
}

fn gate_gemm(driver: &Arc<Driver>, rt: &cud::Runtime) -> Res<bool> {
    let timing = Timing {
        driver: Arc::clone(driver),
    };
    let shapes = [
        Shape {
            name: "h2048_to_h2048",
            in_dim: H,
            out_dim: H,
        },
        Shape {
            name: "h2048_to_i512",
            in_dim: H,
            out_dim: 2 * I,
        },
        Shape {
            name: "i512_to_h2048",
            in_dim: I,
            out_dim: H,
        },
        Shape {
            name: "h2048_to_vocab",
            in_dim: H,
            out_dim: VOCAB,
        },
    ];
    let ms = [1usize, 512, 2057];
    let mut ok = true;
    println!("\n=== gate 1: cudarc BF16 GEMM parity and speed vs tch ===");
    for s in &shapes {
        let (w_rows, w_cols) = (s.out_dim, s.in_dim);
        let wb = bf16_bits(w_rows * w_cols, 0x1234 ^ w_rows as u64 ^ w_cols as u64);
        let w_t = tch_bf16(&wb, &[w_rows as i64, w_cols as i64]);
        let w_c = rt.upload_bf16_bits(&wb)?;
        rt.synchronize()?;
        println!(
            "\n{}: weight [{w_rows}, {w_cols}] bf16 ({:.1} MiB)",
            s.name,
            (wb.len() * 2) as f64 / (1 << 20) as f64
        );
        for &m in &ms {
            let xb = bf16_bits(m * s.in_dim, 0x99 ^ m as u64);
            let x_t = tch_bf16(&xb, &[m as i64, s.in_dim as i64]);
            let x_c = rt.upload_bf16_bits(&xb)?;
            let mut out_c = rt.alloc_bf16(m * w_rows)?;
            rt.synchronize()?;

            // Correctness: the actual projection (transposed weight view).
            // Compare in row chunks so the f32 staging tensor never has to be
            // the whole [2057, vocab] output at once.
            let y_t = x_t.matmul(&w_t.transpose(0, 1)).to_kind(Kind::BFloat16);
            rt.gemm_bf16_nt(&x_c, &w_c, &mut out_c, m, w_rows, s.in_dim)?;
            unsafe { driver.synchronize(tch_stream())? };
            rt.synchronize()?;
            let chunk = 256.min(m);
            let (mut max_abs, mut scale, mut equal, mut exact) = (0.0f64, 0.0f64, 0usize, 0usize);
            let mut start = 0usize;
            while start < m {
                let n = chunk.min(m - start);
                let ct = y_t.narrow(0, start as i64, n as i64).to_kind(Kind::Float);
                let cc = rt.download_f32_range(&out_c, start * w_rows, n * w_rows)?;
                let ht = to_f32_vec(&ct);
                if ht.len() != cc.len() {
                    return Err(format!(
                        "gate 1: output length mismatch {} vs {}",
                        ht.len(),
                        cc.len()
                    )
                    .into());
                }
                for (a, b) in ht.iter().zip(cc.iter()) {
                    max_abs = max_abs.max((*a as f64 - *b as f64).abs());
                    scale = scale.max(a.abs() as f64);
                    exact += usize::from(a.to_bits() == b.to_bits());
                }
                let at = argmax_rows(&ht, n, w_rows);
                let ac = argmax_rows(&cc, n, w_rows);
                equal += at.iter().zip(ac.iter()).filter(|(a, b)| a == b).count();
                start += n;
            }
            let max_rel = if scale > 0.0 { max_abs / scale } else { 0.0 };
            let pass = equal == m && max_rel <= 0.0625;
            ok &= pass;
            println!(
                "  M={m:4}: argmax {equal}/{m}, max|d|={max_abs:.4}, max rel={max_rel:.5}, bit-exact {exact}/{} {}",
                m * w_rows,
                if pass { "PASS" } else { "FAIL" }
            );
            drop(y_t);

            let mut tch_fn = || -> Res<()> {
                let y = x_t.matmul(&w_t.transpose(0, 1));
                std::hint::black_box(&y);
                Ok(())
            };
            // The cudarc side reuses the engine-owned output, which is the
            // M7.1 design; tch allocates its output each call as before.
            let mut cud_fn = || -> Res<()> {
                rt.gemm_bf16_nt(&x_c, &w_c, &mut out_c, m, w_rows, s.in_dim)?;
                Ok(())
            };
            let (tch_times, cud_times) = timing.ab(
                tch_stream(),
                rt.raw_stream(),
                8,
                5,
                &mut tch_fn,
                &mut cud_fn,
            )?;
            let (tb, tm, _) = stats(tch_times.clone());
            let (cb, cm, _) = stats(cud_times.clone());
            let delta = (cm - tm) / tm * 100.0;
            println!(
                "        gemm tch best {tb:.3} med {tm:.3} ms | cudarc best {cb:.3} med {cm:.3} ms | med {delta:+.2}%"
            );
            if std::env::var_os("M7_DIST").is_some() {
                println!("        tch rounds {tch_times:?}");
                println!("        cud rounds {cud_times:?}");
            }
            if m != 1 && delta > 5.0 {
                println!("        FAIL: cudarc is more than 5% slower on a prefill shape");
                ok = false;
            }
        }
    }
    Ok(ok)
}

/// Diagnostic: which cuBLAS algorithm matches ATen on the prefill shapes.
/// The default and tensor-op variants plus the four fixed algorithms and the
/// fast 16-bit compute type cover what ATen could be choosing.
fn gate_sweep(driver: &Arc<Driver>, rt: &cud::Runtime) -> Res<()> {
    use ::cudarc::cublas::sys::{cublasComputeType_t as C, cublasGemmAlgo_t as A};

    let timing = Timing {
        driver: Arc::clone(driver),
    };
    let shapes = [
        ("h2048_to_h2048", H, H),
        ("h2048_to_i512", H, 2 * I),
        ("i512_to_h2048", I, H),
    ];
    let variants: [(&str, cud::GemmTuning); 8] = [
        (
            "default/32F",
            cud::GemmTuning {
                algo: A::CUBLAS_GEMM_DEFAULT,
                compute: C::CUBLAS_COMPUTE_32F,
            },
        ),
        (
            "tensorop/32F",
            cud::GemmTuning {
                algo: A::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                compute: C::CUBLAS_COMPUTE_32F,
            },
        ),
        (
            "algo0/32F",
            cud::GemmTuning {
                algo: A::CUBLAS_GEMM_ALGO0,
                compute: C::CUBLAS_COMPUTE_32F,
            },
        ),
        (
            "algo1/32F",
            cud::GemmTuning {
                algo: A::CUBLAS_GEMM_ALGO1,
                compute: C::CUBLAS_COMPUTE_32F,
            },
        ),
        (
            "algo2/32F",
            cud::GemmTuning {
                algo: A::CUBLAS_GEMM_ALGO2,
                compute: C::CUBLAS_COMPUTE_32F,
            },
        ),
        (
            "algo3/32F",
            cud::GemmTuning {
                algo: A::CUBLAS_GEMM_ALGO3,
                compute: C::CUBLAS_COMPUTE_32F,
            },
        ),
        (
            "default/fast16bf",
            cud::GemmTuning {
                algo: A::CUBLAS_GEMM_DEFAULT,
                compute: C::CUBLAS_COMPUTE_32F_FAST_16BF,
            },
        ),
        (
            "tensorop/fast16bf",
            cud::GemmTuning {
                algo: A::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                compute: C::CUBLAS_COMPUTE_32F_FAST_16BF,
            },
        ),
    ];
    println!("\n=== gate 1 sweep: cuBLAS variants vs tch ===");
    for (name, in_dim, out_dim) in shapes {
        let wb = bf16_bits(out_dim * in_dim, 0x1234 ^ out_dim as u64 ^ in_dim as u64);
        let w_t = tch_bf16(&wb, &[out_dim as i64, in_dim as i64]);
        let w_c = rt.upload_bf16_bits(&wb)?;
        for &m in &[512usize, 2057] {
            let xb = bf16_bits(m * in_dim, 0x99 ^ m as u64);
            let x_t = tch_bf16(&xb, &[m as i64, in_dim as i64]);
            let x_c = rt.upload_bf16_bits(&xb)?;
            let mut out_c = rt.alloc_bf16(m * out_dim)?;
            rt.synchronize()?;
            let mut tch_fn = || -> Res<()> {
                let y = x_t.matmul(&w_t.transpose(0, 1));
                std::hint::black_box(&y);
                Ok(())
            };
            let mut tch_samples = Vec::new();
            for _ in 0..6 {
                tch_samples.push(timing.batch(tch_stream(), 5, &mut tch_fn)?);
            }
            let (tb, tm, _) = stats(tch_samples);
            println!("\n{name} M={m}: tch best {tb:.3} med {tm:.3} ms");
            for (label, tuning) in variants {
                let mut f = || -> Res<()> {
                    rt.gemm_bf16_nt_tuned(&x_c, &w_c, &mut out_c, m, out_dim, in_dim, tuning)?;
                    Ok(())
                };
                let mut samples = Vec::new();
                for _ in 0..6 {
                    samples.push(timing.batch(rt.raw_stream(), 5, &mut f)?);
                }
                let (best, med, _) = stats(samples);
                println!(
                    "  {label:20} best {best:.3} med {med:.3} ms | med {:+.2}%",
                    (med - tm) / tm * 100.0
                );
            }
        }
    }
    Ok(())
}

fn gate_capture(rt: &cud::Runtime, hidden: usize) -> Res<bool> {
    println!("\n=== gate 2: capture/replay into an engine-owned buffer ===");
    let rows = 512usize;
    let xb = bf16_bits(rows * hidden, 0xc0ffee);
    let xb2 = bf16_bits(rows * hidden, 0x5150);
    let wb = bf16_bits(hidden * hidden, 0xbeef);

    // Every buffer is allocated and owned before the capture; nothing inside
    // the captured region allocates or frees.
    let mut x = rt.upload_bf16_bits(&xb)?;
    let w = rt.upload_bf16_bits(&wb)?;
    let mut out = rt.alloc_bf16(rows * hidden)?;
    let mut out_eager2 = rt.alloc_bf16(rows * hidden)?;
    rt.synchronize()?;
    let out_ptr_before = rt.device_ptr(&out);

    rt.gemm_bf16_nt(&x, &w, &mut out, rows, hidden, hidden)?;
    rt.synchronize()?;
    let eager = rt.download_f32(&out)?;

    rt.begin_capture()?;
    rt.gemm_bf16_nt(&x, &w, &mut out, rows, hidden, hidden)?;
    let graph = rt.end_capture()?;
    let out_ptr_after = rt.device_ptr(&out);

    graph.launch()?;
    rt.synchronize()?;
    let replay1 = rt.download_f32(&out)?;
    graph.launch()?;
    rt.synchronize()?;
    let replay2 = rt.download_f32(&out)?;

    let cmp = |a: &[f32], b: &[f32], what: &str| {
        let exact = a
            .iter()
            .zip(b.iter())
            .filter(|(x, y)| x.to_bits() == y.to_bits())
            .count();
        let max = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (*x as f64 - *y as f64).abs())
            .fold(0.0f64, f64::max);
        println!("  {what}: {exact}/{} exact, max|d|={max}", a.len());
        exact == a.len()
    };
    let same = cmp(&eager, &replay1, "replay1 vs eager");
    let det = cmp(&replay1, &replay2, "replay2 vs replay1");

    // New input in the same captured x, then an eager reference and two
    // replays. The graph reads the engine-owned addresses, so a new payload
    // needs no re-capture.
    let x2 = rt.upload_bf16_bits(&xb2)?;
    rt.synchronize()?;
    rt.gemm_bf16_nt(&x2, &w, &mut out_eager2, rows, hidden, hidden)?;
    rt.synchronize()?;
    let eager2 = rt.download_f32(&out_eager2)?;
    // Point the graph at the new payload: the captured graph reads `x`, so
    // copy the new input into that same engine-owned buffer.
    rt.stream()
        .memcpy_dtod(&x2, &mut x)
        .map_err(|e| e.to_string())?;
    rt.synchronize()?;
    graph.launch()?;
    rt.synchronize()?;
    let new1 = rt.download_f32(&out)?;
    graph.launch()?;
    rt.synchronize()?;
    let new2 = rt.download_f32(&out)?;
    let new_det = cmp(&new1, &new2, "new-input replay2 vs replay1");
    let new_same = cmp(&eager2, &new1, "new-input replay vs eager");

    println!(
        "  output ptr stable: {} == {}; graph at {:p}",
        out_ptr_before,
        out_ptr_after,
        graph.cu_graph_exec()
    );
    Ok(same && det && new_det && new_same && out_ptr_before == out_ptr_after)
}

fn gate_interop(driver: &Arc<Driver>, rt: &cud::Runtime, hidden: usize) -> Res<bool> {
    println!("\n=== gate 3: cubin raw-pointer/stream interop ===");
    let rows = 512usize;
    let xb = bf16_bits(rows * hidden, 0x11);
    let rb = bf16_bits(rows * hidden, 0x22);
    let gb = bf16_bits(hidden, 0x33);
    let eps = 1e-6f32;

    // tch-fed: the serving launcher, on torch's stream.
    let x_t = tch_bf16(&xb, &[rows as i64, hidden as i64]);
    let r_t = tch_bf16(&rb, &[rows as i64, hidden as i64]);
    let w_t = tch_bf16(&gb, &[hidden as i64]);
    let sum_t = tch::Tensor::zeros(
        [rows as i64, hidden as i64],
        (Kind::BFloat16, tch::Device::Cuda(0)),
    );
    let out_t = tch::Tensor::zeros(
        [rows as i64, hidden as i64],
        (Kind::BFloat16, tch::Device::Cuda(0)),
    );
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw/config.json",
    )?)?;
    let params = ferrum_torrens::models::qwen3_6::config::parse_text_config(
        cfg.get("text_config").ok_or("no text_config")?,
    )?;
    let gdn = GdnDecode::new(TDevice::Cuda(0), &params, std::ptr::null_mut())?;
    unsafe { gdn.add_rmsnorm(&x_t, &r_t, &w_t, &sum_t, &out_t, eps)? };
    unsafe { driver.synchronize(std::ptr::null_mut())? };
    let tch_sum = to_f32_vec(&sum_t);
    let tch_out = to_f32_vec(&out_t);

    // cudarc-fed: the same cubin, raw ABI, on the cudarc side stream.
    let kernels = cud::LinearKernels::load(Arc::clone(rt.driver()))?;
    let actual_hidden = kernels.hidden()?;
    if actual_hidden != hidden {
        return Err(
            format!("gate 3: cubin HIDDEN={actual_hidden} but test hidden={hidden}").into(),
        );
    }
    let x_c = rt.upload_bf16_bits(&xb)?;
    let r_c = rt.upload_bf16_bits(&rb)?;
    let w_c = rt.upload_bf16_bits(&gb)?;
    let sum_c = rt.alloc_bf16(rows * hidden)?;
    let out_c = rt.alloc_bf16(rows * hidden)?;
    rt.synchronize()?;
    unsafe {
        kernels.add_rmsnorm(
            rt.raw_stream(),
            rt.device_ptr(&x_c),
            rt.device_ptr(&r_c),
            rt.device_ptr(&w_c),
            rt.device_ptr(&sum_c),
            rt.device_ptr(&out_c),
            eps,
            rows,
            hidden,
            hidden as u32,
            hidden as u32,
        )?
    };
    rt.synchronize()?;
    let cand_sum = rt.download_f32(&sum_c)?;
    let cand_out = rt.download_f32(&out_c)?;

    let cmp = |a: &[f32], b: &[f32], what: &str| -> bool {
        let exact = a
            .iter()
            .zip(b.iter())
            .filter(|(x, y)| x.to_bits() == y.to_bits())
            .count();
        let max_abs = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (*x as f64 - *y as f64).abs())
            .fold(0.0f64, f64::max);
        println!("  {what}: {exact}/{} bit-exact, max|d|={max_abs}", a.len());
        exact == a.len()
    };
    let ok =
        cmp(&tch_sum, &cand_sum, "add_rmsnorm sum") & cmp(&tch_out, &cand_out, "add_rmsnorm out");
    Ok(ok)
}

fn main() -> Res<()> {
    let which = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    let driver = Arc::new(Driver::open()?);
    driver.init()?;
    driver.attach_primary_context(0)?;
    // Register ATen's CUDA backend for the tch side of gates 1 and 3.
    ferrum_torrens::infrastructure::torch::load_cuda_backend()
        .map_err(|e| format!("libtorch cuda: {e}"))?;
    // The same driver and primary context the tch side uses; no second context.
    let rt = cud::Runtime::open_with(Arc::clone(&driver), 0)?;
    let hidden = H;

    let started = Instant::now();
    let mut all = true;
    if which == "sweep" {
        gate_sweep(&driver, &rt)?;
        return Ok(());
    }
    if which == "all" || which == "gemm" {
        all &= gate_gemm(&driver, &rt)?;
    }
    if which == "all" || which == "capture" {
        all &= gate_capture(&rt, hidden)?;
    }
    if which == "all" || which == "interop" {
        all &= gate_interop(&driver, &rt, hidden)?;
    }
    println!(
        "\nM7.1a gates 1-3: {} in {:.1}s",
        if all { "PASS" } else { "FAIL" },
        started.elapsed().as_secs_f64()
    );
    if !all {
        std::process::exit(1);
    }
    Ok(())
}
