//! M7.0 spike gates 1-3: candle BF16 GEMM parity/speed against `tch`,
//! capture/replay of candle ops plus one committed cubin on the candle stream,
//! and bit-exact raw-pointer interop for `add_rmsnorm`.
//!
//! Run inside the devshell with:
//!
//!   FERRUM_KERNEL_DIR=<kernels store path> \
//!   CUDA_COMPUTE_CAP=120 NVCC_CCBIN=<gcc-14> NVCC_PREPEND_FLAGS="-I$CUDA_HOME/include" \
//!   cargo run --release --features cuda,candle --example m7_0_gemm -- all
//!
//! This is a spike: it links both libtorch and candle on purpose (gate 1/3
//! compare them) and leaks the captured graph at exit so the graph-owned
//! allocations outlive the test.

use std::sync::Arc;
use std::time::Instant;

use candle_core::{DType, Device as CDevice, Tensor as CTensor};
use ferrum_torrens::domain::expert_kernel::MoEError;
use ferrum_torrens::infrastructure::candle as cand;
use ferrum_torrens::infrastructure::cuda_driver::{CAPTURE_MODE_GLOBAL, CUstream, Driver};
use ferrum_torrens::infrastructure::cuda_gdn::GdnDecode;
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

fn candle_bf16(bits: &[u16], shape: &[usize], dev: &CDevice) -> Res<CTensor> {
    let vals: Vec<bf16> = bits.iter().map(|b| bf16::from_bits(*b)).collect();
    Ok(CTensor::from_vec(vals, shape, dev)?)
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
        candle_stream: CUstream,
        rounds: usize,
        reps: usize,
        mut tch: F,
        mut candle: G,
    ) -> Res<(Vec<f64>, Vec<f64>)>
    where
        F: FnMut() -> Res<()>,
        G: FnMut() -> Res<()>,
    {
        for _ in 0..5 {
            tch()?;
            candle()?;
        }
        unsafe {
            self.driver.synchronize(tch_stream)?;
            self.driver.synchronize(candle_stream)?;
        }
        let (mut a, mut b) = (Vec::with_capacity(rounds), Vec::with_capacity(rounds));
        for round in 0..rounds {
            if round % 2 == 0 {
                a.push(self.batch(tch_stream, reps, &mut tch)?);
                b.push(self.batch(candle_stream, reps, &mut candle)?);
            } else {
                b.push(self.batch(candle_stream, reps, &mut candle)?);
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

fn gate_gemm(driver: &Arc<Driver>, cand_dev: &CDevice, cand_stream: CUstream) -> Res<bool> {
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
    println!("\n=== gate 1: BF16 matmul parity and speed ===");
    for s in &shapes {
        let (w_rows, w_cols) = (s.out_dim, s.in_dim);
        let wb = bf16_bits(w_rows * w_cols, 0x1234 ^ w_rows as u64 ^ w_cols as u64);
        let w_t = tch_bf16(&wb, &[w_rows as i64, w_cols as i64]);
        let w_c = candle_bf16(&wb, &[w_rows, w_cols], cand_dev)?;
        println!(
            "\n{}: weight [{w_rows}, {w_cols}] bf16 ({:.1} MiB)",
            s.name,
            (wb.len() * 2) as f64 / (1 << 20) as f64
        );
        for &m in &ms {
            let xb = bf16_bits(m * s.in_dim, 0x99 ^ m as u64);
            let x_t = tch_bf16(&xb, &[m as i64, s.in_dim as i64]);
            let x_c = candle_bf16(&xb, &[m, s.in_dim], cand_dev)?;

            // Correctness: the actual projection (transposed weight view).
            // Compare in row chunks so the f32 staging tensor never has to be
            // the whole [2057, vocab] output at once.
            let y_t = x_t.matmul(&w_t.transpose(0, 1)).to_kind(Kind::BFloat16);
            let y_c = x_c.matmul(&w_c.t()?)?.to_dtype(DType::BF16)?;
            let chunk = 256.min(m);
            let (mut max_abs, mut scale, mut equal) = (0.0f64, 0.0f64, 0usize);
            let mut start = 0usize;
            while start < m {
                let n = chunk.min(m - start);
                let ct = y_t.narrow(0, start as i64, n as i64).to_kind(Kind::Float);
                let cc = y_c
                    .narrow(0, start, n)?
                    .to_dtype(DType::F32)?
                    .contiguous()?;
                let ht = to_f32_vec(&ct);
                let hc = cand::to_f32_vec(&cc)?;
                if ht.len() != hc.len() {
                    return Err(format!(
                        "gate 1: output length mismatch {} vs {}",
                        ht.len(),
                        hc.len()
                    )
                    .into());
                }
                for (a, b) in ht.iter().zip(hc.iter()) {
                    max_abs = max_abs.max((*a as f64 - *b as f64).abs());
                    scale = scale.max(a.abs() as f64);
                }
                let at = argmax_rows(&ht, n, w_rows);
                let ac = argmax_rows(&hc, n, w_rows);
                equal += at.iter().zip(ac.iter()).filter(|(a, b)| a == b).count();
                start += n;
            }
            let max_rel = if scale > 0.0 { max_abs / scale } else { 0.0 };
            // bf16 has 8 mantissa bits; the same cuBLAS f32 accumulation then
            // rounds to bf16. A 1/16 relative envelope admits the final
            // rounding while still catching a wrong transpose or layout.
            let pass = equal == m && max_rel <= 0.0625;
            ok &= pass;
            println!(
                "  M={m:4}: argmax {equal}/{m}, max|d|={max_abs:.4}, max rel={max_rel:.5} {}",
                if pass { "PASS" } else { "FAIL" }
            );
            drop(y_t);
            drop(y_c);

            let mut tch_fn = || -> Res<()> {
                let y = x_t.matmul(&w_t.transpose(0, 1));
                std::hint::black_box(&y);
                Ok(())
            };
            let mut candle_fn = || -> Res<()> {
                let y = x_c.matmul(&w_c.t()?)?;
                std::hint::black_box(&y);
                Ok(())
            };
            let (tch_times, candle_times) =
                timing.ab(tch_stream(), cand_stream, 8, 5, &mut tch_fn, &mut candle_fn)?;
            let (tb, tm, _) = stats(tch_times);
            let (cb, cm, _) = stats(candle_times);
            let delta = (cm - tm) / tm * 100.0;
            println!(
                "        matmul tch best {tb:.3} med {tm:.3} ms | candle best {cb:.3} med {cm:.3} ms | med {delta:+.2}%"
            );
            if m != 1 && delta > 5.0 {
                println!("        FAIL: candle is more than 5% slower on a prefill shape");
                ok = false;
            }
        }
    }
    Ok(ok)
}

fn probe_capture(cand_dev: &CDevice) -> Res<()> {
    use candle_core::cuda_backend::cudarc::driver::sys::{
        CUgraphInstantiate_flags, CUstreamCaptureMode,
    };
    println!("\n=== capture probe: one candle matmul, cudarc graph ===");
    let (stream, _) = cand::stream(cand_dev)?;
    let x = candle_bf16(&bf16_bits(512 * 2048, 1), &[512, 2048], cand_dev)?;
    let w = candle_bf16(&bf16_bits(2048 * 2048, 2), &[2048, 2048], cand_dev)?;
    let b = candle_bf16(&bf16_bits(2048, 3), &[2048], cand_dev)?;
    let gamma = candle_bf16(&bf16_bits(2048, 4), &[2048], cand_dev)?;
    let gamma32 = gamma.to_dtype(DType::F32)?;
    let run_eager = || -> Res<CTensor> {
        let y = x.matmul(&w.t()?)?.broadcast_add(&b)?;
        let y32 = y.to_dtype(DType::F32)?;
        let r = candle_nn::ops::rms_norm(&y32, &gamma32, 1e-6)?;
        Ok(r.to_dtype(DType::BF16)?.contiguous()?)
    };
    let eager = cand::to_f32_vec(&run_eager()?)?;
    stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)?;
    let y = x.matmul(&w.t()?)?.broadcast_add(&b)?;
    let y32 = y.to_dtype(DType::F32)?;
    let r = candle_nn::ops::rms_norm(&y32, &gamma32, 1e-6)?;
    let r = r.to_dtype(DType::BF16)?.contiguous()?;
    let graph = stream
        .end_capture(CUgraphInstantiate_flags(1))?
        .ok_or("end_capture returned no graph")?;
    stream.synchronize()?;
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
    };
    graph.launch()?;
    stream.synchronize()?;
    let replay1 = cand::to_f32_vec(&r)?;
    cmp(&eager, &replay1, "rms replay1 vs eager");
    graph.launch()?;
    stream.synchronize()?;
    let replay2 = cand::to_f32_vec(&r)?;
    cmp(&replay1, &replay2, "rms replay2 vs replay1");
    std::mem::forget(graph);
    std::mem::forget(y);
    std::mem::forget(y32);
    std::mem::forget(r);
    Ok(())
}
fn gate_capture(
    driver: &Arc<Driver>,
    cand_dev: &CDevice,
    cand_stream: CUstream,
    hidden: usize,
) -> Res<bool> {
    use ferrum_torrens::infrastructure::candle::device_ptr;
    println!("\n=== gate 2: capture/replay ===");
    let rows = 512usize;
    unsafe { cand::disable_event_tracking(cand_dev)? };

    let xb = bf16_bits(rows * hidden, 0xc0ffee);
    let xb2 = bf16_bits(rows * hidden, 0x5150);
    let wb = bf16_bits(hidden * hidden, 0xbeef);
    let bb = bf16_bits(hidden, 0xfeed);
    let gb = bf16_bits(hidden, 0xabc);
    let x = candle_bf16(&xb, &[rows, hidden], cand_dev)?;
    let x2 = candle_bf16(&xb2, &[rows, hidden], cand_dev)?;
    let w = candle_bf16(&wb, &[hidden, hidden], cand_dev)?;
    let b = candle_bf16(&bb, &[hidden], cand_dev)?;
    let gamma = candle_bf16(&gb, &[hidden], cand_dev)?;
    let gamma32 = gamma.to_dtype(DType::F32)?;

    let mk =
        |dev: &CDevice| -> Res<CTensor> { Ok(CTensor::zeros(rows * hidden, DType::BF16, dev)?) };
    let kernels = cand::LinearKernels::load(Arc::clone(driver))?;
    println!(
        "  hidden={} rows={} stream={:p} async_alloc={}",
        hidden,
        rows,
        cand_stream,
        cand_dev
            .as_cuda_device()?
            .cuda_stream()
            .context()
            .has_async_alloc()
    );

    let (cstream, _) = cand::stream(cand_dev)?;
    let run = |x: &CTensor,
               sum: &CTensor,
               out: &CTensor,
               kernels: &cand::LinearKernels,
               stream: CUstream|
     -> Res<()> {
        let y = x.matmul(&w.t()?)?.broadcast_add(&b)?;
        let y32 = y.to_dtype(DType::F32)?;
        let rms32 = candle_nn::ops::rms_norm(&y32, &gamma32, 1e-6)?;
        let rms = rms32.to_dtype(DType::BF16)?.contiguous()?;
        unsafe {
            kernels.add_rmsnorm(
                stream,
                device_ptr(x)?,
                device_ptr(&rms)?,
                device_ptr(&gamma)?,
                device_ptr(sum)?,
                device_ptr(out)?,
                1e-6,
                rows,
                hidden,
                hidden as u32,
                hidden as u32,
            )?
        };
        Ok(())
    };

    // Every eager run happens before the capture, so its allocations have
    // returned to the pool and cannot perturb the graph's memory nodes.
    let sum_e = mk(cand_dev)?;
    let out_e = mk(cand_dev)?;
    run(&x, &sum_e, &out_e, &kernels, cand_stream)?;
    cstream.synchronize()?;
    let eager_sum = cand::to_f32_vec(&sum_e)?;
    let eager_out = cand::to_f32_vec(&out_e)?;
    drop(sum_e);
    drop(out_e);

    let sum_e2 = mk(cand_dev)?;
    let out_e2 = mk(cand_dev)?;
    run(&x2, &sum_e2, &out_e2, &kernels, cand_stream)?;
    cstream.synchronize()?;
    let eager2_sum = cand::to_f32_vec(&sum_e2)?;
    let eager2_out = cand::to_f32_vec(&out_e2)?;
    drop(sum_e2);
    drop(out_e2);
    cstream.synchronize()?;

    let sum_c = mk(cand_dev)?;
    let out_c = mk(cand_dev)?;
    let sum_ptr_before = device_ptr(&sum_c)?;
    let out_ptr_before = device_ptr(&out_c)?;

    let capture_status = || -> Res<bool> {
        use candle_core::cuda_backend::cudarc::driver::sys::CUstreamCaptureStatus;
        Ok(cstream.capture_status()? != CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE)
    };

    unsafe { driver.begin_capture(cand_stream, CAPTURE_MODE_GLOBAL)? };
    if !capture_status()? {
        return Err("gate 2: stream is not capturing after begin_capture".into());
    }
    let y = x.matmul(&w.t()?)?.broadcast_add(&b)?;
    if !capture_status()? {
        return Err("gate 2: capture ended during candle matmul/add".into());
    }
    let y32 = y.to_dtype(DType::F32)?;
    let rms32 = candle_nn::ops::rms_norm(&y32, &gamma32, 1e-6)?;
    if !capture_status()? {
        return Err("gate 2: capture ended during candle rms_norm".into());
    }
    // Keep every intermediate alive: a drop inside the captured region is a
    // free node in the graph and makes later replays invalid.
    let rms = rms32.to_dtype(DType::BF16)?.contiguous()?;
    unsafe {
        kernels.add_rmsnorm(
            cand_stream,
            device_ptr(&x)?,
            device_ptr(&rms)?,
            device_ptr(&gamma)?,
            device_ptr(&sum_c)?,
            device_ptr(&out_c)?,
            1e-6,
            rows,
            hidden,
            hidden as u32,
            hidden as u32,
        )?
    };
    if !capture_status()? {
        return Err("gate 2: capture ended during the add_rmsnorm cubin".into());
    }
    // Free the graph-owned intermediates inside the captured region, after the
    // cubin that reads them. That gives the graph an alloc/free pair per
    // temporary, so it can be relaunched without AUTO_FREE_ON_LAUNCH.
    drop(rms);
    drop(rms32);
    drop(y32);
    drop(y);
    let sum_ptr_after = device_ptr(&sum_c)?;
    let out_ptr_after = device_ptr(&out_c)?;
    let graph = unsafe { driver.end_capture(cand_stream)? };
    let exec = unsafe { driver.graph_instantiate(graph)? };
    unsafe { driver.graph_launch(exec, cand_stream)? };
    cstream.synchronize()?;
    println!(
        "  captured graph={:p} exec={:p}; replay output ptr stable: sum {} == {}, out {} == {}",
        graph, exec, sum_ptr_before, sum_ptr_after, out_ptr_before, out_ptr_after
    );

    // Read back through the driver only: no candle allocation between replays,
    // so nothing can reuse a graph memory node's address.
    let read = |t: &CTensor| cand::bf16_f32_via_d2h(driver, t);
    let replay_a = read(&out_c)?;
    let replay_a_sum = read(&sum_c)?;
    unsafe { driver.graph_launch(exec, cand_stream)? };
    cstream.synchronize()?;
    let replay_b = read(&out_c)?;
    let replay_b_sum = read(&sum_c)?;
    let det = replay_a
        .iter()
        .zip(replay_b.iter())
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    let det_sum = replay_a_sum
        .iter()
        .zip(replay_b_sum.iter())
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    let det_max = replay_a
        .iter()
        .zip(replay_b.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0f64, f64::max);
    let det_sum_max = replay_a_sum
        .iter()
        .zip(replay_b_sum.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0f64, f64::max);
    println!(
        "  replay 1 vs replay 2 (same input): out {det}/{}, max|d|={det_max}; sum {det_sum}/{}, max|d|={det_sum_max}",
        replay_a.len(),
        replay_a_sum.len()
    );

    let sum_equal = replay_a_sum
        .iter()
        .zip(eager_sum.iter())
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    let sum_max = replay_a_sum
        .iter()
        .zip(eager_sum.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0f64, f64::max);
    let equal = replay_a
        .iter()
        .zip(eager_out.iter())
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    let max_abs = replay_a
        .iter()
        .zip(eager_out.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0f64, f64::max);
    let lo = eager_out.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = eager_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "  replay vs eager: sum {sum_equal}/{} exact max|d|={sum_max}; out {equal}/{} exact max|d|={max_abs}; range [{lo}, {hi}]",
        replay_a_sum.len(),
        replay_a.len()
    );

    // New input: H2D into the captured x, then two replays.
    let mut x_bytes = Vec::with_capacity(xb2.len() * 2);
    for v in &xb2 {
        x_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let free_before = driver.mem_info()?.0;
    unsafe { driver.copy_h2d(device_ptr(&x)?, &x_bytes)? };
    unsafe { driver.graph_launch(exec, cand_stream)? };
    cstream.synchronize()?;
    let replay_c = read(&out_c)?;
    let replay_c_sum = read(&sum_c)?;
    unsafe { driver.graph_launch(exec, cand_stream)? };
    cstream.synchronize()?;
    let replay_d = read(&out_c)?;
    let free_after = driver.mem_info()?.0;
    let det2 = replay_c
        .iter()
        .zip(replay_d.iter())
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    let det2_max = replay_c
        .iter()
        .zip(replay_d.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0f64, f64::max);
    let equal2 = replay_c
        .iter()
        .zip(eager2_out.iter())
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    let max2 = replay_c
        .iter()
        .zip(eager2_out.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0f64, f64::max);
    let sum2_equal = replay_c_sum
        .iter()
        .zip(eager2_sum.iter())
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    println!(
        "  new input: replay {det2}/{} exact across runs (max|d|={det2_max}), vs eager out {equal2}/{} exact max|d|={max2}, sum {sum2_equal}/{} exact, free VRAM {} -> {} MiB",
        replay_c.len(),
        replay_c.len(),
        replay_c_sum.len(),
        free_before >> 20,
        free_after >> 20
    );

    let ok = det == replay_a.len()
        && det_sum == replay_a_sum.len()
        && det2 == replay_c.len()
        && sum_max <= 0.5
        && max_abs <= 0.5
        && max2 <= 0.5
        && device_ptr(&sum_c)? == sum_ptr_before
        && device_ptr(&out_c)? == out_ptr_before;

    Ok(ok)
}

fn gate_interop(
    driver: &Arc<Driver>,
    cand_dev: &CDevice,
    cand_stream: CUstream,
    hidden: usize,
) -> Res<bool> {
    use ferrum_torrens::infrastructure::candle::device_ptr;
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

    // candle-fed: the same cubin, raw-ABI, on candle's stream.
    let kernels = cand::LinearKernels::load(Arc::clone(driver))?;
    // The cubin bakes the serving hidden size.
    let actual_hidden = kernels.hidden()?;
    if actual_hidden != hidden {
        return Err(
            format!("gate 3: cubin HIDDEN={actual_hidden} but test hidden={hidden}").into(),
        );
    }
    let x_c = candle_bf16(&xb, &[rows, hidden], cand_dev)?;
    let r_c = candle_bf16(&rb, &[rows, hidden], cand_dev)?;
    let w_c = candle_bf16(&gb, &[hidden], cand_dev)?;
    let sum_c = CTensor::zeros(rows * hidden, DType::BF16, cand_dev)?;
    let out_c = CTensor::zeros(rows * hidden, DType::BF16, cand_dev)?;
    unsafe {
        kernels.add_rmsnorm(
            cand_stream,
            device_ptr(&x_c)?,
            device_ptr(&r_c)?,
            device_ptr(&w_c)?,
            device_ptr(&sum_c)?,
            device_ptr(&out_c)?,
            eps,
            rows,
            hidden,
            hidden as u32,
            hidden as u32,
        )?
    };
    let (cstream, _) = cand::stream(cand_dev)?;
    cstream.synchronize()?;
    let cand_sum = cand::to_f32_vec(&sum_c)?;
    let cand_out = cand::to_f32_vec(&out_c)?;

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
    let cand_dev = cand::device(0)?;
    // The engine owns its single stream and all buffer lifetimes; cudarc's
    // per-tensor event tracking is exactly the host work a captured/serving
    // step does not want.
    unsafe { cand::disable_event_tracking(&cand_dev)? };
    let cand_raw = cand::raw_stream(&cand_dev)?;
    let hidden = H;

    let started = Instant::now();
    let mut all = true;
    if which == "all" || which == "gemm" {
        all &= gate_gemm(&driver, &cand_dev, cand_raw)?;
    }
    if which == "probe" {
        probe_capture(&cand_dev)?;
        return Ok(());
    }
    if which == "all" || which == "capture" {
        all &= gate_capture(&driver, &cand_dev, cand_raw, hidden)?;
    }
    if which == "all" || which == "interop" {
        all &= gate_interop(&driver, &cand_dev, cand_raw, hidden)?;
    }
    println!(
        "\nM7.0 gates 1-3: {} in {:.1}s",
        if all { "PASS" } else { "FAIL" },
        started.elapsed().as_secs_f64()
    );
    if !all {
        std::process::exit(1);
    }
    let _ = MoEError::Bad(String::new());
    Ok(())
}
