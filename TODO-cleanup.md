# Cleanup backlog (proposed, not executed)

Status: **surveyed 2026-11-16, nothing removed yet.** This is a candidate list,
not a plan of record. `TODO.md` still decides what gets worked on; each item
here needs a go-ahead and its own gate before it lands.

Scope: dead code, rejected experiment arms that are still compiled and
reachable through env vars, obsolete benchmarks and probes, unused dependency
weight, and the local FreeToken checkout that sits inside the working tree.

Explicitly **not** in scope:

- **Speculative decode.** `application/spec_decode.rs`, `domain/spec.rs`,
  `models/qwen3_6/mtp.rs`, `[spec]` config, `tests/engine_spec_generate.rs`,
  `tests/cudarc_spec_verify.rs`, `FERRUM_SPEC_*` and `FERRUM_SPEC_PROFILE` all
  stay. That is the next work item (TODO open item 2), not cleanup. One finding
  is recorded under A6: the host MTP draft has no caller today.
- **The CPU oracle path.** `models/qwen3_6/forward.rs`, the non-`cudarc`
  registry arm and the CPU fallback stay: they are the M1/M3 oracle and the
  reason `cargo test --features cudarc` is meaningful on a host with no device.
- **Plan-of-record instruments.** `infrastructure/profile.rs`,
  `infrastructure/host_profile.rs`, `infrastructure/dup.rs`,
  `scripts/cupti-trace/`, `scripts/freetoken_probe/baselines.py`,
  `scripts/route-trace-sim.py` are named in `OPTIMIZATION.md` (line 135) and
  stay unless the plan itself is rewritten.
- **`wip/vmm-slot-banks`.** A branch, not code on `main`; leave it alone.

## How each batch lands

One commit per batch, conventional-commit subject, and `DONE.md` gets the
decision entries for anything that changes what the engine can do (the
experiment removals below). Gates for every batch:

```sh
cargo fmt --check
cargo clippy --all-targets
cargo clippy --all-targets --features cudarc
cargo test --lib                     # currently 435 passing
```

Batches that touch `python/ferrum_kernel/` also run `kernel-selftest`,
`kernel-compile-check` and, before committing, `nix build .#kernels` (the
`kernel/` tree is gitignored; the derivation rebuilds it from tracked sources).

Batches that change the set of loaded cubins (B1, B3) shift the device module
layout, so they also need the canonical decode check, not just unit tests:

```sh
FERRUM_SMOKE_PROMPT="$(cat bench/prompt-clean.txt)" FERRUM_SMOKE_TOKENS=48 \
FERRUM_SMOKE_GRAPH=1 FERRUM_SMOKE_REPS=3 prlimit --as=68719476736 -- <bin>
```

Gate: the 48-token trajectory md5 is still `790b7e4fc852` (teacher-forced ids
or the canonical greedy run, whichever the batch uses) and tg stays inside the
known 36.6-39.0 band with n >= 8 order-balanced reps. Cubin count goes 69 to
67 after B1 and, if B3 drops the runtime variants too, to 65; record the count
in the commit rather than letting it change silently.

---

## Batch A -- dead code, zero behaviour change

### A1. Drop 14 unused methods from `infrastructure/cudarc.rs` (~119 lines)

Each of these appears exactly once in the repo (its definition); no `src/`,
`tests/`, `examples/` or script calls it.

| method | lines |
|---|---|
| `Runtime::alloc_f32_shared` | 4 |
| `Runtime::tensor_from_device_f32` | 14 |
| `Runtime::alloc_u8` | 4 |
| `Runtime::upload_bf16_bits` | 4 |
| `Runtime::download_f32_range` | 11 |
| `Runtime::device_ptr_f32` | 4 |
| `Runtime::device_ptr_u8` | 4 |
| `Runtime::gemm_bf16_nt_tuned` | 50 |
| `Tensor::as_u8_slice` | 3 |
| `Tensor::to_host_bf16_bits` | 8 |
| `Tensor::div_scalar` | 3 |
| `Tensor::sum_all` | 3 |
| `Tensor::max_all` | 3 |
| `LinearKernels::norm_hidden` | 4 |

`gemm_bf16_nt_tuned` is the surviving half of the pre-torch-removal cuBLAS
variant spike (its doc says "so the spike can find the variant ATen picks");
the spike is over and the tuned entry point was never wired to `gemm_bf16_nt`.

After `gemm_bf16_nt_tuned` goes, `GemmTuning` (lines 63-77) has only the
`GemmTuning::default().compute/algo` use inside `gemm_bf16_nt`; either inline
those two constants or keep the struct with just its default. No numerical
change either way.

Gate: unit suite only (no Python, no kernels).

### A2. Drop 6 unused free functions (~29 lines)

- `infrastructure/cuda_moe.rs`: `cache_bytes` (3)
- `models/qwen3_6/forward_cudarc.rs`: `split2` (3), `split3` (7),
  `draft_bytes` (3)
- `models/qwen3_6/vision.rs`: `has_weight` (3), `num_visual_tokens` (10)

`draft_bytes` sits next to the spec-decode draft path but is itself uncalled;
confirm it is not planned for the K-sweep work before deleting.

### A3. Rename `infrastructure/torch.rs` to `infrastructure/device.rs`

The file has no torch left in it: it is device resolution
(`DeviceRequest`/`Device`/`CudaProbe`/`resolve_device`) over the driver probe.
223 lines, 35 references across `src/` and `tests/`. Six other modules import
it (`cuda_attention`, `cuda_gdn`, `cuda_linear`, `cuda_moe`, `kvcache_dev`,
`decode_graph`) plus two test modules. Pure rename; the only risk is a missed
`use` and the `mod` line in `infrastructure/mod.rs`.

The `TODO.md` code map (line ~180) should change in the same commit.

### A4. Config ceremony: three single-variant enums (needs a decision)

`config.rs:39` `MoeStrategy`, `:46` `ExpertLoad`, `:53` `CachePolicy` each have
exactly one variant (`Offload`, `Serial`, `Lru`). The fields
(`moe.strategy`, `moe.expert_load`, `moe.cache_policy`, lines 183/185/191) are
never read outside `config.rs`; they exist so a deployed config can spell them.
Removing them is ~40 lines plus test asserts, but it **breaks the config
schema** (`deny_unknown_fields` would reject old files). Recommendation: leave
them unless the schema is being versioned anyway; if removed, accept the three
keys and ignore them for one release.

### A5. `spec.draft_model_path` is parsed and never read

`config.rs:266` / default at `:279`. No code path uses it; `[spec] mtp` uses the
target checkpoint's own head. Because spec decode is next, this is a
"wire it or delete it" decision, not a straight removal. Lean: delete it and
re-add when a separate draft model is actually supported.

### A6. `models/qwen3_6/mtp.rs` host draft has no caller

`draft_forward` (154 lines) is the only independent reference implementation
of the MTP draft layer; the device path is
`forward_cudarc::draft_forward`. Nothing in `src/` or `tests/` calls the host
one. Since spec decode is next, keep it and **wire it into a gate** (host draft
vs device draft on the same hidden rows) rather than delete it. Record the
missing gate in `TODO.md` open item 2.

---

## Batch B -- rejected experiment arms still compiled and env-reachable

Each of these is a losing A/B arm that survived because it is env-selectable.
Removing one is a behaviour-preserving cleanup only in the sense that the
default path is unchanged; the removed arm must be recorded in `DONE.md`.

### B1. The bf16 shared expert (`FERRUM_SHARED_BF16=1`)

Estimated ~490 lines across 10 files, plus 2 of 69 cubins
(`shared_expert_phase1`, `shared_expert_phase2`). `FERRUM_SHARED_BAKE`
toggles the load-time step that bakes the NVFP4 global into the bf16 weight;
it belongs to this arm and goes with it.

- `python/ferrum_kernel/shared_expert.py` (148) -- the bf16 fused kernels
- `infrastructure/cuda_linear.rs:340-449` `SharedExpertFused` (110)
- `infrastructure/kernel_abi.rs` consts (645-648), name lists (731-732),
  load/launch branches (964-975, 1089-1095), arg structs and packers
  (2169-2245), ~85
- `models/qwen3_6/load_cudarc.rs:399-452` the load-time "bake the NVFP4
  global into the bf16 weight" block, ~54
- `models/qwen3_6/forward_cudarc.rs:1458-1488` the fused arm, plus the
  `shared_expert_fused` field and constructor args, ~35
- `application/registry.rs:720-739` `shared_expert_fused_port`, ~20
- `aot.py` import and dict entries, `signatures.py`, `selftest.py`,
  `compile_check.py`, ~40

Evidence that it lost: decision 166 measured NVFP4 fused 34.30 mean against
bf16 fused 33.67 on the clean prompt, and found the win was **fusing the
epilogue, not the dtype**; the unfused bf16 arm is the slowest of the three.
`FERRUM_SHARED_BF16` has zero references in `tests/`, `scripts/`, `nix/`,
`README.md` or `TODO.md` -- it is only reachable by setting the variable by
hand.

The NVFP4 fused path (`shared_expert_nvfp4.py`, `SharedExpertNvfp4`) is the
default and stays.

Risk: removing the arm drops two loaded modules, which is exactly the kind of
device-layout change decision 166 warned about ("every extra loaded module
shifts the device layout"). Gate on the canonical trajectory md5 plus the
gated `engine_real_generate` oracle, not just `cargo test --lib`.

### B2. `FERRUM_PF_SEGMENTS=all` (~45 lines)

`cuda_moe.rs:1798`, `1807-1812`, `2212-2221` and its unit test `2629-2636`.
It reproduces the pre-R10.10 grouped-GEMM grid (every cache slot instead of
the half's 2E buffer slots) for an A/B that decision R10.10 already settled.
Only `DONE.md`'s history (around its R10.10 entry) and the knob's own unit
test mention it. Removing it deletes the `all_segments` branch and the
`seg_ids` host build for that case.

### B3. `FERRUM_ATTN_SPLITK=0` (the single-launch paged attention)

~110 Rust lines (`cuda_attention.rs:173-190` plus the `AttentionArgs` packing
and `paged_attention_args`/`attention_grid` path in `kernel_abi.rs`) and the
Python kernel it selects (`attention.py:42-106`, ~65 lines), plus
`paged_attention` and `paged_attention_bf16` cubins (2 of 69).

Evidence: decision 158 ported split-K and made it the default (+1.0% overall,
+0.5% isolated), and the non-split path has been the `=0` fallback since.
Caveat: `python/ferrum_kernel/selftest.py:383` still calls
`paged_attention_decode` (the non-split wrapper) as its numerical reference.
Keep that Python function and its kernel registered for the selftest, or
repoint the selftest at `paged_attention_reference`; only the Rust dispatch
arm and the two cubins need to go if the goal is a smaller artifact.

### B4. `FERRUM_FUSED_ROUTER=0` (~15 lines)

`forward_cudarc.rs:1420-1431` (the `topk_renorm` fallback and its duplicate
call under the `FERRUM_DUP` hook). Decision 158 made the fused router the
default (+2.5% isolated) and left the unfused arm as the `=0` fallback. It has
no other caller.

### B5. `FT_ABLATE` (~45 lines)

`forward_cudarc.rs:36-49` plus seven call sites (1850, 1854, 1857, 1869, 1912,
2247, 2315). It skips mixer/attn/gdn/moe/head stages so a prefill run can
attribute time, but the resulting logits are not a forward result. It is not
in the instrument list at `OPTIMIZATION.md:135`; the same attribution now
comes from `FT_PROFILE` on the eager path and CUPTI on the captured path.
Decision 1999 records it as a profiling aid at the time it was added.

### B6. `FERRUM_PF_DEBUG` (~18 lines)

Three print sites (`cuda_moe.rs:1578`, `1763`, `2122`) that dump the retained
expert window and each layer-0 residency. Decision 5322 records the bug it
found; the bug is fixed and gated by
`engine_real_generate::a_warmup_request_does_not_corrupt_the_next_decode`.

### B7. Keep, for now: `FERRUM_NORM_FUSED=0` and `FERRUM_GDN_TAIL=0`

These two off-arms were added one session ago (decisions 174/176) with the
comment "so one binary can be measured against itself". They are small and
still useful for the next node-count bundle; revisit after that bundle lands
and the fusions have survived a re-measurement.

---

## Batch C -- dev probes to retire or promote

These are all default-off `getenv` probes, ~120 lines total. None is in the
`OPTIMIZATION.md` instrument list, so each is either promoted into that list
or retired. Recommendation next to each.

| probe | where | size | recommendation |
|---|---|---:|---|
| `FT_TRACE=1` | `models/qwen3_6/forward.rs:834-858`, ~18 call sites | ~25 | delete; the CPU oracle's stage trace predates `FT_LOGIT_TRACE` and CUPTI, and nothing in `TODO.md`/`OPTIMIZATION.md` cites it |
| `FT_LOGIT_TRACE=1` | `application/generation.rs:58-101`, call at `:727` | ~50 | keep or promote: it is the only tool that answers "near-tie or real divergence" on a captured-arm logit difference; if kept, add it to the instrument list |
| `FT_SLOTS_REPORT` | `application/registry.rs:669-681` | ~14 | delete or fold into the existing `[slots]` info log behind `RUST_LOG`; it is a one-shot print, not a measurement |
| `FT_EMBED_HOST=0` | `models/qwen3_6/load_cudarc.rs:303-316` | ~14 | delete the A/B arm; host-pinned embedding is the default and the device fallback only exists for the never-repeated comparison |
| `FERRUM_PF_DEBUG` | see B6 | ~18 | delete (in B6) |

If `FT_TRACE` goes, the `trace`/`trace_vec` call sites in the CPU oracle go
with it; that is the single largest block in this batch.

---

## Batch D -- obsolete benchmarks, probes, examples and plans

### D1. `examples/m7_2_load_cudarc.rs` (55 lines) and its `[[example]]` entry

The M7.2 milestone demo; `tests/cudarc_load.rs` is the same gate with
assertions and is what the docs cite. `Cargo.toml`'s `[[example]]` block (5
lines, `required-features = ["cudarc"]`) goes too. Nothing else references the
example (`DONE.md:4256` mentions its report as history).

### D2. `tests/vision_prefill_churn.rs` (168 lines)

Its own header says "It is a probe, not a correctness gate"; it measured the
R12 step-0 question (short prefills stay on chunked staged LRU) and decision
145 records the answer. The vision gates that matter are
`tests/vision_smoke.rs` and `tests/vision_cudarc_tower.rs`, which stay.

### D3. `scripts/freetoken_probe/r16_reprofile.py` (473 lines)

The R16 reference re-profile ran on a payload decision 165 invalidated (the
tainted prompt / degenerate trajectories). The canonical reference protocol is
`scripts/freetoken_probe/baselines.py` plus `bench/prompt-clean.txt`; nothing
in `TODO.md` or `OPTIMIZATION.md` points at `r16_reprofile.py`. The
`probe_impl.py`/`sitecustomize.py` pair it depends on stays (baselines and the
torch-profiler trace use it).

### D4. `scripts/cupti-trace/decode-steps.py` (85 lines)

Superseded by `step-budget.py`, whose docstring says decode-steps "answers how
much device time each kernel took" while step-budget answers the span/union
question the plan now decides on. Its only remaining reference is the usage
comment in `scripts/cupti-trace/build.sh:13`; change that comment to
`step-budget.py` in the same commit.

### D5. `python/ferrum_kernel/autotune_min.py` (215 lines)

A one-off timing proof that an autotune sweep does not need torch. The answer
is already recorded in `TODO.md` open item 4 ("A fully torch-free autotune ...
is open but not scheduled"). `autotune.py` (the real sweep) stays. Fold the
conclusion into `DONE.md` and drop the proof script, or keep it as the
starting point for the torch-free harness; decide when item 4 is scheduled.

### D6. `plans/` (4 files, 412 lines) -- optional

All four are completed plans: the prefill/K=4 baseline plan, the closed candle
evaluation (decision 132), and the two multimodal plans whose steps landed as
decisions 145/157. `DONE.md` is the history of record; the candle file is cited
four times from `DONE.md`/`TODO.md` as a decision artifact. Options: (a) fold
each into its `DONE.md` entry and delete `plans/`, updating the four citations;
(b) delete only the two multimodal plans and the prefill plan, keep the candle
evaluation because it is cited; (c) leave as is. No code depends on any of
them.

### D7. Local, gitignored junk in the tree

`triton-tower.log` (a stray log at the repo root), `scripts/__pycache__/`,
`scripts/freetoken_probe/__pycache__/`, `scripts/cupti-trace/__pycache__/`,
`.home/`. All are already ignored (`*.log`, `__pycache__/`, `.home/`), so this
is a local `rm`, not a repo change. Do **not** run a blanket `git clean -Xd`:
it would also delete `.model-cache/`, `kernel/` and `target/`.

---

## Batch E -- the in-tree FreeToken checkout and its caches

`AGENTS.md` says upstream FreeToken is not vendored and should be cloned
outside the tree; `TODO.md:8` still says "Reference code: `./FreeToken`".
Today the checkout lives inside the tree, at a gitignored path:

| path | size | what it is |
|---|---:|---|
| `.model-cache/bench/FreeToken` | 5.7 GiB | the external checkout plus its `.venv`, at commit `14119b8` (matches `ftw-ref-14119b8`) |
| `.model-cache/bench/ftw-ref-14119b8` | 21 GiB | the reference FTW copy the probes serve |
| `.model-cache/bench/uv-cache` | 6.6 GiB | uv's download/build cache |
| `.model-cache/bench/cuda-home` | 148 MiB | CUDA headers and libraries staged for the venv build |
| `.home/.cache/freetoken` | 16 KiB | jailed-agent home cache, unrelated |

It is already ignored by `.gitignore:2` (`.model-cache/`), so nothing is
tracked and no history needs rewriting. The cleanup is about the working tree
and the scripts:

1. Move the checkout to a local reference location outside the repo (for
   example `../freetoken-ref`), keep it checked out at `14119b8`.
2. Teach `scripts/cupti-trace/freetoken-trace.sh` to use
   `FREETOKEN_REF` (and `FREETOKEN_FTW`) instead of the hardcoded
   `$root/.model-cache/bench/FreeToken/.venv` and
   `$root/.model-cache/bench/ftw-ref-14119b8` (lines 20-21). This is the only
   script that hardcodes them; `scripts/freetoken_probe/baselines.py` already
   runs from `sys.executable` and takes `--model`.
3. Add `/FreeToken/` and `/freetoken/` to `.gitignore` as a safety net, since
   `TODO.md:8` invites a root checkout.
4. Fix `TODO.md:8` to say `FREETOKEN_REF` (an external checkout), matching
   `AGENTS.md`.
5. Decide whether the 6.6 GiB `uv-cache` and 21 GiB `ftw-ref` copy stay. Both
   are local-only; the reference FTW is needed to run the reference probes and
   can be rebuilt from the HF snapshot, so it is a disk-budget decision, not a
   repo change.

Gate: `bash -n scripts/cupti-trace/freetoken-trace.sh`, and one reference
trace run if a GPU and the moved checkout are available. Do not delete the
checkout before the move is verified, and never delete the FTW reference copy
while a probe baseline is still being compared against it.

---

## Batch F -- dependency weight

No crate is unused: `futures-core` has exactly one use (`Stream` in the HTTP
module) but removing it would mean reshaping the SSE handler; `half`, `image`,
`tokenizers`, `minijinja`, `base64`, `thiserror`, `toml`, `serde*` are all on
the serving path; `mockall`, `tower`, `serial_test` are dev-only as intended.

### F1. `image` default features pull the whole codec set

`image = "0.25"` with default features brings `exr`, `tiff`, `qoi`, `webp`,
`gif`, `bmp`, `hdr`, `ico`, `pnm`, `tga` and AVIF (`ravif`/`rav1e`) plus
`rayon`. The engine only needs to decode API image input and the unit test
writes a PNG (`image_prep.rs:116-123`). Proposed:

```toml
image = { version = "0.25", default-features = false, features = ["jpeg", "png"] }
```

Add `webp` and/or `gif` only if the API is meant to accept those data URLs.
Gate: `cargo test --lib image_prep`, `cargo clippy --all-targets --features
cudarc`, and one `vision_smoke` run on a real image. This is a build-time and
binary-size win, not a throughput one.

### F2. `tokenizers` default features (optional, small)

The default `progressbar` feature pulls `indicatif`; a server does not need
progress bars. `default-features = false` plus the regex backend and
`esaxx_fast` would trim a little compile time. This needs a check that the
JSON tokenizer's normalizer still builds; low value, do it only if the build
time is worth the experiment.

---

## Batch G -- documentation and reference debt found during the survey

- `TODO.md:188` code map is stale: it lists `scripts/convert_hf_to_ftw.py`
  (deleted; the Rust converter replaced it), an `http.rs` file (now a
  `http/` module), and calls `cuda_moe.rs` "the one kernel" (the ABI has 20
  kernel constants and 69 cubins).
- `TODO.md:74` names a `kernel-autotune` devshell alias. `flake.nix` defines
  only `kernel-selftest`, `kernel-compile-check` and `kernel-aot`; either add
  the alias (`${FERRUM_PYTHON} -m ferrum_kernel.autotune`) or fix the line.
- `TODO.md:8` says the reference is `./FreeToken`; see Batch E.
- `DONE.md:4084` cites `scripts/m7-0-candle-patch.sh` as a historical step.
  The script is gone; either leave the sentence as history or note it is no
  longer in the tree. Same for `scripts/convert_hf_to_ftw.py` references that
  read as instructions rather than history.
- When a probe from Batch B or C is removed, update the instrument list at
  `OPTIMIZATION.md:135` and any `ARCHITECTURE.md` diagram or table that names
  it (`ARCHITECTURE.md:84` and `:152` name `FERRUM_DUP`, which stays).
- `README.md`'s performance table is intentional history (517/2057-prefix
  rows plus the old libtorch column); do not "refresh" it to the 48-token
  clean-prompt standing without also changing the row labels.

---

## Suggested order and size

| batch | content | estimated lines removed | risk |
|---|---|---:|---|
| A | dead methods, dead fns, rename, config decisions | ~150 + 223 moved | none to low |
| B | bf16 shared expert, PF segments, split-K arm, router arm, ablation, PF debug | ~720 + 65 Python + 4 cubins | medium (layout shift) |
| C | four dev probes | ~100 | low |
| D | example, churn probe, R16 script, decode-steps, autotune proof, plans | ~1.4k | none |
| E | move FreeToken checkout, script env, gitignore | 0 (disk + scripts) | low |
| F | image features, optional tokenizers | 0 (dependency closure) | low |
| G | doc fixes | 0 | none |

Practical order: **A, then C, then D, then E, then F, then G, then B** (B last
because it needs the GPU trajectory/oracle gates and a decision entry per
removed arm). A and B can each be one commit; D6/D5 are separate decisions.

## Decisions needed before starting

1. A4: is breaking the config schema (dropping `strategy`/`expert_load`/
   `cache_policy`) acceptable, or keep the ceremony?
2. A5: delete `spec.draft_model_path` now, or wait for the spec-decode work?
3. B3: drop the non-split paged-attention Rust arm and cubins, or keep the
   whole kernel as the selftest reference?
4. D5/D6: delete `autotune_min.py` and `plans/`, or archive them into
   `DONE.md` first?
5. E: where should the local reference checkout live
   (`../freetoken-ref`, `$HOME/...`, or keep it under `.model-cache/`)?
6. F1: which image formats must the API accept (jpeg/png only, or also
   webp/gif)?
