# Measure FreeToken prefill (pp) and batched (K=4) baselines

Goal: close the two missing upstream comparisons, recorded as ours-vs-theirs in
`OPTIMIZATION.md`:

| metric | ours (recorded) | FreeToken (recorded) |
|---|---|---|
| prefill pp @517 tok | 143.5 tok/s (dec 102) | **missing** |
| prefill pp @2057 tok | 156.5 tok/s (dec 103) | **missing** |
| K=4 aggregate decode | 26.2 tok/s (serial 16.6, dec 92) | **missing** |
| bs=1 captured decode | 28.3 tok/s | 30.3 (dec 76/104) |

Their side runs on the same GPU, same FTW, `ft serve --moe-backend offload
--moe-cache-auto` (the config behind their 30.3), no FreeToken file is edited.

## Metric definitions (must match ours exactly)

- **pp** (our `tests/throughput_smoke.rs`): `prompt_tokens / time-to-first-token`,
  greedy, in-process. Prompt = unit `"The quick brown fox jumps over the lazy
  dog. "` (11 tok) repeated to actual lengths **517** (target 512) and **2057**
  (target 2048). Their side: identical prompt text over HTTP `/v1/completions`
  (raw, no chat template), `ttft = first SSE token - request start`,
  `pp = usage.prompt_tokens / ttft`. Frame difference: ours has no HTTP hop
  (~ms vs seconds) - note in the record.
- **K=4 aggregate** (our `tests/dispatcher_batching.rs`): K=4 concurrent
  requests, prompt `"The capital of France is"`, greedy, `ignore_eos`,
  24 tokens each; `aggregate = total_tokens / (t_last_complete - t_first_submit)`.
  Serial phase = the same 4 requests back-to-back, one window. Their side:
  identical prompt/params over HTTP; windows measured the same way.
- Warm-ups discarded; **n=5** pp per length, **5 interleaved S/B rounds** for
  K=4 (order-balanced: alternate starting phase). Report mean, sample sd, min/max.

## Preconditions (AGENTS.md resource limits)

1. Memory gate **waived per user (2026-10-20)**: run regardless of
   `MemAvailable` (23.34 GiB at planning time). This task loads only FreeToken
   (~17 GiB pinned banks + torch); our engine never loads, so our banks are not
   a factor. An OOM kill mid-run loses only logs - retry.
2. Only one model-loading process: `ps` check (only opencode now).
3. Every model-load runs under `prlimit --as=68719476736`.
4. CUDA driver present (`/run/opengl-driver/lib/libcuda.so.1`, cuInit works);
   `CUDA_VISIBLE_DEVICES=0`. `nvidia-smi` is absent - do not depend on it.

## Env recipe (decision 76, adapted to this sandbox)

Venv already repointed to nix python3.12.15; `FreeToken/.venv` has freetoken
(edit), torch 2.11.0+cu130, triton 3.6, flashinfer. Jail inherits `CUDA_HOME`
(cuda-merged-12.9). Wrap every command:

```bash
C=$CUDA_HOME
STD=$(dirname "$(gcc -print-file-name=libstdc++.so.6)")
env -u PYTHONPATH -u LIBTORCH -u LIBTORCH_LIB -u LIBTORCH_INCLUDE \
  PATH="$PWD/FreeToken/.venv/bin:$PATH" CPATH="$C/include" CPLUS_INCLUDE_PATH="$C/include" \
  LIBRARY_PATH="$C/lib:$C/lib/stubs:/run/opengl-driver/lib" \
  LD_LIBRARY_PATH="$C/lib:$STD:/run/opengl-driver/lib" \
  CUDA_VISIBLE_DEVICES=0 FREETOKEN_ALLOW_CUDA_MISMATCH=1 \
  prlimit --as=68719476736 -- <command>
```

Note: `LD_LIBRARY_PATH` must **exclude** the inherited libtorch-2.9 dirs (venv
torch is 2.11; RUNPATH would let the old libs win) but keep nix libstdc++ and
the driver (AGENTS outside-devshell recipe).

## Steps

1. **Env smoke (cheap, no model).**
   `env ... FreeToken/.venv/bin/python -c "import torch, freetoken; print(torch.__version__, torch.cuda.is_available())"`
   Must print `2.11.0 ... True`. Failure -> fix env, do not start the server.
2. **Write driver** `scripts/freetoken_probe/baselines.py` (new, ~250 lines,
   tooling like `probe_impl.py`; no FreeToken edits). It:
   - spawns `python -m freetoken.cli serve --model .model-cache/Qwen3.6-35B-A3B-NVFP4.ftw
     --host 127.0.0.1 --port <free> --moe-backend offload --moe-cache-auto
     --max-running-requests 4 --cuda-graph-max-bs 4 --memory-ratio 0.9
     --max-seq-len-override 8216` (mirrors `bench_decode_moe.py:serve_cmd`,
     but K=4 instead of its hard-coded 1) in its own session group;
   - health-waits on `/health` (reuse bench's wait/stop logic shape: pump log,
     SIGTERM then SIGKILL, 3 s reclaim pause);
   - **phase A warm:** one request per prompt length + one K=4 window, discarded
     (also forces the lazy bs=4 graph capture);
   - **phase B pp:** n=5 each of 517/2057-token prompts, greedy, `max_tokens 4`,
     `/v1/completions` raw with `stream` + `stream_options.include_usage`;
   - **phase C K=4:** 5 interleaved rounds of serial window (4 sequential
     requests) and batched window (4 threads submitted together), prompt
     `"The capital of France is"`, 24 tokens, `ignore_eos`, greedy;
   - reads `/v1/stats` for VRAM; writes rows to
     `<out>/baseline_rows.json` + prints a summary table.
3. **Run** (mem gate + ps gate first), logs to
   `/tmp/opencode/freetoken_baseline/run.log`, server log mirrored by the driver:
   one `ft serve` session total (one load for all phases).
4. **Sanity-check results** against expectations before recording:
   - their pp within a plausible band of ours (same op path; a >2x surprise
     means a frame/config mismatch - re-check before writing it down);
   - their K=4 aggregate > their serial (batching actually engaged; if not,
     `--cuda-graph-max-bs`/`--max-running-requests` wiring is wrong);
   - prompt_tokens come back as 517/2057 (tokenizer identity check; small
     drift -> record actual counts).
5. **Record:**
   - `OPTIMIZATION.md`: new subsection under R9 "Reference baselines
     (2026-10-20): prefill pp + K=4 aggregate" - ours/theirs table, n, sd,
     method, frame caveats, commands, log paths;
   - `TODO.md`: add a checked §8 item for the baselines + one Status sentence
     (the full Status/milestone refresh stays its own queued task).
6. **Gates:** `cargo fmt --check`, `cargo clippy --all-targets`, `cargo test`
   (all currently green; this change is docs + one script). No kernel files
   touched, so `check-kernels.sh` not required.
7. **Commit** (AGENTS Commits, one file set, only if gates green):
   `feat(scripts): measure freetoken prefill and batched decode baselines`.
   `plans/` stays untracked, never committed.

## Deliverable report

Per AGENTS sub-agent format: summary + final `git status`; timing report (n,
mean/sd/min/max per metric, ours-vs-theirs table); correctness (env smoke, gate
results, sanity checks in step 4); justification (frame caveats, what this does
/ does not settle); commands + log paths. "No safe measurement" (e.g. memory
never reaches 24 GiB) is an acceptable report.

## Risks

- Host RAM is tight (23.34 GiB available vs ~17 GiB pinned by the FreeToken
  load); gate waived per user - an OOM kill means a retry, nothing else.
- Cold JIT in the jail: `~/.cache`/`~/.triton` live on tmpfs (fresh per jail),
  so flashinfer + triton kernels recompile at first serve/run - minutes, inside
  the bench's 1800 s timeout; keep logs streaming.
- Frame difference (HTTP vs in-process pp) is small but real; recorded as a
  caveat, not hidden. Optional follow-up (not in this task): run the same
  driver against our own server for a frame-matched pair.

## Appendix: "option to not pin the host banks" - researched (2026-10-20)

Asked while waiving the memory gate; findings, for the record:

- Our host expert banks are **not mmap'd**. `FtwPinnedBanks::load`
  (`infrastructure/banks.rs:196`) does `driver.alloc_pinned` =
  `cuMemHostAlloc(..., MEMHOSTALLOC_DEVICEMAP)`
  (`infrastructure/cuda_driver.rs:651`), then fills them from the FTW with
  `pread` (`read_bank_layer`). The file header's "mmap'd FTW" wording is stale
  (decision 72 dropped `memmap2`). ~17.3 GiB, all 40x256 experts, resident from
  startup.
- **Not pinning would not free any memory.** The pages are ordinary anonymous
  resident memory whether registered with the driver or not (no swap here), so
  `MemAvailable` would not move; the 17.3 GiB stays 17.3 GiB.
- **It would also break the current staging path.** `DEVICEMAP` is what makes
  the buffer reachable from the device: the copy kernel's `src_ptrs` are the
  device aliases of the host banks (zero-copy PCIe reads - the reason `moe_copy`
  reads host memory directly at ~10 GB/s). A pageable/unpinned buffer has no
  device alias, so "unpin" is not a config flag: it requires a second design
  (a device staging buffer + real `cuMemcpyHtoD` per miss, extra VRAM + an
  extra data pass), which would then also be *slower* on the copy stage that is
  already PCIe-bound.
- Real alternatives if host-RAM pressure ever becomes the goal: (a) lazy bank
  fill (host banks as a cache over the FTW, disk-backed misses), or (b)
  file-backed mmap banks with an explicit-H2D path - both are architecture
  slices, not options, and neither helps co-residency with FreeToken's own
  ~17 GiB pinned banks anyway (17.3 + 17 > 32 GiB box).
- Not part of this measurement task; user decision (2026-10-20): **drop it** -
  findings stay in this appendix only, no `TODO.md` entry, no code.
