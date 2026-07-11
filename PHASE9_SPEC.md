# PHASE 9 SPEC — Retire the Binding Constraint (IoBinding / pinned memory / CUDA Graphs)

**STATUS: COMPLETE (2026-07-11).** Target ≥8,500 met: 8,527–8,861 raw-tensor,
9,077–9,180 raw-RGB (now the fastest contract), open-loop ~8,300 @ p50 10 ms.
Winning lever: Task 4 device-resident I/O (binding fork + pinned staging).
Deviations from this spec: the ±10% Task-1 gate proved unsatisfiable on fp32
for ANY IoBinding-capable ORT (~20% ORT-internal regression, accepted with
user sign-off — see PHASE9.md); CUDA Graphs closed as definitively
incompatible with dynamic batching (three independent confirmations); stretch
10,000 not reached. Full numbers + negatives in `PHASE9.md`.

## Context

Phase 8 ended at ~7,500 req/s sustained (raw tensors, FP16, 4 infer lanes),
honest band 6,800–7,500. The measured wall: every `session.Run` carries a
pageable-host→device copy of the 14 MB batch inside the call, and concurrent
Runs serialize on the one CUDA stream (per-Run wall time inflates ~linearly
with lanes). GPU ~85% util, CPU mostly idle. Nothing above this is reachable
inside the standing Phase-0 constraint ("never touch the ORT/cuDNN/binding
setup"), so **this phase deliberately retires that constraint** — scoped,
measured, with a rollback path. The JPEG legacy path and the default (no-flag)
build behavior remain unchanged.

## Feasibility facts (verified 2026-07-11, this working tree)

- Current matched set: binding `yalue/onnxruntime_go v1.11.0` = ORT_API_VERSION
  18 = vendored `libonnxruntime.so` 1.18.1.
- **IoBinding** (`DynamicAdvancedSession.CreateIoBinding`, `RunWithBinding`,
  `BindInput/BindOutput`) exists in binding v1.20.0+ — checked v1.20/v1.25/
  v1.28/v1.31 module sources. Every IoBinding-capable version needs ORT ≥1.22
  (API 22–26), so **binding upgrade and ORT shared-library upgrade are one
  move, not two**. Newest is v1.31.0 (API 26 ≈ ORT 1.23.x).
- Even v1.31.0 creates tensors on **CPU memory only** (`ortMemoryInfo` is
  still hardcoded CPU; `NewTensor`/`NewEmptyTensor`/`NewCustomDataTensor` all
  CPU-backed). Device-resident or pinned-host OrtValues are NOT exposed →
  CUDA Graphs' fixed-device-address requirement cannot be met by the stock
  binding alone. That work is Task 4's custom cgo layer (or a fork).
- **Zero-upgrade lever available NOW:** v1.11.0 already has
  `CUDAProviderOptions.Update(map[string]string)` and engine.go already holds
  a `cudaOpts` — ORT CUDA EP tuning keys (`cudnn_conv_algo_search`,
  `do_copy_in_default_stream`, `arena_extend_strategy`, `gpu_mem_limit`,
  `enable_cuda_graph`, …) can be swept without touching the foundation.
- ORT 1.23.x Linux GPU builds target CUDA 12.x + cuDNN 9 — same families as
  the vendored `cuda-12.6` + `cudnn9` dirs. Exact minor-version compat is
  Task 1's first gate, not an assumption.

## Measurement protocol (binding, from Phase 8's gotcha)

- ≥48,000 requests per level (12k finishes in ~2 s = inside boost-clock ramp;
  Phase 8 saw 4,139–6,971 swings at identical settings on short bursts).
- 4,000-request warm pass after every server start; `/stats?reset=1` between
  runs; run-to-run variance is ±10% — only differences beyond that count.
- Reference configs, measured before/after every tier:
  A) raw fp32 1-lane (contract baseline), B) raw FP16 `-infer-lanes 4`
  (headline), C) JPEG legacy defaults. Phase-8 reference numbers: A ~4,900,
  B ~7,500 (band 6,800–7,500), C ~2,260.

## Task 0 — CUDA provider-option sweep (zero upgrade, fully reversible)

1. Add a `-cuda-opts k=v,k=v` passthrough flag (or temporary hardcode) feeding
   `cudaOpts.Update`. No other engine changes.
2. Sweep, one at a time, on config B: `cudnn_conv_algo_search` (EXHAUSTIVE /
   HEURISTIC / DEFAULT), `do_copy_in_default_stream=0`,
   `arena_extend_strategy=kSameAsRequested`. Also re-run the Phase-8 batch-64
   sweep under EXHAUSTIVE — the batch-size collapse smells like cuDNN algo
   selection; confirming or killing that hypothesis is part of this task.
3. Try `enable_cuda_graph=1` here too: expected to fail or no-op without
   device-bound I/O — record the exact error/behavior as the Task 4
   motivation.
4. Gate: keep only reproducible >10% wins; otherwise record negatives and
   proceed with defaults.

## Task 1 — Foundation upgrade (the constraint retirement)

1. Snapshot rollback path: `mv onnxruntime onnxruntime.ort118` (gitignored,
   keep until phase end); same for anything else replaced.
2. Download the ORT 1.23.x linux-x64-gpu release tgz; place `lib/` under
   `onnxruntime/`. Verify cuDNN-9 minor and CUDA-12.6 lib compat (ldd the
   providers .so against the run.sh LD_LIBRARY_PATH; the Phase-1 GPU-memory
   health check must still prove CUDA placement).
3. `go.mod`: `yalue/onnxruntime_go v1.31.0`. Rebuild. Expect API breakage in
   engine.go to be minor (session/API names checked — `DynamicAdvancedSession`
   and `CUDAProviderOptions` survive to v1.31.0).
4. Gates, all mandatory before any Task-2 work: `go test .` and
   `-tags cpujpeg` green; dog/cat top-1 on all three content types (fp32 AND
   fp16 models — reconvert fp16 with `convert_fp16.py` only if the new ORT
   rejects the current file); configs A/B/C within the ±10% band of Phase-8
   references. A regression outside the band = stop, diagnose, or roll back
   and end the phase with the negative result documented.
5. Update SETUP.md (new ORT version, download URL, any cuDNN delta).

## Task 2 — IoBinding lanes (stock binding, CPU-backed values)

1. Per lane: `CreateIoBinding()` + pre-created input/output tensors, reused
   across batches (one binding per goroutine — bindings are not documented
   thread-safe; lanes already own their batch end-to-end). Pack directly into
   the lane's bound input tensor buffer instead of a pooled slice.
2. This removes per-Run output allocation + input name resolution, NOT the
   H2D copy (values are still CPU-backed). Expectation is modest (+0–10%);
   the point is the scaffolding Task 3/4 need.
3. Variable-size tail batches: bound shapes are fixed — either pad the last
   batch to max-batch (measure the cost) or fall back to plain Run for
   non-full batches (keep whichever measures better; record).
4. Gate: no correctness change (routing stress tests, incl. multi-lane, 0
   swaps); keep only if not a regression. `-io-binding` flag, default off.

## Task 3 — CUDA Graphs attempt (cheap, expected blocked)

1. On the Task-2 scaffolding: `enable_cuda_graph=1`. ORT requires stable
   device addresses across replays; with CPU-backed bound values this should
   fail or fall back — capture the exact behavior.
2. If it unexpectedly works: measure config B, expect the biggest single win
   on WSL2 (per-launch overhead is inflated here; ResNet-18 = ~70 launches
   per Run). If blocked (expected): record, proceed to Task 4.

## Task 4 — Pinned/device memory layer (the hard tier; only if Tasks 0–3
leave the target unmet)

1. Decide fork-vs-shim by diff size: (a) fork the binding to add
   `CreateMemoryInfo("Cuda"/"CudaPinned")` + tensor-from-external-buffer, or
   (b) a separate cgo package calling the ORT C API directly (headers already
   vendored by the binding) that produces OrtValues the binding's
   `IoBinding.BindInput` can accept — requires a way to wrap a raw OrtValue
   in the binding's `Value` interface; check `GetInternals()` symmetry first.
   Pick the smaller, more revertable diff. NO other binding behavior changes.
2. Step 1: pinned-host staging (cudaHostAlloc) for the pack buffer — H2D
   ~2× faster and overlappable. Step 2: device-resident bound input/output +
   retry CUDA Graphs (Task 3 rerun).
3. Memory-safety gates (this is the crash-or-silent-corruption tier):
   routing stress ×10 runs, multi-lane stress, `-race` build of the full
   suite, VRAM leak check (nvidia-smi steady-state over a 10-minute soak).

## Task 5 — Re-measure, report, decide the project's end

1. Final table: configs A/B/C plus each kept lever, closed-loop AND open-loop
   saturation, before/after per tier.
2. Full correctness battery (both models × three content types), vet, both
   test suites, 0 errors/rejects under load.
3. `PHASE9.md` (negatives included — Task 0 and Task 3 are expected to
   produce some), update this spec's status header, update roadmap memory,
   commit (no co-author trailer). Delete `onnxruntime.ort118` rollback dir
   only after the phase is committed green.

## Constraints (standing, updated)

- Phase-0 "don't touch the binding/ORT setup" is RETIRED by this phase —
  that retirement is the phase. Everything else stands:
- Plain HTTP (no gRPC). No TensorRT EP (changes the project's story; explicit
  non-goal). JPEG path behavior and the default no-flag build unchanged.
- Build on the Linux filesystem; PATH needs `~/sdk/go/bin`; runs need the
  run.sh LD_LIBRARY_PATH (update run.sh if lib layout changes).
- Warm before measuring; ≥48k-request levels; kill servers by PID via
  `ss -tlnp` (never `pkill -f` patterns that match the bash command itself).
- No Claude co-author trailer on commits.

## Success criteria

- Target: **≥8,500 req/s** sustained on config B's successor (a real >+13%
  over the 7,500 band-top). Stretch: **≥10,000**.
- Every tier measured with numbers, including losers; Task 0 and Task 3
  negatives are first-class results.
- Correctness identical on all paths and both models; default build behavior
  unchanged; no VRAM growth over a 10-minute soak.
- Rollback path proven (Task 1 gate) before any new-API work began.

## Files to modify

- `go.mod`/`go.sum` (binding v1.31.0), `onnxruntime/` vendored libs (Task 1).
- `engine.go` — `-cuda-opts` (Task 0), IoBinding lanes behind `-io-binding`
  (Task 2). `main.go` — flag plumbing.
- New: `PHASE9.md`; Task 4 (if reached): new cgo package dir or binding fork
  reference; SETUP.md updates.
- Rollback artifacts (`onnxruntime.ort118/`) stay gitignored and local.

## Verification

- After every task: full `go test .`, dog/cat on all content types, config-B
  sweep, `/stats` errors=0 rejects=0.
- Task 1 additionally: A/B/C re-baseline within ±10% of Phase-8 numbers.
- Task 4 additionally: `-race` suite, 10-minute soak, VRAM steady.
- End: open-loop saturation to confirm closed-loop capacity is real.
