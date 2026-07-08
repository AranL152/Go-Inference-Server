# PHASE 7 SPEC — Close the In-Server Gap (feed the idle GPU)

> **STATUS: COMPLETE — see `PHASE7.md` for results.** Summary: ~2,050–2,130 req/s
> (target ≥2,100 met); alloc churn 1.9 MB → 42 KB per request; infer 9.2 → 4.7 ms.
> Task 1 pinned the gap (alloc profile: ReadAll 41% / tensor 30% / batch buffer
> 29%; client contention ruled out). Task 2 pooling all kept, with one deviation:
> the input tensor is Put by the HANDLER after Predict, not by packAndSend —
> tests share one tensor across concurrent Predicts, so the spec'd Put point was
> unsafe. Task 3 hybrid decode: built, measured at 8/4/2 GPU decoders, REJECTED
> (negative result — nvJPEG-on-CUDA-cores inflates inference 2.7–3.8×, net loss).

## Context

Phase 6 moved JPEG decode off the GPU (libjpeg-turbo, `-cpu-preprocess`) and broke
the decode↔inference contention wall: served throughput ~1,490 → **~1,790 req/s**,
per-batch inference 24ms → 8ms, GPU util 63% → 35%. But:

- **Isolated CPU preprocess measured ~2,450/s** (16-way, `TestCPUPreprocessParallel`);
  the live server only reaches ~1,790. That ~650/s gap is CPU overhead *other than*
  decode.
- **The GPU is ~65% idle** — inference-only ceiling is ~5,885/s; the CPU can't feed it.

Phase 7 recovers as much of that gap as measurement supports, then (gated) uses the
idle GPU as a decode spillover. Target: **~2,100–2,500 req/s**. Hard wall beyond
that is Huffman-bound decode (hardware-limited).

**Prime suspects for the gap (from reading the hot path this session):**

| # | Site | Cost at 1,790 req/s |
|---|------|---------------------|
| 1 | `cpu_decode.go` `preprocessTurbo`: `make([]float32, 3*224*224)` per request | ~1.05 GB/s garbage |
| 2 | `engine.go:420` `packAndSend`: `make([]float32, n*inputElems)` per batch (~14MB) + full memcpy | ~1.1 GB/s garbage + copy CPU |
| 3 | `main.go` handler: `io.ReadAll(body)` per request | ~0.5 GB/s garbage (doubling growth) |
| 4 | Loadtest client shares the same 16 cores (isolated benchmark didn't) | unknown — must measure |

Total suspected churn ≈ 2.5–3 GB/s → frequent GC cycles stealing cores from
Huffman decode. But per project discipline: **pin causes by measurement before
fixing; report negative results plainly.**

## Constraints (standing)
- Plain HTTP (no gRPC). Do NOT touch the Phase-0 ORT/cuDNN/CUDA binding setup.
- Build on Linux filesystem. `PATH` needs `~/sdk/go/bin`; runs need
  `LD_LIBRARY_PATH="$PWD/onnxruntime/lib:$PWD/cudnn9:/usr/local/cuda-12.6/lib64:/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu"`.
- No Claude co-author trailer on commits.
- Don't kill processes via `pkill -f`/`pgrep -f` patterns that match the bash
  command itself (recurring exit-144 trap) — check via `curl /health` + kill by PID.

## Task 1 — Instrument & pin the gap (GATE for everything else)

No code changes to the hot path; measurement only.

1. **GC pressure check (cheapest possible):** run the server with `GOGC=1000`
   (env var, zero code change), same loadtest sweep (`-levels 160,192,224 -n 12000`)
   vs the 1,790 baseline. If throughput jumps materially → GC confirmed as a major
   component → Task 2 justified. If flat → allocation pooling is NOT the lever;
   record the negative result and skip to 1.3/Task 3.
2. **Alloc profile:** add `net/http/pprof` import to main.go (diagnostic-only,
   behind the existing mux), grab `go tool pprof -alloc_space` +
   `-inuse_space` during load; confirm which of suspects 1–3 dominates.
3. **Client contention split:** during a run, `pidstat -u -p <server>,<client> 1`
   (or `top -b`) to measure what CPU share the loadtest client itself consumes.
   If the client eats ≥2–3 cores, the "gap" is partly measurement artifact —
   quantify it and note that real capacity is higher than the loadtest shows.
4. Optionally add `runtime.ReadMemStats`-based fields to `/stats`
   (alloc rate MB/s, NumGC) for ongoing observability.

**Deliverable:** a pinned % breakdown of the ~650/s gap (GC vs client vs other).

## Task 2 — Allocation fixes (gated on Task 1 confirming GC)

Apply in order, **measuring after each** (loadtest sweep + alloc profile):

1. **Pool the per-request NCHW tensor** (suspect 1). Package-global
   `inputPool sync.Pool` of `[]float32` (len `inputElems`). `preprocessTurbo`
   (cpu_decode.go) Gets from the pool instead of `make`; `packAndSend`
   (engine.go) Puts each `r.input` back **after** copying it into the batch
   buffer. Safe: every element is overwritten by `preprocessTurbo` (no zeroing
   needed); handlers never touch `input` after `Predict` returns — document this
   ownership handoff in a comment on `Predict`.
2. **Pool the batch buffer** (suspect 2). Pool of `[]float32` sized
   `maxBatch*inputElems`; `packAndSend` Gets one and slices to `n*inputElems`
   for `ort.NewTensor` (NewTensor requires exact length). Carry the raw slice in
   `packedBatch`; return it to the pool in `inferLoop` right after `pb.inT.Destroy()`
   (ORT pins the Go slice only while the OrtValue lives). The 14MB memcpy itself
   stays (it's the H2D staging) — only the allocation goes.
3. **Pool the body read** (suspect 3). Replace `io.ReadAll` in the
   `-cpu-preprocess`/`-gpu-preprocess` handler branches with a pooled
   `bytes.Buffer` (`bytes.Buffer.ReadFrom`), returned after `Preprocess` finishes
   (the cgo decode reads the raw bytes, so lifetime must cover the call).

Keep each step a separate commit-sized change; if a step measures flat, keep it
only if it's strictly simpler, otherwise revert and record.

## Task 3 — Hybrid decode spill to the idle GPU (gated)

**Gate:** only if, after Task 2, CPU is still saturated AND GPU util is still low
(≤~45%) AND throughput < ~2,400.

- Handler decode selection: try a **non-blocking** CPU decoder checkout
  (`select` with `default` on `cpuPool.free`); on miss, try the GPU pool
  (`gpuPool` already exists and both are built); on both busy, block on CPU.
- Run both pools simultaneously (new mode: `-cpu-preprocess -gpu-preprocess`
  together becomes legal = hybrid; currently `main.go` rejects the combo).
- **Watch for the Phase-4 failure mode:** monitor `avg_infer_run_ms` — if GPU
  spill pushes per-batch inference back toward 24ms, cap the GPU share (smaller
  `-gpu-decoders`, e.g. 4–8) until inference stays <~12ms.
- Measure: sweep vs Task-2 result; also record the CPU/GPU decode split counts
  (add per-pool counters to `/stats`).

## Task 4 — Re-measure, correctness, report

1. Full sweep (closed-loop 128–256 + open-loop rates) vs the 1,790 baseline and
   the Phase-4 GPU path.
2. Correctness: dog → Samoyed (~0.90), cat → Egyptian cat (~0.685); run
   `TestPipelineRoutingStress` (0 swaps); full `go test .` green;
   `go vet -tags cpujpeg .` clean.
3. Write `PHASE7.md` (honest before/after, negative results included), update
   `PHASE7_SPEC.md` results sections, update project-roadmap memory.

## Success criteria
- Measured, reproducible throughput gain over ~1,790 (target ≥2,100; stretch ~2,500).
- Cause of the in-server gap pinned with numbers (even for the parts we don't fix).
- Correctness unchanged (top-1 matches, 0 routing swaps, 0 errors under load).
- Default (no-flag) build and behavior unchanged.

## Files to modify
- `engine.go` — batch-buffer pool in `packAndSend`/`inferLoop`; input-slice
  return-to-pool; `packedBatch` gains the raw slice field.
- `cpu_decode.go` — `preprocessTurbo` uses `inputPool`; per-pool decode counters.
- `main.go` — pprof (Task 1), pooled body read, hybrid decode selection (Task 3),
  `/stats` additions (GC + pool split).
- New: `PHASE7_SPEC.md` (this plan, as repo doc), later `PHASE7.md`.

## Verification
- Baseline first: rebuild current main, one sweep to reconfirm ~1,790 on the day.
- After each Task-2 step: `go test .`, sweep `-levels 160,192,224 -n 12000`,
  compare `/stats` (alloc rate, avg_cpu_decode_ms, rejects=0, errors=0).
- End: head-to-head table CPU-preprocess (old vs new) + hybrid if built.
