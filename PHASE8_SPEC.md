# PHASE 8 SPEC — Raw-Input Contract + Infer-Lane Throughput

**STATUS: COMPLETE (2026-07-10).** All five tasks done, including both stretch
items. Raw path ~4,900 → ~7,500 req/s (FP16 + 4 lanes); ~6,429 without FP16
(3 lanes) — target and stretch both beaten. Batch sweep negative (32 stays).
Results and honest negatives in `PHASE8.md`.

## Context

Phase 7 ended at ~2,100–2,130 req/s, decode-bound: ~13 of 16 cores on
Huffman-bound libjpeg-turbo, GPU ~62% idle. Phase-8 groundwork (measured
2026-07-09, this working tree) added a decode-free path and pinned the numbers:

- **Raw-tensor path**: `POST /predict` with `Content-Type: application/x-raw-tensor`,
  body = preprocessed NCHW float32 tensor (602,112 bytes). Server skips all
  preprocessing; tensor goes straight into the pooled `Predict` path.
  Correctness verified (dog → Samoyed 0.909, cat → Egyptian cat 0.684).
- **Loadtest `-raw`**: preprocesses testdata JPEGs once client-side, ships tensors.
- **Measured same-day A/B** (16-core, RTX 4080 SUPER, WSL2, Phase-7 protocol):
  JPEG baseline ~2,200 req/s (GPU 38%); raw closed-loop **4,660–4,962** req/s
  (batches full at 31.8/32, 0 errors); open-loop saturation **~4,900** (offered
  6,000 → achieved 4,882 with p50 9.8→274 ms blowup). GPU util 67–75%,
  server CPU ~2.8/16 cores. Results: `phase8_raw_results.*`, `phase8_jpeg_baseline.*`.

## The contract decision (this phase's premise)

**The server's primary contract becomes pre-decoded tensor input.** The JPEG
path remains as a legacy/edge contract (unchanged, still measured), but the
headline number and all Phase-8 optimization target the raw path.

Rationale / precedent (interview-ready):
- NVIDIA **Triton** and the **KServe v2** inference protocol natively accept raw
  tensors; image decode is the client's job unless a preprocessing ensemble
  (DALI) is explicitly added. This contract is the industry default, not a dodge.
- **MLPerf Inference** rules allow untimed offline preprocessing — the measured
  serving path starts at the preprocessed tensor.
- Video-analytics pipelines (e.g. DeepStream) feed inference from already-decoded
  frames; nobody re-encodes JPEG between stages.
- Architecture answer to "who decodes?": decode is stateless and horizontally
  cheap (CPU nodes); GPU inference is the scarce resource. Separating tiers so
  the GPU does only GPU-shaped work is textbook. On THIS card specifically,
  GPU-side decode was measured and rejected twice (Phases 4–7): no NVJPG engine,
  nvJPEG falls back to CUDA cores and steals SMs from inference (datacenter
  parts — A100/H100 — have dedicated NVJPG engines; that's where server-side
  GPU decode belongs).
- Trade accepted: 602 KB/request vs ~50–100 KB JPEG. Invisible on loopback
  (~3 GB/s measured); over a real network ~4,900 req/s ≈ 24 Gbps → the NIC
  becomes the wall. Mitigation if ever needed: uint8-RGB variant (Task 4).

## Where the new wall is

~4,900 req/s = the single serialized infer lane: gather → pack (14 MB staging
memcpy) → `session.Run` (which does the H2D copy internally — inputs are
CPU-backed Go slices) → split. GPU util tops at ~75%; CPU nearly idle. The
Phase-5 bare probe (no HTTP, no batch assembly) hit **5,885**; end-to-end is at
83% of that.

**Binding feasibility (checked this session, yalue/onnxruntime_go v1.11.0):**
no IOBinding; `ortMemoryInfo` is CPU-only ("we only support CPU" in the source).
So device-resident input tensors, explicit pinned-host buffers, and CUDA Graphs
(which require fixed device-bound buffers) are **out of reach without changing
the binding** — forbidden by the standing Phase-0 constraint. The levers that
remain inside the constraint:

| # | Lever | Cost | Why it might work |
|---|-------|------|-------------------|
| 1 | `-max-batch` sweep 32/64/128 | zero code | amortize per-Run overhead. NOTE: measured per-image Run cost went the WRONG way with batch (0.19 ms/img @ batch 24 vs 0.25 @ 32) — do not assume, measure |
| 2 | N concurrent infer lanes on the ONE shared session (`-infer-lanes`) | small (spawn N `inferLoop` goroutines) | ORT sessions are thread-safe for concurrent Run; overlaps lane A's CPU-side pack/split and implicit H2D with lane B's GPU compute. Phase 5 disproved multiple SESSIONS (VRAM + contention), NOT concurrent Run on one session |
| 3 | FP16 model file (`resnet18_fp16.onnx`), float32 I/O preserved (`keep_io_types`) | model conversion only; zero Go/binding changes (`-model` flag) | ResNet FP32 doesn't touch tensor cores; FP16 typically ~2x compute on Ada. Gated on Python + onnxconverter-common being available |
| 4 | (stretch) uint8-RGB raw variant (150 KB bodies, server does normalize-only) | small handler branch | 4x less bandwidth; makes the contract network-realistic |

## Constraints (standing)
- Plain HTTP (no gRPC). Do NOT touch the Phase-0 ORT/cuDNN/CUDA binding setup —
  which now explicitly includes NOT swapping/patching yalue/onnxruntime_go.
- Build on Linux filesystem. `PATH` needs `~/sdk/go/bin`; runs need the run.sh
  `LD_LIBRARY_PATH`.
- No Claude co-author trailer on commits.
- Don't kill processes via `pkill -f` patterns that match the bash command
  itself (recurring exit-144 trap) — check via `curl /health`, kill by PID.
- JPEG path behavior and default (no-flag) build unchanged.
- Warm before measuring (cold-start artifact ~990 req/s on first level).

## Task 1 — Re-baseline + batch-size sweep (zero code changes)

1. Rebuild, launch (`-cpu-preprocess -quiet`), 4,000-request warm pass.
2. Raw closed-loop sweep (`-raw -levels 128,256,384 -n 12000 -warmup 200`) at
   `-max-batch 32` (baseline), then 64, then 128. `/stats?reset=1` between runs;
   record `avg_infer_run_ms`, `avg_batch_size`, per-image Run cost.
3. Keep the best batch size for all later tasks. If all flat/worse, record the
   negative result and stay at 32.

## Task 2 — Concurrent infer lanes (`-infer-lanes`, default 1 = today's behavior)

1. `engine.go`: spawn N `inferLoop` goroutines consuming the same packed-batch
   channel. Audit first: per-batch state must be goroutine-local (packedBatch
   carries its own tensor + staging slice; unpack workers already route
   per-request — ordering across batches is immaterial). `/stats` busy-frac
   becomes per-lane or aggregate; note which.
2. Measure lanes 1/2/3 at the Task-1 best batch size. Watch `avg_infer_run_ms`:
   if concurrent Runs serialize on the CUDA stream, Run time will inflate
   ~proportionally and throughput stays flat → record negative result, revert to
   1 lane (keep the flag only if strictly simple).
3. Gate for keeping: reproducible gain at 0 errors, correctness suite green.

## Task 3 — FP16 model (gated on tooling availability)

1. Gate: `python3 -c "import onnx, onnxconverter_common"` (venv install OK,
   Linux fs only). If unavailable and not trivially installable, record and skip.
2. Convert: `float16.convert_float_to_float16(..., keep_io_types=True)` →
   `resnet18_fp16.onnx`. I/O stays float32 → Go code unchanged; run server with
   `-model resnet18_fp16.onnx`.
3. Correctness tolerance: top-1 class MUST match (dog Samoyed, cat Egyptian
   cat); confidence within ±0.05 of FP32 reference. Run full `go test .`
   pointing at the fp32 model (default) to prove no regression there.
4. Measure raw sweep vs Task-2 result. Record VRAM delta too (`nvidia-smi`).

## Task 4 — (stretch) uint8-RGB raw variant

Only if Tasks 1–3 leave the server CPU-idle and GPU <90% util AND time permits:
`Content-Type: application/x-raw-rgb` = 224×224×3 uint8 body (150,528 B);
server does normalize-into-pooled-tensor (vectorizable, ~0.3 ms). Loadtest
`-raw-rgb`. Measures the network-realistic contract's cost.

## Task 5 — Re-measure, correctness, report

1. Final head-to-head table: JPEG legacy path (unchanged, one sweep) vs raw
   fp32 vs raw + best levers; closed-loop AND open-loop saturation point.
2. Correctness: dog/cat top-1 on BOTH paths, `TestPipelineRoutingStress` 0 swaps,
   `go test .` + `-tags cpujpeg` green, `go vet` clean, 0 errors/rejects under load.
3. Write `PHASE8.md` (honest, negative results included), update this spec's
   status header, update project-roadmap memory. Commit (no co-author trailer).

## Success criteria
- Contract documented + defensible (this file) — raw primary, JPEG legacy.
- Raw-path throughput measurably above the 4,900 baseline; target **≥5,500**,
  stretch ≥5,885 (match the Phase-5 bare probe) without FP16; FP16 reported
  separately if built.
- Every lever measured with numbers, including the ones that lose.
- Correctness unchanged on both paths; default build/behavior unchanged.

## Files to modify
- `engine.go` — `-infer-lanes` (Task 2); nothing else in the hot path expected.
- `main.go` — flag plumbing; Task 4 branch if reached.
- `loadtest/` — Task 4 `-raw-rgb` if reached.
- New: `resnet18_fp16.onnx` (Task 3), `PHASE8.md`, this spec.
- Already in tree from groundwork: raw path in `main.go`, `loadtest/raw.go`,
  `phase8_raw_results.*`, `phase8_jpeg_baseline.*`.

## Verification
- Warm pass before every measured sweep; `/stats?reset=1` between levels/runs.
- After each task: `go test .`, raw sweep, `/stats` (errors=0, rejects=0).
- End: open-loop saturation run to confirm the closed-loop number is real capacity.
