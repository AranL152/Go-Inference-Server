# PHASE 9 — Retiring the Binding Constraint (device-resident I/O)

**Result: ~8,600–8,900 req/s raw-tensor, ~9,100 req/s raw-RGB** (closed-loop
sustained, FP16 + `-device-binding` + 5 infer lanes), open-loop saturation
~8,300 with p50 still ~10 ms. Same-day, same-rig comparison: the Phase-8 best
config measured 6,249 on the new foundation the morning of these runs, so the
phase's levers added **+38%**; against Phase 8's recorded 7,500 band it's
~+15–22% (day-to-day rig drift is ±10%, documented in Phase 8). Target ≥8,500
**met**; stretch 10,000 not reached. Correctness exact on every path, 0
errors/rejects in every measured run, `-race` clean, 10-minute soak VRAM-flat.
Measured 2026-07-11.

The phase deliberately retired the Phase-0 "never touch the ORT/binding
foundation" constraint. That cost one real, permanent regression (below) and
bought the lever that finally moved the wall.

## Task 0 — CUDA provider-option sweep (zero upgrade): 1 keeper, later obsoleted

On the OLD stack (ORT 1.18.1), same-day baselines, config B = raw FP16
4 lanes:

| option | result |
|---|---|
| `cudnn_conv_algo_search=HEURISTIC` | +18% on config B (6,652–6,846 vs 5,348–6,349, 3-for-3) — but **−16% on fp32 1-lane**; config-specific, not a default |
| `cudnn_conv_algo_search=EXHAUSTIVE` | no effect; also does NOT fix the Phase-8 batch-64 penalty (hypothesis killed: 16.0 vs 16.3 ms/Run) |
| `do_copy_in_default_stream=0` | −12%, rejected |
| `arena_extend_strategy=kSameAsRequested` | −16%, rejected |
| `enable_cuda_graph=1` | breaks after first shape: binding can't wrap post-capture outputs (`ONNXType = ONNX_TYPE_UNKNOWN`), inference 500s |

The `-cuda-opts k=v,...` passthrough flag is the permanent artifact. ORT
1.27's default algo selection caught up with HEURISTIC (6,249 default vs
6,115 HEURISTIC on the new stack), so the keeper lasted exactly one task —
recorded because that's how it went.

## Task 1 — Foundation upgrade: ORT 1.18.1 → 1.27.1, binding v1.11.0 → v1.31.0

- Zero Go code changes needed; both test suites green; both models load
  unmodified; correctness exact on all three content types.
- cuDNN 9.10.2 and CUDA 12.6 libs work as-is (verified via ldd).
- **Accepted regression:** raw fp32 is ~20% slower per Run (6.8 → 8.2 ms at
  batch 32) on EVERY IoBinding-capable ORT — 1.22.0, 1.26.0, 1.27.1 measured
  identical. Not `cudnn_conv_algo_search` (both modes tried), not `use_tf32`,
  not cuDNN minor (9.24.0 tried). The choice was "old fp32 speed" XOR
  "IoBinding API"; taken deliberately (user sign-off) because every remaining
  phase lever lives on FP16, where the new stack measures equal-or-better.
  fp32 users: pin the ORT 1.18.1 + binding v1.11.0 matched set (commit
  history; SETUP.md Phase-9 note).

## Task 2 — IoBinding with CPU-backed values: neutral, but bought the map

Throughput 6,195–6,270 vs 6,089–6,249 default — no win, as predicted (the
pageable H2D still happens). Kept behind `-io-binding` (default off) as
scaffolding. Two behaviors discovered here, both load-bearing for Task 4:

1. **ORT does not write a pre-bound CPU output value on a CUDA session** —
   reads of the bound tensor return garbage (992/1000 wrong in the routing
   stress). The only supported read path is `GetBoundOutputValues()`.
2. **Bound CPU inputs' device copies are cached across runs** — refilling the
   buffer does nothing; every Run reuses the FIRST upload. The C API's
   invalidation hook (`SynchronizeBoundInputs`) isn't exposed by the Go
   binding; re-binding the input every batch is the workaround.

`TestPipelineRoutingStressIoBinding` exists because it caught both.

## Task 3 — CUDA Graphs: definitively incompatible with dynamic batching

Tried on the old stack (Task 0), the new stack with CPU-backed binding, and
again with device-resident fixed addresses (Task 4). Same failure every time:
after graph capture, any Run with a different batch shape returns
`ONNX_TYPE_UNKNOWN` outputs (85 errors/2000 even with device binding — the
unavoidable partial-batch plain-Run path breaks it). CUDA Graphs requires
every Run to have one static shape; a dynamic batcher cannot promise that.
Closed as a negative with three independent confirmations.

## Task 4 — Device-resident I/O: the phase's win

~90 lines added to a local fork of the binding
(`onnxruntime_go_fork/`, wired via a go.mod replace directive):
`CreateMemoryInfo("Cuda", ...)` + a `DeviceTensor` type wrapping
externally-owned CUDA device memory as a bindable ONNX value. Plus
`cudamem.go` (~60 lines): cudaMalloc/cudaHostAlloc/cudaMemcpy via libcudart.

`-device-binding` gives full batches engine-owned device buffers bound once;
the gatherer packs into **pinned** host staging and the lane does the H2D
itself, runs against fixed device addresses, copies 128 KB of logits back,
and recycles the slot. ORT's internal pageable staging copy is gone from the
serialized path. Partial batches use the plain path unchanged.

| config (FP16, batch 32, new stack, same day) | req/s | per-Run ms |
|---|---:|---:|
| 4 lanes, default (Phase-8 equivalent)         | 6,089–6,249 | 20.4 |
| 4 lanes, `-device-binding`                    | 8,282–8,401 | 15.0 |
| **5 lanes, `-device-binding`**                | **8,527–8,861** | 17.7 |
| 6 lanes, `-device-binding`                    | 8,188–8,710 | 18.8 |

Gates: routing stress (device variant) passed first run, 0 swaps; `-race`
clean; 10-minute soak = 289,212 requests, 0 rejects, VRAM climbing for ~3
rounds then **flat at 5,878 MiB for 6 straight rounds** (ORT arena steady
state for 5 lanes, not a leak — cap with `gpu_mem_limit` on smaller cards).

## Task 5 — Final numbers (all 0 errors)

| path / config (2026-07-11, same rig) | req/s |
|---|---:|
| JPEG legacy (fp32 defaults, unchanged)             | ~2,000 |
| raw fp32, 1 lane (carries the Task-1 regression)   | ~4,000 |
| raw tensor, FP16, 4 lanes (Phase-8 best, new stack)| ~6,250 |
| raw tensor, FP16, 5 lanes, device-binding          | **8,527–8,861** |
| **raw RGB uint8, FP16, 5 lanes, device-binding**   | **9,077–9,180** |

- Open-loop: offered 11,000 → achieved 8,300, GPU 91%, p50 10.1 ms — the
  saturation knee no longer melts latency (Phase 8 saturated at p50 244 ms).
- **The contract inversion is the headline surprise:** with ORT's H2D copy
  off the hot path, shoveling 602 KB float-tensor bodies through HTTP costs
  more than the server-side normalize of 150 KB uint8 bodies. The
  network-realistic contract (`application/x-raw-rgb`) is now the FASTEST
  contract at 4× less bandwidth (~11 Gbps vs ~42 it would take in float32).
- Correctness: dog Samoyed / cat Egyptian cat exact on jpeg + raw + raw-rgb,
  both models; `go test .`, `-tags cpujpeg`, `go vet` green.

## Journey

345 (Phase 2) → 1,463 (P4) → 1,790 (P6) → 2,130 (P7) → 7,500 (P8) →
**~8,700 raw / ~9,100 raw-RGB (P9)**, every step measured, every loser kept
on the books.

## Recommended operating points

- Throughput: `-cpu-preprocess -quiet -max-batch 32 -infer-lanes 5 -model resnet18_fp16.onnx -device-binding`
- Clients should send `application/x-raw-rgb` (fastest AND lightest).
- Default (no flags): unchanged behavior, plain Run path, fp32
  (note: fp32 carries the Task-1 ~20% ORT regression).
