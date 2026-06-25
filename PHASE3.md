# Phase 3 — Dynamic Batching Coordinator

**Headline (honest): end-to-end throughput did *not* improve — it went from the
Phase 2 baseline of ~345 req/s to ~338 req/s peak (≈0.98×), and GPU utilization
stayed low (~15–32%). Batching is correctly implemented and the batched GPU path
has ~12× headroom in isolation (≈4,200 req/s at full batch-32), but it is starved:
the real bottleneck is CPU-side image preprocessing (~38 ms/image), which sits
*upstream* of the batch queue. Because requests spend ~38 ms decoding/resizing
before they ever reach the collector, they trickle in one at a time and the
realized batch size stays ≈1–1.7 even at concurrency 128 — so the GPU never gets
fed a full batch under the real workload.**

This is a negative end-to-end result with a clear, measured cause, plus proof
that the batching machinery itself works. The fix (faster/parallel preprocessing)
is the natural Phase 4 target.

## What was built

A dynamic batching coordinator replacing Phase 1's per-request mutex-then-infer:

- **Collector goroutine (single GPU owner).** `Engine.collect()` in `engine.go`.
  HTTP handlers no longer call inference directly — `Engine.Predict()` submits a
  `batchRequest{input, resultCh}` to a channel and blocks on its private result
  channel.
- **Two batching triggers, whichever fires first** (both configurable):
  - **Max batch size** — `-max-batch` (default **32**).
  - **Max wait time** — `-max-wait` (default **5ms**). Bounds the extra latency an
    unlucky lone request can incur.
- **Variable-N batched inference.** The model (`resnet18-v1-7`) already declares a
  dynamic batch dim (`data:[N,3,224,224] → [N,1000]`), so no re-export was needed.
  Phase 1's `AdvancedSession` with a fixed `[1,…]` tensor reused in place was
  replaced with ORT's **`DynamicAdvancedSession`**: each batch builds a fresh
  `[N,3,224,224]` input tensor and passes a `nil` output so ORT allocates the
  correctly-shaped `[N,1000]` output. One `session.Run()` per batch.
- **Correct result routing.** `splitBatchLogits()` splits the `[N,1000]` output
  row-by-row, with row `i` routed back to `batch[i]`'s channel — strict index
  correspondence, unit-tested without a GPU.
- **Observability.** `Engine.BatchStats()` tracks cumulative batches/requests and
  the realized average batch size (this is what exposed the starvation).

### Batching parameters chosen

| Param | Value | Rationale |
|---|---|---|
| `-max-batch` | 32 | Fills the GPU efficiently (isolation test saturates throughput at 32); ResNet-18 batch-32 is only ~2 GB GPU memory. |
| `-max-wait` | 5 ms | Caps added tail latency for lone requests. *(Turned out to be moot — see below — because batches rarely fill before the GPU drains them.)* |

## Correctness — passes, including under batching

`./test_phase3.sh` (sets the same lib paths as `run.sh`, runs `go test -v`):

- **`TestMixedBatchNoSwap`** — a dog and a cat submitted concurrently into the
  **same** batch each get their own correct answer: **dog → 258 Samoyed**,
  **cat → 285 Egyptian cat**, not swapped. ✅
- **`TestKnownAnswersConcurrent`** — 100 concurrent dog/cat requests batched
  together; every one returns its correct individual class. ✅
- **`TestSplitBatchLogits_Routing` / `_OrderSwaps` / `_LengthMismatch`** — pure
  unit tests proving results follow their row (no fixed/aliased mapping). ✅

All pass. Batching does not corrupt or mix up results.

## Load-test results (same tool, same levels as Phase 2)

`./loadtest/loadtest -levels 1,4,16,64,128 -n 1500 -warmup 50` against the batched
server (`resnet18.onnx`, CUDA EP, RTX 4080, 16 vCPU; default `-max-batch 32
-max-wait 5ms`). Raw: `phase3_results.csv` / `phase3_results.json`.

| Concurrency | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) | max (ms) | GPU util avg | GPU util max |
|------------:|-------------------:|---------:|---------:|---------:|----------:|---------:|-------------:|-------------:|
|   1 |  23.9 |  47.3 |  61.6 |  63.9 |  41.8 |   77.9 | 29% | 37% |
|   4 | 112.5 |  42.2 |  57.3 |  62.1 |  35.5 |  504.8 | 22% | 83% |
|  16 | 277.8 |  64.3 |  85.3 | 513.7 |  57.3 |  665.8 | 28% | 74% |
|  64 | 221.4 | 270.2 | 682.7 | 758.9 | 285.6 |  783.9 | 32% | 68% |
| 128 | 337.7 | 330.8 | 764.8 | 880.2 | 371.5 | 1110.6 | 15% | 71% |

(Re-running with per-request logging disabled (`-quiet`) gave the same numbers,
ruling out log contention as the cause.)

## Before / after vs PHASE2.md

| Concurrency | Phase 2 baseline (req/s) | Phase 3 batched (req/s) | Change |
|------------:|-------------------------:|------------------------:|-------:|
|   1 |  28.7 |  23.9 | 0.83× |
|   4 | 133.8 | 112.5 | 0.84× |
|  16 | 342.2 | 277.8 | 0.81× |
|  64 | 342.7 | 221.4 | 0.65× |
| 128 | 330.8 | 337.7 | 1.02× |
| **peak** | **~345** | **~338** | **≈0.98× (no improvement)** |

GPU compute utilization at the plateau: **~38% (Phase 2) → ~15–32% (Phase 3)** —
still mostly idle, and if anything *lower*. The expected rise toward saturation
**did not happen** under the real workload.

## Why no improvement — measured, not guessed

Three measurements pin the cause:

**1. The realized batch size is ≈1, not 32.** `Engine.BatchStats()` during the
c=64/128 runs reports an average batch size of **~1.3–1.7** (most batches are a
single request). The two triggers fire on an essentially empty queue — there is
almost never more than one preprocessed request waiting when the collector wakes.

**2. Preprocessing costs ~38 ms/image, single-threaded.**
`BenchmarkPreprocess` (JPEG decode + bilinear resize + normalize) =
**38.6 ms/op**. Across 16 cores that is a hard ceiling of ~16/0.0386 ≈ **415
req/s** for the preprocessing stage *alone* — and the closed-loop load generator
runs on the same 16 cores, pushing the realized ceiling down to ~345. Preprocessing
sits **upstream** of the batch queue, so it both caps total throughput *and*
spaces requests out so they can't accumulate into a batch.

**3. The batched GPU path has ~12× headroom in isolation.**
`TestEngineThroughputPreprocessedConcurrent` preprocesses one image once, then
fires concurrent `Predict()` calls reusing that tensor (bypassing decode):

| Concurrency | Throughput (req/s) | Realized batch size |
|------------:|-------------------:|--------------------:|
|   1 |    86 |  1.0 |
|  16 | 1,381 | 16.0 |
|  64 | 3,098 | 32.0 |
| 128 | 4,174 | 32.0 |
| 256 | 4,216 | 32.0 |

When the queue is actually fed, batches fill to 32 and throughput reaches
**~4,200 req/s (~12× the baseline)**. So the batching coordinator is correct and
highly effective; it is simply **starved** end-to-end.

### Reconciling with Phase 2's analysis

Phase 2 attributed the 345 plateau to the mutex-serialized GPU critical section
(~2.9 ms ⇒ ~345/s). Phase 3 removed that mutex — and throughput stayed ~345,
revealing that **the GPU serialization was not the binding constraint**; CPU
preprocessing (~415/s ceiling, ~345/s after sharing cores with the load tool) was.
The ~38% idle GPU in Phase 2 was the GPU *waiting to be fed*, not the GPU waiting
on a lock. Removing the lock without speeding up the feeder cannot help — and the
per-batch tensor alloc/copy/destroy on the single collector goroutine adds a small
amount of CPU overhead that competes with preprocessing, which is why several
levels are slightly *below* baseline.

## Conclusion & next step

- ✅ Correctness preserved through the batching path (dog→Samoyed, cat→Egyptian cat,
  including mixed batches — no swaps).
- ✅ Batching coordinator implemented correctly with both triggers, variable N,
  and verified routing.
- ❌ **No end-to-end throughput gain (~345 → ~338 req/s).** Honestly reported.
- 🔎 **Root cause (measured): CPU image preprocessing at ~38 ms/image is the
  bottleneck and starves the batch queue (realized batch ≈1).** The batched GPU
  path itself does ~4,200 req/s / batch-32 in isolation.

**Phase 4 implication:** to realize the batching win, attack preprocessing —
parallelize/offload JPEG decode + resize (e.g. a worker pool feeding the
collector, SIMD/native resize, or GPU-side preprocessing). Only then will real
traffic fill batches and push the RTX 4080 toward the ~4,200 req/s it already
demonstrates here.

## Files

- `engine.go` — `DynamicAdvancedSession` + batching collector (`collect`,
  `runBatch`, `splitBatchLogits`, `BatchStats`).
- `main.go` — `-max-batch`, `-max-wait`, `-quiet` flags; collector lifecycle.
- `routing_test.go` — pure routing/no-swap unit tests.
- `engine_integration_test.go` — mixed-batch + known-answer-under-load GPU tests.
- `engine_throughput_test.go` — engine-isolation throughput (the ~12× headroom proof).
- `preprocess_bench_test.go` — preprocessing cost benchmark (the bottleneck proof).
- `test_phase3.sh` — runs all of the above with the correct lib paths.
- `phase3_results.csv` / `phase3_results.json` — raw load-test numbers.

## Reproduce

```bash
./run.sh -max-batch 32 -max-wait 5ms &          # start the batched server on :8080
CGO_ENABLED=0 ~/sdk/go/bin/go build -o loadtest/loadtest ./loadtest
./loadtest/loadtest -levels 1,4,16,64,128 -n 1500 -warmup 50 -out phase3_results
./test_phase3.sh                                 # correctness + isolation evidence
./test_phase3.sh -run BenchmarkPreprocess -bench . -benchtime 3s
```
