# Phase 4 — GPU Image Preprocessing (nvJPEG + NPP)

**Headline (honest, and positive this time): end-to-end throughput rose from the
~345 req/s baseline to ~1,460 req/s — a ~4.2× improvement — by moving JPEG decode
and resize off the CPU and onto the GPU (nvJPEG decode + NPP resize).** The Phase 3
investigation had proved the real bottleneck was CPU JPEG decode (~80% of a ~38 ms
preprocessing cost), not the batching machinery. Phase 4 attacked that bottleneck
directly. Per-image preprocessing dropped ~6.6× (38.6 ms → 5.9 ms for the large
image), batches finally fill (realized batch size ~1.3 → 20–29 at high
concurrency, as a *side effect* of feeding the queue faster), and predictions are
unchanged (dog → Samoyed, cat → Egyptian cat).

The new ceiling is **not** CPU decode and **not** raw GPU saturation. Direct
instrumentation (a `/stats` per-stage timer + an open-loop load mode — see "Where
the new bottleneck is") shows it is the **single collector goroutine's serialized
per-batch cycle** (gather-wait + CPU tensor assembly + `session.Run()` + logit
split, none overlapping), with the collector busy-fraction pinned at 0.97–0.99.
Both CPU and GPU show idle headroom precisely because that one lane is unpipelined.
Open-loop confirms ~1,450 req/s is genuine server capacity (zero overruns), not a
measurement artifact. Fixing it (collector pipelining) is Phase 5.

## Why this was the right target (recap of the investigation)

Phase 3 built a correct dynamic batching coordinator but throughput didn't move
(~345 → ~338 req/s). The front-loaded Phase 4 investigation established, by
measurement:

1. **Throughput was preprocessing-gated, not batch-gated.** The GPU drains a queue
   ~10× faster than CPU preprocessing fills it, so the batch queue could never
   build depth (realized batch ≈1.3) regardless of `-max-wait` or concurrency.
2. **The cost was JPEG decode, and it scales with resolution** (`BenchmarkDecode`
   / `Resize` / `Normalize`):

   | Stage | dog.jpg (1546×1213) | cat.jpg (335×500) |
   |---|---:|---:|
   | **JPEG decode** | **31.8 ms (82%)** | 3.3 ms (44%) |
   | Resize | 6.9 ms (18%) | 2.4 ms (32%) |
   | Normalize | 0.26 ms (<1%) | 0.26 ms (3%) |

3. **Real CPU parallelism is ~8-wide** (Ryzen 7 7800X3D, 8 physical / 16 threads):
   load-mix avg ~22 ms ÷ 8 ≈ 355/s, matching the measured 345 baseline.
4. **Pure Go can't fix decode** (`image/jpeg` is slow and can't DCT-downscale), so
   any fix needs a C dependency. With the GPU ~60% idle, GPU decode was the move.

## What was built

The Go coordinator (HTTP server, batching collector, result routing) is
**unchanged**. Only the per-request preprocessing leaf was replaced, behind a flag.

- **`gpu_decode.go` (cgo).** nvJPEG decode + NPP bilinear resize + center-crop,
  entirely on the GPU, returning the same `[]float32` NCHW tensor the collector
  already consumes. The cheap normalize (~0.26 ms) stays on the CPU.
- **Decoder-context pool.** A fixed set of reusable contexts (persistent nvJPEG
  handle/state, a CUDA stream, an NPP stream context bound to it, and
  grow-on-demand device buffers). Handlers borrow a context, decode, return it.
  Pool size (`-gpu-decoders`, default 8) bounds concurrent GPU decodes and GPU
  memory. Reuse was a *performance necessity*, not just concurrency control (see
  below).
- **`main.go` flags.** `-gpu-preprocess` (off by default; CPU path retained as
  fallback) and `-gpu-decoders N`. PNG would fall through to CPU; nvJPEG is
  JPEG-only.

No change to the Phase-0 build setup was required: `run.sh` already puts
`/usr/local/cuda-12.6/lib64` on `LD_LIBRARY_PATH` (where `libnvjpeg`, `libnppig`,
`libnppc`, `libcudart` live), and the cgo `CFLAGS`/`LDFLAGS` point at the matching
CUDA 12.6 toolkit. The ORT/cuDNN stack still loads on GPU unchanged.

## The hardware finding (does this GPU decode JPEGs in dedicated silicon?)

**No.** `nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, …)` returns `ARCH_MISMATCH`
(rc=-7) on this RTX 4080 SUPER — the dedicated NVJPG decode engine is a
datacenter-GPU feature (A100/H100-class); consumer Ada GeForce does not expose it.
So decode runs on `NVJPEG_BACKEND_GPU_HYBRID` — the CUDA cores do the parallel
decode stages (dequant, IDCT, color-convert), **sharing the SMs with inference.**
This is why the eventual ceiling is GPU-pipeline contention, and it is the measured
reason GPU util can't be read as "spare capacity." Despite the contention,
throughput still rose ~4.2×.

## Correctness — predictions unchanged

- **`TestGPUDecodeParity`** — GPU vs `image/jpeg`: mean |Δ| = 0.38 (dog) / 0.27
  (cat), 100% of channels within ±3. ✅
- **`TestGPUResizeParity`** — NPP linear vs CPU half-pixel bilinear diverges at the
  pixel level under heavy downscale (dog mean |Δ| = 6.5) but…
- **`TestGPUTop1Parity`** — …top-1 is **identical** through the real engine:
  dog → 258 Samoyed (CPU 0.9091 / GPU 0.9061), cat → 285 Egyptian cat (CPU 0.6843
  / GPU 0.7041). The resize divergence is cosmetic. ✅
- **`TestGPUPoolPreprocess`** — 16 goroutines hammering the pool, no
  cross-contamination, dog tensor reproducible. ✅
- All Phase 3 tests (mixed-batch no-swap, known-answers-concurrent, routing) still
  pass. ✅ Live HTTP check with `-gpu-preprocess`: dog → Samoyed, cat → Egyptian
  cat. ✅

## Per-image preprocessing cost (the direct win)

`BenchmarkGPUPoolPreprocess` (decode + resize + normalize, handles reused):

| Image | CPU (full) | GPU naive (handle/call) | **GPU pooled** | Speedup vs CPU |
|---|---:|---:|---:|---:|
| dog.jpg | 38.6 ms | 15.3 ms | **5.88 ms** | **6.6×** |
| cat.jpg | ~6 ms | 4.6 ms | **1.05 ms** | **~6×** |

Handle reuse cut the dog from 15.3 → 5.88 ms (2.6×) — nvJPEG handle/state creation
is ~4 ms of fixed cost, which is why the *naive* per-call path was actually slower
than CPU for the small image. The pool erases it.

## End-to-end throughput — before/after (same machine, same load tool)

CPU path is the default build; GPU path is `-gpu-preprocess -gpu-decoders 8`.
`./loadtest/loadtest` (closed-loop). Raw: `phase4_cpu_results.*`,
`phase4_gpu_results.*`.

| Concurrency | CPU req/s | GPU req/s | Speedup |
|------------:|----------:|----------:|--------:|
|   4 |  91.1 |   293.5 | 3.2× |
|  16 | 316.1 |   865.9 | 2.7× |
|  32 | 320.5 | 1,211.9 | 3.8× |
|  64 | 307.2 | 1,454.3 | 4.7× |
| 128 | 357.0 | 1,462.6 | 4.1× |
| 256 | 349.0 | 1,449.4 | 4.2× |
| **peak** | **~357** | **~1,463** | **~4.1×** |

Realized batch size: Phase 3 ≈ 1.3 → Phase 4 cumulative ~4.3, with individual
high-concurrency batches of **20–29** (the low-concurrency levels drag the average
down). Faster preprocessing raised the arrival rate enough that the queue finally
accumulates — the original Phase 4 "make batches fill" goal, achieved indirectly.

## Where the new bottleneck is (measured — 3 diagnostics)

The first cut of this section guessed "pipeline serialization." That was an
inference by elimination (CPU ~55% idle, GPU ~59% util, pool 8≈16, flat throughput
c=128→256), and it leaned on `nvidia-smi` utilization as if it meant spare
capacity — which it does **not** (util = fraction of time ≥1 kernel ran, not
fraction of compute used). So I instrumented the server (`/stats`: per-stage
timing + collector busy-fraction) and added an **open-loop** load mode, and
measured it directly. The picture is more precise — and corrects two things I'd
overstated.

**1. The ~1,450 ceiling is real server capacity, not a closed-loop artifact.**
Open-loop (fire at a fixed offered rate regardless of responses) achieves ~1,442
req/s with **zero overruns and zero errors** before latency degrades. So the
closed-loop generator was *not* the cap.

**2. The collector/inference lane is the serial bottleneck.** `collector_busy_frac`
is **0.97–0.99 in every run** — the single collector goroutine is essentially
never idle on an empty queue. Throughput = `batch_size / per-batch-cycle`, and the
cycle is fully serialized on that one goroutine.

**3. The per-batch cycle decomposes into measured, non-overlapping parts**
(`/stats`, stable open-loop point, batch ≈ 10):

| Stage (per batch) | time | notes |
|---|---:|---|
| gather wait | ≤ 5 ms | `maxWait`; ~half the cycle at small batches |
| assemble + Run + split (`avg_runbatch_ms`) | 5.15 ms | of which… |
| &nbsp;&nbsp;• `session.Run()` (`avg_infer_run_ms`) | 3.15 ms | the actual inference |
| &nbsp;&nbsp;• assemble (15 MB CPU memcpy + H2D) + `splitBatchLogits` | **~2.0 ms** | `runbatch − run`, serialized, no overlap |

So **both CPU and GPU show idle headroom precisely because the lane is
unpipelined**: the GPU sits idle during the gather wait, the CPU tensor
marshalling, and the H2D/D2H/sync portions; the CPU sits idle during `Run()`.
That is the textbook signature of a single serial stage, and it explains the
misleading 50–65 % GPU util and ~55 % CPU idle.

**4. Correction to my earlier "decode inflates inference 3×" claim — it's a
load-dependent contention *collapse*, not a fixed factor.** `avg_infer_run_ms`
varies with load: ~3 ms (light) → ~7 ms (batch-32, healthy) → ~19 ms (backlogged).
Because this card has no NVJPG engine, decode runs on the CUDA cores; when the
queue backs up, all decoder contexts stay maxed and contend with inference,
inflating `Run` and *reducing* throughput (1,442 → 1,191) while latency explodes
(33 ms → ~2 s). So pushing past the knee makes it worse, not better. The
measured `Run` is also CPU-scheduling-sensitive: on the CPU-preprocess server,
where 8 cores are saturated decoding, the collector goroutine is starved and `Run`
wall-time balloons to ~40 ms — which is *why isolation hits 4,200/s* (collector
has a free core and the GPU to itself) and the served path cannot.

**Fix direction (Phase 5, out of scope here):** pipeline the collector —
double-buffer (assemble the next batch's input tensor *during* the current
`Run()`), move `splitBatchLogits` off the collector goroutine, put decode and
inference on separate CUDA streams, and/or run multiple inference lanes. Tuning
`maxWait` trades latency for batch fullness. None of this was needed to bank the
~4.2× win.

## Conclusion

- ✅ **~4.2× end-to-end throughput** (~357 → ~1,463 req/s), same machine.
- ✅ **~6.6× cheaper preprocessing** (38.6 → 5.9 ms/large image) via GPU decode+resize.
- ✅ **Batches fill** (≈1.3 → 20–29) as a side effect of the faster feeder.
- ✅ **Predictions unchanged**; Go coordinator untouched; Phase-0 build protected.
- 🔎 **Measured cause of the new ceiling (3 diagnostics):** the single collector
  goroutine is the serial lane (busy-fraction 0.97–0.99); its per-batch cycle is
  gather-wait + serialized CPU tensor-assembly/split + `Run()`, so neither CPU nor
  GPU saturates. Open-loop confirms ~1,450 req/s is real server capacity (0
  overruns). A secondary load-dependent decode↔inference contention (no NVJPG
  engine on this card) causes a throughput *collapse* if pushed past the knee.
  Fix is collector pipelining — Phase 5.

## Files

- `gpu_decode.go` — cgo: nvJPEG decode + NPP resize/crop, decoder-context pool,
  `normalizeRGBI`.
- `gpu_decode_test.go` — backend probe, decode parity, resize parity, top-1 parity.
- `gpu_pool_test.go` — pool concurrency correctness + `BenchmarkGPUPoolPreprocess`.
- `preprocess.go` — split into `resizeForModel` / `normalizeCHW` for stage benchmarks.
- `preprocess_bench_test.go` — per-stage decode/resize/normalize benchmarks.
- `main.go` — `-gpu-preprocess`, `-gpu-decoders` flags; CPU fallback retained.
- `phase4_cpu_results.*` / `phase4_gpu_results.*` — raw before/after load numbers.
- `PHASE4_SPEC.md` — the spec this phase executed.

## Reproduce

```bash
# Build (CUDA 12.6 nvJPEG/NPP linked via cgo; same lib paths as run.sh).
./run.sh -gpu-preprocess -gpu-decoders 8 &            # GPU preprocessing server
./run.sh &                                            # CPU baseline server (default)

CGO_ENABLED=0 ~/sdk/go/bin/go build -o loadtest/loadtest ./loadtest
./loadtest/loadtest -levels 4,16,32,64,128,256 -n 8000 -warmup 300 -out phase4_gpu_results

# Correctness + per-image cost (sets the ORT/CUDA lib paths):
./test_phase3.sh -run 'TestGPU|TestNvjpeg'
./test_phase3.sh -run '^$' -bench 'BenchmarkGPUPoolPreprocess|BenchmarkDecode' -benchtime 2s
```
