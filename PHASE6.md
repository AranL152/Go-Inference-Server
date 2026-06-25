# Phase 6 — Break the GPU decode↔inference contention wall

**Result: ~1,490 → ~1,790 req/s (+20%), by moving JPEG decode off the GPU onto
the CPU (libjpeg-turbo SIMD). The decode↔inference GPU contention is eliminated
(per-batch inference 24ms → 8ms), but the win is modest because the bottleneck
simply moved to CPU decode, which is Huffman-bound. The GPU now sits ~65% idle —
there is large inference headroom (~5,885/s) that the CPU decode path can't yet
feed.**

This was a measurement-driven phase with two negative/limiting results pinned to
hardware causes, one bug caught by a correctness check, and an honest win smaller
than the theoretical ceiling.

## 1. Where Phase 5 left us (the wall)

Throughput decomposition (each stage measured alone, this RTX 4080 SUPER / WSL2):

| Stage measured in isolation | Throughput |
|---|---|
| Inference only (no decode) | ~5,885 req/s |
| GPU decode only (no inference) | ~2,100 req/s (saturates at 16 nvJPEG ctx) |
| **Combined, served** | **~1,490 req/s** |

Combined < either stage alone ⇒ decode and inference **serialize on the GPU**.
This card has no dedicated NVJPG engine (ARCH_MISMATCH), so nvJPEG runs the
GPU_HYBRID backend on the *same CUDA cores* as ONNX Runtime inference; the kernels
contend rather than overlap. That contention is the wall.

## 2. Gate 1 — faster *GPU* decode via nvJPEG downscale: IMPOSSIBLE here

Idea: we need ~256px from ~1546px, so decode at 1/4 on the GPU to cut decode work
while keeping decode on the GPU. Built a standalone gate (`gpu_scaled_bench.go`,
decoupled nvJPEG API, decode-only, sync each):

- scale NONE → 3.10 ms/decode (works)
- scale 1/2, 1/4, 1/8 → **all fail, `INVALID_PARAMETER`**

Pinned cause, straight from `nvjpeg.h` line 701: `// works only with the hardware
decoder backend` above `nvjpegDecodeParamsSetScaleFactor`. Scale-factor downscale
runs **only on the NVJPG hardware engine**, which this consumer Ada card lacks. Not
a tuning problem — silicon. Lever dead.

## 3. Pivot — CPU scaled decode (libjpeg-turbo), decode OFF the GPU

The old CPU baseline (Phase 2, ~345/s) was **Go's pure-Go `image/jpeg` at full
resolution** — the slowest decoder doing the most work. We had never tried SIMD
libjpeg-turbo, nor scaled decode. Two stacking ideas, and moving decode off the
GPU also removes the contention entirely (GPU → inference only, 5,885 ceiling).

Probe (`cpu_decode.go` + `cpu_decode_test.go`, `-tags cpujpeg`):

- **Single-thread decode:** full 7.37ms · 1/2 6.60 · 1/4 5.95 · 1/8 5.18.
  Scaling is only ~1.24× at 1/4 → **decode is Huffman-bound** (entropy decode of
  the full bitstream dominates; DCT scaling only cuts IDCT/upsample).
  The real win: **SIMD 7.4ms vs Go's pure-Go ~38ms = 5.2×.**
- **16-way decode-only aggregate:** ~3,200/s — beats the 2,100 GPU ceiling.

## 4. Integration (`-cpu-preprocess`)

libjpeg-turbo decode → reuse the proven CPU `resizeForModel`/`normalizeCHW` tail.
Path A is, by design, "swap the decoder." New flag `-cpu-preprocess` /
`-cpu-decoders` (pool of N decoders, one libjpeg handle per goroutine — not
thread-safe); Phase-5 pipeline + backpressure unchanged; GPU does inference only.

Three things measurement forced along the way:

1. **The resize tail was a co-bottleneck.** Integrated full preprocess peaked at
   only ~1,686/s — the decode-only 3,200 had ignored that the pure-Go bilinear
   resize+normalize costs ~2.5ms/img (the GPU path hid this on NPP). **Fix:**
   `preprocessTurbo` fuses resize-shorter-256 + center-crop-224 + normalize into
   one pass over only the 224×224 output (float32, no intermediate image, no alpha,
   no second crop pass). Tail 2.5ms → ~0.8ms; aggregate 1,686 → ~2,450.

2. **A correctness bug, caught by checking predictions.** Hardcoding decode at 1/4
   upscaled small images: cat.jpg (335×500) → 83px → upscaled to 256, detail
   destroyed → top-1 flipped Egyptian cat (0.70) → tabby (0.25). **Fix:**
   `chooseScale` picks the most-aggressive DCT factor whose scaled shorter side is
   still ≥ 256 (never upscale). After fix, top-1 matches the reference:
   dog Samoyed 0.906→0.901, cat Egyptian cat 0.704→0.685.

3. **In-server CPU contention.** The isolated 2,450 doesn't survive the live server:
   the same 16 cores also do tensor assembly (15MB memcpy/batch), HTTP, JSON, GC.
   Under load CPU decode rises 5.3ms → ~9ms.

## 5. Head-to-head (same machine, same sweep, this session)

| Path | Peak req/s | GPU util | inference Run / batch |
|---|---|---|---|
| GPU-preprocess (Phase 4) | ~1,586 | 63% | **24 ms (contended)** |
| **CPU-preprocess (Phase 6)** | **~1,790** | 35% | **8 ms (uncontended)** |

The thesis is proven directly: moving decode off the GPU drops per-batch inference
from 24ms to 8ms (3×) — that's the contention disappearing. Net throughput
+13% over the live GPU path, +20% over the historical 1,490 baseline. Correctness
preserved (top-1 matches; 0 errors across 37k requests).

## 6. Honest verdict & what's left

- **Win is real but modest (+20%), not the ~2.15× the decomposition suggested.**
  The contention wall is gone, but the bottleneck simply *moved* to CPU decode,
  which is Huffman-bound (SIMD already; scaling barely helps) and shares 16 cores
  with the rest of the server. The test's small cat image (little to downscale)
  also caps the gain; a large-image workload would benefit more.
- **The GPU is now ~65% idle (35% util).** Inference can do ~5,885/s; the CPU feeds
  it only ~1,790. The resource balance flipped from GPU-bound to CPU-bound.
- **Remaining levers (future):** (a) more/faster CPU cores — decode scales with
  cores; (b) a hybrid split that sends *some* decodes to the idle GPU to use that
  headroom without re-saturating it; (c) pinned-memory / async H2D to overlap the
  input copy; (d) a faster entropy decoder. Plus the still-pending prod hardening
  (metrics/TLS/auth).

## Files
- `cpu_decode.go` — libjpeg-turbo (TurboJPEG API) decoder + pool + adaptive scale +
  fused `preprocessTurbo`. Requires `libturbojpeg0-dev` (system) / `-lturbojpeg`.
- `cpu_decode_test.go` (`-tags cpujpeg`) — single-thread scale sweep + parallel
  aggregate benchmarks (slow; kept out of the normal suite).
- `gpu_scaled_bench.go` / `gpu_scaled_bench_test.go` — the nvJPEG downscale gate
  (records that GPU scale-factor decode is hardware-only / unavailable here).
- `main.go` — `-cpu-preprocess` / `-cpu-decoders` flags, handler branch, `/stats`
  (`avg_cpu_decode_ms`, `avg_cpu_resize_normalize_ms`).
- `PHASE6_SPEC.md` — spec + §0b–0e measured results (gate + probe + decision).
