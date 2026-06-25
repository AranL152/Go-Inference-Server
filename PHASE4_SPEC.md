# PHASE 4 SPEC — GPU Image Preprocessing (nvJPEG + NPP)

> **Lineage note.** The original Phase 4 goal was "make dynamic batching actually
> fill." The front-loaded investigation (kept below) *disproved* that as a useful
> goal on this hardware and redirected us here. This spec is the redefined Phase 4:
> attack the real, measured ceiling — CPU JPEG decode — by moving decode + resize
> onto the GPU. Treat the investigation section as the justification that earns the
> right to cross the "no GPU preprocessing" line the previous spec drew.

## 1. Context — what the investigation established (measured, not assumed)

- **Throughput is preprocessing-gated, not batch-gated.** The GPU batched path
  sustains ~4,200 req/s at batch-32 in isolation; under real traffic it stays at
  ~345 req/s. Removing the Phase 1 mutex (Phase 3) did not help.
- **Batching cannot help on this box.** The collector is a queue whose consumer
  (GPU, ~4,200/s) drains ~10× faster than its producer (CPU preprocessing,
  ~415/s ceiling) fills it. A queue drained faster than it fills never builds
  depth, so realized batch size stays ≈1.3 regardless of `-max-wait` or offered
  concurrency. Bigger batches would only trade latency for the *same* throughput.
- **The bottleneck is JPEG decode, and it scales with image resolution.** Splitting
  the ~39 ms/image cost (`preprocess_bench_test.go`):

  | Stage | dog.jpg (1546×1213) | cat.jpg (335×500) |
  |---|---:|---:|
  | **JPEG decode** | **31.8 ms (82%)** | 3.3 ms (44%) |
  | Resize (bilinear) | 6.9 ms (18%) | 2.4 ms (32%) |
  | Normalize → NCHW | 0.26 ms (<1%) | 0.26 ms (3%) |

- **Real parallelism is ~8-wide, not 16.** CPU is a Ryzen 7 7800X3D (8 physical
  cores / 16 threads). Load-mix avg ~22 ms/image ÷ 8 cores ≈ 355/s — matching the
  measured 345 baseline. The earlier "16-core / ~415/s ceiling" was optimistic.
- **Pure-Go cannot fix decode.** `image/jpeg` is slow and cannot DCT-downscale
  (decode directly to ~256 px). Every real decode fix needs a C dependency
  (libjpeg-turbo on CPU, or nvJPEG on GPU). Since we're paying the cgo cost
  regardless, and the GPU sits ~70% idle, we decode on the GPU.

**Conclusion the spec acts on:** CPU JPEG decode is the confirmed hard ceiling.
Move decode + resize to the GPU; keep everything else.

## 2. Goal & success criteria

Raise end-to-end `POST /predict` throughput meaningfully above the ~345 req/s
baseline by eliminating CPU JPEG decode as the bottleneck.

- **Primary:** end-to-end throughput materially exceeds 345 req/s under the same
  load test (`-levels 1,4,16,64,128 -n 1500`). Report the honest number and where
  the new bottleneck lands (GPU decode? PCIe copy? inference? Go collector?).
- **Correctness preserved:** GPU-preprocessed predictions match the CPU path —
  dog → 258 Samoyed, cat → 285 Egyptian cat, including mixed batches and the
  existing no-swap routing tests. Numeric drift from a different resize kernel is
  allowed *only if top-1 is unchanged* on the test set.
- **Per-stage re-measurement:** new decode+resize cost on GPU vs the 31.8/6.9 ms
  CPU numbers; realized batch size after the change (expected to finally rise as
  the feeder speeds up).
- **Build setup protected:** the Phase 0 ORT/cuDNN/CUDA stack still loads on GPU;
  `run.sh` still works. A negative or partial result with a pinned cause is still
  a valid outcome (Phase 3 precedent).

## 3. Confirmed environment (probed)

- nvJPEG: `/usr/local/cuda-12.6/.../libnvjpeg.so` (+ `nvjpeg.h`)
- NPP resize: `libnppig.so`, core `libnppc.so` (+ `nppi_geometry_transforms.h`)
- `libcudart.so`, `nvcc` present; CUDA **12.6** matches the ORT CUDA-12 build.
- `run.sh` already puts `/usr/local/cuda-12.6/lib64` on `LD_LIBRARY_PATH` (verify
  it symlinks to `targets/x86_64-linux/lib`; add the targets path if not).

## 4. Architecture — minimal blast radius

Keep the Go coordinator (HTTP server, batching collector, result routing)
**completely unchanged.** Replace only the per-request preprocessing leaf.

```
BEFORE:  handler → preprocessReader (CPU decode+resize+normalize) → []float32 → collector → batched inference
AFTER:   handler → gpuPreprocess (nvJPEG decode + NPP resize on GPU) → []float32 → collector → batched inference
```

The GPU preprocessor returns the **same `[]float32` of length `inputElems`
(3×224×224, NCHW, normalized)** the collector already consumes. The collector,
batching, and inference path do not change. The big image is decoded/resized on
the GPU; only the small 224×224 result crosses PCIe.

### Decision points (resolve during implementation, measure don't assume)
1. **Normalize on GPU vs CPU.** Normalize is 0.26 ms — negligible. Option A:
   NPP does decode→resize→normalize, return float32 NCHW (fully GPU, smaller
   PCIe copy of floats). Option B: GPU returns 224×224×3 uint8, Go does the
   normalize (smaller cgo surface, ~0.26 ms CPU/req is irrelevant at these rates).
   **Default: Option B** for a smaller, safer cgo surface; revisit if the host
   copy shows up in profiling.
2. **Decoder concurrency.** nvJPEG state is not freely thread-safe. Use a **pool
   of decoder contexts** (handle/state/stream/device buffers), size configurable
   via a flag (e.g. `-gpu-decoders 8`). Each handler borrows one, decodes,
   returns it. The pool size becomes the new concurrency knob and a natural
   backpressure point.
3. **Per-request vs batched decode.** Start with **per-request** GPU decode
   (one image per cgo call) — simplest, and the existing collector still batches
   the *inference*. Batched nvJPEG decode is a later optimization, only if
   per-request decode-launch overhead shows up as the new ceiling.
4. **Resize convention parity.** CPU path uses bilinear, half-pixel
   (align_corners=False), shorter-side-256 then center-crop-224. Configure NPP
   resize to match as closely as possible; validate by top-1 parity, not bit
   equality.

## 5. Scope

**In:** nvJPEG decode + NPP resize (+ crop) behind a cgo boundary; a decoder-context
pool; a `gpuPreprocess([]byte) ([]float32, error)` entry point swapped into the
handler behind a flag (`-gpu-preprocess`, default off until validated); correctness
parity tests; before/after throughput + per-stage + batch-size re-measurement;
`PHASE4.md` report.

**Out:** batched nvJPEG decode (later, only if needed); GPU-resident handoff to
ORT without a host round-trip (later); backpressure/hardening (Phase 5);
multi-GPU (Phase 6); replacing the Go coordinator with Triton/DALI (explicitly
never — that deletes the project).

## 6. Implementation tasks (ordered to de-risk early)

1. **Standalone cgo decode smoke test, off the server.** A tiny `cgo` file +
   `go test` that: inits nvJPEG once, decodes `testdata/dog.jpg` on the GPU,
   copies RGB back to host, and compares against `image/jpeg` decode (allow small
   per-pixel tolerance). **Goal: prove the libs link against the protected stack
   and decode correctly before touching anything else.** This is the gate.
   **Also report the nvJPEG backend chosen** — try `nvjpegCreateEx` with
   `NVJPEG_BACKEND_HARDWARE` and log whether this RTX 4080 SUPER exposes the
   dedicated NVJPG decode engine (decode on separate silicon, no contention with
   inference) or falls back to `NVJPEG_BACKEND_GPU_HYBRID` (CUDA cores, shares
   SMs with inference). This finding directly informs the decode-vs-inference
   contention risk in §7 and should be recorded in `PHASE4.md`.
2. **Add NPP resize + center-crop** to the smoke path; compare the resized result
   against `bilinearResize`/`resizeForModel` (top-1 parity after normalize).
3. **Decoder-context pool** with configurable size; concurrency-safe checkout.
4. **`gpuPreprocess` entry point** returning `[]float32` (NCHW, normalized),
   wired behind `-gpu-preprocess` flag; CPU path remains the default/fallback.
5. **Correctness suite:** dog→Samoyed, cat→Egyptian cat, mixed-batch no-swap, all
   passing through the GPU path. Reuse existing integration tests with the flag on.
6. **Re-measure:** per-stage GPU decode+resize cost; full load sweep
   (`-levels 1,4,16,64,128`); realized batch size; GPU util; before/after table.
7. **`PHASE4.md`** — honest report: cause (decode), fix, before/after throughput
   and batch size, new bottleneck location, and any parity caveats.

## 7. Risks & mitigations

- **Breaking the Phase 0 build (highest risk).** nvJPEG/NPP add cgo `LDFLAGS`/
  include paths. *Mitigation:* match CUDA 12.6 exactly; keep the GPU path behind a
  build that still produces a working CPU-path binary; task 1 is a standalone gate
  before integration; never edit the existing ORT lib-path ordering in `run.sh`,
  only append.
- **Resize/decode numeric divergence changing predictions.** *Mitigation:*
  validate by top-1 parity on the labeled test images, not bit equality; tune NPP
  interpolation mode if a known image flips class.
- **GPU now shared between decode and inference.** The 4,200/s inference-only
  ceiling will drop somewhat. *Mitigation:* measure the real combined ceiling;
  it should still land well above the ~345–450 CPU path. Report it honestly.
- **nvJPEG thread-safety / GPU OOM under load.** *Mitigation:* bounded decoder-
  context pool caps concurrent decodes and device memory.
- **Per-request cgo/launch overhead becomes the new floor.** *Mitigation:* if
  measured, move to batched nvJPEG decode (deferred task).

## 8. Deliverables

- `gpu_preprocess.go` (+ cgo) — nvJPEG decode + NPP resize + decoder pool.
- Parity + correctness tests through the GPU path.
- `-gpu-preprocess` / `-gpu-decoders` flags in `main.go`; CPU path retained.
- Updated `run.sh`/build notes for the added lib/include paths (append-only).
- Before/after table (throughput, per-stage cost, realized batch size, GPU util).
- `PHASE4.md` honest report, including where the bottleneck moved.
```
