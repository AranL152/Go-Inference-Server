# PHASE 6 SPEC — Overlap GPU Decode and Inference (Break the ~1,500 Wall)

> **Lineage.** Phase 4 put JPEG decode on the GPU (~4.2×, to ~1,450). Phase 5
> pipelined the CPU side and *disproved* the idea that the collector lane was the
> bottleneck — freeing it (infer busy-fraction 0.99→0.10) gave no throughput gain.
> The wall is GPU-side: decode (nvJPEG/NPP on the CUDA cores — this card has no
> dedicated NVJPG engine) and inference (ORT) run on one GPU and do **not overlap**,
> leaving it stuck at ~63% util while neither stage alone is saturated (inference
> can do ~5,900/s in isolation). Phase 6 attacks that non-overlap directly.

## 0b. Task 1 RESULT (profile + decomposition) — changes the target

The Task-1 investigation ran, and it **reframes Phase 6**. Two methods:

- **nsys under WSL2 traces CUDA *API* calls but NOT GPU kernel execution**
  ("does not contain CUDA kernel data"). The API trace shows the synchronous
  `cudaMemcpy2D` (decode D2H) is 36% of CUDA-API time and ORT uses CUDA graphs +
  `cudaEventSynchronize` — but it can't show kernel overlap directly. Fell back to:
- **Throughput decomposition** (`TestDecodeOnlyThroughput`, `ORT_PROBE=1`):

  | Path | throughput | GPU-ms/req |
  |---|---:|---:|
  | Inference only (probe) | 5,885/s | 0.17 |
  | **Decode only (8 / 16 / 32 ctx)** | **1,820 / 2,113 / 2,083/s** | ~0.5 |
  | Combined (served) | ~1,490/s | 0.67 |

**Findings:**
1. **Decode is the dominant cost and the hard ceiling — ~2,100 req/s**, saturating
   at 16 contexts (more don't help). nvJPEG GPU_HYBRID (no hardware NVJPG engine on
   this card) is the wall. Inference (5,885) is NOT the limit.
2. Combined (1,490) is *below* decode-only (2,113) → decode and inference contend
   (~30% penalty), they don't cleanly overlap.
3. **Therefore the overlap fix's absolute ceiling is the decode-only rate
   (~2,100), not inference (~5,900).** Best case ~1,490 → ~2,100 = **+40%**, and
   realistically less. The earlier 2,500–3,500 target was wrong.

**Revised strategy:** overlapping decode/inference is now a *modest* (+~40%),
*complex* win bounded by a decode wall. The bigger lever is **faster decode
itself** — and on this consumer card that means **DCT-downscale decoding**
(libjpeg-turbo scaled decode: the source is 1546 px but we only need 256 px, so
decoding at 1/4 scale does ~16× less IDCT work). That, not GPU overlap, is the path
past ~2,100 without a hardware JPEG engine or a second GPU.

## 0. What the Phase 6 dig-in already found (read before starting)

We tried the obvious cheap fix first and it **did not work** — record this so it
isn't re-attempted blindly:

- The pooled decode path ended with a **synchronous `cudaMemcpy2D`** (D2H of the
  224×224 crop) which runs on the **default stream** and forces a device-wide
  implicit sync against ORT's inference stream — a textbook overlap-killer.
- We changed it to `cudaMemcpy2DAsync` on the per-context stream + a single
  per-stream sync, rebuilt, and re-measured: **no improvement** (c=256 1505→1430,
  util still ~67%). Reverted.

**Conclusion:** the serialization is *deeper* than the D2H copy — most likely
**(a) nvJPEG GPU_HYBRID's internal CPU-side Huffman + per-image synchronization**,
and/or **(b) ORT's CUDA execution provider synchronizing the device per `Run`**.
Neither is fixable by guessing. **Phase 6 must start by profiling the GPU
timeline**, not by editing kernels.

## 0c. RESUME STATE — execution paused here (read this first to continue)

**Decision (user):** do BOTH faster-decode (#1) AND decode/inference overlap (#2).
Coherent path: keep decode **on the GPU but downscaled** so #2 still applies.
(If #1 were done via CPU libjpeg-turbo, decode leaves the GPU and #2 becomes moot —
that's the fallback only if GPU downscale doesn't help.)

**Measured ceilings so far (this hardware):** inference-only 5,885/s; decode-only
~2,100/s (saturates at 16 nvJPEG contexts); combined served ~1,490/s. Decode is the
wall.

**nvJPEG scale-factor downscale IS available** (`NVJPEG_SCALE_1_BY_2/4/8`) — but only
via the **decoupled API**, not the simple `nvjpegDecode` we use today. We need
256px from a 1546px source, so ~1/4 scale is the target (then NPP-resize 386→256).

**IMMEDIATE NEXT STEP (in progress) — the GATE before any integration:** write a
minimal standalone benchmark that decodes ONE image at scale NONE vs 1/2 vs 1/4 vs
1/8 (decode-only, sync each) and reports ms/decode.
- If 1/4 is materially faster than NONE → scaled GPU decode works → build it into
  the decoder pool, then do the overlap fix (#2).
- If 1/4 ≈ NONE → decode is **Huffman-bound** (scale only cuts IDCT, not entropy
  decode) → **pivot #1 to libjpeg-turbo CPU scaled decode** (fast, frees the GPU for
  inference-only ~5,885, naturally overlaps — subsumes #2). New cgo dep: libjpeg-turbo.

**Decoupled nvJPEG API call sequence (signatures already verified in nvjpeg.h):**
- once: `nvjpegCreateEx` → `nvjpegDecoderCreate(handle, NVJPEG_BACKEND_GPU_HYBRID, &decoder)`
  → `nvjpegDecoderStateCreate(handle, decoder, &state)` → `nvjpegJpegStreamCreate(handle,&jstream)`
  → `nvjpegDecodeParamsCreate(handle,&params)` → `nvjpegDecodeParamsSetOutputFormat(params, NVJPEG_OUTPUT_RGBI)`
  → `nvjpegDecodeParamsSetScaleFactor(params, NVJPEG_SCALE_1_BY_4)`; `cudaStreamCreate`.
- per image: `nvjpegJpegStreamParse(handle, data, len, save_metadata=0, save_stream=0, jstream)`
  → `nvjpegJpegStreamGetFrameDimensions(jstream,&w,&h)` (scaled out ≈ ceil(w/4)×ceil(h/4);
  alloc device RGBI buf scaledW*scaledH*3, pitch scaledW*3)
  → `nvjpegDecodeJpeg(handle, decoder, state, jstream, &outImg, params, stream)` → `cudaStreamSynchronize`.
- `nvjpegDecodeJpeg(handle, decoder, decoder_state, jpeg_bitstream, destination*, decode_params, stream)`.

**Tree state:** clean; builds; full suite passes. Added `p6_decodeonly_test.go`
(decode-only throughput, guarded by `ORT_PROBE=1`, skipped in normal runs). nsys
profiling script at `/tmp/p6profile.sh`, report `/tmp/p6prof.nsys-rep` (CUDA API
trace only — WSL2 gives no GPU kernel data; use the throughput-decomposition method
instead). Phases 4 & 5 complete and uncommitted. Engine is the Phase-5 pipeline
(gather/pack → 1 infer lane → unpack workers) + backpressure (`-max-inflight`,
gate before decode) + `ensureORT` (init once/process).

**Task state:** #16 done; #17 faster-decode = IN PROGRESS at the gate above;
#18 overlap = pending; #19 re-measure + PHASE6.md = pending.

**Lib paths for any run/build:** `export LD_LIBRARY_PATH="$PWD/onnxruntime/lib:$PWD/cudnn9:/usr/local/cuda-12.6/lib64:/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"`,
`PATH` includes `~/sdk/go/bin`. nvJPEG/NPP cgo flags already in `gpu_decode.go`.

## 0d. GATE RESULT (2026-06-23) — scaled GPU decode is IMPOSSIBLE on this card

Wrote the gate (`gpu_scaled_bench.go` + `gpu_scaled_bench_test.go`, decoupled nvJPEG
API, decode-only, sync each, 500 iters, `ORT_PROBE=1`). Result:

- scale=NONE → 3.10 ms/decode (single context), out 1546×1213. **Works.**
- scale=1/2, 1/4, 1/8 → ALL fail with `INVALID_PARAMETER` (rc=-262) at
  `nvjpegDecodeJpegHost`.

**Pinned cause (from nvjpeg.h line 701, not inference):**
`// works only with the hardware decoder backend` directly above
`nvjpegDecodeParamsSetScaleFactor`. nvJPEG scale-factor downscale runs ONLY on the
dedicated NVJPG hardware engine. This RTX 4080 SUPER has no NVJPG engine
(ARCH_MISMATCH, Phase 4), so it falls back to GPU_HYBRID, which rejects any scale
factor != NONE. **Lever #1 as written (faster GPU decode via nvJPEG scaling) is dead
on this hardware — not a tuning problem, a silicon one.**

**Consequence:** the "keep decode on GPU but downscaled" plan (§0c) is not achievable.
The two remaining levers are now mutually exclusive paths, not "both":
- **Path A — CPU scaled decode (libjpeg-turbo DCT 1/2..1/8).** Moves decode OFF the
  GPU entirely → kills decode↔inference SM contention → GPU becomes inference-only
  (5,885 ceiling); CPU scaled decode is cheap and parallel across 16 cores. Higher
  ceiling, but reverses Phase 4's "decode on GPU" and reintroduces H2D. Needs dev
  headers: `libjpeg-turbo8-dev`/`libjpeg62-turbo-dev` (runtime `libjpeg.so.8` present,
  headers NOT). This is the §0c pre-authorized pivot.
- **Path B — overlap fix only (#2) on existing full-res GPU decode.** Separate
  non-blocking streams + drop per-decode device sync. Stays decode-bound: ceiling
  ~2,100, realistic 1,490 → ~1,800-2,000. No new dependency.

## 0e. PATH A PROBE RESULT (2026-06-23) — CPU scaled decode WINS, integrate it

Probe behind `-tags cpujpeg` (`cpu_decode.go` + `cpu_decode_test.go`), libjpeg-turbo
classic API (scale_num/denom, JDCT_FASTEST), headers from libjpeg-turbo8-dev deb
extracted to scratch (no system install), linked `-l:libjpeg.so.8`.

Single-thread scale sweep (dog.jpg 1546×1213):
- full 7.37ms · 1/2 6.60ms · 1/4 5.95ms · 1/8 5.18ms. Scaling barely helps
  (1/4 only 1.24× over full) → decode is **Huffman-bound** (entropy decode of the
  full bitstream dominates; DCT scaling only cuts IDCT/upsample). The REAL win is
  **SIMD libjpeg-turbo 7.4ms vs Go pure-Go image/jpeg ~38ms = 5.2×.**

16-way parallel aggregate at 1/4 (nproc=16): 4w 1131 · 8w 2118 · 12w 2724 ·
**16w 3222 (peak)** · 24w 3187 · 32w 3161. Plateaus ~3,200 (cores saturated).

**Decision (measured):** ~3,200 CPU decode/s beats the GPU decode-only ceiling
(~2,100) by +52% AND moves decode OFF the GPU → GPU runs inference-only (5,885).
New combined ceiling = min(3,200, 5,885) ≈ 3,200 vs today's 1,490 (~2.15× theory).
Caveat: 3,200 is decode-only; in-server the 16 cores also do normalize / tensor
assembly / GC / HTTP, so expect ~2,400–2,900 real, 3,200 as stretch. Keep 1/4
scale (shrinks downstream CPU resize + memory traffic even though decode savings
are modest).

**Integration plan (Path A):**
1. Promote `cpu_decode.go` to a non-tagged, real preprocessing path: libjpeg-turbo
   decode@1/4 → CPU resize-shorter-256 → center-crop 224 → normalizeCHW, all host.
   Resolve the header/lib wiring properly (vendor headers or document the deb), drop
   the scratch-path CFLAGS.
2. New flag `-cpu-preprocess` (parallel to `-gpu-preprocess`); handler picks decoder.
   Keep the gpu path intact for comparison. Pool = N goroutines each own jpegDecoder
   (handles NOT thread-safe). Reuse the Phase-5 pipeline + backpressure unchanged.
3. GPU does inference only — confirm no nvJPEG/NPP on the hot path in this mode.
4. Measure end-to-end (open-loop + /stats) vs 1,490; correctness (0 swaps, top-1);
   compare CPU vs GPU preprocess. Write PHASE6.md with the honest before/after.

This SUPERSEDES the old #18 "overlap GPU decode/inference" (Path B) — decode no
longer runs on the GPU, so there's nothing to overlap there. #18 is moot under Path A.

## 1. Goal & success criteria

Increase sustained throughput above the ~1,500 req/s wall by making GPU decode and
inference overlap (use the idle ~37%), OR produce a profiled, pinned conclusion
that they cannot overlap on this stack/hardware and why.

- **Primary:** sustained throughput materially above 1,500 req/s. **Honest target:
  2,500–3,500** (the idle 37% bounds the upside; decode genuinely consumes GPU
  compute, so the ~5,900 inference-only number is NOT reachable). **Magnitude is
  unknown until measured — a negative result with a profiled cause is an
  acceptable outcome** (Phase 3/5 precedent).
- **Correctness preserved:** dog→Samoyed, cat→Egyptian cat; 0 swaps
  (`TestPipelineRoutingStress`); full suite + `-race` green.
- **No regressions:** backpressure still sheds gracefully; Phase-0 build intact.

## 2. Investigation first (the gate — do NOT change kernels before this)

1. **Profile the GPU timeline under load with `nsys`** (`/usr/local/cuda-12.6/bin/nsys`).
   Capture a closed-loop c=128 run and inspect: do nvJPEG/NPP (decode) kernels and
   ORT (inference) kernels ever run concurrently, or strictly alternate? Identify
   every forced synchronization (default-stream ops, `cudaDeviceSynchronize`,
   nvJPEG host syncs, ORT EP syncs).
   - **WSL2 caveat:** nsys GPU sampling can be limited under WSL2. Fallback:
     CUDA events around decode vs Run, plus `ncu`/`nvprof`, to bound overlap.
2. **Pin which component forces the serialization:**
   - *ORT:* does the CUDA EP run on the default stream / call a device sync per
     `Run`? Check ORT run options / `RunOptions`, and whether a non-default stream
     can be supplied (IO binding with a user stream).
   - *nvJPEG:* does GPU_HYBRID sync the host per `nvjpegDecode`? Compare against
     **batched** nvJPEG decode (one call for many images) to see if the sync is
     per-image (amortizable) or intrinsic.
3. **Report the timeline finding** before writing any fix — this is what makes the
   eventual result trustworthy.

## 3. Fix candidates (apply based on the profile, measure each)

- **A. GPU-resident handoff (eliminate the GPU→CPU→GPU round trip).** Today decode
  copies the crop D2H, the CPU normalizes, then the collector copies H2D to build
  the ORT input — two PCIe crossings per request. Instead: **normalize on the GPU**
  (NPP or a small kernel) into the NCHW float tensor on-device, and feed inference
  via **ORT IO binding** of that GPU tensor. Removes both copies and the CPU
  normalize, and lets decode-stream and inference-stream be the only GPU work —
  the prerequisite for real overlap. Biggest change, biggest potential.
- **B. Explicit stream separation + pinned memory.** Ensure no decode op touches
  the default stream; any unavoidable host copy uses pinned buffers (per decoder
  context) so it can truly overlap compute via `cudaMemcpy*Async`.
- **C. Batched nvJPEG decode.** If the profile shows per-image decode sync is the
  cost, decode N images per `nvjpegDecodeBatched` call to amortize it. Fits the
  existing batch boundary (the gatherer already groups requests).
- **D. CUDA graphs for inference** to cut per-`Run` launch/sync overhead, if the
  profile shows that as a contributor.
- **E. (Pivot, if overlap proves impossible) move decode OFF the inference GPU:**
  libjpeg-turbo on CPU returns the GPU to inference-only (~5,900 ceiling, bounded
  by CPU decode rate with turbo). Worth it only if the profile says GPU overlap is
  intrinsically blocked by nvJPEG GPU_HYBRID on this card.

## 4. Scope

**In:** GPU-timeline profiling; one or more of fixes A–E driven by it; re-measure
(open-loop + closed-loop + `/stats` + nsys); `PHASE6.md`.

**Out:** multi-GPU / second card (its own phase); model re-export beyond adding
preprocessing ops if fix A needs it; production observability/TLS/auth (hardening
phase); changing the Phase-0 ORT/cuDNN/CUDA stack.

## 5. Tasks (ordered to de-risk)

1. **nsys/CUDA-event profile** of the current pipeline under load — produce the
   decode-vs-inference overlap timeline and the list of forced syncs. **Gate.**
2. Based on the gate, prototype the **single highest-leverage** fix (likely A:
   GPU-resident handoff via ORT IO binding) behind a flag, CPU/Phase-5 path retained.
3. **Correctness gate:** dog/cat top-1, `TestPipelineRoutingStress` (0 swaps),
   `-race`, full suite — before any throughput claim.
4. Re-profile to confirm overlap actually happens now (not just throughput moved).
5. **Re-measure** before/after vs ~1,500; locate the new bottleneck.
6. **`PHASE6.md`** — honest report: the timeline finding, the fix, before/after,
   new ceiling, or a pinned negative result.

## 6. Risks & mitigations

- **ORT IO binding correctness/complexity (fix A).** Binding a GPU tensor + GPU
  normalize must produce bit-equivalent-enough results to keep top-1. *Mitigation:*
  reuse the Phase 4 parity tests; gate on `TestPipelineRoutingStress`.
- **The overlap may be intrinsically blocked** (nvJPEG GPU_HYBRID host sync, or
  ORT device sync with no escape). *Mitigation:* that's a valid negative result —
  report it from the profile and consider pivot E.
- **nsys under WSL2 limited.** *Mitigation:* CUDA-event timing fallback.
- **GPU memory** from on-device tensors / IO binding. *Mitigation:* bounded
  in-flight (already in place) + measure GPU mem.

## 7. Deliverables

- Profile artifacts (nsys timeline / CUDA-event numbers) showing the serialization.
- The chosen fix behind a flag; Phase-5 path retained as fallback.
- Correctness suite green (+ `-race`); before/after throughput + `/stats` + overlap
  confirmation.
- `PHASE6.md` — honest outcome, including a pinned negative result if overlap can't
  be achieved.
