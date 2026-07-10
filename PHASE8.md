# PHASE 8 — Raw-Input Contract + Infer-Lane Throughput

**Result: ~4,900 → ~7,500 req/s** (raw-tensor path, closed-loop sustained,
FP16 model + 4 concurrent infer lanes). Without FP16: **~6,400** at 3 lanes,
already past both the ≥5,500 target and the 5,885 Phase-5 bare-probe stretch
goal. JPEG legacy path unchanged (~2,260). All correctness gates green,
0 errors / 0 rejects in every measured run. Measured 2026-07-10 on the
Phase-7 rig (16-core, RTX 4080 SUPER, WSL2).

## The contract (premise, decided in the spec)

The server's primary contract is now pre-decoded input; JPEG remains a legacy
path, unchanged. Rationale and precedent (Triton/KServe raw tensors, MLPerf
untimed preprocessing, decode-tier separation) are in `PHASE8_SPEC.md`. Two
raw variants ship:

- `application/x-raw-tensor` — 602,112-byte preprocessed NCHW float32 tensor;
  server does HTTP + batching + inference only. The headline path.
- `application/x-raw-rgb` — 150,528-byte 224×224 interleaved-RGB uint8; server
  does one fused normalize-into-pooled-tensor pass (`normalizeRGB8`). The
  network-realistic variant: 4× smaller bodies (~6.9 Gbps at its peak vs
  ~28 Gbps for tensors at theirs).

## Measurement protocol change (important for reproducing)

Phase-8 throughput makes a 12,000-request level finish in ~2 s, which is
inside the GPU's boost-clock ramp on this desktop card — early short-burst
runs swung from 5,481 to 6,971 req/s at identical settings. All comparison
numbers below use **48,000 requests per level** (~6–10 s sustained) after a
4,000-request warm pass. Residual run-to-run variance is still ~±10% (WSL2,
shared desktop GPU); conclusions below only rest on differences bigger than
that, and the ordering of configurations was stable across repeats.

## Task 1 — max-batch sweep (raw, 1 lane, fp32): negative, stay at 32

Per-image Run cost gets WORSE with batch size on this card, catastrophically
so at 128 (cuDNN picks slower kernels / larger workspaces at higher batch):

| max-batch | best throughput | avg_infer_run_ms | avg batch | ms/image |
|----------:|----------------:|-----------------:|----------:|---------:|
| 32        | 5,145           | 7.7              | 31.9      | 0.24     |
| 64        | 4,574           | 16.4             | 62.4      | 0.26     |
| 128       | 1,749           | 78.1             | 72.5      | 1.08     |

(Files: `phase8_t1_b32/b64/b128.*`. Short-protocol runs, but the losses are
far outside noise.) Batch 32 kept for everything below.

## Task 2 — concurrent infer lanes (`-infer-lanes`, one shared session): kept

N `inferLoop` goroutines issue concurrent `session.Run` calls on the ONE
shared ORT session (sessions are thread-safe; Phase 5 only disproved multiple
*sessions*). Per-Run wall time inflates roughly linearly with lanes — the
Runs mostly serialize on the CUDA stream — but overlapping lane A's CPU-side
pack/split and implicit H2D copy with lane B's GPU compute still nets +31%:

| lanes (fp32, batch 32) | best sustained req/s | avg_infer_run_ms |
|-----------------------:|---------------------:|-----------------:|
| 1 (default)            | 4,903                | 7.1              |
| 2                      | 5,941                | 12.1             |
| 3                      | **6,429**            | 15.0             |
| 4                      | 6,057                | 19.0             |
| 5                      | 5,721 (short proto)  | 31.3             |
| 6                      | 5,468 (short proto)  | 32.6             |

(Files: `phase8_t2_lanes*_long*.*`.) GPU util rises 67% → ~86–91% at 3 lanes.
Default stays 1 (exact Phase 5–7 behavior); flag is opt-in.
`TestPipelineRoutingStressMultiLane` proves 1,000 interleaved dog/cat requests
across 2 lanes route with 0 swaps. With >1 lane, `/stats` busy/idle aggregates
across lanes (busyFrac reads as the average per-lane busy fraction).

## Task 3 — FP16 model: kept, +17% on top of lanes (new lane optimum: 4)

`convert_fp16.py` converts with `keep_io_types=True`, so I/O stays float32 and
the Go side is unchanged (`-model resnet18_fp16.onnx`). The old GluonCV export
needed three fixups first (opset 8→13, IR version 3→8, initializer-inputs
dropped) — details in the script; without them ORT fails to load the converted
model with "Tensor element type mismatch. 10 != 1".

- Correctness: dog → Samoyed 0.9008 (fp32 ref 0.909), cat → Egyptian cat
  0.6858 (ref 0.684). Top-1 identical, confidence Δ < 0.01 (gate was ±0.05).
- Model file 46.8 MB → 29.8 MB. Session GPU alloc is small enough (+42 MiB)
  to trip the Phase-1 "may have fallen back to CPU" heuristic warning — it's
  false here: GPU util hits 90% under load, confirmed CUDA execution.
- At 3 lanes FP16 is throughput-neutral (~6,200): compute got cheaper (GPU
  util dropped 86%→79% at the same req/s) but the wall was elsewhere.
- The cheaper Runs shift the lane knee to 4: **7,633 peak, 7,467–7,559 on
  repeat** (`phase8_t3_fp16_lanes4*.*`). 5 lanes regresses (5,798).

## Task 4 — uint8-RGB raw variant: built, measured

`application/x-raw-rgb` + loadtest `-raw-rgb`. On the best config (FP16, 4
lanes): **5,767 req/s** sustained (`phase8_t4_rawrgb_long.*`) — ~23% below the
raw-tensor path (the per-request normalize costs ~1.7 cores at this rate and
competes with the pack loop), but still 2.5× the JPEG path with 4× less
network than raw tensors. Correctness: Samoyed 0.9092 / Egyptian cat 0.6865.

## Task 5 — final head-to-head (all sustained closed-loop, 0 errors)

| path / config                              | req/s | GPU util |
|--------------------------------------------|------:|---------:|
| JPEG legacy (fp32, 1 lane, defaults)        | 2,264 | ~39%     |
| raw tensor, fp32, 1 lane (Phase-8 baseline) | 4,903 | ~67%     |
| raw tensor, fp32, 3 lanes                   | 6,429 | ~86%     |
| raw tensor, FP16, 4 lanes (headline)        | 7,500± | ~85%    |
| raw RGB uint8, FP16, 4 lanes                | 5,767 | ~81%     |

Open-loop saturation (FP16, 4 lanes, raw tensors): offered 7,500 → achieved
6,434 with p50 14.4 ms; offered 9,000 → achieved 6,803 with p50 244 ms (queue
inflation past saturation), 0 errors/overruns at every rate. The open-loop
ceiling (~6,800) sits ~10% under the best closed-loop bursts — consistent with
the measured run-to-run variance band; the honest capacity claim is
**~6,800–7,500 req/s**.

Correctness suite: dog/cat top-1 on all three content types on the default
fp32 build; `go test .` (incl. both routing-stress tests) and
`go test -tags cpujpeg .` green; `go vet` clean.

## What moved the number

Serialized-lane overhead (pack + implicit H2D inside `Run`), not GPU compute,
was the Phase-7 wall. Concurrent Runs on one session recover most of it
without touching the binding (IOBinding/pinned buffers/CUDA Graphs remain out
of reach in yalue/onnxruntime_go v1.11.0 — verified, `ortMemoryInfo` is
CPU-only). FP16 then lowers the GPU-side cost enough to feed a 4th lane.

## Negative results (kept honest)

- Batch 64/128: worse to catastrophic (table above). Do not raise max-batch.
- FP16 at 3 lanes: no gain — only pays off once lanes can use the freed GPU.
- Lanes ≥5: per-Run serialization eats the overlap win at either precision.
- Short measurement bursts (12k requests) are unreliable at these rates on
  this rig; they produced both the 6,971 outlier and 4,139 underread.

## Recommended operating points

- Throughput: `-cpu-preprocess -quiet -max-batch 32 -infer-lanes 4 -model resnet18_fp16.onnx`
- FP32-only: `-infer-lanes 3`
- Default (no flags): unchanged Phase-7 behavior, 1 lane, fp32.
