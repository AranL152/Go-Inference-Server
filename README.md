# go-inference-server

A GPU inference server for ResNet-18 written in Go (ONNX Runtime, CUDA execution
provider), taken from **345 to ~9,100 req/s on a single RTX 4080 SUPER** across
nine measured optimization phases — a 26× improvement on the same hardware, with
every step profiled, every hypothesis tested, and every negative result kept on
the books.

```
345 ──► 1,463 ──► 1,790 ──► 2,130 ──► 7,500 ──► 9,100 req/s
 P2       P4       P6        P7        P8        P9
```

## The journey

Each phase attacked the bottleneck the previous phase's measurements exposed.
Full write-ups live in `PHASE1.md`–`PHASE9.md` (specs in `PHASE*_SPEC.md`, raw
load-test data in `results/`).

| Phase | Lever | Result (req/s) | What the measurements said |
|---|---|---:|---|
| 1 | Long-running HTTP server, persistent CUDA session | — | baseline correctness |
| 2 | Load-test harness (`loadtest/`) | 345 | mutex serializes inference; GPU ~62% idle |
| 3 | Dynamic batching coordinator | 338 | **no gain** — real bottleneck was CPU JPEG decode (realized batch ≈ 1.3) |
| 4 | GPU preprocessing (nvJPEG + NPP via cgo) | 1,463 | batches finally fill (20–29); new wall: single collector goroutine |
| 5 | Pipelined collector + backpressure (`-max-inflight`) | 1,500 | **+3% only** — disproved Phase 4's theory; real wall was decode↔inference SM contention |
| 6 | CPU decode via libjpeg-turbo SIMD (`-cpu-preprocess`) | 1,790 | GPU util 63%→35%, per-batch Run 24→8 ms; bottleneck moved to Huffman-bound CPU decode |
| 7 | Allocation pooling (pprof-guided `sync.Pool`s) | 2,130 | 1.9 MB→42 KB garbage/request; hybrid CPU+GPU decode built, measured, **rejected** |
| 8 | Raw-tensor contract + concurrent infer lanes + FP16 | 7,500 | decode off the wire entirely; lanes overlap H2D with compute |
| 9 | Device-resident I/O binding (forked Go binding + `cudaMalloc`/pinned staging) | **8,700 raw / 9,100 raw-RGB** | ORT's pageable H2D removed from the serialized path; per-Run 20.4→15.0 ms |

At the final operating point the GPU sits at ~91% utilization under open-loop
saturation with p50 ≈ 10 ms — the remaining headroom is in the compute itself
(TensorRT EP, INT8), not the serving path.

## Architecture

```
POST /predict ──► backpressure gate (-max-inflight, sheds 503 BEFORE decode)
                        │
              preprocess (per content type)
                ├─ image/jpeg      → libjpeg-turbo SIMD pool (or nvJPEG, or hybrid)
                ├─ x-raw-rgb       → server-side normalize only (150 KB uint8)   ◄ fastest
                └─ x-raw-tensor    → memcpy into pooled NCHW tensor (602 KB f32)
                        │
              dynamic batcher (gather ≤ max-batch or max-wait)
                        │  packs into PINNED host staging
              N infer lanes on ONE shared ORT session (-infer-lanes)
                        │  engine-owned cudaMalloc device buffers, bound once;
                        │  lane does its own H2D, Run, 128 KB logits D2H
              unpack workers ──► per-request top-1 response
```

Key design points:

- **Dynamic batching** with a gather window (`-max-batch 32 -max-wait 5ms`);
  batch 32 is measured optimal — per-image Run cost *rises* with batch on this
  model/card (0.24 ms @ 32 → 1.08 ms @ 128).
- **Concurrent infer lanes** on a single session: per-Run time inflates
  (CUDA-stream serialization) but overlapping pack + H2D with compute nets +31%.
- **Device-resident I/O** (`-device-binding`): ~90 lines added to a local fork
  of `onnxruntime_go` (`onnxruntime_go_fork/`, wired via a go.mod replace)
  expose CUDA memory-info and a `DeviceTensor`; `cudamem.go` drives
  cudaMalloc/cudaHostAlloc/cudaMemcpy via libcudart.
- **Allocation discipline**: pooled request bodies, input tensors, and batch
  staging buffers — 45× less garbage per request, ~9 GCs per load sweep instead
  of ~1,000.
- **Backpressure before the expensive work**: overload p50 went from 3.7 s to
  60 ms by gating ahead of decode.

## Findings worth reading (including the failures)

The phase docs record what *didn't* work with the same rigor as what did:

- **The contract inversion** (Phase 9): once ORT's H2D copy left the hot path,
  the network-realistic 150 KB uint8-RGB contract *beats* the 602 KB float32
  tensor contract — HTTP byte-shoveling costs more than server-side normalize.
- **CUDA Graphs is incompatible with dynamic batching** — three independent
  confirmations; any Run with a new batch shape breaks post-capture.
- **Two undocumented ORT IoBinding behaviors** found via a routing stress test:
  CUDA sessions never write pre-bound CPU outputs (read via
  `GetBoundOutputValues` only), and bound CPU inputs' device copies are cached
  across runs (re-bind per batch).
- **This RTX 4080 SUPER has no NVJPG hardware decode engine** — nvJPEG falls
  back to CUDA cores and fights inference for SMs, which is why CPU decode won
  (Phase 6) and hybrid spill lost (Phase 7).
- **ORT ≥ 1.22 carries a ~20% fp32 per-Run regression** vs 1.18.1 on this
  model (measured identical on 1.22/1.26/1.27) — accepted deliberately because
  every remaining lever lives on FP16.

## Running it

Requires Linux (built under WSL2), an NVIDIA GPU, CUDA 12.6, cuDNN 9, ONNX
Runtime 1.27.1 GPU build, and `libturbojpeg0-dev`. See `SETUP.md` for the full
environment bring-up (the `onnxruntime/` and `cudnn9/` library directories are
not committed — too large for GitHub).

```bash
# Build + run at the recommended throughput operating point
./run.sh -cpu-preprocess -quiet -max-batch 32 -infer-lanes 5 \
         -model resnet18_fp16.onnx -device-binding

# Classify an image
curl -X POST --data-binary @testdata/dog.jpg localhost:8080/predict
# {"class_index":258,"class_name":"Samoyed","confidence":0.90}

# Fastest client contract: pre-decoded 224x224 RGB, 150,528 bytes
curl -X POST -H 'Content-Type: application/x-raw-rgb' \
     --data-binary @img.rgb localhost:8080/predict
```

Endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /predict` | inference — `image/jpeg`, `application/x-raw-rgb`, or `application/x-raw-tensor` |
| `GET /health` | liveness + device placement check |
| `GET /stats` | human-readable per-stage timing averages (`?reset=1` between load levels) |
| `GET /metrics` | Prometheus exposition: pipeline counters, `/predict` latency histogram, decoder-pool and Go runtime stats |
| `/debug/pprof/` | live CPU/alloc profiling |

Load testing (closed- and open-loop):

```bash
cd loadtest && go build .
./loadtest -levels 64,128,256 -n 48000 -warmup 1000 -raw-rgb   # ≥48k per level — shorter runs sit inside GPU boost-clock ramp
```

## Repo layout

- `main.go`, `engine.go`, `preprocess.go`, `metrics.go` — server, batching
  pipeline, CPU preprocess, Prometheus endpoint
- `cpu_decode.go` / `gpu_decode.go` / `hybrid_decode.go` — the three decoder
  pools (libjpeg-turbo / nvJPEG+NPP / CPU-primary-GPU-overflow)
- `cudamem.go` — CUDA allocation helpers for device-resident binding
- `onnxruntime_go_fork/` — local binding fork adding `DeviceTensor` (~90 lines)
- `loadtest/` — the measurement harness every phase's numbers come from
- `PHASE*.md` — the investigation record; `results/` — raw CSV/JSON per run
- `convert_fp16.py` — fp32 → fp16 model conversion (keep_io_types)
