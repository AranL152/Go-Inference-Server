# go-inference-server

An HTTP inference server for ResNet-18 written in Go, running on ONNX Runtime's
CUDA execution provider. It loads the model onto the GPU once, batches incoming
requests dynamically, and serves top-1 ImageNet predictions.

The point of the repo is throughput on one card: it went from 345 req/s to about
9,100 req/s on a single RTX 4080 SUPER, and everything in `loadtest/` exists so
that number is measured rather than claimed.

## What's in it

- `main.go` — HTTP layer, request-body handling, three request contracts
- `engine.go` — the ORT session, dynamic batcher, infer lanes, unpack workers,
  backpressure, I/O binding paths
- `preprocess.go` — pure-Go decode, resize, center-crop, normalize
- `cpu_decode.go` — libjpeg-turbo (SIMD) decoder pool, the default decode path
- `gpu_decode.go` — nvJPEG + NPP decoder pool (decode and resize on the GPU)
- `hybrid_decode.go` — router that uses CPU decoders first and the GPU as overflow
- `cudamem.go` — cudaMalloc / cudaHostAlloc / cudaMemcpy wrappers for the
  device-resident binding path
- `metrics.go` — hand-rolled Prometheus endpoint
- `onnxruntime_go_fork/` — local fork of `github.com/yalue/onnxruntime_go` adding
  a `DeviceTensor` (~90 lines), wired in through a `go.mod` replace
- `loadtest/` — closed-loop and open-loop load generator with nvidia-smi sampling
- `convert_fp16.py` — fp32 to fp16 model conversion
- `yolo.go`, `detect.go` — YOLOX letterbox preprocessing, box decoding and NMS
- `jobs.go` — dataset detection jobs behind the web UI
- `sysstats.go` — live GPU, CPU, memory and pipeline samples for the UI
- `ui/` — Svelte web UI, built into `webui/` and embedded in the binary
- `export_yolox.py` — YOLOX-S export with a dynamic batch dimension

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /predict` | inference; `image/jpeg`, `application/x-raw-rgb`, or `application/x-raw-tensor` |
| `GET /health` | liveness plus device-placement check |
| `GET /stats` | per-stage timing averages (`?reset=1` to zero between load levels) |
| `GET /metrics` | Prometheus exposition: pipeline counters, latency histogram, decoder pools, Go runtime |
| `/debug/pprof/` | live CPU and allocation profiles |
| `/` | web UI (detector models) |
| `/api/jobs` | start, list, cancel and page through dataset detection jobs |
| `/api/system` | GPU, CPU, memory and pipeline samples, last 120 seconds |

## Running it

Linux with an NVIDIA GPU, CUDA 12.6, cuDNN 9, an ONNX Runtime 1.27.1 GPU build,
and `libturbojpeg0-dev`. The `onnxruntime/` and `cudnn9/` library directories are
not committed; `SETUP.md` covers how to get them and what the version
constraints are.

```bash
# throughput operating point
./run.sh -cpu-preprocess -quiet -max-batch 32 -infer-lanes 5 \
         -model resnet18_fp16.onnx -device-binding

# classify an image
curl -X POST --data-binary @testdata/dog.jpg localhost:8080/predict
# {"class_index":258,"class_name":"Samoyed","confidence":0.90}

# pre-decoded 224x224 RGB body (150,528 bytes), the fastest contract
curl -X POST -H 'Content-Type: application/x-raw-rgb' \
     --data-binary @img.rgb localhost:8080/predict
```

Tests: `./gotest.sh` (unit tests plus GPU integration tests) and `./smoke.sh`
against a running server. Load tests:

```bash
cd loadtest && go build .
./loadtest -levels 64,128,256 -n 48000 -warmup 1000 -raw-rgb
```

Use at least ~48k requests per level. Shorter runs finish inside the GPU's
boost-clock ramp and read low.

## Object detection demo

The server also runs a detector. With a detector model, `/predict` returns
boxes instead of a class, and the web UI at `/` has three tabs:

- **Detect**: point it at a folder of JPEGs on the server, or upload images
  or a whole folder, and list the classes to sort into (blank means all 80
  COCO classes). Each job shows a live progress bar.
- **Results**: images grouped by detected class with per-category counts and
  boxes drawn on each image.
- **Performance**: a Task Manager style view of GPU utilization, VRAM, power,
  clocks, every CPU core, memory, throughput, latency and batch size.

```bash
./demo.sh                     # YOLOX-S on TensorRT FP16, UI at http://localhost:8080
curl -X POST --data-binary @testdata/street.jpg localhost:8080/predict
# {"width":768,"height":576,"detections":[{"class_name":"bicycle","confidence":0.95,"box":[124,120,560,421]}, ...]}
```

The model is YOLOX-S (Megvii, Apache-2.0), exported by `export_yolox.py` with
a dynamic batch dimension and box decoding inside the graph. Throughput over
all 5,000 COCO val2017 images, end to end (file read, JPEG decode, letterbox,
inference, NMS):

| Configuration | images/s |
|---|---|
| CUDA FP32, 1 lane | 178 |
| TensorRT FP16 | 560 |
| + device-binding | 696 |
| + 3 infer lanes, 256 job workers | 996 |
| + letterbox column tables and a no-resize fast path | 1,173 |

Accuracy through the full Go pipeline is 40.4 COCO mAP (0.5:0.95) on val2017
for both FP32 and TensorRT FP16, against 40.5 published for YOLOX-S. Measure it
with `eval_coco.py` against a server started with `-conf 0.001 -iou 0.65`.

TensorRT INT8 (`quantize_yolox.py`, Conv-only QDQ, MinMax calibration on 512
val2017 images, scored on the other 4,488) was tried and not adopted: 38.7 mAP
against 40.4 for FP16 on the same images, and 1,234 against 1,228 images/s,
because the GPU is not the bottleneck here.

Rebuild the UI with `cd ui && pnpm install && pnpm run publish`.

## How it works

```
POST /predict -> in-flight gate (-max-inflight, sheds 503 before decode)
                     |
              preprocess by content type
                ├─ image/jpeg          libjpeg-turbo pool (or nvJPEG, or hybrid)
                ├─ x-raw-rgb           normalize only (150 KB uint8 body)
                └─ x-raw-tensor        memcpy into a pooled NCHW tensor (602 KB)
                     |
              gatherer: batch up to -max-batch, or flush after -max-wait,
              packing into pinned host staging
                     |
              N infer lanes on one shared ORT session (-infer-lanes)
              engine-owned device buffers bound once; each lane does its own
              H2D, Run, and 128 KB logits D2H
                     |
              unpack workers -> per-request top-1 response
```

Notable details:

- Batch 32 is the measured optimum on this model and card. Per-image Run cost
  rises with batch size (0.24 ms at 32, 1.08 ms at 128), so bigger batches lose.
- Several infer lanes share one session. Per-Run time inflates because CUDA work
  serializes, but overlapping pack and H2D with compute still nets about +31%.
- `-device-binding` binds cudaMalloc'd buffers once and keeps ORT's pageable
  host-to-device copy out of the serialized path.
- Request bodies, input tensors, and batch staging buffers are pooled: about
  42 KB of garbage per request instead of 1.9 MB.
- Backpressure is taken before decode, so overload sheds instead of queueing.

At the final operating point the GPU runs at about 91% utilization under
open-loop saturation with a p50 around 10 ms. What is left is in the compute
itself (TensorRT execution provider, INT8), not in the serving path.

## Improvements

Each step was driven by whatever the previous measurement pointed at.

1. **Persistent server and session (345 req/s).** Loading the model and creating
   the CUDA session once at startup instead of per request; a per-request session
   re-uploads the weights every time.
2. **Load-test harness.** Closed-loop and open-loop generator with latency
   percentiles and GPU sampling; every number below comes from it.
3. **Dynamic batching (338 req/s).** No throughput change, because realized batch
   size was 1.3: requests never queued up long enough to batch. It did show where
   the time actually went.
4. **GPU preprocessing with nvJPEG and NPP (1,463 req/s).** Moving JPEG decode and
   resize off the CPU let batches fill to 20-29 images and quadrupled throughput.
5. **Pipelined gather/infer/unpack plus backpressure (1,500 req/s).** Only +3% on
   throughput, but the in-flight gate cut overload p50 from 3.7 s to 60 ms.
6. **CPU decode via libjpeg-turbo (1,790 req/s).** SIMD decode on the CPU freed
   the GPU from decode work: GPU utilization dropped 63% to 35% and per-batch Run
   went from 24 ms to 8 ms.
7. **Allocation pooling (2,130 req/s).** pprof pointed at request bodies, input
   tensors, and batch staging; pooling all three cut garbage per request from
   1.9 MB to 42 KB and GC runs per sweep from ~1,000 to ~9.
8. **Raw contracts, infer lanes, FP16 (7,500 req/s).** Letting clients send
   pre-decoded bodies took decode off the server entirely, concurrent lanes
   overlapped copy with compute, and FP16 weights cut Run time further.
9. **Device-resident I/O binding (8,700 raw / 9,100 raw-RGB req/s).** About 90
   lines added to a fork of the Go binding expose CUDA memory info and a
   `DeviceTensor`, so the engine owns its device buffers and pinned staging;
   per-Run time went from 20.4 ms to 15.0 ms.
10. **Prometheus endpoint.** The counters that were already being collected for
    the tuning work, exported at `/metrics` with a `/predict` latency histogram.

## Challenges

- **Batching alone did nothing.** Throughput was flat until realized batch size
  was instrumented, which showed batches of 1.3 and pointed at CPU JPEG decode as
  the actual limit.
- **A wrong diagnosis cost a phase of work.** Pipelining was built on the theory
  that a single collector goroutine was the ceiling; it returned 3%, and the real
  wall turned out to be GPU decode and inference contending for SMs.
- **This card has no NVJPG hardware decoder.** nvJPEG silently runs on CUDA
  cores, so GPU decode steals SMs from inference and tripled per-batch Run time
  when it carried every decode. That is why CPU decode won.
- **nvJPEG scale-factor decode does not save time.** Entropy decode happens
  before scaling, so 1/4-scale decode costs about the same as full size, and
  cheaper GPU decode was not available as a lever.
- **Hybrid CPU-primary/GPU-overflow decode was rejected.** It was built and
  measured; spilling to the GPU cost more inference time than the extra decode
  capacity was worth.
- **CUDA Graphs is incompatible with dynamic batching.** Confirmed three
  independent ways: any Run with a new batch shape invalidates the captured
  graph, so capture cannot survive variable batch sizes.
- **Two undocumented ORT IoBinding behaviors.** A CUDA session never writes a
  pre-bound CPU output (read it with `GetBoundOutputValues` instead), and a bound
  CPU input's device copy is cached across runs, so it must be re-bound per
  batch. Both surfaced as 992 of 1000 wrong answers in a routing stress test.
- **ORT 1.22 and later carry a ~20% fp32 regression.** Per-Run fp32 time is ~20%
  worse than 1.18.1 on this model, identical on 1.22, 1.26 and 1.27, and not
  fixable through provider options. Accepted, since the remaining wins are on
  FP16.
- **Bigger batches are slower per image.** Per-image Run cost rises from 0.24 ms
  at batch 32 to 1.08 ms at batch 128, so the batch-size knob has an optimum well
  below the maximum.
- **The fastest contract was the smaller one.** Once ORT's host-to-device copy
  left the hot path, the 150 KB uint8 RGB body beat the 602 KB float32 tensor
  body: moving bytes over HTTP costs more than normalizing them server-side.
- **The FP16 conversion needed model surgery.** The GluonCV export is opset 8
  with every weight listed as a graph input, so it needed an opset upgrade and
  initializer-input removal before `keep_io_types` conversion would load.
- **Short load runs lie.** Runs under ~48k requests per level land inside the
  GPU's boost-clock ramp and report low; all comparisons use longer runs.
