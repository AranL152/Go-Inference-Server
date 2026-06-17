# Phase 1 — Single-Request GPU Inference Server

**Verdict: 🟢 GREEN.** The Phase 0 one-shot program is now a long-running HTTP
server. It loads ResNet-18 onto the RTX 4080 once at startup (ONNX Runtime CUDA
EP, reused for every request) and serves `POST /predict` and `GET /health`.
Predictions are correct on recognizable images, and inference is confirmed on the
GPU.

## How to run

```bash
./run.sh            # builds ./server and runs it on :8080
./run.sh -addr :9000   # flags pass through (also -model, -labels, -lib)
```

Startup log (model loaded on GPU):

```
loaded 1000 class labels from imagenet_classes.txt
loading model resnet18.onnx onto GPU via CUDA execution provider...
=== model loaded on GPU (allocated +74 MiB; compute-app PID listed: true) ===
listening on :8080  (POST /predict, GET /health)
```

## How to test

With the server running, in another terminal:

```bash
./test.sh                  # /health + dog + cat + a malformed-body case
# or by hand:
curl -X POST http://localhost:8080/predict --data-binary @testdata/dog.jpg
curl http://localhost:8080/health
```

`POST /predict` accepts either a raw image body (`--data-binary @file`) or a
multipart upload (`-F image=@file`). JPEG and PNG are supported.

## Test results (this run)

| Image            | What it is                | Top-1 prediction        | Confidence |
|------------------|---------------------------|-------------------------|-----------:|
| `testdata/dog.jpg` | white fluffy dog (the canonical PyTorch sample image, a Samoyed) | **Samoyed** (258) | 0.909 |
| `testdata/cat.jpg` | a cat                     | **Egyptian cat** (285)  | 0.684 |

These are genuinely correct, not just "returned a number":
- The dog image really is a Samoyed → top-1 Samoyed at 91%.
- The cat → "Egyptian cat", one of ImageNet's three adjacent, routinely-confused
  cat classes (Egyptian cat / tabby / tiger cat). Correctly a cat.

Error handling verified:
- `GET /predict` → `405` `{"error":"use POST with an image body"}`
- malformed body → `400` `{"error":"could not decode image (expected JPEG or PNG): ..."}`
- `GET /health` → `200` `{"device":"cuda","model_loaded":true,"status":"ok"}`

## GPU confirmation (same method as Phase 0)

- Session creation allocated GPU memory (`+74 MiB` delta) — above the 50 MiB
  CPU-fallback threshold.
- Our server PID appears in `nvidia-smi --query-compute-apps`:
  ```
  13077, [Not Found], [N/A]      # ./server — process_name/used_memory are the WSL2 [N/A] quirk
  ```
  PID-in-compute-apps is the definitive signal; under WSL2 the per-process memory
  always reads `[N/A]`, exactly as in Phase 0.

## What's in the prediction response

```json
{"class_index": 258, "class_name": "Samoyed", "confidence": 0.909}
```

`confidence` is the softmax probability of the top-1 class. `class_name` is
omitted if no labels file is loaded (numeric `class_index` is the documented
fallback). Labels: `imagenet_classes.txt` (1000 lines, standard ILSVRC2012
ordering — index 207 = "golden retriever", etc.).

## Preprocessing (what resnet18-v1-7 expects)

Model I/O confirmed from the file: input `data` `[N,3,224,224]`, output
`resnetv15_dense0_fwd` `[N,1000]` (GluonCV `resnetv15`). The ONNX-model-zoo recipe:

1. resize shorter side to 256 (bilinear, half-pixel/align_corners=False — matches PIL)
2. center-crop 224×224
3. scale to `[0,1]`, normalize per channel: mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`
4. NCHW layout

Phase 0's flat-0.5 dummy input is replaced by this real pipeline
(`preprocess.go`, dependency-free so the Phase 0 cgo build is untouched).

## Code layout

- `main.go`        — flags, startup, HTTP routing (`/predict`, `/health`), error handling, graceful shutdown.
- `engine.go`      — persistent CUDA session created once; in-place input reuse; softmax/top-1; mutex serializes inference (Phase 1 = one request at a time).
- `preprocess.go`  — JPEG/PNG decode, resize-256, center-crop-224, ImageNet normalize, NCHW.
- `gpu.go`         — `nvidia-smi` helpers carried over from Phase 0.
- `run.sh`         — same PATH/LD_LIBRARY_PATH as Phase 0; builds `./server`.
- `test.sh`        — end-to-end smoke test.
- `testdata/`      — `dog.jpg`, `cat.jpg` sample images with known answers.
- `phase0_main.go.bak` — the original Phase 0 one-shot, preserved (not compiled).

## Deliberately NOT in Phase 1

No concurrency beyond Go's default (inference is serialized on purpose — Phase 2
will load-test to expose that), no batching, no backpressure/metrics, still plain
HTTP (no gRPC).
