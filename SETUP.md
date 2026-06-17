# Phase 0 — GPU Inference Path Validation

**Verdict: 🟢 GREEN.** A Go program loads an ONNX model and runs inference on the
RTX 4080 via ONNX Runtime's CUDA execution provider. Proceed to Phase 1.

## How to run

```bash
./run.sh
```

Expected tail of output:

```
GPU memory used   : 0 MiB -> 406 MiB (delta +406 MiB)
nvidia-smi compute apps:
<pid>, [N/A]
=== GREEN: inference executed on the CUDA execution provider (GPU). ===
```

## Environment found

| Component   | Version                              | Notes                               |
|-------------|--------------------------------------|-------------------------------------|
| GPU         | RTX 4080 (16 GB)                     | driver 595.97, CUDA driver br. 13.2 |
| CUDA Toolkit| 12.6 (nvcc)                          | libs in `/usr/local/cuda-12.6`      |
| cuDNN       | 8.9.7 system **+ 9.10.2** (pip)      | **9.x is the one we needed**        |
| Go          | 1.26.4                               | installed to `~/sdk/go` (no sudo)   |
| ONNX Runtime| 1.18.1 (cuda12 build)                | `onnxruntime/lib/`                  |
| Go binding  | `github.com/yalue/onnxruntime_go` v1.11.0 | pinned — see below            |

## The two version-compatibility traps (the riskiest part, as predicted)

1. **Go binding ↔ ORT API version.** The binding's `@latest` (v1.31.0) targets
   ORT API 26 and fails against ORT 1.18.1 ("API version 26 not available,
   only [1,18] supported"). **Fix:** pin the binding to **v1.11.0**, the newest
   release that targets ORT API 18. (See `go.mod`.)

2. **ORT cuda12 build links cuDNN 9, not cuDNN 8.** The system cuDNN is 8.9.7,
   but `libonnxruntime_providers_cuda.so` needs `libcudnn.so.9`. **Fix:** cuDNN
   9.10.2 already existed in a pip package; its libs are copied into `cudnn9/`
   and put on `LD_LIBRARY_PATH`.

If you ever upgrade ORT: 1.19+ also needs cuDNN 9 (fine), but you must move the
Go binding to a matching tag (binding API version must be ≤ the ORT version).

## WSL2 notes

- GPU works in WSL2 — **no need to switch to native Windows/Ubuntu.**
- `libcuda.so` comes from the Windows driver via `/usr/lib/wsl/lib`.
- WSL2's `nvidia-smi` reports per-process memory as `[N/A]`, so the program's
  primary GPU proof is the **total GPU-memory delta** (0 → ~406 MiB), with the
  PID-in-compute-apps list as a secondary check.
- Build on the Linux filesystem (`~/go-inference-server`), **not** the
  `/mnt/c/...` Windows mount — that path has a space and slow I/O that break
  cgo/Go builds.

## Files

- `main.go`   — the ~30-line validation (commented).
- `run.sh`    — sets `PATH` + `LD_LIBRARY_PATH`, builds, runs.
- `resnet18.onnx` — ResNet-18 v1-7 (ONNX model zoo).
- `onnxruntime/`  — extracted ORT 1.18.1 cuda12 native libraries.
- `cudnn9/`        — cuDNN 9 libraries the CUDA EP loads at runtime.
