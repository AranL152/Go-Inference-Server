# Environment setup

The server needs an NVIDIA GPU, CUDA 12.6, cuDNN 9, an ONNX Runtime GPU build,
and libjpeg-turbo. Two directories the build depends on are not committed
because they are large and machine-specific: `onnxruntime/` and `cudnn9/`.

## Versions this was built against

| Component | Version | Notes |
|---|---|---|
| GPU | RTX 4080 SUPER (16 GB) | WSL2, driver from the Windows host |
| CUDA toolkit | 12.6 | `/usr/local/cuda-12.6` |
| cuDNN | 9.10.2 | copied into `cudnn9/` |
| ONNX Runtime | 1.27.1, cuda12 build | extracted into `onnxruntime/` |
| Go binding | `github.com/yalue/onnxruntime_go` v1.31.0 | forked locally, see below |
| Go | 1.26.4 | installed to `~/sdk/go` |
| libjpeg-turbo | `libturbojpeg0-dev` | apt package |

## Getting the native libraries

1. Download `onnxruntime-linux-x64-gpu_cuda12-1.27.1.tgz` from the
   microsoft/onnxruntime GitHub releases and extract it to `onnxruntime/`, so
   that `onnxruntime/lib/libonnxruntime.so` exists.
2. Put cuDNN 9's shared libraries in `cudnn9/`. The pip package
   (`nvidia-cudnn-cu12`) is one source; copying the `.so` files out of its
   `lib/` directory is enough.
3. `sudo apt install libturbojpeg0-dev`.

`run.sh` sets `LD_LIBRARY_PATH` to pick these up in the right order:
`onnxruntime/lib`, then `cudnn9`, then `/usr/local/cuda-12.6/lib64`, then
`/usr/lib/wsl/lib`. Order matters: the CUDA-12 ORT build links cuDNN 9, and a
system cuDNN 8 earlier on the path will be found first and fail to load the CUDA
provider.

## Version compatibility

The Go binding declares an ORT API version and fails against any runtime older
than that ("API version 26 not available, only [1,18] supported"). The binding
tag and the ORT build have to be matched: binding v1.11.0 targets ORT API 18
(ORT 1.18.x), v1.31.0 targets API 26 (ORT 1.22+). If you change one, change the
other.

`go.mod` replaces the binding with `onnxruntime_go_fork/`, a local copy that
adds a `DeviceTensor` type and CUDA memory-info accessors (about 90 lines) so
the engine can bind its own device buffers. Everything else in that directory is
upstream.

Known cost of the newer runtime: fp32 per-Run time is about 20% worse on ORT
1.22 and later than on 1.18.1 for this model, measured identical on 1.22, 1.26
and 1.27. FP16 is unaffected. Pinning ORT 1.18.1 with binding v1.11.0 restores
the old fp32 speed but loses IoBinding.

## WSL2 notes

- `libcuda.so` comes from the Windows driver through `/usr/lib/wsl/lib`.
- `nvidia-smi` reports per-process memory as `[N/A]`, so the server proves GPU
  placement with the total GPU-memory delta across session creation and treats
  the compute-apps PID list as a secondary signal.
- Build on the Linux filesystem, not under `/mnt/c`. The Windows mount is slow
  and its paths break cgo builds.

## FP16 model

`resnet18_fp16.onnx` is generated from `resnet18.onnx`:

```bash
pip install onnx onnxconverter-common
python3 convert_fp16.py
```

The script upgrades the opset and drops initializer-inputs before conversion;
the comments in it explain why this specific export needs both.

## Verifying the setup

```bash
./run.sh                 # should log "model loaded on GPU (allocated +N MiB)"
./smoke.sh               # health check plus two known-answer predictions
./gotest.sh              # unit tests and GPU integration tests
```

If the server logs that it allocated only a few MiB of GPU memory, ORT fell back
to CPU: the CUDA provider failed to load, almost always a cuDNN or CUDA library
path problem.
