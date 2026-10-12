#!/usr/bin/env bash
# Build and run the GPU inference HTTP server.
# The LD_LIBRARY_PATH order below is load-bearing; see the note under it.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Go toolchain (installed to ~/sdk/go, no sudo needed).
export PATH="$HOME/sdk/go/bin:$HOME/go/bin:$PATH"

# Runtime library search path. Order matters:
#   onnxruntime/lib  -> libonnxruntime.so + libonnxruntime_providers_cuda.so
#   tensorrt10       -> libnvinfer.so.10 for -trt (pip tensorrt-cu12-libs==10.16.1.11
#                       --target ~/tensorrt10; ORT 1.27 needs TRT 10, not ~/TensorRT-8.6)
#   cudnn9           -> libcudnn.so.9 (the CUDA-12 ORT build links cuDNN 9, not 8)
#   cuda-12.6/lib64  -> libcublas / libcufft / libcurand / libcudart
#   /usr/lib/wsl/lib -> libcuda.so (provided by the Windows NVIDIA driver via WSL)
export LD_LIBRARY_PATH="$HERE/onnxruntime/lib:$HOME/tensorrt10/tensorrt_libs:$HERE/cudnn9:/usr/local/cuda-12.6/lib64:/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

cd "$HERE"
CGO_ENABLED=1 go build -o server .
exec ./server "$@"
