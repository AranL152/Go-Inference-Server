#!/usr/bin/env bash
# Build and run the Phase 1 GPU inference HTTP server.
# (Same PATH + LD_LIBRARY_PATH setup as Phase 0 — do not change the lib paths.)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Go toolchain (installed to ~/sdk/go in Phase 0 setup).
export PATH="$HOME/sdk/go/bin:$HOME/go/bin:$PATH"

# Runtime library search path. Order matters:
#   onnxruntime/lib  -> libonnxruntime.so + libonnxruntime_providers_cuda.so
#   cudnn9           -> libcudnn.so.9 (the CUDA-12 ORT build links cuDNN 9, not 8)
#   cuda-12.6/lib64  -> libcublas / libcufft / libcurand / libcudart
#   /usr/lib/wsl/lib -> libcuda.so (provided by the Windows NVIDIA driver via WSL)
export LD_LIBRARY_PATH="$HERE/onnxruntime/lib:$HERE/cudnn9:/usr/local/cuda-12.6/lib64:/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

cd "$HERE"
CGO_ENABLED=1 go build -o server .
exec ./server "$@"
