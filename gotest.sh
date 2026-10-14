#!/usr/bin/env bash
# Correctness tests: pure routing unit tests plus GPU integration tests
# (mixed-batch no-swap, known-answer-under-load). Same lib paths as run.sh.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/sdk/go/bin:$HOME/go/bin:$PATH"
# go-nvml wraps every NVML call including deprecated ones
export CGO_CFLAGS="${CGO_CFLAGS:--O2 -g} -Wno-deprecated-declarations"
export LD_LIBRARY_PATH="$HERE/onnxruntime/lib:$HERE/cudnn9:/usr/local/cuda-12.6/lib64:/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
cd "$HERE"
CGO_ENABLED=1 go test -v "$@"
