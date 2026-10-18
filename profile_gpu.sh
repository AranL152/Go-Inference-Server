#!/usr/bin/env bash
# Trace the GPU side of the real server under steady load with Nsight Systems
# writes profiles/MODEL.nsys-rep for the Nsight Systems UI and profiles/MODEL.nsys.json for the dashboard
# needs an Nsight Systems that supports the installed driver, e.g. sudo apt install nsight-systems-2026.1.3
# RATE=400 ./profile_gpu.sh yolox_s_u8.onnx -max-batch 16 -infer-lanes 3 -device-binding -trt trt_cache
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
MODEL=$1
shift
PORT=8097
RATE=${RATE:-400}
WARM=4
CAPTURE=${CAPTURE:-3}
OUT=profiles/${MODEL%.onnx}
SESSION=gpuprof$$
mkdir -p profiles

NSYS=$(ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | sort -V | tail -1)
NSYS=${NSYS:-/usr/local/cuda-12.6/bin/nsys}
echo "using $NSYS"

export PATH="$HOME/sdk/go/bin:$PATH"
export CGO_CFLAGS="${CGO_CFLAGS:--O2 -g} -Wno-deprecated-declarations"
export LD_LIBRARY_PATH="$HERE/onnxruntime/lib:$HOME/tensorrt10/tensorrt_libs:$HERE/cudnn9:/usr/local/cuda-12.6/lib64:/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
CGO_ENABLED=1 go build -o server .
(cd loadtest && go build -o loadtest .)

"$NSYS" launch --trace=cuda --session="$SESSION" ./server -addr :$PORT -quiet -jobs-dir "" -model "$MODEL" "$@" &
trap '"$NSYS" shutdown --kill sigterm --session="$SESSION" >/dev/null 2>&1 || true' EXIT
until curl -sf localhost:$PORT/health >/dev/null; do sleep 1; done

# detectors get JPEGs and classifiers raw rgb bodies, at a fixed offered rate
BODY=()
if ! curl -s localhost:$PORT/api/config | grep -q '"detector":true'; then BODY=(-raw-rgb); fi
loadtest/loadtest -addr http://localhost:$PORT -images testdata -imagefiles dog.jpg,cat.jpg,street.jpg \
	-rates "$RATE" -rate-dur $((WARM + CAPTURE + 2))s -warmup 0 -out "$OUT.load" "${BODY[@]}" >/dev/null &
LOAD=$!
sleep $WARM
"$NSYS" start --session="$SESSION" -o "$OUT" -f true
sleep "$CAPTURE"
"$NSYS" stop --session="$SESSION"
wait $LOAD || true
rm -f "$OUT.load".*

"$NSYS" export --type sqlite -f true -o "$OUT.sqlite" "$OUT.nsys-rep" >/dev/null
python3 summarize_nsys.py "$OUT.sqlite" "$MODEL" > "$OUT.nsys.json"
rm "$OUT.sqlite"
echo "wrote $OUT.nsys.json and $OUT.nsys-rep"
