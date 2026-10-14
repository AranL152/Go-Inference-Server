#!/usr/bin/env bash
# Profile one steady state batch of the real server with Nsight Compute
# writes profiles/MODEL.ncu-rep for the Nsight Compute UI and profiles/MODEL.json for the dashboard
# needs GPU performance counter access for all users in the NVIDIA Control Panel on Windows
# ./profile_kernels.sh yolox_s_u8.onnx -trt trt_cache
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
MODEL=$1
shift
PORT=8099
BATCH=16
OUT=profiles/${MODEL%.onnx}
mkdir -p profiles
export PATH="$HOME/sdk/go/bin:$PATH"
export CGO_CFLAGS="${CGO_CFLAGS:--O2 -g} -Wno-deprecated-declarations"
export LD_LIBRARY_PATH="$HERE/onnxruntime/lib:$HOME/tensorrt10/tensorrt_libs:$HERE/cudnn9:/usr/local/cuda-12.6/lib64:/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
CGO_ENABLED=1 go build -o server .

/usr/local/cuda-12.6/bin/ncu --profile-from-start off --target-processes all \
	--section SpeedOfLight --section Occupancy --section LaunchStats \
	--metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active \
	-f -o "$OUT" ./server -addr :$PORT -quiet -jobs-dir "" -model "$MODEL" -max-batch $BATCH \
	-max-wait 50ms -infer-lanes 1 -device-binding "$@" &
NCU=$!
until curl -sf localhost:$PORT/health >/dev/null; do sleep 1; done

# detectors get a JPEG and classifiers a raw rgb body
if curl -s localhost:$PORT/api/config | grep -q '"detector":true'; then
	send() { curl -s -o /dev/null -X POST --data-binary @testdata/street.jpg localhost:$PORT/predict; }
else
	head -c 150528 /dev/zero > profiles/.rgb
	send() { curl -s -o /dev/null -X POST -H 'Content-Type: application/x-raw-rgb' --data-binary @profiles/.rgb localhost:$PORT/predict; }
fi
batch() {
	local pids=()
	for i in $(seq $BATCH); do send & pids+=($!); done
	wait "${pids[@]}"
}

batch; batch
curl -s -X POST localhost:$PORT/debug/cuda-profiler/start >/dev/null
batch
curl -s -X POST localhost:$PORT/debug/cuda-profiler/stop >/dev/null
kill -INT $(pgrep -P $NCU -f "server -addr :$PORT" || pgrep -f "server -addr :$PORT")
wait $NCU || true
rm -f profiles/.rgb

/usr/local/cuda-12.6/bin/ncu --import "$OUT.ncu-rep" --csv --page raw > "$OUT.csv"
python3 summarize_profile.py "$OUT.csv" "$MODEL" > "$OUT.json"
rm "$OUT.csv"
echo "wrote $OUT.json and $OUT.ncu-rep"
