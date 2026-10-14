#!/usr/bin/env bash
# YOLOX object detection demo with the web UI at http://localhost:8080
# first start builds the TensorRT engine into trt_cache which takes a minute or two
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/run.sh" -model yolox_s_u8.onnx -quiet -max-batch 16 -infer-lanes 3 \
	-device-binding -trt "$HERE/trt_cache" "$@"
