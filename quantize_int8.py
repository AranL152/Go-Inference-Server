# Quantize resnet18.onnx (fp32) to INT8 with static QDQ quantization:
# per-channel symmetric int8 weights, int8 activations calibrated on real
# images. float32 I/O is unchanged, so the Go server needs zero changes.
#
# Requires: pip install onnx onnxruntime-gpu numpy pillow   (venv is fine)
# Usage:    python quantize_int8.py CALIB_DIR [method] [out.onnx]
#   CALIB_DIR: directory tree of JPEGs (e.g. an ImageNet train subset); 512
#              are sampled. method: minmax | percentile | entropy
#   QUANT_OPS (env): op types that get QDQ, default Conv. Under TensorRT,
#   Conv-only keeps top-1 within 0.1 pt of fp16 and is the fastest; adding
#   Add (int8 residual sums) costs ~1.5 pt; quantizing all ops also puts QDQ
#   around MaxPool and TensorRT finds no int8 kernel for the fused
#   conv0+relu+pool layer, failing the engine build.
#   QUANT_EXCLUDE (env): comma-separated node names to leave in float.
#
# Biases stay float (QuantizeBias=False): TensorRT rejects an int32
# DequantizeLinear on a bias ("only activation datatypes allowed"), which
# splits the graph off TRT; it fuses the float bias into the int8 layer itself.
#
# Same opset/initializer fixups as convert_fp16.py (old GluonCV export):
# per-channel QDQ needs opset >= 13 (DequantizeLinear axis attribute), and the
# quantizer only treats weights as constants once they are no longer inputs.
import glob
import os
import sys

import numpy as np
import onnx
from onnx import version_converter
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def preprocess(path):
    """Shorter side -> 256, center-crop 224, NCHW float32, ImageNet-normalized
    (the server's preprocessing contract)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BILINEAR)
    w, h = img.size
    l, t = (w - 224) // 2, (h - 224) // 2
    a = np.asarray(img.crop((l, t, l + 224, t + 224)), np.float32) / 255.0
    return ((a.transpose(2, 0, 1) - MEAN) / STD)[None]


class Reader(CalibrationDataReader):
    def __init__(self, paths, name):
        self.it = iter(paths)
        self.name = name

    def get_next(self):
        p = next(self.it, None)
        return None if p is None else {self.name: preprocess(p)}


def main():
    calib_dir = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else "percentile"
    out = sys.argv[3] if len(sys.argv) > 3 else "resnet18_int8.onnx"

    m = onnx.load("resnet18.onnx")
    m = version_converter.convert_version(m, 13)
    m.ir_version = 8
    init_names = {i.name for i in m.graph.initializer}
    real_inputs = [i for i in m.graph.input if i.name not in init_names]
    del m.graph.input[:]
    m.graph.input.extend(real_inputs)
    del m.graph.value_info[:]
    onnx.save(m, "resnet18_fp32_op13.tmp.onnx")
    quant_pre_process("resnet18_fp32_op13.tmp.onnx", "resnet18_prep.tmp.onnx")

    paths = sorted(glob.glob(os.path.join(calib_dir, "**", "*.JPEG"), recursive=True))
    rng = np.random.default_rng(0)
    paths = list(rng.choice(paths, size=min(512, len(paths)), replace=False))

    methods = {"minmax": CalibrationMethod.MinMax,
               "percentile": CalibrationMethod.Percentile,
               "entropy": CalibrationMethod.Entropy}
    quantize_static(
        "resnet18_prep.tmp.onnx", out, Reader(paths, real_inputs[0].name),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        per_channel=True, calibrate_method=methods[method],
        op_types_to_quantize=os.environ.get("QUANT_OPS", "Conv").split(","),
        nodes_to_exclude=[n for n in os.environ.get("QUANT_EXCLUDE", "").split(",") if n],
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True,
                       "QuantizeBias": False,
                       "CalibPercentile": 99.999})
    for f in ("resnet18_fp32_op13.tmp.onnx", "resnet18_prep.tmp.onnx"):
        os.remove(f)
    print(f"wrote {out} ({method}, {len(paths)} calibration images)")


if __name__ == "__main__":
    main()
