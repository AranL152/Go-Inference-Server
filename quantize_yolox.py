# Quantize yolox_s.onnx to INT8 QDQ for TensorRT with the same recipe as quantize_int8.py
#
# Conv only QDQ with per channel symmetric weights and float biases
# activations calibrated on letterboxed images in the server input format
# python quantize_yolox.py CALIB_DIR [out.onnx] [minmax or percentile]
# percentile keeps whole activation tensors and needs far more than 16 GB of RAM at 640 pixels
import glob
import os
import sys

import cv2
import numpy as np
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process


def letterbox(path, size=640):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    out = np.full((size, size, 3), 114, np.uint8)
    out[:int(h * r), :int(w * r)] = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
    return out.transpose(2, 0, 1)[None].astype(np.float32)


class Reader(CalibrationDataReader):
    def __init__(self, paths):
        self.it = iter(paths)

    def get_next(self):
        p = next(self.it, None)
        return None if p is None else {"images": letterbox(p)}


calib_dir = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "yolox_s_int8.onnx"
method = sys.argv[3] if len(sys.argv) > 3 else "minmax"
paths = sorted(glob.glob(os.path.join(calib_dir, "**", "*.jpg"), recursive=True))[:512]
quant_pre_process("yolox_s.onnx", "yolox_prep.tmp.onnx", skip_symbolic_shape=True)
quantize_static(
    "yolox_prep.tmp.onnx", out, Reader(paths),
    quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
    per_channel=True, calibrate_method={"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile}[method], op_types_to_quantize=["Conv"],
    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True, "QuantizeBias": False,
                   "CalibPercentile": 99.999})
os.remove("yolox_prep.tmp.onnx")
print(f"wrote {out} from {len(paths)} calibration images with {method}")
