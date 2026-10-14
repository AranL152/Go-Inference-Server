# Export YOLOX-S to ONNX with a dynamic batch dimension and box decoding in the graph
#
# Requires the YOLOX source on PYTHONPATH plus torch onnx and the yolox_s.pth checkpoint
# git clone https://github.com/Megvii-BaseDetection/YOLOX
# curl -LO https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
# PYTHONPATH=YOLOX python export_yolox.py yolox_s.pth yolox_s.onnx
#
# Input  images  N x 3 x 640 x 640 float32 BGR 0 to 255 letterboxed with pad value 114
# Output output  N x 8400 x 85 rows of cx cy w h objectness then 80 class scores
import sys

import torch
from yolox.exp import get_exp

ckpt, out = sys.argv[1], sys.argv[2]
exp = get_exp(None, "yolox-s")
model = exp.get_model()
model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
model.eval()
model.head.decode_in_inference = True

torch.onnx.export(
    model, torch.zeros(1, 3, 640, 640), out,
    input_names=["images"], output_names=["output"],
    dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17, dynamo=False)

# pin the non batch output dims so the server can size buffers from the model alone
import onnx
m = onnx.load(out)
dims = m.graph.output[0].type.tensor_type.shape.dim
dims[1].dim_value, dims[2].dim_value = 8400, 85
onnx.save(m, out)
print("wrote", out)
