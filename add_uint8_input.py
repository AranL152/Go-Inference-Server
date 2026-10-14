# Give a model a uint8 input so the server ships 4x fewer bytes and the GPU does the float math
#
# python add_uint8_input.py yolox IN.onnx OUT.onnx
#   uint8 N x 3 x H x W BGR in then Cast to float which YOLOX expects as is
# python add_uint8_input.py resnet IN.onnx OUT.onnx
#   uint8 N x 224 x 224 x 3 RGB in which is the raw rgb request body byte for byte
#   then Cast and Transpose to NCHW and ImageNet normalize
# Cast comes first because TensorRT rejects uint8 intermediate tensors
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

kind, src, dst = sys.argv[1:4]
m = onnx.load(src)
old = m.graph.input[0]
dims = [d.dim_param or d.dim_value for d in old.type.tensor_type.shape.dim]
name = old.name
inner = name + "_f32"
for n in m.graph.node:
    n.input[:] = [inner if i == name else i for i in n.input]

if kind == "yolox":
    shape = dims
    nodes = [helper.make_node("Cast", [name], [inner], to=TensorProto.FLOAT)]
    inits = []
elif kind == "resnet":
    shape = [dims[0], dims[2], dims[3], dims[1]]
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    scale = (1 / (255 * std)).reshape(1, 3, 1, 1)
    bias = (-mean / std).reshape(1, 3, 1, 1)
    inits = [numpy_helper.from_array(scale, "u8_scale"), numpy_helper.from_array(bias, "u8_bias")]
    nodes = [
        helper.make_node("Cast", [name], ["u8_f32"], to=TensorProto.FLOAT),
        helper.make_node("Transpose", ["u8_f32"], ["u8_nchw"], perm=[0, 3, 1, 2]),
        helper.make_node("Mul", ["u8_nchw", "u8_scale"], ["u8_scaled"]),
        helper.make_node("Add", ["u8_scaled", "u8_bias"], [inner]),
    ]
else:
    sys.exit("kind must be yolox or resnet")

new_in = helper.make_tensor_value_info(name, TensorProto.UINT8, shape)
m.graph.input.remove(old)
m.graph.input.insert(0, new_in)
for n in reversed(nodes):
    m.graph.node.insert(0, n)
m.graph.initializer.extend(inits)
onnx.checker.check_model(m)
onnx.save(m, dst)
print("wrote", dst, "input", name, shape, "uint8")
