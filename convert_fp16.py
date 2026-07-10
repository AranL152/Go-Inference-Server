# Convert resnet18.onnx (fp32) to resnet18_fp16.onnx: fp16 weights/compute,
# float32 I/O preserved (keep_io_types) so the Go server needs zero changes.
#
# Requires: pip install onnx onnxconverter-common   (venv is fine)
#
# The three fixups before conversion are all needed for this specific model —
# an old GluonCV export (opset 8, IR version 3) that lists every weight
# initializer as a graph input:
#   1. opset 8 -> 13: the converter emits Cast/graph patterns the old opset
#      can't legally hold ("Tensor element type mismatch. 10 != 1" at load).
#   2. drop initializer-inputs: keep_io_types pins ALL graph inputs to fp32,
#      but the initializers themselves get converted to fp16 -> same mismatch.
#      IR version must be >= 4 for initializers to be allowed out of inputs.
#   3. clear value_info: stale fp32 type annotations on internal edges.
import onnx
from onnx import version_converter
from onnxconverter_common import float16

m = onnx.load("resnet18.onnx")
m = version_converter.convert_version(m, 13)
m.ir_version = 8

init_names = {i.name for i in m.graph.initializer}
real_inputs = [i for i in m.graph.input if i.name not in init_names]
del m.graph.input[:]
m.graph.input.extend(real_inputs)
del m.graph.value_info[:]

m16 = float16.convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=True)
m16.ir_version = 8
onnx.checker.check_model(m16)
onnx.save(m16, "resnet18_fp16.onnx")
print("wrote resnet18_fp16.onnx; inputs:",
      [(i.name, i.type.tensor_type.elem_type) for i in m16.graph.input])
