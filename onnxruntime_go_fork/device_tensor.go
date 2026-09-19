// Fork addition for go-inference-server: float32 tensors backed by
// externally managed CUDA device memory. The stock library only creates
// CPU-backed values, which forces ORT to stage every IoBinding input through
// a cached device copy and to allocate outputs itself. A DeviceTensor lets the
// caller own the device buffers instead: allocating them (with cudaMalloc, for
// example), copying data in and out, and freeing them after the tensor is
// destroyed.
package onnxruntime_go

// #include "onnxruntime_wrapper.h"
import "C"
import (
	"fmt"
	"unsafe"
)

// cudaMemoryInfos caches one OrtMemoryInfo per CUDA device. Entries live for
// the process lifetime (ORT copies the info into each created value, but
// keeping the cache avoids per-tensor create/release churn). Not synchronized:
// create device tensors from one goroutine at a time (setup code).
var cudaMemoryInfos = map[int]*C.OrtMemoryInfo{}

// DeviceTensor is a float32 tensor whose data lives in CUDA device memory
// owned by the CALLER. It satisfies the Value interface so it can be bound
// via IoBinding.BindInput/BindOutput. Because the memory is already on the
// session's device, ORT uses the buffer in place: no cached-copy staleness
// (the stock CPU-value plus IoBinding path reuses the first upload forever)
// and no ORT-side output allocation.
type DeviceTensor struct {
	shape    Shape
	ortValue *C.OrtValue
}

// NewCudaFloat32Tensor wraps sizeBytes of CUDA device memory at devicePtr
// (e.g. from cudaMalloc, on device deviceID) as a float32 ONNX tensor of the
// given shape. sizeBytes must equal shape.FlattenedSize()*4. The device
// memory is NOT owned by the tensor: Destroy releases only the OrtValue;
// free the device buffer afterwards, never before.
func NewCudaFloat32Tensor(s Shape, devicePtr unsafe.Pointer, sizeBytes uintptr,
	deviceID int) (*DeviceTensor, error) {
	if !IsInitialized() {
		return nil, NotInitializedError
	}
	if e := s.Validate(); e != nil {
		return nil, fmt.Errorf("Invalid tensor shape: %w", e)
	}
	if expected := uintptr(s.FlattenedSize()) * 4; sizeBytes != expected {
		return nil, fmt.Errorf("shape %s needs %d bytes, got %d",
			s, expected, sizeBytes)
	}
	memInfo, ok := cudaMemoryInfos[deviceID]
	if !ok {
		status := C.CreateCudaMemoryInfoWrapper(C.int(deviceID), &memInfo)
		if status != nil {
			return nil, fmt.Errorf("creating CUDA memory info: %w",
				statusToError(status))
		}
		cudaMemoryInfos[deviceID] = memInfo
	}
	var ortValue *C.OrtValue
	status := C.CreateOrtTensorWithShape(devicePtr, C.size_t(sizeBytes),
		(*C.int64_t)(unsafe.Pointer(&s[0])), C.int64_t(len(s)), memInfo,
		C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ortValue)
	if status != nil {
		return nil, fmt.Errorf("ORT API error creating device tensor: %s",
			statusToError(status))
	}
	return &DeviceTensor{shape: s.Clone(), ortValue: ortValue}, nil
}

func (t *DeviceTensor) Destroy() error {
	C.ReleaseOrtValue(t.ortValue)
	t.ortValue = nil
	t.shape = nil
	return nil
}

func (t *DeviceTensor) DataType() C.ONNXTensorElementDataType {
	return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
}

func (t *DeviceTensor) GetShape() Shape { return t.shape.Clone() }

func (t *DeviceTensor) GetONNXType() ONNXType { return ONNXTypeTensor }

func (t *DeviceTensor) GetInternals() *ValueInternalData {
	return &ValueInternalData{ortValue: t.ortValue}
}

// ZeroContents is a no-op: the library cannot write CUDA memory. Zero the
// buffer with cudaMemset from the owning side if needed.
func (t *DeviceTensor) ZeroContents() {}
