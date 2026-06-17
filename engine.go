// Persistent GPU inference engine.
//
// Wraps a single ONNX Runtime session (CUDA execution provider) that is created
// ONCE at server startup and reused for every request. Creating a session per
// request would re-upload the model weights to the GPU each time — wasteful.
//
// Concurrency note (Phase 1): inference is deliberately serialized with a mutex.
// The session holds one fixed input tensor and one fixed output tensor; we write
// the next image into the input tensor's backing slice in place, then Run(). That
// is inherently single-request-at-a-time. Real concurrency/batching is Phase 2+.
package main

import (
	"fmt"
	"math"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// ResNet-18 (resnet18-v1-7, GluonCV resnetv15) tensor geometry.
const (
	channels  = 3
	imageSize = 224
	numClass  = 1000
)

// Prediction is the result of one inference.
type Prediction struct {
	ClassIndex int
	ClassName  string
	Confidence float64
}

// Engine owns the persistent ORT session and everything it needs to run.
type Engine struct {
	mu      sync.Mutex // serializes Run() + the in-place tensor write it depends on
	session *ort.AdvancedSession
	input   *ort.Tensor[float32]
	output  *ort.Tensor[float32]
	inBuf   []float32 // == input.GetData(); written in place before each Run
	outBuf  []float32 // == output.GetData(); read after each Run

	options  *ort.SessionOptions
	cudaOpts *ort.CUDAProviderOptions

	labels   []string // ImageNet class names, index-aligned; may be nil
	inName   string
	outName  string
	gpuDelta int // MiB of GPU memory the session allocated at startup
}

// NewEngine initializes ONNX Runtime, builds the CUDA session, and loads labels.
// The caller must have set the shared-library path and is responsible for calling
// Close() at shutdown. labels may be nil (numeric class index fallback).
func NewEngine(modelPath, libPath string, labels []string) (*Engine, error) {
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("initialize ONNX Runtime: %w", err)
	}

	// Read the real tensor names from the model rather than hard-coding them.
	ins, outs, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		ort.DestroyEnvironment()
		return nil, fmt.Errorf("read model I/O info: %w", err)
	}
	e := &Engine{
		labels:  labels,
		inName:  ins[0].Name,
		outName: outs[0].Name,
	}

	// CUDA execution provider — the line that puts work on the RTX 4080.
	e.options, err = ort.NewSessionOptions()
	if err != nil {
		ort.DestroyEnvironment()
		return nil, fmt.Errorf("create session options: %w", err)
	}
	e.cudaOpts, err = ort.NewCUDAProviderOptions()
	if err != nil {
		e.Close()
		return nil, fmt.Errorf("create CUDA provider options: %w", err)
	}
	if err := e.cudaOpts.Update(map[string]string{"device_id": "0"}); err != nil {
		e.Close()
		return nil, fmt.Errorf("configure CUDA provider: %w", err)
	}
	if err := e.options.AppendExecutionProviderCUDA(e.cudaOpts); err != nil {
		e.Close()
		return nil, fmt.Errorf("enable CUDA execution provider: %w", err)
	}

	// Persistent input/output tensors. We reuse these for every request.
	e.input, err = ort.NewEmptyTensor[float32](ort.NewShape(1, channels, imageSize, imageSize))
	if err != nil {
		e.Close()
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	e.output, err = ort.NewEmptyTensor[float32](ort.NewShape(1, numClass))
	if err != nil {
		e.Close()
		return nil, fmt.Errorf("create output tensor: %w", err)
	}
	e.inBuf = e.input.GetData()
	e.outBuf = e.output.GetData()

	// Measure GPU memory across session creation: this is where ORT copies the
	// model weights + CUDA context onto the card. A real GPU session moves the
	// needle; a silent CPU fallback would not.
	memBefore := gpuMemUsedMiB()
	e.session, err = ort.NewAdvancedSession(modelPath,
		[]string{e.inName}, []string{e.outName},
		[]ort.ArbitraryTensor{e.input}, []ort.ArbitraryTensor{e.output},
		e.options)
	if err != nil {
		e.Close()
		return nil, fmt.Errorf("create inference session: %w", err)
	}
	e.gpuDelta = gpuMemUsedMiB() - memBefore

	return e, nil
}

// OnGPU reports whether session creation allocated GPU memory. ORT silently falls
// back to CPU if the CUDA EP fails to load, so this is our proof of GPU placement.
func (e *Engine) OnGPU() bool { return e.gpuDelta > 50 }

// GPUDeltaMiB is the GPU memory allocated when the session was created.
func (e *Engine) GPUDeltaMiB() int { return e.gpuDelta }

// Predict runs one image (already preprocessed into a length-3*224*224 NCHW
// float slice) through the persistent session and returns the top-1 class.
func (e *Engine) Predict(input []float32) (Prediction, error) {
	if len(input) != len(e.inBuf) {
		return Prediction{}, fmt.Errorf("input length %d, want %d", len(input), len(e.inBuf))
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	copy(e.inBuf, input) // write into the tensor ORT reads
	if err := e.session.Run(); err != nil {
		return Prediction{}, fmt.Errorf("inference run: %w", err)
	}

	idx, conf := softmaxTop1(e.outBuf)
	p := Prediction{ClassIndex: idx, Confidence: conf}
	if e.labels != nil && idx < len(e.labels) {
		p.ClassName = e.labels[idx]
	}
	return p, nil
}

// softmaxTop1 returns the argmax index and its softmax probability.
func softmaxTop1(logits []float32) (int, float64) {
	top := 0
	maxLogit := logits[0]
	for i, v := range logits {
		if v > maxLogit {
			maxLogit, top = v, i
		}
	}
	var sum float64
	for _, v := range logits {
		sum += math.Exp(float64(v - maxLogit))
	}
	return top, 1.0 / sum // exp(max-max)=1, divided by sum
}

// Close tears down the session and ONNX Runtime environment. Safe to call on a
// partially constructed Engine.
func (e *Engine) Close() {
	if e.session != nil {
		e.session.Destroy()
	}
	if e.input != nil {
		e.input.Destroy()
	}
	if e.output != nil {
		e.output.Destroy()
	}
	if e.cudaOpts != nil {
		e.cudaOpts.Destroy()
	}
	if e.options != nil {
		e.options.Destroy()
	}
	ort.DestroyEnvironment()
}
