// Persistent GPU inference engine with a dynamic batching pipeline.
//
// One ONNX Runtime session (CUDA execution provider) is created at startup and
// reused for every request; a session per request would re-upload the model
// weights to the GPU each time.
//
// HTTP handlers submit a preprocessed image plus a private result channel and
// block. A gatherer goroutine assembles waiting requests into one batch
// (bounded by max batch size or max wait, whichever fires first), an infer lane
// runs a single session.Run on the [N,3,224,224] input, and unpack workers
// split the [N,1000] output back to each caller. Batching is what keeps the GPU
// busy: at batch 1 it sat around 38% utilization.
//
// The model (resnet18-v1-7) declares a dynamic batch dimension
// ([N,3,224,224] -> [N,1000]), so the session is a DynamicAdvancedSession and
// the input tensor is built per batch.
package main

import (
	"errors"
	"fmt"
	"log"
	"math"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	ort "github.com/yalue/onnxruntime_go"
)

// ErrServerBusy is returned by Predict when the engine is at its in-flight limit
// (backpressure). Handlers should map it to HTTP 503 so overload sheds load
// instead of building an unbounded queue, which collapses latency to seconds.
var ErrServerBusy = errors.New("server busy: in-flight limit reached")

// Pipeline tuning. The channels are buffered just enough to let the gatherer
// pack the next batch and the workers drain results while a lane runs.
const (
	inferChDepth  = 2 // packed batches the gatherer may stay ahead by
	unpackChDepth = 16
	numUnpackers  = 4 // split logits + reply in parallel, off the infer lane
)

// numInferLanes is how many inferLoop goroutines run concurrent session.Run
// calls on the one shared session (ORT sessions are thread-safe). More than one
// lane overlaps a lane's H2D copy with another lane's GPU compute. Set from the
// -infer-lanes flag before NewEngine.
var numInferLanes = 1

// cudaExtraOpts holds extra CUDA execution-provider options (ORT tuning keys
// such as cudnn_conv_algo_search) merged into the provider config at session
// creation. Set from the -cuda-opts flag before NewEngine.
var cudaExtraOpts map[string]string

// trtCacheDir, when non-empty, puts the TensorRT execution provider ahead of
// CUDA (CUDA keeps any node TRT rejects) and caches built engines there; the
// first start builds them (~20-60s), later starts load in under a second.
// The engine always allows fp16 kernels; trtInt8 also enables int8, which is
// how a QDQ model (quantize_int8.py) gets real int8 kernels. Set from the
// -trt / -trt-int8 flags before NewEngine.
var (
	trtCacheDir string
	trtInt8     bool
)

// useIoBinding switches full batches to pre-bound fixed-shape I/O tensors via
// ORT IoBinding: no per-batch NewTensor/Destroy and no ORT-side output
// allocation per Run. Partial (tail) batches always take the plain Run path, so
// shapes stay exact and low-load latency is unchanged. Set from the -io-binding
// flag before NewEngine.
var useIoBinding bool

// numBindingSlots is how many pre-bound tensor sets circulate through
// gather -> infer -> unpack when -io-binding is on. Slots block the gatherer
// when exhausted (same backpressure style as inferCh); they return to the
// pool as unpack finishes, so this only needs to cover the pipeline depth.
const numBindingSlots = 8

// bindingSlot is one reusable pre-bound I/O set: a fixed [maxBatch,C,H,W]
// input tensor, a fixed [maxBatch,numClass] output tensor, and the IoBinding
// tying both to the session. inData aliases the input tensor's backing array.
type bindingSlot struct {
	in      *ort.Tensor[float32]
	out     *ort.Tensor[float32]
	binding *ort.IoBinding
	inData  []float32
}

// useDeviceBinding switches full batches to device-resident pre-bound I/O
// (requires the forked binding plus cudamem.go): the engine owns cudaMalloc'd
// input/output buffers bound once via IoBinding, packs into pinned host
// staging, and does the H2D/D2H copies itself. ORT then runs entirely against
// fixed device addresses, with no pageable staging copy inside Run and no
// per-Run output allocation. Set from -device-binding before NewEngine; takes
// precedence over -io-binding.
var useDeviceBinding bool

// deviceSlot is one reusable device-resident I/O set. stage is pinned host
// memory (cudaHostAlloc) viewed as []float32; devIn/devOut are device buffers
// the bound tensors alias. The slot recycles as soon as the lane has copied
// the logits out (D2H) — unpack never touches slot memory.
type deviceSlot struct {
	devIn, devOut unsafe.Pointer
	stagePtr      unsafe.Pointer
	stage         []float32 // pinned, len maxBatch*inputElems
	in, out       *ort.DeviceTensor
	binding       *ort.IoBinding
}

// ResNet-18 (resnet18-v1-7, GluonCV resnetv15) tensor geometry.
const (
	channels  = 3
	imageSize = 224
	numClass  = 1000
	// inputElems is the per-image flattened NCHW length (3*224*224).
	inputElems = channels * imageSize * imageSize
)

// inputPool recycles per-request NCHW tensors, which at ~600KB each were ~30%
// of all allocation. Every producer (preprocessTurbo, normalizeCHW,
// normalizeRGB8) overwrites all inputElems elements, so Gets need no zeroing.
// The HTTP handler returns the slice here after Predict — see the ownership note
// on Predict. Storing the slice header boxes it into an interface (24B alloc per
// Put); negligible next to the 600KB it saves.
var inputPool = sync.Pool{New: func() any { return make([]float32, inputElems) }}

// Prediction is the result of one inference.
type Prediction struct {
	ClassIndex int
	ClassName  string
	Confidence float64
	Detections []Detection // detector models only in model input pixels
}

// batchResult is delivered back to a single waiting caller.
type batchResult struct {
	pred Prediction
	err  error
}

// batchRequest is one queued inference: a preprocessed image and the private
// channel its result must be routed back on. The channel is buffered (size 1) so
// the collector never blocks delivering a result even if the caller has gone away.
type batchRequest struct {
	input    []float32 // length inputElems, NCHW
	resultCh chan batchResult
	submitT  time.Time // when the handler submitted it (for queue-wait timing)
}

// packedBatch is a gathered batch with its input tensor already assembled (the
// 15 MB CPU copy + H2D). The gatherer produces these so the pack cost overlaps
// the inferrer running the PREVIOUS batch.
type packedBatch struct {
	// slot is non-nil when this batch was packed into a pre-bound slot
	// (-io-binding, full batches only); inT/raw are nil in that case.
	slot *bindingSlot
	// dslot is non-nil when packed into a device-resident slot
	// (-device-binding, full batches only); slot/inT/raw are nil then.
	dslot *deviceSlot
	inT   ort.Value
	raw   []float32       // full-capacity staging buffer backing inT; returned to batchBufPool
	batch []*batchRequest // ordered; row i of the output belongs to batch[i]
}

// inferredBatch is a completed inference handed to an unpack worker for logit
// splitting + result delivery (off the inference lane's critical path).
type inferredBatch struct {
	outT  *ort.Tensor[float32]
	batch []*batchRequest
	// logits is set instead of outT on the -device-binding path: the lane
	// already copied the [n*numClass] logits to this pooled host slice (and
	// recycled its device slot); unpack returns it to logitsPool.
	logits []float32
}

// Engine owns the persistent ORT session and the inference pipeline.
//
// Gather, infer and unpack run as separate stages so the non-compute work
// overlaps inference; with all three on one goroutine, throughput capped at
// ~1,450 req/s while the session itself could do ~5,900.
//
//	handlers -> submitCh -> [gather+pack] -> inferCh -> [infer] -> unpackCh -> [unpack+reply]
//	                         1 goroutine              N lanes                  N workers
//
// There is one session for the whole engine: a probe with several sessions on
// the same device measured slower than one, so extra lanes share this session.
type Engine struct {
	session *ort.DynamicAdvancedSession

	options  *ort.SessionOptions
	cudaOpts *ort.CUDAProviderOptions

	labels  []string // ImageNet class names, index-aligned; may be nil
	inName  string
	outName string

	// model geometry read from the ONNX file
	inDims   []int64 // C H W
	outDims  []int64 // per image output shape
	inElems  int
	outElems int
	detect   bool

	// IoBinding slot pool (-io-binding). nil when disabled.
	// slotPool circulates slots gather -> infer -> unpack -> back; slots keeps
	// them all for Close to destroy after the pipeline drains.
	slotPool chan *bindingSlot
	slots    []*bindingSlot

	// Device-resident slot pool (-device-binding). nil when
	// disabled. Slots circulate gather -> infer -> back (the lane copies
	// logits out before recycling); logitsPool recycles those host copies.
	devSlotPool chan *deviceSlot
	devSlots    []*deviceSlot
	logitsPool  sync.Pool

	// Batching policy (configurable so it can be tuned later).
	maxBatch int
	maxWait  time.Duration

	// batchBufPool recycles the batch staging buffer; at ~14MB per batch it was
	// ~30% of all allocation. Buffers are always maxBatch*inputElems long;
	// packAndSend slices to the realized batch size (NewTensor requires exact
	// length). Returned in inferLoop right after inT.Destroy() — ORT pins the Go
	// slice only while the OrtValue lives.
	batchBufPool sync.Pool

	submitCh chan *batchRequest  // handlers -> gatherer
	inferCh  chan *packedBatch   // gatherer -> infer lane
	unpackCh chan *inferredBatch // infer lane -> unpack workers
	stopCh   chan struct{}       // closed to ask the pipeline to drain + exit
	doneCh   chan struct{}       // closed once all stages have exited
	unpackWG sync.WaitGroup      // tracks the unpack workers
	inferWG  sync.WaitGroup      // tracks the infer lanes; last one out closes unpackCh
	started  bool                // true once the pipeline goroutines are running

	// inflight is a counting semaphore bounding concurrent in-flight requests
	// (backpressure). nil = unlimited. Set via SetMaxInflight before serving.
	inflight    chan struct{}
	statRejects int64 // requests rejected by backpressure (atomic)

	// Batching observability (atomic): cumulative batches run and requests served.
	// Lets us confirm batches actually form and report the realized average size.
	statBatches int64
	statReqs    int64

	// Timing observability (atomic, nanoseconds). statWaitNs sums per-request
	// queue wait (submit -> infer start); statRunNs sums per-batch session.Run()
	// duration; statRunBatchNs sums the infer lane's per-batch processing (Run +
	// handoff). statBusyNs/statIdleNs split the INFER LANE's wall-clock into time
	// processing vs blocked waiting for a packed batch. A busy fraction near 1.0
	// means the pipeline keeps inference saturated.
	statWaitNs     int64
	statRunNs      int64
	statRunBatchNs int64
	statBusyNs     int64
	statIdleNs     int64

	gpuDelta int // MiB of GPU memory the session allocated at startup
}

// TimingStats returns the average per-request queue wait (submit -> infer start),
// the average per-batch session.Run() time, the average per-batch infer-lane
// processing time (Run + handoff), and the infer-lane busy-fraction (time
// processing vs blocked waiting for a packed batch). A busy-fraction near 1.0
// means the pipeline keeps the GPU lane saturated.
func (e *Engine) TimingStats() (avgWaitMs, avgRunMs, avgBatchMs, busyFrac float64) {
	batches := atomic.LoadInt64(&e.statBatches)
	reqs := atomic.LoadInt64(&e.statReqs)
	waitNs := atomic.LoadInt64(&e.statWaitNs)
	runNs := atomic.LoadInt64(&e.statRunNs)
	rbNs := atomic.LoadInt64(&e.statRunBatchNs)
	busy := atomic.LoadInt64(&e.statBusyNs)
	idle := atomic.LoadInt64(&e.statIdleNs)
	if reqs > 0 {
		avgWaitMs = float64(waitNs) / float64(reqs) / 1e6
	}
	if batches > 0 {
		avgRunMs = float64(runNs) / float64(batches) / 1e6
		avgBatchMs = float64(rbNs) / float64(batches) / 1e6
	}
	if busy+idle > 0 {
		busyFrac = float64(busy) / float64(busy+idle)
	}
	return
}

// ResetStats zeroes all cumulative counters so a single load level can be
// measured in isolation (call between concurrency levels).
func (e *Engine) ResetStats() {
	atomic.StoreInt64(&e.statBatches, 0)
	atomic.StoreInt64(&e.statReqs, 0)
	atomic.StoreInt64(&e.statWaitNs, 0)
	atomic.StoreInt64(&e.statRunNs, 0)
	atomic.StoreInt64(&e.statRunBatchNs, 0)
	atomic.StoreInt64(&e.statBusyNs, 0)
	atomic.StoreInt64(&e.statIdleNs, 0)
}

// BatchStats returns cumulative batches run, requests served, and realized
// average batch size since startup.
func (e *Engine) BatchStats() (batches, reqs int64, avg float64) {
	batches = atomic.LoadInt64(&e.statBatches)
	reqs = atomic.LoadInt64(&e.statReqs)
	if batches > 0 {
		avg = float64(reqs) / float64(batches)
	}
	return
}

// CumulativeStats returns the raw monotonic counters behind TimingStats and
// BatchStats. The Prometheus endpoint (metrics.go) needs the cumulative sums,
// not averages: Prometheus computes rates and ratios from counters itself.
// Note ResetStats (via /stats?reset=1) zeroes these; Prometheus rate() treats
// that as an ordinary counter reset.
func (e *Engine) CumulativeStats() (batches, reqs, rejects, waitNs, runNs, runBatchNs, busyNs, idleNs int64) {
	return atomic.LoadInt64(&e.statBatches),
		atomic.LoadInt64(&e.statReqs),
		atomic.LoadInt64(&e.statRejects),
		atomic.LoadInt64(&e.statWaitNs),
		atomic.LoadInt64(&e.statRunNs),
		atomic.LoadInt64(&e.statRunBatchNs),
		atomic.LoadInt64(&e.statBusyNs),
		atomic.LoadInt64(&e.statIdleNs)
}

// ensureORT initializes the process-global ONNX Runtime environment exactly once.
// ORT does not support repeated Initialize/Destroy cycles in a single process, so
// every engine shares one environment for the process lifetime.
var (
	ortInitOnce sync.Once
	ortInitErr  error
)

func ensureORT(libPath string) error {
	ortInitOnce.Do(func() {
		ort.SetSharedLibraryPath(libPath)
		ortInitErr = ort.InitializeEnvironment()
	})
	return ortInitErr
}

// NewEngine initializes ONNX Runtime, builds the CUDA session, loads labels, and
// starts the batching collector goroutine. maxBatch and maxWait define the two
// batching triggers. labels may be nil (numeric class index fallback). The caller
// must call Close() at shutdown.
func NewEngine(modelPath, libPath string, labels []string, maxBatch int, maxWait time.Duration) (*Engine, error) {
	if maxBatch < 1 {
		maxBatch = 1
	}
	// ORT's environment is a process-global initialized exactly once. Repeatedly
	// Initialize/Destroy-ing it (e.g. one engine per test, or a future multi-engine
	// setup) corrupts ORT and segfaults; so we init once and never tear the
	// environment down per-engine (the process exit reclaims it).
	if err := ensureORT(libPath); err != nil {
		return nil, fmt.Errorf("initialize ONNX Runtime: %w", err)
	}

	// Read the real tensor names from the model rather than hard-coding them.
	ins, outs, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		return nil, fmt.Errorf("read model I/O info: %w", err)
	}
	inD, outD := ins[0].Dimensions, outs[0].Dimensions
	if len(inD) != 4 || len(outD) < 2 {
		return nil, fmt.Errorf("unsupported model shapes in %v out %v", inD, outD)
	}
	e := &Engine{
		labels:   labels,
		inName:   ins[0].Name,
		outName:  outs[0].Name,
		inDims:   inD[1:],
		outDims:  outD[1:],
		inElems:  int(inD[1] * inD[2] * inD[3]),
		outElems: 1,
		detect:   len(outD) == 3,
		maxBatch: maxBatch,
		maxWait:  maxWait,
		submitCh: make(chan *batchRequest, maxBatch*4),
		inferCh:  make(chan *packedBatch, inferChDepth),
		unpackCh: make(chan *inferredBatch, unpackChDepth),
		stopCh:   make(chan struct{}),
		doneCh:   make(chan struct{}),
	}
	for _, d := range e.outDims {
		e.outElems *= int(d)
	}
	if e.inElems <= 0 || e.outElems <= 0 {
		return nil, fmt.Errorf("model needs fixed non batch dims in %v out %v", inD, outD)
	}
	e.batchBufPool.New = func() any { return make([]float32, maxBatch*e.inElems) }

	// CUDA execution provider — the line that puts work on the RTX 4080.
	e.options, err = ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("create session options: %w", err)
	}
	if trtCacheDir != "" {
		if err := e.appendTensorRT(); err != nil {
			e.Close()
			return nil, err
		}
	}
	e.cudaOpts, err = ort.NewCUDAProviderOptions()
	if err != nil {
		e.Close()
		return nil, fmt.Errorf("create CUDA provider options: %w", err)
	}
	cudaOptMap := map[string]string{"device_id": "0"}
	for k, v := range cudaExtraOpts {
		cudaOptMap[k] = v
	}
	if err := e.cudaOpts.Update(cudaOptMap); err != nil {
		e.Close()
		return nil, fmt.Errorf("configure CUDA provider: %w", err)
	}
	if err := e.options.AppendExecutionProviderCUDA(e.cudaOpts); err != nil {
		e.Close()
		return nil, fmt.Errorf("enable CUDA execution provider: %w", err)
	}

	// Measure GPU memory across session creation: this is where ORT copies the
	// model weights + CUDA context onto the card. A real GPU session moves the
	// needle; a silent CPU fallback would not.
	//
	// A DynamicAdvancedSession takes its input/output tensors per Run() call,
	// which is what lets the batch dimension N vary.
	memBefore := gpuMemUsedMiB()
	e.session, err = ort.NewDynamicAdvancedSession(modelPath,
		[]string{e.inName}, []string{e.outName}, e.options)
	if err != nil {
		e.Close()
		return nil, fmt.Errorf("create inference session: %w", err)
	}
	e.gpuDelta = gpuMemUsedMiB() - memBefore

	// Device-resident I/O slots (-device-binding): cudaMalloc'd buffers bound
	// once; the engine does its own pinned-host H2D/D2H, ORT computes against
	// fixed device addresses. Wins over -io-binding when both are set.
	if useDeviceBinding {
		e.logitsPool.New = func() any { return make([]float32, maxBatch*e.outElems) }
		e.devSlotPool = make(chan *deviceSlot, numBindingSlots)
		for i := 0; i < numBindingSlots; i++ {
			slot, err := e.newDeviceSlot()
			if err != nil {
				e.Close()
				return nil, fmt.Errorf("create device-binding slot %d: %w", i, err)
			}
			e.devSlots = append(e.devSlots, slot)
			e.devSlotPool <- slot
		}
	}

	// Pre-bound I/O slots (-io-binding): fixed-shape tensors bound once, so
	// full batches skip per-Run tensor creation and ORT output allocation.
	if useIoBinding && !useDeviceBinding {
		e.slotPool = make(chan *bindingSlot, numBindingSlots)
		for i := 0; i < numBindingSlots; i++ {
			slot, err := e.newBindingSlot()
			if err != nil {
				e.Close()
				return nil, fmt.Errorf("create io-binding slot %d: %w", i, err)
			}
			e.slots = append(e.slots, slot)
			e.slotPool <- slot
		}
	}

	// Start the pipeline: 1 gatherer -> N infer lanes -> N unpack workers.
	// Multiple lanes run concurrent Run calls on the one shared session;
	// unpackCh is closed only after the last lane exits.
	e.started = true
	e.unpackWG.Add(numUnpackers)
	for i := 0; i < numUnpackers; i++ {
		go e.unpackLoop()
	}
	lanes := numInferLanes
	if lanes < 1 {
		lanes = 1
	}
	e.inferWG.Add(lanes)
	for i := 0; i < lanes; i++ {
		go e.inferLoop()
	}
	go func() { e.inferWG.Wait(); close(e.unpackCh) }()
	go e.collect()
	go func() { e.unpackWG.Wait(); close(e.doneCh) }()
	return e, nil
}

// OnGPU reports whether session creation allocated GPU memory. ORT silently falls
// back to CPU if the CUDA EP fails to load, so this is our proof of GPU placement.
func (e *Engine) OnGPU() bool { return e.gpuDelta > 50 }

// GPUDeltaMiB is the GPU memory allocated when the session was created.
func (e *Engine) GPUDeltaMiB() int { return e.gpuDelta }

// SetMaxInflight bounds the number of requests that may be in flight at once.
// n <= 0 means unlimited. Call once before serving traffic. Beyond the limit,
// Predict returns ErrServerBusy so the server sheds load instead of queueing
// unboundedly, which collapsed latency to seconds under overload.
func (e *Engine) SetMaxInflight(n int) {
	if n <= 0 {
		e.inflight = nil
		return
	}
	e.inflight = make(chan struct{}, n)
}

// Rejects returns the cumulative number of requests shed by backpressure.
func (e *Engine) Rejects() int64 { return atomic.LoadInt64(&e.statRejects) }

// Acquire takes an in-flight slot for backpressure, returning false if the limit
// is reached (the caller should shed with 503). It must be paired with Release.
// Handlers acquire at request entry, ahead of the expensive decode, so excess
// load is shed immediately instead of piling into the decoder pool. Unlimited
// (SetMaxInflight(0)) always returns true.
func (e *Engine) Acquire() bool {
	if e.inflight == nil {
		return true
	}
	select {
	case e.inflight <- struct{}{}:
		return true
	default:
		atomic.AddInt64(&e.statRejects, 1)
		return false
	}
}

// Release returns an in-flight slot taken by Acquire.
func (e *Engine) Release() {
	if e.inflight != nil {
		<-e.inflight
	}
}

// Predict submits one preprocessed image to the pipeline and blocks until its
// result comes back. This is the only entry point handlers use; they never touch
// the session directly. Backpressure is handled by Acquire/Release around the
// whole request (including decode), not here.
//
// Ownership: the engine copies input during packing and drops its reference, so
// once Predict returns the caller is the sole owner and SHOULD recycle the slice
// via inputPool.Put (the HTTP handler does). The engine deliberately does not Put
// it itself: tests share one tensor across many concurrent Predicts, and pooling
// a slice that other calls still read would corrupt batches.
func (e *Engine) Predict(input []float32) (Prediction, error) {
	if len(input) != e.inElems {
		return Prediction{}, fmt.Errorf("input length %d, want %d", len(input), e.inElems)
	}

	req := &batchRequest{input: input, resultCh: make(chan batchResult, 1), submitT: time.Now()}

	select {
	case e.submitCh <- req:
	case <-e.stopCh:
		return Prediction{}, fmt.Errorf("engine shutting down")
	}

	res := <-req.resultCh
	return res.pred, res.err
}

// collect is the gatherer stage: it forms a batch (bounded by maxBatch or
// maxWait) and assembles its [N,3,224,224] input tensor, then hands the packed
// batch to an infer lane and immediately loops to gather the next, so the pack
// cost (a 15 MB copy) overlaps the lane running the previous batch. On shutdown
// it drains pending requests and closes inferCh, unwinding the rest of the
// pipeline.
func (e *Engine) collect() {
	for {
		// Block for the first request of a batch (or shutdown).
		var first *batchRequest
		select {
		case first = <-e.submitCh:
		case <-e.stopCh:
			e.drainStop()
			close(e.inferCh)
			return
		}

		batch := make([]*batchRequest, 0, e.maxBatch)
		batch = append(batch, first)

		timer := time.NewTimer(e.maxWait)
	gather:
		for len(batch) < e.maxBatch {
			select {
			case r := <-e.submitCh:
				batch = append(batch, r)
			case <-timer.C:
				break gather
			case <-e.stopCh:
				break gather
			}
		}
		timer.Stop()

		e.packAndSend(batch)

		select {
		case <-e.stopCh:
			e.drainStop()
			close(e.inferCh)
			return
		default:
		}
	}
}

// appendTensorRT registers the TensorRT execution provider on e.options. The
// optimization profile spans batch 1..maxBatch so partial batches reuse the
// same engine instead of triggering a rebuild.
//
// ORT graph optimizations are turned off for TRT sessions: TensorRT fuses the
// graph itself, and ORT's WeightBiasQuantization pass rewrites a QDQ model's
// float conv biases into int32 DequantizeLinear nodes, which TensorRT rejects
// ("only activation datatypes allowed"), splitting the graph off TRT.
func (e *Engine) appendTensorRT() error {
	if err := e.options.SetGraphOptimizationLevel(ort.GraphOptimizationLevelDisableAll); err != nil {
		return fmt.Errorf("disable graph optimizations: %w", err)
	}
	opts, err := ort.NewTensorRTProviderOptions()
	if err != nil {
		return fmt.Errorf("create TensorRT provider options: %w", err)
	}
	defer opts.Destroy()
	shape := func(n int) string {
		return fmt.Sprintf("%s:%dx%dx%dx%d", e.inName, n, e.inDims[0], e.inDims[1], e.inDims[2])
	}
	err = opts.Update(map[string]string{
		"device_id":               "0",
		"trt_fp16_enable":         "1",
		"trt_int8_enable":         map[bool]string{false: "0", true: "1"}[trtInt8],
		"trt_engine_cache_enable": "1",
		"trt_engine_cache_path":   trtCacheDir,
		"trt_timing_cache_enable": "1",
		"trt_timing_cache_path":   trtCacheDir,
		"trt_profile_min_shapes":  shape(1),
		"trt_profile_opt_shapes":  shape(e.maxBatch),
		"trt_profile_max_shapes":  shape(e.maxBatch),
	})
	if err != nil {
		return fmt.Errorf("configure TensorRT provider: %w", err)
	}
	if err := e.options.AppendExecutionProviderTensorRT(opts); err != nil {
		return fmt.Errorf("enable TensorRT execution provider: %w", err)
	}
	return nil
}

// newBindingSlot creates one pre-bound I/O set for the -io-binding path.
func (e *Engine) newBindingSlot() (*bindingSlot, error) {
	in, err := ort.NewEmptyTensor[float32](
		e.inShape(e.maxBatch))
	if err != nil {
		return nil, fmt.Errorf("input tensor: %w", err)
	}
	out, err := ort.NewEmptyTensor[float32](e.outShape(e.maxBatch))
	if err != nil {
		in.Destroy()
		return nil, fmt.Errorf("output tensor: %w", err)
	}
	b, err := e.session.CreateIoBinding()
	if err != nil {
		in.Destroy()
		out.Destroy()
		return nil, fmt.Errorf("create binding: %w", err)
	}
	if err := b.BindInput(e.inName, in); err == nil {
		err = b.BindOutput(e.outName, out)
	}
	if err != nil {
		b.Destroy()
		in.Destroy()
		out.Destroy()
		return nil, fmt.Errorf("bind I/O: %w", err)
	}
	return &bindingSlot{in: in, out: out, binding: b, inData: in.GetData()}, nil
}

// newDeviceSlot creates one device-resident pre-bound I/O set for the
// -device-binding path: device input/output buffers, pinned host staging,
// and an IoBinding tying the device tensors to the session.
func (e *Engine) newDeviceSlot() (s *deviceSlot, err error) {
	inBytes := e.maxBatch * e.inElems * 4
	outBytes := e.maxBatch * e.outElems * 4
	s = &deviceSlot{}
	defer func() {
		if err != nil {
			e.destroyDeviceSlot(s)
		}
	}()
	if s.devIn, err = cudaMalloc(inBytes); err != nil {
		return nil, err
	}
	if s.devOut, err = cudaMalloc(outBytes); err != nil {
		return nil, err
	}
	if s.stagePtr, err = cudaHostAlloc(inBytes); err != nil {
		return nil, err
	}
	s.stage = unsafe.Slice((*float32)(s.stagePtr), e.maxBatch*e.inElems)
	s.in, err = ort.NewCudaFloat32Tensor(
		e.inShape(e.maxBatch),
		s.devIn, uintptr(inBytes), 0)
	if err != nil {
		return nil, fmt.Errorf("device input tensor: %w", err)
	}
	s.out, err = ort.NewCudaFloat32Tensor(
		e.outShape(e.maxBatch), s.devOut, uintptr(outBytes), 0)
	if err != nil {
		return nil, fmt.Errorf("device output tensor: %w", err)
	}
	if s.binding, err = e.session.CreateIoBinding(); err != nil {
		return nil, fmt.Errorf("create binding: %w", err)
	}
	if err = s.binding.BindInput(e.inName, s.in); err == nil {
		err = s.binding.BindOutput(e.outName, s.out)
	}
	if err != nil {
		return nil, fmt.Errorf("bind device I/O: %w", err)
	}
	return s, nil
}

// destroyDeviceSlot frees a (possibly partially-constructed) device slot:
// ORT objects first, then the raw buffers they alias.
func (e *Engine) destroyDeviceSlot(s *deviceSlot) {
	if s == nil {
		return
	}
	if s.binding != nil {
		s.binding.Destroy()
	}
	if s.in != nil {
		s.in.Destroy()
	}
	if s.out != nil {
		s.out.Destroy()
	}
	if s.stagePtr != nil {
		cudaFreeHost(s.stagePtr)
	}
	if s.devIn != nil {
		cudaFree(s.devIn)
	}
	if s.devOut != nil {
		cudaFree(s.devOut)
	}
}

// packAndSend assembles the batch's input tensor and forwards it to the infer
// lane. A blocking send provides natural backpressure: if the lane is still busy,
// the gatherer waits here rather than packing unbounded batches ahead.
func (e *Engine) packAndSend(batch []*batchRequest) {
	n := len(batch)
	if n == 0 {
		return
	}
	if e.devSlotPool != nil && n == e.maxBatch {
		// Full batch on the -device-binding path: pack into the slot's PINNED
		// host staging; the lane does the H2D copy itself.
		slot := <-e.devSlotPool
		for i, r := range batch {
			copy(slot.stage[i*e.inElems:(i+1)*e.inElems], r.input)
			r.input = nil
		}
		e.inferCh <- &packedBatch{dslot: slot, batch: batch}
		return
	}
	if e.slotPool != nil && n == e.maxBatch {
		// Full batch on the -io-binding path: pack straight into a pre-bound
		// slot's input tensor. Blocking on the pool is the same backpressure
		// as the inferCh send below; slots return as unpack workers finish.
		slot := <-e.slotPool
		for i, r := range batch {
			copy(slot.inData[i*e.inElems:(i+1)*e.inElems], r.input)
			r.input = nil
		}
		e.inferCh <- &packedBatch{slot: slot, batch: batch}
		return
	}
	raw := e.batchBufPool.Get().([]float32)
	data := raw[:n*e.inElems] // NewTensor requires the exact batch length
	for i, r := range batch {
		copy(data[i*e.inElems:(i+1)*e.inElems], r.input)
		r.input = nil // engine is done with the caller's tensor after this copy
	}
	inT, err := ort.NewTensor(e.inShape(n), data)
	if err != nil {
		e.batchBufPool.Put(raw)
		failBatch(batch, fmt.Errorf("create input tensor: %w", err))
		return
	}
	e.inferCh <- &packedBatch{inT: inT, raw: raw, batch: batch}
}

// inferLoop is a GPU-owning lane: it runs one packed batch at a time and hands
// the output to the unpack workers. Splitting logits and replying happen
// downstream, in parallel, so they never stall the lane. numInferLanes
// of these run concurrently against the one shared session; the inferWG waiter
// in NewEngine closes unpackCh after the last lane exits. With more than one
// lane the busy/idle stats aggregate, so busyFrac reads as the average per-lane
// busy fraction.
func (e *Engine) inferLoop() {
	defer e.inferWG.Done()
	for {
		idleStart := time.Now()
		pb, ok := <-e.inferCh
		if !ok {
			return
		}
		atomic.AddInt64(&e.statIdleNs, int64(time.Since(idleStart)))
		busyStart := time.Now()

		n := len(pb.batch)
		b := atomic.AddInt64(&e.statBatches, 1)
		atomic.AddInt64(&e.statReqs, int64(n))

		// Queue wait: submit -> infer start.
		var waitNs int64
		for _, r := range pb.batch {
			waitNs += int64(busyStart.Sub(r.submitT))
		}
		atomic.AddInt64(&e.statWaitNs, waitNs)
		if b%200 == 0 {
			_, _, avg := e.BatchStats()
			log.Printf("batching: %d batches run, avg batch size %.1f (last=%d)", b, avg, n)
		}

		if pb.dslot != nil {
			// -device-binding path: pinned H2D into the bound device input,
			// run against fixed device addresses, D2H the logits to a pooled
			// host slice, recycle the slot immediately. Device-resident bound
			// values are used in place by ORT — none of the cached-copy
			// staleness of CPU-backed bound inputs applies.
			s := pb.dslot
			runT := time.Now()
			runErr := cudaMemcpyH2D(s.devIn, s.stagePtr, n*e.inElems*4)
			if runErr == nil {
				runErr = e.session.RunWithBinding(s.binding)
			}
			var logits []float32
			if runErr == nil {
				logits = e.logitsPool.Get().([]float32)
				runErr = cudaMemcpyD2H(unsafe.Pointer(&logits[0]), s.devOut,
					n*e.outElems*4)
				if runErr != nil {
					e.logitsPool.Put(logits)
				}
			}
			atomic.AddInt64(&e.statRunNs, int64(time.Since(runT)))
			e.devSlotPool <- s
			if runErr != nil {
				failBatch(pb.batch, fmt.Errorf("inference run (device): %w", runErr))
				atomic.AddInt64(&e.statBusyNs, int64(time.Since(busyStart)))
				continue
			}
			e.unpackCh <- &inferredBatch{logits: logits, batch: pb.batch}
			atomic.AddInt64(&e.statRunBatchNs, int64(time.Since(busyStart)))
			atomic.AddInt64(&e.statBusyNs, int64(time.Since(busyStart)))
			continue
		}

		if pb.slot != nil {
			// -io-binding path: the input tensor is pre-bound and refilled in
			// place, avoiding a per-batch NewTensor/Destroy of a 14 MB pin.
			//
			// Two ORT behaviors to watch here. The output has to be read back
			// with GetBoundOutputValues, because a CUDA session does not
			// reliably write a pre-bound CPU output value (reading the bound
			// tensor directly gave 992/1000 garbage answers); this is also the
			// binding library's documented read pattern. And the input has to
			// be re-bound every batch, because ORT caches the bound input's
			// device copy across runs (SynchronizeBoundInputs, the C-API
			// invalidation hook, is not exposed by the Go binding), so without
			// it every Run reuses the first upload (992/1000 wrong).
			// Re-binding the refilled tensor forces a fresh H2D copy.
			runT := time.Now()
			runErr := pb.slot.binding.BindInput(e.inName, pb.slot.in)
			if runErr == nil {
				runErr = e.session.RunWithBinding(pb.slot.binding)
			}
			atomic.AddInt64(&e.statRunNs, int64(time.Since(runT)))
			if runErr != nil {
				e.slotPool <- pb.slot
				failBatch(pb.batch, fmt.Errorf("inference run (bound): %w", runErr))
				atomic.AddInt64(&e.statBusyNs, int64(time.Since(busyStart)))
				continue
			}
			outs, gErr := pb.slot.binding.GetBoundOutputValues()
			if gErr == nil && len(outs) != 1 {
				gErr = fmt.Errorf("expected 1 bound output, got %d", len(outs))
			}
			var outT *ort.Tensor[float32]
			if gErr == nil {
				var castOK bool
				outT, castOK = outs[0].(*ort.Tensor[float32])
				if !castOK {
					gErr = fmt.Errorf("unexpected bound output type %T", outs[0])
				}
			}
			if gErr != nil {
				for _, v := range outs {
					if v != nil {
						v.Destroy()
					}
				}
				e.slotPool <- pb.slot
				failBatch(pb.batch, fmt.Errorf("read bound output: %w", gErr))
				atomic.AddInt64(&e.statBusyNs, int64(time.Since(busyStart)))
				continue
			}
			// The retrieved value is a fresh OrtValue owned by us; the slot
			// (input + binding) can recycle immediately.
			e.slotPool <- pb.slot
			e.unpackCh <- &inferredBatch{outT: outT, batch: pb.batch}
			atomic.AddInt64(&e.statRunBatchNs, int64(time.Since(busyStart)))
			atomic.AddInt64(&e.statBusyNs, int64(time.Since(busyStart)))
			continue
		}

		// nil output -> ORT allocates the correctly-shaped [N,1000] output.
		outputs := []ort.Value{nil}
		runT := time.Now()
		runErr := e.session.Run([]ort.Value{pb.inT}, outputs)
		atomic.AddInt64(&e.statRunNs, int64(time.Since(runT)))
		pb.inT.Destroy()
		e.batchBufPool.Put(pb.raw) // safe once the OrtValue no longer pins it
		if runErr != nil {
			failBatch(pb.batch, fmt.Errorf("inference run: %w", runErr))
			atomic.AddInt64(&e.statBusyNs, int64(time.Since(busyStart)))
			continue
		}
		outT, ok := outputs[0].(*ort.Tensor[float32])
		if !ok {
			if outputs[0] != nil {
				outputs[0].Destroy()
			}
			failBatch(pb.batch, fmt.Errorf("unexpected output tensor type %T", outputs[0]))
			atomic.AddInt64(&e.statBusyNs, int64(time.Since(busyStart)))
			continue
		}

		e.unpackCh <- &inferredBatch{outT: outT, batch: pb.batch}
		atomic.AddInt64(&e.statRunBatchNs, int64(time.Since(busyStart)))
		atomic.AddInt64(&e.statBusyNs, int64(time.Since(busyStart)))
	}
}

// unpackLoop is a worker that splits a finished batch's [N,1000] logits into
// per-request predictions and delivers each to its caller, preserving index
// correspondence (row i belongs to batch[i]). Runs in parallel with the infer
// lanes and the other unpack workers.
func (e *Engine) unpackLoop() {
	defer e.unpackWG.Done()
	for ib := range e.unpackCh {
		var logits []float32
		if ib.logits != nil {
			// -device-binding: lane-copied host logits; pooled slice is
			// [maxBatch*numClass], slice to the (always-full) batch.
			logits = ib.logits[:len(ib.batch)*e.outElems]
		} else {
			logits = ib.outT.GetData() // length n*numClass, row-major [N,1000]
		}
		var preds []Prediction
		var err error
		if e.detect {
			preds, err = e.splitDetections(logits, len(ib.batch))
		} else {
			preds, err = splitBatchLogits(logits, len(ib.batch), e.labels)
		}
		if ib.logits != nil {
			e.logitsPool.Put(ib.logits)
		} else {
			ib.outT.Destroy()
		}
		if err != nil {
			failBatch(ib.batch, err)
			continue
		}
		for i, r := range ib.batch {
			r.resultCh <- batchResult{pred: preds[i]}
		}
	}
}

// drainStop fails any requests still queued after a shutdown signal so no caller
// blocks forever.
func (e *Engine) drainStop() {
	for {
		select {
		case r := <-e.submitCh:
			r.resultCh <- batchResult{err: fmt.Errorf("engine shutting down")}
		default:
			return
		}
	}
}

// splitBatchLogits turns a flat [N*numClass] logits slice into N top-1
// Predictions, one per row. Kept as a pure function so result-routing
// correctness (no swapped/corrupted answers) can be unit-tested without a GPU.
func splitBatchLogits(logits []float32, n int, labels []string) ([]Prediction, error) {
	if len(logits) != n*numClass {
		return nil, fmt.Errorf("output length %d, want %d (%d x %d)", len(logits), n*numClass, n, numClass)
	}
	preds := make([]Prediction, n)
	for i := 0; i < n; i++ {
		row := logits[i*numClass : (i+1)*numClass]
		idx, conf := softmaxTop1(row)
		p := Prediction{ClassIndex: idx, Confidence: conf}
		if labels != nil && idx < len(labels) {
			p.ClassName = labels[idx]
		}
		preds[i] = p
	}
	return preds, nil
}

// detection thresholds set from flags before serving
var (
	detConfThresh float32 = 0.3
	detIoUThresh  float32 = 0.45
)

func (e *Engine) splitDetections(out []float32, n int) ([]Prediction, error) {
	if len(out) != n*e.outElems {
		return nil, fmt.Errorf("output length %d want %d", len(out), n*e.outElems)
	}
	numCls := int(e.outDims[len(e.outDims)-1]) - 5
	preds := make([]Prediction, n)
	for i := range preds {
		row := out[i*e.outElems : (i+1)*e.outElems]
		preds[i].Detections = decodeDetections(row, numCls, e.labels, detConfThresh, detIoUThresh)
	}
	return preds, nil
}

func (e *Engine) inShape(n int) ort.Shape {
	return ort.NewShape(append([]int64{int64(n)}, e.inDims...)...)
}

func (e *Engine) outShape(n int) ort.Shape {
	return ort.NewShape(append([]int64{int64(n)}, e.outDims...)...)
}

// InputElems is the per image input length the model expects
func (e *Engine) InputElems() int { return e.inElems }

// Detector reports whether the model outputs boxes rather than class scores
func (e *Engine) Detector() bool { return e.detect }

// failBatch delivers the same error to every request in a failed batch.
func failBatch(batch []*batchRequest, err error) {
	for _, r := range batch {
		r.resultCh <- batchResult{err: err}
	}
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

// Close stops the collector, then tears down the session and ONNX Runtime
// environment. Safe to call on a partially constructed Engine.
func (e *Engine) Close() {
	// Stop the collector if it is running.
	if e.stopCh != nil {
		select {
		case <-e.stopCh:
			// already closed
		default:
			close(e.stopCh)
		}
	}
	if e.started && e.doneCh != nil {
		select {
		case <-e.doneCh:
		case <-time.After(5 * time.Second):
		}
	}

	// Slots are only touched by pipeline goroutines, which have all exited by
	// here (doneCh), so the bindings and their tensors are safe to destroy.
	for _, s := range e.slots {
		s.binding.Destroy()
		s.in.Destroy()
		s.out.Destroy()
	}
	e.slots = nil
	for _, s := range e.devSlots {
		e.destroyDeviceSlot(s)
	}
	e.devSlots = nil

	if e.session != nil {
		e.session.Destroy()
	}
	if e.cudaOpts != nil {
		e.cudaOpts.Destroy()
	}
	if e.options != nil {
		e.options.Destroy()
	}
	// Intentionally NOT calling ort.DestroyEnvironment(): the environment is a
	// process-global initialized once by ensureORT and shared by all engines;
	// tearing it down here would break a subsequent NewEngine (and crashes ORT
	// under repeated init/destroy). Process exit reclaims it.
}
