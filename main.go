// HTTP front end for the ResNet-18 inference server.
//
// The model is loaded onto the GPU once at startup (ONNX Runtime, CUDA
// execution provider) and reused for every request:
//
//	POST /predict  JPEG/PNG image, pre-decoded RGB, or a preprocessed tensor;
//	               returns the top-1 ImageNet class as JSON
//	GET  /health   200 if the model session is loaded
//	GET  /stats    per-stage timing averages
//	GET  /metrics  Prometheus exposition
//
// This file handles HTTP, request-body decoding and observability; batching and
// inference live in engine.go.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	_ "net/http/pprof" // registers /debug/pprof/* on http.DefaultServeMux
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

const maxImageBytes = 20 << 20 // 20 MiB request-body cap

// rawTensorContentType marks a /predict body that is already the preprocessed
// NCHW float32 tensor (native byte order, length inputElems*4), so the server
// does HTTP + batching + inference and nothing else.
const rawTensorContentType = "application/x-raw-tensor"

// rawRGBContentType marks a /predict body that is a decoded, resized,
// center-cropped 224x224 image as interleaved RGB uint8 (150,528 bytes).
// Bodies are 4x smaller than x-raw-tensor and the server only has to
// normalize them into NCHW.
const rawRGBContentType = "application/x-raw-rgb"

func main() {
	addr := flag.String("addr", ":8080", "HTTP listen address")
	modelPath := flag.String("model", "resnet18.onnx", "path to the ONNX model")
	labelsPath := flag.String("labels", "", "path to class labels (default imagenet_classes.txt or coco_classes.txt for detectors)")
	libPath := flag.String("lib", "onnxruntime/lib/libonnxruntime.so", "path to libonnxruntime.so")
	maxBatch := flag.Int("max-batch", 32, "dynamic batching: max requests per inference batch")
	maxWait := flag.Duration("max-wait", 5*time.Millisecond, "dynamic batching: max time to wait assembling a batch before running it")
	quiet := flag.Bool("quiet", false, "suppress per-request prediction logging (avoids log contention under load)")
	gpuPreprocess := flag.Bool("gpu-preprocess", false, "decode+resize images on the GPU via nvJPEG+NPP instead of on the CPU")
	gpuDecoders := flag.Int("gpu-decoders", 8, "size of the GPU decoder-context pool (max concurrent GPU decodes) when -gpu-preprocess is set")
	cpuPreprocess := flag.Bool("cpu-preprocess", false, "decode images on the CPU via libjpeg-turbo (SIMD, DCT prescale) instead of Go's pure-Go image/jpeg, keeping decode off the GPU")
	cpuDecoders := flag.Int("cpu-decoders", 16, "size of the libjpeg-turbo decoder pool (max concurrent CPU decodes) when -cpu-preprocess is set")
	maxInflight := flag.Int("max-inflight", 0, "backpressure: max concurrent in-flight requests (0 = unlimited); excess returns HTTP 503")
	inferLanes := flag.Int("infer-lanes", 1, "concurrent inference lanes on the shared ORT session; >1 overlaps the H2D copy with GPU compute")
	cudaOptsFlag := flag.String("cuda-opts", "", "extra CUDA execution-provider options as k=v[,k=v...], e.g. cudnn_conv_algo_search=EXHAUSTIVE")
	ioBinding := flag.Bool("io-binding", false, "run full batches through pre-bound fixed-shape I/O tensors (ORT IoBinding): skips per-batch tensor create/destroy and output allocation")
	deviceBinding := flag.Bool("device-binding", false, "run full batches through device-resident pre-bound I/O with pinned-host staging: engine-owned cudaMalloc buffers, H2D/D2H under engine control; takes precedence over -io-binding")
	trt := flag.String("trt", "", "run the model through the TensorRT execution provider (fp16 allowed), caching built engines in this directory; needs TensorRT 10 on LD_LIBRARY_PATH and a model whose weights are initializers, not graph inputs (e.g. resnet18_op13.onnx or resnet18_int8.onnx)")
	confFlag := flag.Float64("conf", 0.3, "detector score threshold")
	iouFlag := flag.Float64("iou", 0.45, "detector NMS IoU threshold")
	trtInt8Flag := flag.Bool("trt-int8", false, "with -trt: also allow int8 kernels; use with a QDQ model from quantize_int8.py")
	flag.Parse()

	detConfThresh, detIoUThresh = float32(*confFlag), float32(*iouFlag)
	if *labelsPath == "" {
		*labelsPath = "imagenet_classes.txt"
		if modelIsDetector(*modelPath, *libPath) {
			*labelsPath = "coco_classes.txt"
		}
	}
	labels, err := loadLabels(*labelsPath)
	if err != nil {
		log.Printf("WARNING: could not load labels from %q (%v) — responses will use numeric class index only", *labelsPath, err)
	} else {
		log.Printf("loaded %d class labels from %s", len(labels), *labelsPath)
	}

	log.Printf("loading model %s onto GPU via CUDA execution provider...", *modelPath)
	numInferLanes = *inferLanes // must be set before NewEngine spawns the pipeline
	useIoBinding = *ioBinding
	useDeviceBinding = *deviceBinding
	trtCacheDir, trtInt8 = *trt, *trtInt8Flag
	if trtCacheDir != "" {
		log.Printf("TensorRT execution provider enabled (int8=%v, engine cache %s); first start builds engines", trtInt8, trtCacheDir)
	}
	if *cudaOptsFlag != "" {
		cudaExtraOpts = map[string]string{}
		for _, kv := range strings.Split(*cudaOptsFlag, ",") {
			k, v, ok := strings.Cut(kv, "=")
			if !ok || k == "" {
				log.Fatalf("FATAL: -cuda-opts entry %q is not k=v", kv)
			}
			cudaExtraOpts[k] = v
		}
		log.Printf("extra CUDA provider options: %v", cudaExtraOpts)
	}
	engine, err := NewEngine(*modelPath, *libPath, labels, *maxBatch, *maxWait)
	if err != nil {
		log.Fatalf("FATAL: could not initialize inference engine: %v", err)
	}
	defer engine.Close()
	engine.SetMaxInflight(*maxInflight)
	log.Printf("dynamic batching enabled: max-batch=%d, max-wait=%s, max-inflight=%d, infer-lanes=%d", *maxBatch, *maxWait, *maxInflight, *inferLanes)

	// Session creation is where ORT copies the weights onto the card, so a
	// GPU memory delta is the proof of placement: ORT falls back to CPU
	// silently if the CUDA provider fails to load.
	if engine.OnGPU() {
		log.Printf("=== model loaded on GPU (allocated %+d MiB; compute-app PID listed: %v) ===",
			engine.GPUDeltaMiB(), processOnGPU())
	} else {
		log.Printf("WARNING: session allocated only %+d MiB of GPU memory — ORT may have fallen back to CPU",
			engine.GPUDeltaMiB())
	}

	// Decoder selection. JPEG decode is the dominant per-request CPU cost, so
	// it can run on libjpeg-turbo, on the GPU (nvJPEG + NPP), or split across
	// both with the GPU taking overflow only.
	var gpuPool *gpuDecoderPool
	var cpuPool *cpuDecoderPool
	var hybrid *hybridPreprocessor
	switch {
	case *gpuPreprocess && *cpuPreprocess:
		// CPU decodes while it has slack; the GPU is overflow capacity only.
		cpuPool, err = newCPUDecoderPool(*cpuDecoders)
		if err != nil {
			log.Fatalf("FATAL: could not initialize CPU decoder pool: %v", err)
		}
		defer cpuPool.Close()
		gpuPool, err = newGPUDecoderPool(*gpuDecoders)
		if err != nil {
			log.Fatalf("FATAL: could not initialize GPU decoder pool: %v", err)
		}
		defer gpuPool.Close()
		hybrid = &hybridPreprocessor{cpu: cpuPool, gpu: gpuPool}
		log.Printf("HYBRID preprocessing enabled: %d CPU (libjpeg-turbo) decoders primary, %d GPU (nvJPEG %s) decoders as overflow",
			*cpuDecoders, *gpuDecoders, gpuPool.Backend())
	case *gpuPreprocess:
		gpuPool, err = newGPUDecoderPool(*gpuDecoders)
		if err != nil {
			log.Fatalf("FATAL: could not initialize GPU decoder pool: %v", err)
		}
		defer gpuPool.Close()
		log.Printf("GPU preprocessing enabled: %d decoder contexts, nvJPEG backend=%s", *gpuDecoders, gpuPool.Backend())
	case *cpuPreprocess:
		cpuPool, err = newCPUDecoderPool(*cpuDecoders)
		if err != nil {
			log.Fatalf("FATAL: could not initialize CPU decoder pool: %v", err)
		}
		defer cpuPool.Close()
		log.Printf("CPU preprocessing enabled: %d libjpeg-turbo decoders (SIMD, 1/4 DCT scale); GPU runs inference only", *cpuDecoders)
	default:
		log.Printf("CPU preprocessing (default, pure-Go image/jpeg); pass -cpu-preprocess (libjpeg-turbo) or -gpu-preprocess (nvJPEG)")
	}

	if engine.Detector() && cpuPool == nil {
		// detector preprocessing always runs on the libjpeg-turbo pool
		cpuPool, err = newCPUDecoderPool(*cpuDecoders)
		if err != nil {
			log.Fatalf("FATAL: could not initialize CPU decoder pool: %v", err)
		}
		defer cpuPool.Close()
		log.Printf("detector model: letterbox %dx%d on %d libjpeg-turbo decoders, conf=%.2f iou=%.2f",
			letterboxSize, letterboxSize, *cpuDecoders, *confFlag, *iouFlag)
	}

	srv := &server{engine: engine, modelName: filepath.Base(*modelPath), quiet: *quiet, gpuPool: gpuPool, cpuPool: cpuPool, hybrid: hybrid}
	mux := http.NewServeMux()
	mux.HandleFunc("/predict", instrument(&predictMetrics, srv.handlePredict))
	mux.HandleFunc("/health", srv.handleHealth)
	mux.HandleFunc("/stats", srv.handleStats)
	mux.HandleFunc("/metrics", srv.handleMetrics)
	(&jobStore{s: srv}).routes(mux)
	mux.HandleFunc("GET /api/system", startSysMonitor(engine).handleSystem)
	// net/http/pprof registers on the default mux; expose it through ours so
	// `go tool pprof http://.../debug/pprof/{profile,allocs}` works.
	mux.Handle("/debug/pprof/", http.DefaultServeMux)

	httpSrv := &http.Server{Addr: *addr, Handler: mux}

	// Graceful shutdown on SIGINT/SIGTERM.
	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
		<-sig
		log.Println("shutting down...")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = httpSrv.Shutdown(ctx)
	}()

	log.Printf("listening on %s  (POST /predict, GET /health, GET /stats, GET /metrics)", *addr)
	if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("FATAL: server error: %v", err)
	}
	log.Println("stopped")
}

type server struct {
	engine    *Engine
	modelName string
	quiet     bool
	gpuPool   *gpuDecoderPool     // non-nil when -gpu-preprocess is set
	cpuPool   *cpuDecoderPool     // non-nil when -cpu-preprocess is set (libjpeg-turbo)
	hybrid    *hybridPreprocessor // non-nil when both are set (CPU primary, GPU overflow)
}

type predictResponse struct {
	ClassIndex int     `json:"class_index"`
	ClassName  string  `json:"class_name,omitempty"`
	Confidence float64 `json:"confidence"`
}

type errorResponse struct {
	Error string `json:"error"`
}

// bodyBufPool recycles request-body staging buffers; io.ReadAll's doubling
// growth was ~40% of all allocation under load. The buffer must stay checked
// out until Preprocess returns, because the cgo decoders read the raw bytes
// in place.
var bodyBufPool = sync.Pool{New: func() any { return new(bytes.Buffer) }}

// readBodyPooled drains r into a pooled buffer. On success the caller owns the
// buffer and must return it via bodyBufPool.Put when done with buf.Bytes().
func readBodyPooled(r io.Reader) (*bytes.Buffer, error) {
	buf := bodyBufPool.Get().(*bytes.Buffer)
	buf.Reset()
	if _, err := buf.ReadFrom(r); err != nil {
		bodyBufPool.Put(buf)
		return nil, err
	}
	return buf, nil
}

func (s *server) handlePredict(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "use POST with an image body")
		return
	}

	// Backpressure: take an in-flight slot BEFORE the expensive GPU decode, so
	// overload is shed immediately rather than piling into the decoder pool.
	if !s.engine.Acquire() {
		writeError(w, http.StatusServiceUnavailable, "server busy, retry later")
		return
	}
	defer s.engine.Release()

	body, err := imageBody(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	defer body.Close()

	if s.engine.Detector() {
		s.handleDetect(w, body)
		return
	}

	var input []float32
	format := "jpeg"
	if strings.HasPrefix(r.Header.Get("Content-Type"), rawTensorContentType) {
		// The client already did decode+resize+normalize; the body is the
		// tensor itself.
		buf, rerr := readBodyPooled(body)
		if rerr != nil {
			writeError(w, http.StatusBadRequest, "could not read tensor body: "+rerr.Error())
			return
		}
		if buf.Len() != inputElems*4 {
			bodyBufPool.Put(buf)
			writeError(w, http.StatusBadRequest, fmt.Sprintf("raw tensor must be %d bytes (3x224x224 float32), got %d", inputElems*4, buf.Len()))
			return
		}
		input = inputPool.Get().([]float32)
		copy(unsafe.Slice((*byte)(unsafe.Pointer(&input[0])), inputElems*4), buf.Bytes())
		bodyBufPool.Put(buf)
		format = "raw"
	} else if strings.HasPrefix(r.Header.Get("Content-Type"), rawRGBContentType) {
		// The client did decode+resize+crop; the server only runs the
		// /255 + mean/std normalize into the pooled NCHW tensor (~0.3 ms).
		buf, rerr := readBodyPooled(body)
		if rerr != nil {
			writeError(w, http.StatusBadRequest, "could not read rgb body: "+rerr.Error())
			return
		}
		if buf.Len() != inputElems {
			bodyBufPool.Put(buf)
			writeError(w, http.StatusBadRequest, fmt.Sprintf("raw rgb must be %d bytes (224x224x3 uint8), got %d", inputElems, buf.Len()))
			return
		}
		input = inputPool.Get().([]float32)
		normalizeRGB8(buf.Bytes(), input)
		bodyBufPool.Put(buf)
		format = "raw-rgb"
	} else if s.hybrid != nil || s.cpuPool != nil || s.gpuPool != nil {
		// All three JPEG paths (hybrid / libjpeg-turbo / nvJPEG) need the
		// compressed bytes in one buffer. JPEG only; PNG falls to the
		// pure-Go path below.
		buf, rerr := readBodyPooled(body)
		if rerr != nil {
			writeError(w, http.StatusBadRequest, "could not read image body: "+rerr.Error())
			return
		}
		switch {
		case s.hybrid != nil:
			// CPU decoders while they have slack, GPU for overflow.
			input, err = s.hybrid.Preprocess(buf.Bytes())
		case s.cpuPool != nil:
			// CPU path (libjpeg-turbo): SIMD decode keeps preprocessing off the
			// GPU so the GPU runs inference only.
			input, err = s.cpuPool.Preprocess(buf.Bytes())
		default:
			// GPU path: nvJPEG + NPP decode+resize on the GPU.
			input, err = s.gpuPool.Preprocess(buf.Bytes())
		}
		bodyBufPool.Put(buf) // decode is done with the raw bytes
		if err != nil {
			writeError(w, http.StatusBadRequest, "could not decode image (expected JPEG): "+err.Error())
			return
		}
	} else {
		// CPU path (default): decode+resize on the CPU.
		input, format, err = preprocessReader(body)
		if err != nil {
			// Covers malformed images and non-image / wrong-content-type bodies:
			// the decoder can't identify the bytes as JPEG/PNG.
			writeError(w, http.StatusBadRequest, "could not decode image (expected JPEG or PNG): "+err.Error())
			return
		}
	}

	pred, err := s.engine.Predict(input)
	// Predict has copied the tensor into the batch buffer, so recycle it.
	inputPool.Put(input)
	if err != nil {
		log.Printf("inference error: %v", err)
		writeError(w, http.StatusInternalServerError, "inference failed")
		return
	}

	if !s.quiet {
		log.Printf("predict: format=%s -> class=%d (%s) conf=%.4f", format, pred.ClassIndex, pred.ClassName, pred.Confidence)
	}
	writeJSON(w, http.StatusOK, predictResponse{
		ClassIndex: pred.ClassIndex,
		ClassName:  pred.ClassName,
		Confidence: pred.Confidence,
	})
}

// handleStats reports the per-stage timing breakdown. GET /stats returns
// cumulative averages; GET /stats?reset=1 zeroes the counters first, which is
// how a single load level is measured in isolation.
func (s *server) handleStats(w http.ResponseWriter, r *http.Request) {
	if r.URL.Query().Get("reset") == "1" {
		s.engine.ResetStats()
		if s.gpuPool != nil {
			s.gpuPool.ResetStats()
		}
		if s.cpuPool != nil {
			s.cpuPool.ResetStats()
		}
		writeJSON(w, http.StatusOK, map[string]any{"reset": true})
		return
	}
	batches, reqs, avgBatch := s.engine.BatchStats()
	avgWaitMs, avgRunMs, avgBatchMs, busyFrac := s.engine.TimingStats()
	out := map[string]any{
		"requests":            reqs,
		"batches":             batches,
		"avg_batch_size":      avgBatch,
		"avg_queue_wait_ms":   avgWaitMs,  // submit -> batch run start
		"avg_infer_run_ms":    avgRunMs,   // just session.Run() per batch
		"avg_runbatch_ms":     avgBatchMs, // assemble tensor + Run + split logits
		"collector_busy_frac": busyFrac,   // infer lane: processing vs blocked on empty queue
		"rejects":             s.engine.Rejects(),
	}
	if s.gpuPool != nil {
		d, n, cnt := s.gpuPool.PreprocStats()
		out["avg_gpu_decode_ms"] = d
		out["avg_cpu_normalize_ms"] = n
		out["preproc_count"] = cnt
	}
	if s.cpuPool != nil {
		d, n, cnt := s.cpuPool.PreprocStats()
		out["avg_cpu_decode_ms"] = d
		out["avg_cpu_resize_normalize_ms"] = n
		out["preproc_count"] = cnt
	}
	if s.hybrid != nil {
		c, g := s.hybrid.Split()
		out["decode_cpu"] = c
		out["decode_gpu"] = g
	}
	// Cumulative allocation and GC count since start.
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	out["total_alloc_mb"] = float64(ms.TotalAlloc) / (1 << 20)
	out["num_gc"] = ms.NumGC
	writeJSON(w, http.StatusOK, out)
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"status":       "ok",
		"model_loaded": s.engine != nil,
		"device":       deviceLabel(s.engine),
	})
}

func deviceLabel(e *Engine) string {
	if e != nil && e.OnGPU() {
		return "cuda"
	}
	return "cpu"
}

// imageBody returns the image bytes from either a multipart form (field "image"
// or "file") or the raw request body, capped at maxImageBytes. This keeps the
// plain `curl --data-binary @img.jpg` path working while also accepting uploads.
func imageBody(r *http.Request) (io.ReadCloser, error) {
	r.Body = http.MaxBytesReader(nil, r.Body, maxImageBytes)
	if ct := r.Header.Get("Content-Type"); strings.HasPrefix(ct, "multipart/form-data") {
		file, _, err := r.FormFile("image")
		if err != nil {
			file, _, err = r.FormFile("file")
		}
		if err != nil {
			return nil, errors.New(`multipart request without an "image" or "file" field`)
		}
		return file, nil
	}
	return r.Body, nil
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, errorResponse{Error: msg})
}

// loadLabels reads a newline-delimited label file (one class name per line).
func loadLabels(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var labels []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		labels = append(labels, strings.TrimSpace(sc.Text()))
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return labels, nil
}
