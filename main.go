// Phase 1 — single-request GPU inference HTTP server.
//
// A long-running server that loads ResNet-18 onto the GPU once at startup
// (ONNX Runtime CUDA execution provider, validated in Phase 0) and serves:
//
//	POST /predict  — body is a JPEG/PNG image; returns top-1 ImageNet class as JSON
//	GET  /health   — 200 if the model session is loaded
//
// Phase 1 is deliberately one-request-at-a-time (the Engine serializes Run()).
// Concurrency, batching, and production hardening come in later phases.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

const maxImageBytes = 20 << 20 // 20 MiB request-body cap

func main() {
	addr := flag.String("addr", ":8080", "HTTP listen address")
	modelPath := flag.String("model", "resnet18.onnx", "path to the ONNX model")
	labelsPath := flag.String("labels", "imagenet_classes.txt", "path to ImageNet class labels (optional)")
	libPath := flag.String("lib", "onnxruntime/lib/libonnxruntime.so", "path to libonnxruntime.so")
	maxBatch := flag.Int("max-batch", 32, "dynamic batching: max requests per inference batch")
	maxWait := flag.Duration("max-wait", 5*time.Millisecond, "dynamic batching: max time to wait assembling a batch before running it")
	quiet := flag.Bool("quiet", false, "suppress per-request prediction logging (avoids log contention under load)")
	gpuPreprocess := flag.Bool("gpu-preprocess", false, "decode+resize images on the GPU via nvJPEG+NPP instead of on the CPU (Phase 4)")
	gpuDecoders := flag.Int("gpu-decoders", 8, "size of the GPU decoder-context pool (max concurrent GPU decodes) when -gpu-preprocess is set")
	cpuPreprocess := flag.Bool("cpu-preprocess", false, "decode images on the CPU via libjpeg-turbo (SIMD, 1/4 DCT scale) instead of Go's pure-Go image/jpeg, keeping decode off the GPU (Phase 6)")
	cpuDecoders := flag.Int("cpu-decoders", 16, "size of the libjpeg-turbo decoder pool (max concurrent CPU decodes) when -cpu-preprocess is set")
	maxInflight := flag.Int("max-inflight", 0, "backpressure: max concurrent in-flight requests (0 = unlimited); excess returns HTTP 503")
	flag.Parse()

	labels, err := loadLabels(*labelsPath)
	if err != nil {
		log.Printf("WARNING: could not load labels from %q (%v) — responses will use numeric class index only", *labelsPath, err)
	} else {
		log.Printf("loaded %d class labels from %s", len(labels), *labelsPath)
	}

	log.Printf("loading model %s onto GPU via CUDA execution provider...", *modelPath)
	engine, err := NewEngine(*modelPath, *libPath, labels, *maxBatch, *maxWait)
	if err != nil {
		log.Fatalf("FATAL: could not initialize inference engine: %v", err)
	}
	defer engine.Close()
	engine.SetMaxInflight(*maxInflight)
	log.Printf("dynamic batching enabled: max-batch=%d, max-wait=%s, max-inflight=%d", *maxBatch, *maxWait, *maxInflight)

	// Prove GPU placement the same way Phase 0 did: session creation must have
	// allocated GPU memory. ORT silently falls back to CPU otherwise.
	if engine.OnGPU() {
		log.Printf("=== model loaded on GPU (allocated %+d MiB; compute-app PID listed: %v) ===",
			engine.GPUDeltaMiB(), processOnGPU())
	} else {
		log.Printf("WARNING: session allocated only %+d MiB of GPU memory — ORT may have fallen back to CPU",
			engine.GPUDeltaMiB())
	}

	// Optional GPU preprocessing path (Phase 4). When enabled, JPEG decode +
	// resize run on the GPU (nvJPEG + NPP) instead of saturating the CPU, which
	// the Phase 3/4 investigation identified as the real throughput ceiling.
	if *gpuPreprocess && *cpuPreprocess {
		log.Fatalf("FATAL: -gpu-preprocess and -cpu-preprocess are mutually exclusive")
	}
	var gpuPool *gpuDecoderPool
	var cpuPool *cpuDecoderPool
	switch {
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

	srv := &server{engine: engine, quiet: *quiet, gpuPool: gpuPool, cpuPool: cpuPool}
	mux := http.NewServeMux()
	mux.HandleFunc("/predict", srv.handlePredict)
	mux.HandleFunc("/health", srv.handleHealth)
	mux.HandleFunc("/stats", srv.handleStats)

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

	log.Printf("listening on %s  (POST /predict, GET /health)", *addr)
	if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("FATAL: server error: %v", err)
	}
	log.Println("stopped")
}

type server struct {
	engine  *Engine
	quiet   bool
	gpuPool *gpuDecoderPool // non-nil when -gpu-preprocess is set
	cpuPool *cpuDecoderPool // non-nil when -cpu-preprocess is set (libjpeg-turbo)
}

type predictResponse struct {
	ClassIndex int     `json:"class_index"`
	ClassName  string  `json:"class_name,omitempty"`
	Confidence float64 `json:"confidence"`
}

type errorResponse struct {
	Error string `json:"error"`
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

	var input []float32
	format := "jpeg"
	if s.cpuPool != nil {
		// CPU path (libjpeg-turbo): SIMD decode keeps preprocessing off the GPU so
		// the GPU runs inference only. JPEG-only (like the GPU path); PNGs aren't
		// exercised by the load test.
		raw, rerr := io.ReadAll(body)
		if rerr != nil {
			writeError(w, http.StatusBadRequest, "could not read image body: "+rerr.Error())
			return
		}
		input, err = s.cpuPool.Preprocess(raw)
		if err != nil {
			writeError(w, http.StatusBadRequest, "could not decode image on CPU (expected JPEG): "+err.Error())
			return
		}
	} else if s.gpuPool != nil {
		// GPU path: nvJPEG needs the raw compressed bytes, so read the body and
		// decode+resize on the GPU. (nvJPEG is JPEG-only; PNGs would need the CPU
		// fallback — not exercised by the load test, which sends JPEG.)
		raw, rerr := io.ReadAll(body)
		if rerr != nil {
			writeError(w, http.StatusBadRequest, "could not read image body: "+rerr.Error())
			return
		}
		input, err = s.gpuPool.Preprocess(raw)
		if err != nil {
			writeError(w, http.StatusBadRequest, "could not decode/resize image on GPU (expected JPEG): "+err.Error())
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

// handleStats reports the timing breakdown used to locate the throughput
// bottleneck (Phase 4 diagnostics). GET /stats returns the cumulative averages;
// GET /stats?reset=1 zeroes the counters first (call between load levels to
// measure one level in isolation).
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
