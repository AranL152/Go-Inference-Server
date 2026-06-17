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
	flag.Parse()

	labels, err := loadLabels(*labelsPath)
	if err != nil {
		log.Printf("WARNING: could not load labels from %q (%v) — responses will use numeric class index only", *labelsPath, err)
	} else {
		log.Printf("loaded %d class labels from %s", len(labels), *labelsPath)
	}

	log.Printf("loading model %s onto GPU via CUDA execution provider...", *modelPath)
	engine, err := NewEngine(*modelPath, *libPath, labels)
	if err != nil {
		log.Fatalf("FATAL: could not initialize inference engine: %v", err)
	}
	defer engine.Close()

	// Prove GPU placement the same way Phase 0 did: session creation must have
	// allocated GPU memory. ORT silently falls back to CPU otherwise.
	if engine.OnGPU() {
		log.Printf("=== model loaded on GPU (allocated %+d MiB; compute-app PID listed: %v) ===",
			engine.GPUDeltaMiB(), processOnGPU())
	} else {
		log.Printf("WARNING: session allocated only %+d MiB of GPU memory — ORT may have fallen back to CPU",
			engine.GPUDeltaMiB())
	}

	srv := &server{engine: engine}
	mux := http.NewServeMux()
	mux.HandleFunc("/predict", srv.handlePredict)
	mux.HandleFunc("/health", srv.handleHealth)

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
	engine *Engine
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

	body, err := imageBody(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	defer body.Close()

	input, format, err := preprocessReader(body)
	if err != nil {
		// Covers malformed images and non-image / wrong-content-type bodies:
		// the decoder can't identify the bytes as JPEG/PNG.
		writeError(w, http.StatusBadRequest, "could not decode image (expected JPEG or PNG): "+err.Error())
		return
	}

	pred, err := s.engine.Predict(input)
	if err != nil {
		log.Printf("inference error: %v", err)
		writeError(w, http.StatusInternalServerError, "inference failed")
		return
	}

	log.Printf("predict: format=%s -> class=%d (%s) conf=%.4f", format, pred.ClassIndex, pred.ClassName, pred.Confidence)
	writeJSON(w, http.StatusOK, predictResponse{
		ClassIndex: pred.ClassIndex,
		ClassName:  pred.ClassName,
		Confidence: pred.Confidence,
	})
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
