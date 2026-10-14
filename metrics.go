// Prometheus metrics endpoint.
//
// GET /metrics exposes the counters the server already keeps (engine
// batching/timing atomics, decoder-pool timings, Go runtime stats) in the
// Prometheus text exposition format, plus a request-latency histogram for
// /predict. Hand-rolled rather than pulling in the prometheus client: the
// endpoint is a page of code and the server has no other dependencies.
//
// Notes:
//   - Everything cumulative is exported as a counter (_total, _seconds_total)
//     so Prometheus can rate() it; /stats keeps serving the human-readable
//     averages. /stats?reset=1 zeroes the engine counters, which Prometheus
//     sees as an ordinary counter reset.
//   - The histogram buckets span 1 ms to 2.5 s: p50 at saturation is ~10 ms,
//     and overload latencies before backpressure reached seconds.
package main

import (
	"fmt"
	"net/http"
	"runtime"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// latencyBuckets are the histogram upper bounds in seconds (+Inf is implicit).
var latencyBuckets = []float64{0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5}

// latencyHistogram is a fixed-bucket histogram with atomic counters, safe for
// concurrent Observe from every handler goroutine. Bucket counts are stored
// per-bucket and accumulated into Prometheus's cumulative form at scrape time.
type latencyHistogram struct {
	buckets []int64 // len(latencyBuckets)+1 slots; last = +Inf overflow
	count   int64
	sumNs   int64
}

func (h *latencyHistogram) Observe(d time.Duration) {
	sec := d.Seconds()
	i := sort.SearchFloat64s(latencyBuckets, sec) // first bucket with bound >= sec
	atomic.AddInt64(&h.buckets[i], 1)
	atomic.AddInt64(&h.count, 1)
	atomic.AddInt64(&h.sumNs, int64(d))
}

// requestMetrics instruments /predict: one histogram plus per-status-code
// request counters (populated lazily — only codes actually returned appear).
type requestMetrics struct {
	latency latencyHistogram
	byCode  sync.Map // int status code -> *int64
}

var predictMetrics = requestMetrics{
	latency: latencyHistogram{buckets: make([]int64, len(latencyBuckets)+1)},
}

func (m *requestMetrics) record(code int, d time.Duration) {
	m.latency.Observe(d)
	c, ok := m.byCode.Load(code)
	if !ok {
		c, _ = m.byCode.LoadOrStore(code, new(int64))
	}
	atomic.AddInt64(c.(*int64), 1)
}

// statusRecorder captures the status code a handler writes so record can label
// the request counter. Defaults to 200 for handlers that never call WriteHeader.
type statusRecorder struct {
	http.ResponseWriter
	code int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.code = code
	r.ResponseWriter.WriteHeader(code)
}

// instrument wraps a handler with latency + status-code recording.
func instrument(m *requestMetrics, h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		rec := &statusRecorder{ResponseWriter: w, code: http.StatusOK}
		start := time.Now()
		h(rec, r)
		d := time.Since(start)
		m.record(rec.code, d)
		recentLatency.add(d)
	}
}

// handleMetrics serves the Prometheus text exposition format (version 0.0.4).
func (s *server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")

	batches, reqs, rejects, waitNs, runNs, runBatchNs, busyNs, idleNs := s.engine.CumulativeStats()

	counter := func(name, help string, v float64) {
		fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s counter\n%s %g\n", name, help, name, name, v)
	}
	gauge := func(name, help string, v float64) {
		fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s gauge\n%s %g\n", name, help, name, name, v)
	}

	// Engine pipeline counters.
	counter("inference_requests_total", "Requests that completed the batching pipeline.", float64(reqs))
	counter("inference_batches_total", "Dynamic batches executed.", float64(batches))
	counter("inference_rejected_total", "Requests shed by -max-inflight backpressure (HTTP 503).", float64(rejects))
	counter("inference_queue_wait_seconds_total", "Sum of per-request queue wait (submit to batch start).", float64(waitNs)/1e9)
	counter("inference_run_seconds_total", "Sum of per-batch ORT session.Run time.", float64(runNs)/1e9)
	counter("inference_batch_cycle_seconds_total", "Sum of per-batch infer-lane processing (Run + handoff).", float64(runBatchNs)/1e9)
	counter("inference_lane_busy_seconds_total", "Infer-lane wall-clock spent processing batches.", float64(busyNs)/1e9)
	counter("inference_lane_idle_seconds_total", "Infer-lane wall-clock spent waiting for a batch.", float64(idleNs)/1e9)

	// Decoder-pool preprocess timings (whichever pools are active). The pools
	// track avg + count; reconstruct the cumulative sums Prometheus wants.
	if s.cpuPool != nil {
		d, n, cnt := s.cpuPool.PreprocStats()
		counter("preprocess_cpu_images_total", "Images decoded by the libjpeg-turbo pool.", float64(cnt))
		counter("preprocess_cpu_decode_seconds_total", "Sum of CPU JPEG decode time.", d*float64(cnt)/1e3)
		counter("preprocess_cpu_normalize_seconds_total", "Sum of CPU resize+normalize time.", n*float64(cnt)/1e3)
	}
	if s.gpuPool != nil {
		d, n, cnt := s.gpuPool.PreprocStats()
		counter("preprocess_gpu_images_total", "Images decoded by the nvJPEG pool.", float64(cnt))
		counter("preprocess_gpu_decode_seconds_total", "Sum of GPU decode+resize time.", d*float64(cnt)/1e3)
		counter("preprocess_gpu_normalize_seconds_total", "Sum of CPU normalize time on the GPU path.", n*float64(cnt)/1e3)
	}
	if s.hybrid != nil {
		c, g := s.hybrid.Split()
		counter("preprocess_hybrid_cpu_total", "Hybrid-path decodes taken by the CPU pool.", float64(c))
		counter("preprocess_hybrid_gpu_total", "Hybrid-path decodes spilled to the GPU pool.", float64(g))
	}

	// HTTP layer: /predict status codes + latency histogram.
	fmt.Fprintf(w, "# HELP http_requests_total /predict requests by status code.\n# TYPE http_requests_total counter\n")
	codes := []int{}
	predictMetrics.byCode.Range(func(k, _ any) bool { codes = append(codes, k.(int)); return true })
	sort.Ints(codes)
	for _, code := range codes {
		v, _ := predictMetrics.byCode.Load(code)
		fmt.Fprintf(w, "http_requests_total{path=\"/predict\",code=\"%d\"} %d\n", code, atomic.LoadInt64(v.(*int64)))
	}

	fmt.Fprintf(w, "# HELP http_request_duration_seconds /predict latency.\n# TYPE http_request_duration_seconds histogram\n")
	cum := int64(0)
	for i, bound := range latencyBuckets {
		cum += atomic.LoadInt64(&predictMetrics.latency.buckets[i])
		fmt.Fprintf(w, "http_request_duration_seconds_bucket{path=\"/predict\",le=\"%s\"} %d\n",
			strconv.FormatFloat(bound, 'g', -1, 64), cum)
	}
	count := atomic.LoadInt64(&predictMetrics.latency.count)
	fmt.Fprintf(w, "http_request_duration_seconds_bucket{path=\"/predict\",le=\"+Inf\"} %d\n", count)
	fmt.Fprintf(w, "http_request_duration_seconds_sum{path=\"/predict\"} %g\n", float64(atomic.LoadInt64(&predictMetrics.latency.sumNs))/1e9)
	fmt.Fprintf(w, "http_request_duration_seconds_count{path=\"/predict\"} %d\n", count)

	// Go runtime; GC pressure is a first-class signal here.
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	gauge("go_goroutines", "Current number of goroutines.", float64(runtime.NumGoroutine()))
	gauge("go_memstats_heap_alloc_bytes", "Bytes of allocated heap objects.", float64(ms.HeapAlloc))
	counter("go_memstats_alloc_bytes_total", "Cumulative bytes allocated for heap objects.", float64(ms.TotalAlloc))
	counter("go_gc_cycles_total", "Completed GC cycles.", float64(ms.NumGC))
}
