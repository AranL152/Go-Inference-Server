// Per-batch pipeline timings and how well batches fill
package main

import (
	"math"
	"net/http"
	"slices"
	"sync"
	"time"
)

const traceHistory = 4096

// batchSpan is one batch's trip through the pipeline. Times are nanoseconds
// since the engine started, on the monotonic clock. Lanes that do not split
// the copies out of the run leave H2DEnd at Infer and ComputeEnd at RunEnd, so
// their whole run shows as compute with the copies inside it.
type batchSpan struct {
	Seq    int64 `json:"seq"`
	N      int   `json:"n"`
	Lane   int   `json:"lane"`
	Worker int   `json:"worker"`
	Split  bool  `json:"split"` // H2D, compute and D2H timed separately

	Gather     int64 `json:"gather"`      // first request taken off the queue
	Pack       int64 `json:"pack"`        // batch closed, copy into the staging buffer starts
	Packed     int64 `json:"packed"`      // staging copy done, waiting for a free lane
	Infer      int64 `json:"infer"`       // a lane picked the batch up
	H2DEnd     int64 `json:"h2d_end"`     // input copied to the GPU
	ComputeEnd int64 `json:"compute_end"` // model finished
	RunEnd     int64 `json:"run_end"`     // output copied back to the host
	PostStart  int64 `json:"post_start"`  // an unpack worker picked the batch up
	PostEnd    int64 `json:"post_end"`    // results split and handed to callers
}

// batchTracer keeps the most recent finished spans in a ring
type batchTracer struct {
	mu   sync.Mutex
	ring []batchSpan
	next int64 // sequence number of the next span added
}

func newBatchTracer(size int) *batchTracer {
	return &batchTracer{ring: make([]batchSpan, size)}
}

func (t *batchTracer) add(s batchSpan) {
	t.mu.Lock()
	s.Seq = t.next
	t.ring[t.next%int64(len(t.ring))] = s
	t.next++
	t.mu.Unlock()
}

// newest is the sequence number of the latest span, or -1 before the first
func (t *batchTracer) newest() int64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.next - 1
}

// since returns the spans with a sequence number above seq that are still in
// the ring, oldest first. Pass -1 for everything held.
func (t *batchTracer) since(seq int64) []batchSpan {
	t.mu.Lock()
	defer t.mu.Unlock()
	from := max(seq+1, t.next-int64(len(t.ring)), 0)
	out := make([]batchSpan, 0, t.next-min(from, t.next))
	for i := from; i < t.next; i++ {
		out = append(out, t.ring[i%int64(len(t.ring))])
	}
	return out
}

// batchingStats describes how well dynamic batching is filling batches
type batchingStats struct {
	MaxBatch    int     `json:"max_batch"`
	MaxWaitMs   float64 `json:"max_wait_ms"`
	Batches     int     `json:"batches"`
	Sizes       []int   `json:"sizes"` // batches of each size, index 0 holds size 1
	AvgSize     float64 `json:"avg_size"`
	FullPct     float64 `json:"full_pct"`
	FillP50Ms   float64 `json:"fill_p50_ms"` // first request in to batch closed
	FillP99Ms   float64 `json:"fill_p99_ms"`
	LaneWaitP50 float64 `json:"lane_wait_p50_ms"` // packed batch waiting for a free lane
	LaneWaitP99 float64 `json:"lane_wait_p99_ms"`
}

// summarizeBatches describes the spans that reached a lane at or after from
func summarizeBatches(spans []batchSpan, maxBatch int, maxWait time.Duration, from int64) batchingStats {
	st := batchingStats{MaxBatch: maxBatch, MaxWaitMs: float64(maxWait) / 1e6, Sizes: make([]int, maxBatch)}
	var fill, laneWait []float64
	images := 0
	for _, s := range spans {
		if s.Infer < from || s.N < 1 || s.N > maxBatch {
			continue
		}
		st.Batches++
		st.Sizes[s.N-1]++
		images += s.N
		fill = append(fill, float64(s.Pack-s.Gather)/1e6)
		laneWait = append(laneWait, float64(s.Infer-s.Packed)/1e6)
	}
	if st.Batches == 0 {
		return st
	}
	st.AvgSize = float64(images) / float64(st.Batches)
	st.FullPct = 100 * float64(st.Sizes[maxBatch-1]) / float64(st.Batches)
	rank := func(v []float64, q float64) float64 {
		slices.Sort(v)
		return v[max(int(math.Ceil(q*float64(len(v))))-1, 0)]
	}
	st.FillP50Ms, st.FillP99Ms = rank(fill, 0.5), rank(fill, 0.99)
	st.LaneWaitP50, st.LaneWaitP99 = rank(laneWait, 0.5), rank(laneWait, 0.99)
	return st
}

const batchingWindow = 10 * time.Second

func (e *Engine) handleBatching(w http.ResponseWriter, r *http.Request) {
	from := e.now() - int64(batchingWindow)
	writeJSON(w, http.StatusOK, summarizeBatches(e.tracer.since(-1), e.maxBatch, e.maxWait, from))
}
