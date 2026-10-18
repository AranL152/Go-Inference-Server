// Per-batch pipeline timings
package main

import "sync"

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
