package main

import (
	"math"
	"runtime/metrics"
	"testing"
)

func TestHistQuantile(t *testing.T) {
	buckets := []float64{0, 1, 2, 3, math.Inf(1)}
	prev := &metrics.Float64Histogram{Buckets: buckets, Counts: []uint64{1, 0, 0, 0}}
	cur := &metrics.Float64Histogram{Buckets: buckets, Counts: []uint64{1, 90, 9, 1}}
	near(t, "p50", histQuantile(prev, cur, 0.50), 2, 0)
	near(t, "p99", histQuantile(prev, cur, 0.99), 3, 0)
	// the last bucket is open ended, so its lower edge is the best bound
	near(t, "p100", histQuantile(prev, cur, 1), 3, 0)
	near(t, "nothing new", histQuantile(cur, cur, 0.99), 0, 0)
	near(t, "no prev", histQuantile(nil, cur, 0.5), 2, 0)
	near(t, "no histogram", histQuantile(nil, nil, 0.5), 0, 0)
}

func TestFillRuntime(t *testing.T) {
	prev := runtimeCounters{allocBytes: 1000, allocObjects: 10, gcCycles: 5, gcCPU: 1.0, totalCPU: 10, idleCPU: 4}
	cur := runtimeCounters{allocBytes: 1000 + 100*4096, allocObjects: 10 + 100*3, gcCycles: 9, gcCPU: 1.3, totalCPU: 12, idleCPU: 4.5, heapLive: 64 << 20}
	var s sysSample
	fillRuntime(&s, prev, cur, 100, 2)
	near(t, "bytes per request", s.AllocBytesPerReq, 4096, 1e-9)
	near(t, "allocs per request", s.AllocsPerReq, 3, 1e-9)
	near(t, "GCs per second", s.GCPerSec, 2, 1e-9)
	// 0.3 s of GC out of 1.5 s of CPU the process actually used
	near(t, "GC share of CPU", s.GCCPUPct, 20, 1e-9)
	near(t, "live heap MB", s.HeapLiveMB, 64, 1e-9)

	var idle sysSample
	fillRuntime(&idle, cur, cur, 0, 1)
	if idle.AllocBytesPerReq != 0 || idle.GCPerSec != 0 || idle.GCCPUPct != 0 {
		t.Errorf("an idle interval should read zero, got %+v", idle)
	}
}

var sinkBytes []byte

// Allocating a known amount per fake request should read back as that amount.
// Small object counts reach the runtime metrics when per CPU caches flush, so
// they are checked over enough allocations for that lag to wash out.
func TestAllocPerRequestMeasured(t *testing.T) {
	const n, size = 200, 64 << 10
	before := readRuntime()
	for range n {
		sinkBytes = make([]byte, size)
	}
	after := readRuntime()
	var s sysSample
	fillRuntime(&s, before, after, n, 1)
	near(t, "bytes per request", s.AllocBytesPerReq, size, size*0.1)

	const small = 100_000
	before = readRuntime()
	for range small {
		sinkBytes = make([]byte, 16)
	}
	after = readRuntime()
	fillRuntime(&s, before, after, small, 1)
	near(t, "small allocs per request", s.AllocsPerReq, 1, 0.05)
}
