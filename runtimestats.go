// Go runtime allocation, GC and scheduler readings for the dashboard
package main

import (
	"math"
	"runtime/metrics"
)

var runtimeMetricNames = []string{
	"/gc/heap/allocs:bytes",
	"/gc/heap/allocs:objects",
	"/gc/cycles/total:gc-cycles",
	"/cpu/classes/gc/total:cpu-seconds",
	"/cpu/classes/total:cpu-seconds",
	"/cpu/classes/idle:cpu-seconds",
	"/gc/heap/live:bytes",
	"/sched/latencies:seconds",
	"/sched/pauses/total/gc:seconds",
}

// runtimeCounters is one snapshot of the cumulative runtime metrics
type runtimeCounters struct {
	allocBytes, allocObjects, gcCycles uint64
	gcCPU, totalCPU, idleCPU           float64
	heapLive                           uint64
	sched, gcPause                     *metrics.Float64Histogram
}

func readRuntime() runtimeCounters {
	s := make([]metrics.Sample, len(runtimeMetricNames))
	for i, n := range runtimeMetricNames {
		s[i].Name = n
	}
	metrics.Read(s)
	u := func(i int) uint64 {
		if s[i].Value.Kind() == metrics.KindUint64 {
			return s[i].Value.Uint64()
		}
		return 0
	}
	f := func(i int) float64 {
		if s[i].Value.Kind() == metrics.KindFloat64 {
			return s[i].Value.Float64()
		}
		return 0
	}
	h := func(i int) *metrics.Float64Histogram {
		if s[i].Value.Kind() == metrics.KindFloat64Histogram {
			return s[i].Value.Float64Histogram()
		}
		return nil
	}
	return runtimeCounters{
		allocBytes: u(0), allocObjects: u(1), gcCycles: u(2),
		gcCPU: f(3), totalCPU: f(4), idleCPU: f(5),
		heapLive: u(6), sched: h(7), gcPause: h(8),
	}
}

// histQuantile returns the q quantile of the observations added between two
// snapshots of a cumulative histogram, as the upper edge of its bucket
func histQuantile(prev, cur *metrics.Float64Histogram, q float64) float64 {
	if cur == nil {
		return 0
	}
	counts := make([]uint64, len(cur.Counts))
	var total uint64
	for i, c := range cur.Counts {
		if prev != nil && i < len(prev.Counts) && prev.Counts[i] <= c {
			c -= prev.Counts[i]
		}
		counts[i] = c
		total += c
	}
	if total == 0 {
		return 0
	}
	target := uint64(math.Ceil(q * float64(total)))
	var run uint64
	for i, c := range counts {
		run += c
		if run >= target {
			if hi := cur.Buckets[i+1]; !math.IsInf(hi, 1) {
				return hi
			}
			return cur.Buckets[i]
		}
	}
	return 0
}

// fillRuntime sets the allocation, GC and scheduler fields of s from two
// snapshots dt seconds apart during which reqs requests were served
func fillRuntime(s *sysSample, prev, cur runtimeCounters, reqs, dt float64) {
	d := func(a, b uint64) float64 {
		if b < a {
			return 0
		}
		return float64(b - a)
	}
	if reqs > 0 {
		s.AllocBytesPerReq = d(prev.allocBytes, cur.allocBytes) / reqs
		s.AllocsPerReq = d(prev.allocObjects, cur.allocObjects) / reqs
	}
	if dt > 0 {
		s.GCPerSec = d(prev.gcCycles, cur.gcCycles) / dt
	}
	if used := (cur.totalCPU - prev.totalCPU) - (cur.idleCPU - prev.idleCPU); used > 0 {
		s.GCCPUPct = min(100, 100*max(cur.gcCPU-prev.gcCPU, 0)/used)
	}
	s.HeapLiveMB = float64(cur.heapLive) / (1 << 20)
	s.GCPauseP99Us = histQuantile(prev.gcPause, cur.gcPause, 0.99) * 1e6
	s.SchedP99Us = histQuantile(prev.sched, cur.sched, 0.99) * 1e6
}
