package main

import (
	"sync"
	"testing"
	"time"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
)

// runRequests sends n copies of input through the engine from conc goroutines
// and fails the test on any error
func runRequests(t *testing.T, e *Engine, input []float32, n, conc int) {
	t.Helper()
	var wg sync.WaitGroup
	next := make(chan struct{}, n)
	for range n {
		next <- struct{}{}
	}
	close(next)
	errs := make(chan error, n)
	for range conc {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range next {
				if _, err := e.Predict(input); err != nil {
					errs <- err
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatalf("predict: %v", err)
	}
}

// waitForSpans waits until spans after seq cover want requests. Callers get
// their result before the unpack worker records the span, so it can lag a bit.
func waitForSpans(t *testing.T, e *Engine, seq int64, want int) []batchSpan {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		spans := e.tracer.since(seq)
		got := 0
		for _, s := range spans {
			got += s.N
		}
		if got >= want || time.Now().After(deadline) {
			if got != want {
				t.Fatalf("spans cover %d requests, want %d", got, want)
			}
			return spans
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// checkDashboardMatchesWork pushes a known number of requests through the
// engine and checks that what the dashboard samples and what the timeline
// draws both add back up to that work
func checkDashboardMatchesWork(t *testing.T, e *Engine, n, conc int) []batchSpan {
	t.Helper()
	dog := loadAndPreprocess(t, "testdata/dog.jpg")
	runRequests(t, e, dog, 8, 8) // warm up so the timed run is steady

	m := &sysMonitor{engine: e}
	sp := newSampler(m, nil)
	prev := m.counters()
	seq := e.tracer.newest()

	runRequests(t, e, dog, n, conc)
	spans := waitForSpans(t, e, seq, n)
	s := sp.step(time.Now())
	cur := m.counters()

	if got := cur.reqs - prev.reqs; got != int64(n) {
		t.Errorf("engine counted %d requests, sent %d", got, n)
	}
	if got := cur.batches - prev.batches; got != int64(len(spans)) {
		t.Errorf("engine counted %d batches, timeline has %d", got, len(spans))
	}
	near(t, "avg batch", s.AvgBatch, float64(n)/float64(len(spans)), 1e-9)

	var h2d, compute, d2h, post int64
	for _, b := range spans {
		if b.N < 1 || b.N > e.maxBatch {
			t.Errorf("batch %d has %d images, max is %d", b.Seq, b.N, e.maxBatch)
		}
		if b.Lane < 0 || b.Lane >= max(numInferLanes, 1) || b.Worker < 0 || b.Worker >= numUnpackers {
			t.Errorf("batch %d ran on lane %d worker %d", b.Seq, b.Lane, b.Worker)
		}
		order := []int64{b.Gather, b.Pack, b.Packed, b.Infer, b.H2DEnd, b.ComputeEnd, b.RunEnd, b.PostStart, b.PostEnd}
		for i := 1; i < len(order); i++ {
			if order[i] < order[i-1] {
				t.Errorf("batch %d stage %d at %d is before stage %d at %d", b.Seq, i, order[i], i-1, order[i-1])
				break
			}
		}
		h2d += b.H2DEnd - b.Infer
		compute += b.ComputeEnd - b.H2DEnd
		d2h += b.RunEnd - b.ComputeEnd
		post += b.PostEnd - b.PostStart
	}

	// compute, D2H and post come from the same clock readings as the
	// counters, so they match to the nanosecond
	if got := cur.computeNs - prev.computeNs; got != compute {
		t.Errorf("compute counter %d ns, spans sum to %d ns", got, compute)
	}
	if got := cur.d2hNs - prev.d2hNs; got != d2h {
		t.Errorf("D2H counter %d ns, spans sum to %d ns", got, d2h)
	}
	if got := cur.postNs - prev.postNs; got != post {
		t.Errorf("post counter %d ns, spans sum to %d ns", got, post)
	}
	// the span's H2D starts when the lane takes the batch, a few bookkeeping
	// calls before the copy clock starts, so it can only be a little longer
	if got := cur.h2dNs - prev.h2dNs; got > h2d || h2d-got > int64(len(spans))*int64(time.Millisecond)/10 {
		t.Errorf("H2D counter %d ns, spans sum to %d ns", got, h2d)
	}
	near(t, "compute ms per batch", s.ComputeMs, float64(compute)/float64(len(spans))/1e6, 1e-9)

	if s.LaneBusy <= 0 || s.LaneBusy > 100 {
		t.Errorf("lane busy %.1f%% out of range", s.LaneBusy)
	}
	if s.QueueMs <= 0 {
		t.Errorf("queue wait %.3f ms, want > 0", s.QueueMs)
	}
	t.Logf("%d requests in %d batches (avg %.2f), compute %.3f ms/batch, lane busy %.0f%%, queue %.2f ms",
		n, len(spans), s.AvgBatch, s.ComputeMs, s.LaneBusy, s.QueueMs)
	return spans
}

// On the plain path the copies happen inside session.Run, so the timeline
// shows no separate H2D or D2H and the whole run as compute.
func TestDashboardMatchesWorkPlain(t *testing.T) {
	e := newTestEngine(t, 8, 3*time.Millisecond)
	defer e.Close()
	spans := checkDashboardMatchesWork(t, e, 640, 32)
	for _, b := range spans {
		if b.Split || b.RunEnd != b.ComputeEnd {
			t.Fatalf("plain path batch %d reported split copies: %+v", b.Seq, b)
		}
	}
}

// With device binding full batches time H2D, compute and D2H separately and
// two lanes both carry work.
func TestDashboardMatchesWorkDeviceBinding(t *testing.T) {
	oldLanes, oldDev := numInferLanes, useDeviceBinding
	numInferLanes, useDeviceBinding = 2, true
	t.Cleanup(func() { numInferLanes, useDeviceBinding = oldLanes, oldDev })

	e := newTestEngine(t, 8, 5*time.Millisecond)
	defer e.Close()
	spans := checkDashboardMatchesWork(t, e, 1600, 64)

	split, lanes := 0, map[int]bool{}
	for _, b := range spans {
		lanes[b.Lane] = true
		if b.Split {
			split++
			if b.N != e.maxBatch {
				t.Errorf("batch %d of %d images took the device path, which is for full batches", b.Seq, b.N)
			}
			if b.H2DEnd <= b.Infer || b.RunEnd <= b.ComputeEnd {
				t.Errorf("split batch %d has an empty copy: %+v", b.Seq, b)
			}
		}
	}
	if split == 0 {
		t.Fatal("no batch took the device binding path")
	}
	if len(lanes) != 2 {
		t.Errorf("work ran on lanes %v, want both", lanes)
	}
	t.Logf("%d of %d batches split their copies", split, len(spans))
}

// NVML's reference clock must tick in microseconds for violationPct to be right
func TestNVMLReferenceClockMicroseconds(t *testing.T) {
	if testing.Short() {
		t.Skip("needs a GPU")
	}
	g, err := openGPU()
	if err != nil {
		t.Skipf("no NVML: %v", err)
	}
	a, r := g.dev.GetViolationStatus(nvml.PERF_POLICY_POWER)
	if r != nvml.SUCCESS {
		t.Skipf("violation status unsupported: %s", nvml.ErrorString(r))
	}
	start := time.Now()
	time.Sleep(500 * time.Millisecond)
	b, _ := g.dev.GetViolationStatus(nvml.PERF_POLICY_POWER)
	elapsedUs := float64(time.Since(start).Microseconds())
	near(t, "reference clock delta µs", float64(b.ReferenceTime-a.ReferenceTime), elapsedUs, elapsedUs*0.2)
}

// Under steady inference NVML should see the GPU working and the engine's
// memory resident
func TestGPUBusyUnderLoad(t *testing.T) {
	e := newTestEngine(t, 32, 3*time.Millisecond)
	defer e.Close()
	g, err := openGPU()
	if err != nil {
		t.Skipf("no NVML: %v", err)
	}
	dog := loadAndPreprocess(t, "testdata/dog.jpg")
	stop := make(chan struct{})
	var wg sync.WaitGroup
	for range 64 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
					e.Predict(dog)
				}
			}
		}()
	}
	time.Sleep(1500 * time.Millisecond)
	s := g.sample()
	close(stop)
	wg.Wait()

	if s.Util <= 0 || s.Util > 100 {
		t.Errorf("GPU busy %.0f%% under load, want 1..100", s.Util)
	}
	if s.MemTotalMB <= 0 || s.MemUsedMB <= 0 || s.MemUsedMB > s.MemTotalMB {
		t.Errorf("GPU memory %.0f of %.0f MB", s.MemUsedMB, s.MemTotalMB)
	}
	if s.PowerW <= 0 || s.PowerMaxW <= 0 {
		t.Errorf("power %.0f W of %.0f W limit", s.PowerW, s.PowerMaxW)
	}
	t.Logf("%s busy %.0f%%, %.0f MB used, %.0f W, PCIe in %.0f MB/s", s.Name, s.Util, s.MemUsedMB, s.PowerW, s.PCIeRxMBs)
}

// Batch staging and output buffers are recycled, so steady traffic should
// allocate kilobytes per request, not the megabytes a fresh batch buffer costs
func TestEngineAllocationPerRequest(t *testing.T) {
	e := newTestEngine(t, 8, 3*time.Millisecond)
	defer e.Close()
	dog := loadAndPreprocess(t, "testdata/dog.jpg")
	runRequests(t, e, dog, 640, 32) // fill the free lists

	const n = 640
	before := readRuntime()
	runRequests(t, e, dog, n, 32)
	after := readRuntime()
	var s sysSample
	fillRuntime(&s, before, after, n, 1)
	t.Logf("%.1f KB and %.0f objects per request", s.AllocBytesPerReq/1024, s.AllocsPerReq)
	if s.AllocBytesPerReq > 64<<10 {
		t.Errorf("%.0f KB allocated per request, want under 64 KB", s.AllocBytesPerReq/1024)
	}
}
