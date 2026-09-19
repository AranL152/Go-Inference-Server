package main

import (
	"sync"
	"testing"
	"time"
)

// TestPipelineRoutingStressIoBinding is the multi-lane routing stress with
// -io-binding on: full batches run through pre-bound slots
// that are REUSED across batches and recycled by unpack workers, so a slot
// returned to the pool too early (or bound outputs read after reuse) would
// surface here as swapped answers. Small maxBatch + many concurrent requests
// keeps most batches full so the binding path dominates.
func TestPipelineRoutingStressIoBinding(t *testing.T) {
	if testing.Short() {
		t.Skip("skip GPU test in -short")
	}
	oldLanes, oldBind := numInferLanes, useIoBinding
	numInferLanes = 2
	useIoBinding = true
	t.Cleanup(func() { numInferLanes, useIoBinding = oldLanes, oldBind })

	e := newTestEngine(t, 16, 3*time.Millisecond)
	defer e.Close()

	dog := loadAndPreprocess(t, "testdata/dog.jpg")
	cat := loadAndPreprocess(t, "testdata/cat.jpg")

	const n = 1000
	type job struct {
		want  int
		input []float32
	}
	jobs := make([]job, n)
	for i := 0; i < n; i++ {
		if i%2 == 0 {
			jobs[i] = job{classSamoyed, dog}
		} else {
			jobs[i] = job{classEgyptianCat, cat}
		}
	}

	got := make([]int, n)
	errs := make([]error, n)
	var wg sync.WaitGroup
	for i := range jobs {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			p, err := e.Predict(jobs[i].input)
			errs[i] = err
			got[i] = p.ClassIndex
		}(i)
	}
	wg.Wait()

	swaps := 0
	for i := range jobs {
		if errs[i] != nil {
			t.Fatalf("job %d: %v", i, errs[i])
		}
		if got[i] != jobs[i].want {
			swaps++
		}
	}
	if swaps > 0 {
		t.Errorf("%d/%d requests got the wrong answer with io-binding slots (slot recycling bug)", swaps, n)
	}
	_, _, avg := e.BatchStats()
	t.Logf("routed %d mixed requests through bound slots with 0 swaps; avg batch size %.1f", n, avg)
}

// TestPipelineRoutingStressDeviceBinding is the same stress with
// -device-binding on: full batches go through engine-owned
// CUDA device buffers with pinned-host staging and engine-driven H2D/D2H.
// Wrong answers here would mean copy/compute ordering bugs or slot recycling
// races on the device path.
func TestPipelineRoutingStressDeviceBinding(t *testing.T) {
	if testing.Short() {
		t.Skip("skip GPU test in -short")
	}
	oldLanes, oldDev := numInferLanes, useDeviceBinding
	numInferLanes = 2
	useDeviceBinding = true
	t.Cleanup(func() { numInferLanes, useDeviceBinding = oldLanes, oldDev })

	e := newTestEngine(t, 16, 3*time.Millisecond)
	defer e.Close()

	dog := loadAndPreprocess(t, "testdata/dog.jpg")
	cat := loadAndPreprocess(t, "testdata/cat.jpg")

	const n = 1000
	type job struct {
		want  int
		input []float32
	}
	jobs := make([]job, n)
	for i := 0; i < n; i++ {
		if i%2 == 0 {
			jobs[i] = job{classSamoyed, dog}
		} else {
			jobs[i] = job{classEgyptianCat, cat}
		}
	}

	got := make([]int, n)
	errs := make([]error, n)
	var wg sync.WaitGroup
	for i := range jobs {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			p, err := e.Predict(jobs[i].input)
			errs[i] = err
			got[i] = p.ClassIndex
		}(i)
	}
	wg.Wait()

	swaps := 0
	for i := range jobs {
		if errs[i] != nil {
			t.Fatalf("job %d: %v", i, errs[i])
		}
		if got[i] != jobs[i].want {
			swaps++
		}
	}
	if swaps > 0 {
		t.Errorf("%d/%d wrong answers with device-resident binding (copy ordering or slot race)", swaps, n)
	}
	_, _, avg := e.BatchStats()
	t.Logf("routed %d mixed requests through device slots with 0 swaps; avg batch size %.1f", n, avg)
}
