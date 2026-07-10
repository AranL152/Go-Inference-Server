package main

import (
	"sync"
	"testing"
	"time"
)

// TestPipelineRoutingStressMultiLane is TestPipelineRoutingStress with TWO
// concurrent infer lanes sharing the one session (Phase 8, -infer-lanes 2).
// The new hazard vs the single-lane test: batches complete out of order across
// lanes, so any hidden ordering assumption in the gather→infer→unpack handoffs
// would surface here as swapped answers or a double-close panic at shutdown.
func TestPipelineRoutingStressMultiLane(t *testing.T) {
	if testing.Short() {
		t.Skip("skip GPU test in -short")
	}
	old := numInferLanes
	numInferLanes = 2
	t.Cleanup(func() { numInferLanes = old })

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
		t.Errorf("%d/%d requests got the wrong answer with 2 infer lanes (cross-lane routing bug)", swaps, n)
	}
	_, _, avg := e.BatchStats()
	t.Logf("routed %d mixed requests across 2 lanes with 0 swaps; avg batch size %.1f", n, avg)
}
