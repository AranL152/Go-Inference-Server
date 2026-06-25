package main

import (
	"sync"
	"testing"
	"time"
)

// TestPipelineRoutingStress hammers the Phase 5 pipeline with many concurrent
// mixed dog/cat requests at a small batch size, so batches form constantly and
// the parallel unpack workers are busy. Every caller must get ITS image's answer
// back — a routing bug introduced by the gather→infer→unpack handoffs would show
// up as a cat returning Samoyed (or vice versa) here.
func TestPipelineRoutingStress(t *testing.T) {
	if testing.Short() {
		t.Skip("skip GPU test in -short")
	}
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
		t.Errorf("%d/%d requests got the wrong answer (routing bug in the pipeline)", swaps, n)
	}
	_, _, avg := e.BatchStats()
	t.Logf("routed %d mixed requests with 0 swaps; avg batch size %.1f", n, avg)
}
