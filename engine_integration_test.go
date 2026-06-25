package main

import (
	"os"
	"sync"
	"testing"
	"time"
)

// Known-answer ImageNet indices for the Phase 1 correctness fixtures.
const (
	classSamoyed     = 258 // testdata/dog.jpg
	classEgyptianCat = 285 // testdata/cat.jpg
)

// loadAndPreprocess reads a test image and runs it through the real preprocessing
// pipeline (same path the HTTP handler uses).
func loadAndPreprocess(t *testing.T, path string) []float32 {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("fixture %s missing: %v", path, err)
	}
	defer f.Close()
	in, _, err := preprocessReader(f)
	if err != nil {
		t.Fatalf("preprocess %s: %v", path, err)
	}
	return in
}

// newTestEngine builds a real CUDA engine for integration tests. Requires the
// ONNX Runtime libraries on LD_LIBRARY_PATH (see test_phase3.sh). Skips in -short.
func newTestEngine(t *testing.T, maxBatch int, maxWait time.Duration) *Engine {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping GPU integration test in -short mode")
	}
	if _, err := os.Stat("resnet18.onnx"); err != nil {
		t.Skipf("model missing: %v", err)
	}
	labels, _ := loadLabels("imagenet_classes.txt")
	e, err := NewEngine("resnet18.onnx", "onnxruntime/lib/libonnxruntime.so", labels, maxBatch, maxWait)
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	if !e.OnGPU() {
		t.Logf("WARNING: session allocated only %d MiB — may be CPU fallback", e.GPUDeltaMiB())
	}
	return e
}

// TestMixedBatchNoSwap submits a dog and a cat concurrently into the SAME batch
// (large wait window forces them to be gathered together) and confirms each
// caller gets its own correct answer back — dog -> Samoyed, cat -> Egyptian cat,
// not swapped or corrupted. This is the core batching-correctness guarantee.
func TestMixedBatchNoSwap(t *testing.T) {
	// Long wait window + batch room => both requests land in one Run().
	e := newTestEngine(t, 8, 150*time.Millisecond)
	defer e.Close()

	dog := loadAndPreprocess(t, "testdata/dog.jpg")
	cat := loadAndPreprocess(t, "testdata/cat.jpg")

	type out struct {
		name string
		pred Prediction
		err  error
	}
	results := make([]out, 2)
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); p, err := e.Predict(dog); results[0] = out{"dog", p, err} }()
	go func() { defer wg.Done(); p, err := e.Predict(cat); results[1] = out{"cat", p, err} }()
	wg.Wait()

	for _, r := range results {
		if r.err != nil {
			t.Fatalf("%s: predict error: %v", r.name, r.err)
		}
	}
	if results[0].pred.ClassIndex != classSamoyed {
		t.Errorf("dog -> class %d (%s), want %d (Samoyed)", results[0].pred.ClassIndex, results[0].pred.ClassName, classSamoyed)
	}
	if results[1].pred.ClassIndex != classEgyptianCat {
		t.Errorf("cat -> class %d (%s), want %d (Egyptian cat)", results[1].pred.ClassIndex, results[1].pred.ClassName, classEgyptianCat)
	}
}

// TestKnownAnswersConcurrent re-runs the Phase 1 correctness check under load:
// many concurrent dog/cat requests that get batched together must each still
// return the right individual answer (no cross-contamination across a batch).
func TestKnownAnswersConcurrent(t *testing.T) {
	e := newTestEngine(t, 32, 5*time.Millisecond)
	defer e.Close()

	dog := loadAndPreprocess(t, "testdata/dog.jpg")
	cat := loadAndPreprocess(t, "testdata/cat.jpg")

	const each = 50
	type job struct {
		want  int
		input []float32
	}
	jobs := make([]job, 0, each*2)
	for i := 0; i < each; i++ {
		jobs = append(jobs, job{classSamoyed, dog}, job{classEgyptianCat, cat})
	}

	var wg sync.WaitGroup
	errs := make([]error, len(jobs))
	got := make([]int, len(jobs))
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

	for i := range jobs {
		if errs[i] != nil {
			t.Fatalf("job %d: %v", i, errs[i])
		}
		if got[i] != jobs[i].want {
			t.Errorf("job %d: got class %d, want %d", i, got[i], jobs[i].want)
		}
	}
}
