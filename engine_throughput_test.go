package main

import (
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestEngineThroughputPreprocessedConcurrent isolates the batching engine from the
// CPU preprocessing stage: it preprocesses ONE image up front, then fires many
// concurrent Predict() calls reusing that tensor. If the engine's batching path
// has headroom (and preprocessing is the real bottleneck), throughput here should
// far exceed the end-to-end ~345 req/s, and realized batch size should be large.
func TestEngineThroughputPreprocessedConcurrent(t *testing.T) {
	if testing.Short() {
		t.Skip("skip GPU test in -short")
	}
	if _, err := os.Stat("resnet18.onnx"); err != nil {
		t.Skipf("model missing: %v", err)
	}
	labels, _ := loadLabels("imagenet_classes.txt")
	e, err := NewEngine("resnet18.onnx", "onnxruntime/lib/libonnxruntime.so", labels, 32, 5*time.Millisecond)
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	defer e.Close()

	f, _ := os.Open("testdata/dog.jpg")
	defer f.Close()
	in, _, err := preprocessReader(f)
	if err != nil {
		t.Fatalf("preprocess: %v", err)
	}

	for _, conc := range []int{1, 16, 64, 128, 256} {
		const total = 4000
		var done int64
		var wg sync.WaitGroup
		start := time.Now()
		startBatches, startReqs, _ := e.BatchStats()
		for w := 0; w < conc; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for atomic.AddInt64(&done, 1) <= total {
					if _, err := e.Predict(in); err != nil {
						t.Errorf("predict: %v", err)
						return
					}
				}
			}()
		}
		wg.Wait()
		elapsed := time.Since(start)
		endBatches, endReqs, _ := e.BatchStats()
		nb := endBatches - startBatches
		nr := endReqs - startReqs
		avg := 0.0
		if nb > 0 {
			avg = float64(nr) / float64(nb)
		}
		t.Logf("conc=%-3d  %.0f req/s  avg_batch=%.1f  (%d reqs in %d batches, %s)",
			conc, float64(total)/elapsed.Seconds(), avg, nr, nb, elapsed.Round(time.Millisecond))
	}
}
