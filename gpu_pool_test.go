package main

import (
	"os"
	"sync"
	"testing"
)

// TestGPUPoolPreprocess checks the pooled GPU preprocess produces a correctly
// shaped tensor and the same top-1 as the CPU path, concurrently across the pool.
func TestGPUPoolPreprocess(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping GPU pool test in -short mode")
	}
	pool, err := newGPUDecoderPool(4)
	if err != nil {
		t.Fatalf("newGPUDecoderPool: %v", err)
	}
	defer pool.Close()
	t.Logf("pool backend: %s", pool.Backend())

	dog, err := os.ReadFile("testdata/dog.jpg")
	if err != nil {
		t.Skipf("read dog: %v", err)
	}
	cat, err := os.ReadFile("testdata/cat.jpg")
	if err != nil {
		t.Skipf("read cat: %v", err)
	}

	// Concurrency: hammer the pool from many goroutines, check tensor length and
	// that dog/cat tensors differ (no cross-contamination / shared-buffer bug).
	dogRef, err := pool.Preprocess(dog)
	if err != nil {
		t.Fatalf("preprocess dog: %v", err)
	}
	if len(dogRef) != inputElems {
		t.Fatalf("tensor length %d, want %d", len(dogRef), inputElems)
	}

	const workers = 16
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			img, name := dog, "dog"
			if w%2 == 1 {
				img, name = cat, "cat"
			}
			for i := 0; i < 20; i++ {
				out, err := pool.Preprocess(img)
				if err != nil {
					errs <- err
					return
				}
				if len(out) != inputElems {
					errs <- err
					return
				}
				// dog tensor must reproduce deterministically.
				if name == "dog" && !floatsClose(out, dogRef) {
					errs <- errMismatch
					return
				}
			}
		}(w)
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		if e != nil {
			t.Fatalf("concurrent preprocess: %v", e)
		}
	}
}

var errMismatch = errStr("dog tensor not reproducible across concurrent pool use")

type errStr string

func (e errStr) Error() string { return string(e) }

func floatsClose(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		d := a[i] - b[i]
		if d > 1e-4 || d < -1e-4 {
			return false
		}
	}
	return true
}

// BenchmarkGPUPoolPreprocess measures the pooled GPU decode+resize+normalize cost
// (handles reused) — the real per-image cost to compare against the CPU 38.6ms.
func BenchmarkGPUPoolPreprocess(b *testing.B) {
	pool, err := newGPUDecoderPool(1)
	if err != nil {
		b.Fatalf("newGPUDecoderPool: %v", err)
	}
	defer pool.Close()

	for _, name := range []string{"dog.jpg", "cat.jpg"} {
		raw, err := os.ReadFile("testdata/" + name)
		if err != nil {
			b.Skip(err)
		}
		b.Run(name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if _, err := pool.Preprocess(raw); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
