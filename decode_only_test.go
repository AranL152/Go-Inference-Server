package main

import (
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestDecodeOnlyThroughput measures GPU decode+resize+normalize throughput with NO
// inference, at concurrency matching the decoder pool. Combined with the probe's
// inference-only ~5,885 req/s, this decides whether decode and inference serialize
// on the GPU: if 1/combined ≈ 1/decode_only + 1/infer_only, they're fully serial.
// Run with ORT_PROBE=1 (uses its own pool; standalone).
func TestDecodeOnlyThroughput(t *testing.T) {
	if os.Getenv("ORT_PROBE") == "" {
		t.Skip("standalone; run with ORT_PROBE=1")
	}
	dog, err := os.ReadFile("testdata/dog.jpg")
	if err != nil {
		t.Skip(err)
	}
	cat, err := os.ReadFile("testdata/cat.jpg")
	if err != nil {
		t.Skip(err)
	}
	imgs := [][]byte{dog, cat}

	for _, poolSize := range []int{8, 16, 24, 32, 48} {
		pool, err := newGPUDecoderPool(poolSize)
		if err != nil {
			t.Fatalf("pool: %v", err)
		}
		// concurrency == pool size (more would just block on checkout).
		conc := poolSize
		const total = 12000
		var done int64
		var wg sync.WaitGroup
		start := time.Now()
		for w := 0; w < conc; w++ {
			wg.Add(1)
			go func(w int) {
				defer wg.Done()
				for {
					n := atomic.AddInt64(&done, 1)
					if n > total {
						return
					}
					if _, err := pool.Preprocess(imgs[int(n)%2]); err != nil {
						t.Errorf("preprocess: %v", err)
						return
					}
				}
			}(w)
		}
		wg.Wait()
		el := time.Since(start)
		t.Logf("decode-only pool=%-2d conc=%-2d -> %.0f req/s  (%d imgs in %s)",
			poolSize, conc, float64(total)/el.Seconds(), total, el.Round(time.Millisecond))
		pool.Close()
	}
}
