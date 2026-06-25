//go:build cpujpeg

package main

import (
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestCPUDecodeScaleSweep — single-thread libjpeg-turbo decode of one image at
// full / 1/2 / 1/4 / 1/8, reporting ms/decode and speedup. Confirms the SIMD
// decode cost and that scaling is only a modest (Huffman-bound) gain.
// Run: go test -tags cpujpeg -run TestCPUDecodeScaleSweep -v .
func TestCPUDecodeScaleSweep(t *testing.T) {
	jpeg, err := os.ReadFile("testdata/dog.jpg")
	if err != nil {
		t.Skip(err)
	}
	d, err := newJPEGDecoder()
	if err != nil {
		t.Fatalf("decoder: %v", err)
	}
	defer d.Close()

	const iters = 1000
	scales := []struct {
		name       string
		num, denom int
	}{
		{"full", 1, 1},
		{"1/2 ", 1, 2},
		{"1/4 ", 1, 4},
		{"1/8 ", 1, 8},
	}

	var fullMs float64
	for i, sc := range scales {
		if _, err := d.DecodeRGBAScale(jpeg, sc.num, sc.denom); err != nil { // warmup
			t.Fatalf("scale %s: %v", sc.name, err)
		}
		start := time.Now()
		var w, hh int
		for n := 0; n < iters; n++ {
			img, err := d.DecodeRGBAScale(jpeg, sc.num, sc.denom)
			if err != nil {
				t.Fatalf("scale %s: %v", sc.name, err)
			}
			w, hh = img.Rect.Dx(), img.Rect.Dy()
		}
		ms := float64(time.Since(start).Nanoseconds()) / float64(iters) / 1e6
		if i == 0 {
			fullMs = ms
		}
		t.Logf("scale=%s out=%dx%-4d  %.3f ms/decode  (%.0f decode/s single-thread)  speedup_vs_full=%.2fx",
			sc.name, w, hh, ms, 1000.0/ms, fullMs/ms)
	}
}

// TestCPUPreprocessParallel — the real in-server number: full Preprocess
// (decode@1/4 + resize + normalize) across N worker goroutines, measuring
// AGGREGATE req/s. This is what must clear ~2,100 (GPU decode-only ceiling) for
// Path A to win.
// Run: go test -tags cpujpeg -run TestCPUPreprocessParallel -v .
func TestCPUPreprocessParallel(t *testing.T) {
	dog, err := os.ReadFile("testdata/dog.jpg")
	if err != nil {
		t.Skip(err)
	}
	cat, err := os.ReadFile("testdata/cat.jpg")
	if err != nil {
		t.Skip(err)
	}
	imgs := [][]byte{dog, cat}

	for _, workers := range []int{4, 8, 12, 16, 24, 32} {
		pool, err := newCPUDecoderPool(workers)
		if err != nil {
			t.Fatalf("pool: %v", err)
		}
		total := int64(1000 * workers) // ~constant wall time per config
		var done int64
		var wg sync.WaitGroup
		start := time.Now()
		for wkr := 0; wkr < workers; wkr++ {
			wg.Add(1)
			go func() {
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
			}()
		}
		wg.Wait()
		el := time.Since(start)
		dec, norm, _ := pool.PreprocStats()
		t.Logf("workers=%-2d -> %.0f req/s  (%d in %s)  [decode %.2fms + resize/norm %.2fms per img]",
			workers, float64(total)/el.Seconds(), total, el.Round(time.Millisecond), dec, norm)
		pool.Close()
	}
}
