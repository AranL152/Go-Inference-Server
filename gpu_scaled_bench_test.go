package main

import (
	"os"
	"testing"
)

// TestScaledDecodeGate decodes one real image at scale NONE / 1/2 / 1/4 / 1/8
// (decode only, synced each time) and prints ms/decode.
//
// Only ~256px of a ~1546px source is needed, so 1/4 is plenty:
//   - 1/4 materially faster than NONE: scaled GPU decode is worth pooling.
//   - 1/4 about equal to NONE: decode is Huffman-bound and GPU scaling cannot
//     help.
//
// Standalone (creates its own nvJPEG objects). Run with ORT_PROBE=1.
func TestScaledDecodeGate(t *testing.T) {
	if os.Getenv("ORT_PROBE") == "" {
		t.Skip("standalone gate; run with ORT_PROBE=1")
	}
	jpeg, err := os.ReadFile("testdata/dog.jpg")
	if err != nil {
		t.Skip(err)
	}

	backend := backendGPUHybrid
	if ok, _ := gpuBackendAvailable(backendHardware); ok {
		backend = backendHardware
	}
	t.Logf("backend=%s  source=%d bytes", backend, len(jpeg))

	const iters = 500
	scales := []struct {
		name  string
		scale int
	}{
		{"NONE", 0},
		{"1/2 ", 1},
		{"1/4 ", 2},
		{"1/8 ", 3},
	}

	var noneMs float64
	for i, sc := range scales {
		ms, w, h, err := scaledDecodeBench(jpeg, backend, sc.scale, iters)
		if err != nil {
			// nvjpegDecodeParamsSetScaleFactor "works only with the hardware
			// decoder backend" (nvjpeg.h). This card has no NVJPG engine, so any
			// scale != NONE fails with INVALID_PARAMETER. That IS the gate result:
			// scaled GPU decode is unavailable here -> pivot lever #1 to CPU.
			t.Logf("scale=%s UNSUPPORTED on backend %s: %v", sc.name, backend, err)
			continue
		}
		if i == 0 {
			noneMs = ms
		}
		speedup := noneMs / ms
		t.Logf("scale=%s out=%dx%-4d  %.3f ms/decode  (%.0f decode/s)  speedup_vs_NONE=%.2fx",
			sc.name, w, h, ms, 1000.0/ms, speedup)
	}
}
