package main

import (
	"bytes"
	"image"
	"math"
	"os"
	"testing"
	"time"
)

// TestNvjpegBackends reports which nvJPEG backends this GPU provides. The headline
// Phase 4 question: is the dedicated HARDWARE (NVJPG) engine available — decode on
// separate silicon, no contention with inference — or only GPU_HYBRID (CUDA cores)?
func TestNvjpegBackends(t *testing.T) {
	for _, b := range []nvjpegBackend{backendDefault, backendHybrid, backendGPUHybrid, backendHardware} {
		ok, rc := gpuBackendAvailable(b)
		status := "UNAVAILABLE"
		if ok {
			status = "available"
		}
		t.Logf("nvJPEG backend %-24s : %s (rc=%d)", b.String(), status, rc)
	}
	hw, _ := gpuBackendAvailable(backendHardware)
	if hw {
		t.Logf(">>> RTX 4080 SUPER HAS the dedicated NVJPG hardware decode engine — decode runs off the CUDA cores.")
	} else {
		t.Logf(">>> No dedicated NVJPG engine — decode uses GPU_HYBRID (CUDA cores, shares SMs with inference).")
	}
}

// TestGPUDecodeParity decodes the test images on the GPU and checks the pixels
// match Go's image/jpeg decode within tolerance (different IDCT/color-convert
// rounding means we expect small per-pixel differences, not bit equality).
func TestGPUDecodeParity(t *testing.T) {
	for _, name := range []string{"dog.jpg", "cat.jpg"} {
		raw, err := os.ReadFile("testdata/" + name)
		if err != nil {
			t.Skipf("read %s: %v", name, err)
		}

		// Reference: Go's pure-Go decoder.
		refImg, _, err := image.Decode(bytes.NewReader(raw))
		if err != nil {
			t.Fatalf("%s: image/jpeg decode: %v", name, err)
		}
		ref := toRGBA(refImg)
		rw, rh := ref.Rect.Dx(), ref.Rect.Dy()

		// GPU decode. Prefer the hardware engine if present, else GPU_HYBRID.
		backend := backendGPUHybrid
		if ok, _ := gpuBackendAvailable(backendHardware); ok {
			backend = backendHardware
		}
		pix, gw, gh, err := gpuDecodeRGB(raw, backend)
		if err != nil {
			t.Fatalf("%s: gpuDecodeRGB: %v", name, err)
		}
		if gw != rw || gh != rh {
			t.Fatalf("%s: dimension mismatch: gpu %dx%d vs ref %dx%d", name, gw, gh, rw, rh)
		}

		// Compare R,G,B per pixel.
		var sumAbs, maxAbs float64
		within := 0
		n := gw * gh
		for i := 0; i < n; i++ {
			ro := ref.PixOffset(i%gw, i/gw) // ref is RGBA, 4 bytes/px
			for c := 0; c < 3; c++ {
				d := math.Abs(float64(pix[i*3+c]) - float64(ref.Pix[ro+c]))
				sumAbs += d
				if d > maxAbs {
					maxAbs = d
				}
				if d <= 3 {
					within++
				}
			}
		}
		meanAbs := sumAbs / float64(n*3)
		pctWithin := 100 * float64(within) / float64(n*3)
		t.Logf("%s (%dx%d, backend=%s): mean|Δ|=%.3f max|Δ|=%.0f within±3=%.2f%%",
			name, gw, gh, backend, meanAbs, maxAbs, pctWithin)

		if meanAbs > 2.0 {
			t.Errorf("%s: mean abs pixel diff %.3f too high — decoders disagree", name, meanAbs)
		}
		if pctWithin < 95.0 {
			t.Errorf("%s: only %.2f%% of channels within ±3 — decoders disagree", name, pctWithin)
		}
	}
}

// TestGPUResizeParity checks the GPU decode+resize+crop (nvJPEG + NPP) against the
// CPU resizeForModel + center-crop at the pixel level. NPP's linear interpolation
// uses a slightly different sampling convention than our half-pixel bilinear, so
// we expect modest per-pixel differences — the authoritative check is end-to-end
// top-1 parity (Task 5). This just confirms the GPU crop is sane and close.
func TestGPUResizeParity(t *testing.T) {
	backend := backendGPUHybrid
	if ok, _ := gpuBackendAvailable(backendHardware); ok {
		backend = backendHardware
	}
	for _, name := range []string{"dog.jpg", "cat.jpg"} {
		raw, err := os.ReadFile("testdata/" + name)
		if err != nil {
			t.Skipf("read %s: %v", name, err)
		}

		// CPU reference: decode + resize-shorter-256 + center-crop 224.
		refImg, _, err := image.Decode(bytes.NewReader(raw))
		if err != nil {
			t.Fatalf("%s: decode: %v", name, err)
		}
		resized, cx, cy := resizeForModel(refImg)

		// GPU path.
		gpu, err := gpuDecodeResizeRGB(raw, backend)
		if err != nil {
			t.Fatalf("%s: gpuDecodeResizeRGB: %v", name, err)
		}

		var sumAbs, maxAbs float64
		within := 0
		for y := 0; y < imageSize; y++ {
			for x := 0; x < imageSize; x++ {
				ro := resized.PixOffset(cx+x, cy+y)
				gi := (y*imageSize + x) * 3
				for c := 0; c < 3; c++ {
					d := math.Abs(float64(gpu[gi+c]) - float64(resized.Pix[ro+c]))
					sumAbs += d
					if d > maxAbs {
						maxAbs = d
					}
					if d <= 8 {
						within++
					}
				}
			}
		}
		n := imageSize * imageSize * 3
		meanAbs := sumAbs / float64(n)
		pctWithin := 100 * float64(within) / float64(n)
		t.Logf("%s (backend=%s): resize mean|Δ|=%.3f max|Δ|=%.0f within±8=%.2f%%",
			name, backend, meanAbs, maxAbs, pctWithin)

		// Loose bound — conventions differ; Task 5 is the real top-1 gate.
		if meanAbs > 12.0 {
			t.Errorf("%s: resize mean abs diff %.3f unexpectedly large", name, meanAbs)
		}
	}
}

// TestGPUTop1Parity is the decisive check: run the GPU-preprocessed tensor through
// the real engine and confirm top-1 matches the CPU path (dog -> Samoyed, cat ->
// Egyptian cat). This is what validates the NPP resize convention — pixel-level
// divergence only matters if it changes the prediction.
func TestGPUTop1Parity(t *testing.T) {
	e := newTestEngine(t, 8, 50*time.Millisecond)
	defer e.Close()

	backend := backendGPUHybrid
	if ok, _ := gpuBackendAvailable(backendHardware); ok {
		backend = backendHardware
	}

	cases := []struct {
		name string
		want int
	}{
		{"dog.jpg", classSamoyed},
		{"cat.jpg", classEgyptianCat},
	}
	for _, tc := range cases {
		raw, err := os.ReadFile("testdata/" + tc.name)
		if err != nil {
			t.Skipf("read %s: %v", tc.name, err)
		}

		// CPU path prediction (reference).
		cpuPred, err := e.Predict(loadAndPreprocess(t, "testdata/"+tc.name))
		if err != nil {
			t.Fatalf("%s: cpu predict: %v", tc.name, err)
		}

		// GPU path prediction.
		rgb, err := gpuDecodeResizeRGB(raw, backend)
		if err != nil {
			t.Fatalf("%s: gpuDecodeResizeRGB: %v", tc.name, err)
		}
		gpuPred, err := e.Predict(normalizeRGBI(rgb))
		if err != nil {
			t.Fatalf("%s: gpu predict: %v", tc.name, err)
		}

		t.Logf("%s: CPU -> %d (%s) conf=%.4f | GPU -> %d (%s) conf=%.4f",
			tc.name, cpuPred.ClassIndex, cpuPred.ClassName, cpuPred.Confidence,
			gpuPred.ClassIndex, gpuPred.ClassName, gpuPred.Confidence)

		if gpuPred.ClassIndex != tc.want {
			t.Errorf("%s: GPU path -> class %d (%s), want %d", tc.name, gpuPred.ClassIndex, gpuPred.ClassName, tc.want)
		}
		if gpuPred.ClassIndex != cpuPred.ClassIndex {
			t.Errorf("%s: GPU top-1 %d != CPU top-1 %d", tc.name, gpuPred.ClassIndex, cpuPred.ClassIndex)
		}
	}
}
