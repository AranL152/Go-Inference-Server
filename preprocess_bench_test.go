package main

import (
	"bytes"
	"image"
	"os"
	"testing"
)

// BenchmarkPreprocess measures the full decode→resize→normalize pipeline (the
// Phase 3 headline ~38ms number).
func BenchmarkPreprocess(b *testing.B) {
	raw, err := os.ReadFile("testdata/dog.jpg")
	if err != nil {
		b.Skip(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := preprocessReader(bytes.NewReader(raw)); err != nil {
			b.Fatal(err)
		}
	}
}

// The next three benchmarks split that ~38ms into its stages so we can see which
// one dominates — the measurement that decides whether the fix is a CPU resize
// rewrite, ORT-graph offload, or (out of scope) GPU JPEG decode.

// BenchmarkDecode isolates JPEG decode (pure-Go image/jpeg).
func BenchmarkDecode(b *testing.B) {
	for _, name := range []string{"dog.jpg", "cat.jpg"} {
		raw, err := os.ReadFile("testdata/" + name)
		if err != nil {
			b.Skip(err)
		}
		b.Run(name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if _, _, err := image.Decode(bytes.NewReader(raw)); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkResize isolates toRGBA + bilinear resize (decode done once, up front).
func BenchmarkResize(b *testing.B) {
	for _, name := range []string{"dog.jpg", "cat.jpg"} {
		img := mustDecode(b, name)
		b.Run(name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				resizeForModel(img)
			}
		})
	}
}

// BenchmarkNormalize isolates the center-crop + per-channel normalize → NCHW
// step (decode + resize done once, up front).
func BenchmarkNormalize(b *testing.B) {
	for _, name := range []string{"dog.jpg", "cat.jpg"} {
		img := mustDecode(b, name)
		resized, cx, cy := resizeForModel(img)
		b.Run(name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				normalizeCHW(resized, cx, cy)
			}
		})
	}
}

func mustDecode(b *testing.B, name string) image.Image {
	b.Helper()
	raw, err := os.ReadFile("testdata/" + name)
	if err != nil {
		b.Skip(err)
	}
	img, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		b.Fatal(err)
	}
	return img
}
