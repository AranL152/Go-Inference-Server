package main

import (
	"image"
	"math/rand"
	"os"
	"testing"
)

// letterboxRef is the original float32 letterbox, kept to check the fixed point one against
func letterboxRef(img *image.RGBA, out []byte) float32 {
	sw, sh := img.Rect.Dx(), img.Rect.Dy()
	r := min(float32(letterboxSize)/float32(sw), float32(letterboxSize)/float32(sh))
	rw, rh := int(float32(sw)*r), int(float32(sh)*r)
	const plane = letterboxSize * letterboxSize
	b, g, rd := out[:plane], out[plane:2*plane], out[2*plane:3*plane]

	// source column offsets and weights are the same for every row
	var x0s, x1s [letterboxSize]int
	var wxs [letterboxSize]float32
	inv := 1 / r
	for x := 0; x < rw; x++ {
		fx := (float32(x)+0.5)*inv - 0.5
		x0 := max(floor32(fx), 0)
		x0s[x], x1s[x] = x0*4, min(x0+1, sw-1)*4
		wxs[x] = max(fx-float32(x0), 0)
	}
	for y := 0; y < letterboxSize; y++ {
		i := y * letterboxSize
		if y >= rh {
			fillRef(b[i:i+letterboxSize], g[i:i+letterboxSize], rd[i:i+letterboxSize])
			continue
		}
		fillRef(b[i+rw:i+letterboxSize], g[i+rw:i+letterboxSize], rd[i+rw:i+letterboxSize])
		if rw == sw && rh == sh {
			row := img.Pix[y*img.Stride:]
			for x := 0; x < rw; x++ {
				p := row[x*4 : x*4+3]
				rd[i+x], g[i+x], b[i+x] = p[0], p[1], p[2]
			}
			continue
		}
		fy := (float32(y)+0.5)*inv - 0.5
		y0 := max(floor32(fy), 0)
		wy := max(fy-float32(y0), 0)
		row0 := img.Pix[y0*img.Stride:]
		row1 := img.Pix[min(y0+1, sh-1)*img.Stride:]
		for x := 0; x < rw; x++ {
			a0, a1, wx := x0s[x], x1s[x], wxs[x]
			for c, dst := range [3][]byte{rd, g, b} {
				top := float32(row0[a0+c]) + (float32(row0[a1+c])-float32(row0[a0+c]))*wx
				bot := float32(row1[a0+c]) + (float32(row1[a1+c])-float32(row1[a0+c]))*wx
				dst[i+x] = uint8(top + (bot-top)*wy + 0.5)
			}
		}
	}
	return r
}

func fillRef(planes ...[]byte) {
	for _, p := range planes {
		for i := range p {
			p[i] = letterboxPad
		}
	}
}

// decoded loads a test JPEG through the same decode the server uses
func decoded(t testing.TB, path string) *image.RGBA {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Skip(err)
	}
	d, err := newJPEGDecoder()
	if err != nil {
		t.Skip(err)
	}
	img, _, _, err := d.decodeFitLong(b, letterboxSize)
	if err != nil {
		t.Fatal(err)
	}
	return img
}

func randomRGBA(w, h int, seed int64) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	rand.New(rand.NewSource(seed)).Read(img.Pix)
	return img
}

// The fixed point letterbox must land within one level of the float one on
// every pixel, for downscales, upscales and exact fits
func TestLetterboxMatchesFloatReference(t *testing.T) {
	imgs := map[string]*image.RGBA{
		"dog.jpg":        decoded(t, "testdata/dog.jpg"),
		"street.jpg":     decoded(t, "testdata/street.jpg"),
		"noise 1000x700": randomRGBA(1000, 700, 1),
		"noise 320x500":  randomRGBA(320, 500, 2),
		"noise 640x427":  randomRGBA(640, 427, 3),
		"noise 641x2":    randomRGBA(641, 2, 4),
	}
	for name, img := range imgs {
		want := make([]byte, 3*letterboxSize*letterboxSize)
		got := make([]byte, len(want))
		rw := letterboxRef(img, want)
		rg := letterboxInto(img, got)
		if rw != rg {
			t.Errorf("%s: scale %v, reference %v", name, rg, rw)
		}
		worst, off := 0, 0
		for i := range want {
			d := int(got[i]) - int(want[i])
			if d < 0 {
				d = -d
			}
			if d > 1 {
				off++
			}
			worst = max(worst, d)
		}
		if worst > 1 {
			t.Errorf("%s: %d pixels differ by more than 1, worst %d", name, off, worst)
		}
	}
}

func BenchmarkLetterbox(b *testing.B) {
	img := decoded(b, "testdata/dog.jpg")
	out := make([]byte, 3*letterboxSize*letterboxSize)
	b.ReportAllocs()
	for b.Loop() {
		letterboxInto(img, out)
	}
}

func BenchmarkLetterboxFloatReference(b *testing.B) {
	img := decoded(b, "testdata/dog.jpg")
	out := make([]byte, 3*letterboxSize*letterboxSize)
	for b.Loop() {
		letterboxRef(img, out)
	}
}
