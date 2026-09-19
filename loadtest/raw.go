// Client-side preprocessing for the raw request modes.
//
// With -raw (or -raw-rgb) the loadtest preprocesses each test image once at
// startup, using the same recipe as the server (shorter side to 256 via
// bilinear half-pixel resize, center-crop 224, and for -raw also /255 plus
// ImageNet mean/std into NCHW), then sends those bytes on every request. The
// server then measures HTTP + batching + inference without decode.
//
// The preprocessing is a copy of the server's preprocess.go: this tool cannot
// import the server (both are package main), and the copy runs once per image
// at startup, so its speed does not matter.
package main

import (
	"bytes"
	"image"
	"image/draw"
	_ "image/jpeg"
	"math"
	"unsafe"
)

const (
	rawImageSize   = 224
	rawResizeShort = 256
	rawChannels    = 3
	rawTensorBytes = rawChannels * rawImageSize * rawImageSize * 4
	rawRGBBytes    = rawChannels * rawImageSize * rawImageSize
)

var (
	rawMean = [rawChannels]float32{0.485, 0.456, 0.406}
	rawStd  = [rawChannels]float32{0.229, 0.224, 0.225}
)

// rawTensorPayload decodes a JPEG and returns the preprocessed NCHW float32
// tensor as native-byte-order bytes, ready to POST as application/x-raw-tensor.
// Loadtest and server run on the same machine, so native order is safe.
func rawTensorPayload(jpegBytes []byte) ([]byte, error) {
	img, _, err := image.Decode(bytes.NewReader(jpegBytes))
	if err != nil {
		return nil, err
	}
	t := rawPreprocess(img)
	out := make([]byte, rawTensorBytes)
	copy(out, unsafe.Slice((*byte)(unsafe.Pointer(&t[0])), rawTensorBytes))
	return out, nil
}

// rawRGBPayload decodes a JPEG and returns the resized, center-cropped
// 224×224 image as interleaved RGB uint8 (150,528 bytes), ready to POST as
// application/x-raw-rgb — the network-realistic raw contract: the server
// does only the normalize step.
func rawRGBPayload(jpegBytes []byte) ([]byte, error) {
	img, _, err := image.Decode(bytes.NewReader(jpegBytes))
	if err != nil {
		return nil, err
	}
	src := rawToRGBA(img)
	sw, sh := src.Rect.Dx(), src.Rect.Dy()
	var rw, rh int
	if sw < sh {
		rw = rawResizeShort
		rh = int(math.Round(float64(rawResizeShort) * float64(sh) / float64(sw)))
	} else {
		rh = rawResizeShort
		rw = int(math.Round(float64(rawResizeShort) * float64(sw) / float64(sh)))
	}
	resized := rawBilinearResize(src, rw, rh)
	cx := (rw - rawImageSize) / 2
	cy := (rh - rawImageSize) / 2

	out := make([]byte, rawRGBBytes)
	for y := 0; y < rawImageSize; y++ {
		for x := 0; x < rawImageSize; x++ {
			off := resized.PixOffset(cx+x, cy+y)
			i := (y*rawImageSize + x) * 3
			out[i] = resized.Pix[off]
			out[i+1] = resized.Pix[off+1]
			out[i+2] = resized.Pix[off+2]
		}
	}
	return out, nil
}

func rawPreprocess(img image.Image) []float32 {
	src := rawToRGBA(img)
	sw, sh := src.Rect.Dx(), src.Rect.Dy()

	var rw, rh int
	if sw < sh {
		rw = rawResizeShort
		rh = int(math.Round(float64(rawResizeShort) * float64(sh) / float64(sw)))
	} else {
		rh = rawResizeShort
		rw = int(math.Round(float64(rawResizeShort) * float64(sw) / float64(sh)))
	}
	resized := rawBilinearResize(src, rw, rh)
	cx := (rw - rawImageSize) / 2
	cy := (rh - rawImageSize) / 2

	plane := rawImageSize * rawImageSize
	out := make([]float32, rawChannels*plane)
	for y := 0; y < rawImageSize; y++ {
		for x := 0; x < rawImageSize; x++ {
			off := resized.PixOffset(cx+x, cy+y)
			r := float32(resized.Pix[off]) / 255.0
			g := float32(resized.Pix[off+1]) / 255.0
			b := float32(resized.Pix[off+2]) / 255.0
			i := y*rawImageSize + x
			out[0*plane+i] = (r - rawMean[0]) / rawStd[0]
			out[1*plane+i] = (g - rawMean[1]) / rawStd[1]
			out[2*plane+i] = (b - rawMean[2]) / rawStd[2]
		}
	}
	return out
}

func rawToRGBA(img image.Image) *image.RGBA {
	if rgba, ok := img.(*image.RGBA); ok && rgba.Rect.Min == image.Pt(0, 0) {
		return rgba
	}
	b := img.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(dst, dst.Bounds(), img, b.Min, draw.Src)
	return dst
}

func rawBilinearResize(src *image.RGBA, dstW, dstH int) *image.RGBA {
	sw, sh := src.Rect.Dx(), src.Rect.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, dstW, dstH))
	scaleX := float64(sw) / float64(dstW)
	scaleY := float64(sh) / float64(dstH)

	for dy := 0; dy < dstH; dy++ {
		fy := (float64(dy)+0.5)*scaleY - 0.5
		y0 := int(math.Floor(fy))
		wy := fy - float64(y0)
		y1 := rawClamp(y0+1, 0, sh-1)
		y0 = rawClamp(y0, 0, sh-1)

		for dx := 0; dx < dstW; dx++ {
			fx := (float64(dx)+0.5)*scaleX - 0.5
			x0 := int(math.Floor(fx))
			wx := fx - float64(x0)
			x1 := rawClamp(x0+1, 0, sw-1)
			x0 = rawClamp(x0, 0, sw-1)

			o00 := src.PixOffset(x0, y0)
			o10 := src.PixOffset(x1, y0)
			o01 := src.PixOffset(x0, y1)
			o11 := src.PixOffset(x1, y1)
			do := dst.PixOffset(dx, dy)

			for c := 0; c < 4; c++ {
				top := rawLerp(float64(src.Pix[o00+c]), float64(src.Pix[o10+c]), wx)
				bot := rawLerp(float64(src.Pix[o01+c]), float64(src.Pix[o11+c]), wx)
				dst.Pix[do+c] = uint8(math.Round(rawLerp(top, bot, wy)))
			}
		}
	}
	return dst
}

func rawLerp(a, b, t float64) float64 { return a + (b-a)*t }

func rawClamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
