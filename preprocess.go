// Image preprocessing for ResNet-18 (resnet18-v1-7 / GluonCV).
//
// The ONNX model zoo recipe for this model:
//   1. resize so the SHORTER side is 256 px (preserve aspect ratio)
//   2. center-crop 224x224
//   3. scale pixels to [0,1], then normalize per channel with
//      mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
//   4. lay out as NCHW (channel-planar): [1, 3, 224, 224]
//
// Resizing uses bilinear interpolation with the half-pixel coordinate convention
// (align_corners=False), matching PIL/torchvision so predictions line up with the
// reference recipe. Implemented without external deps to keep the cgo build setup
// from Phase 0 untouched.
package main

import (
	"fmt"
	"image"
	"image/draw"
	_ "image/jpeg" // register JPEG decoder
	_ "image/png"  // register PNG decoder
	"io"
	"math"
)

var (
	imagenetMean = [channels]float32{0.485, 0.456, 0.406}
	imagenetStd  = [channels]float32{0.229, 0.224, 0.225}
)

const resizeShort = 256

// preprocessReader decodes an image from r and returns the preprocessed NCHW
// float tensor (length 3*224*224) plus the detected format ("jpeg"/"png").
func preprocessReader(r io.Reader) ([]float32, string, error) {
	img, format, err := image.Decode(r)
	if err != nil {
		return nil, "", fmt.Errorf("decode image: %w", err)
	}
	return preprocess(img), format, nil
}

// preprocess turns a decoded image into the model's input tensor.
func preprocess(img image.Image) []float32 {
	src := toRGBA(img)
	sw, sh := src.Rect.Dx(), src.Rect.Dy()

	// Step 1: target size after resizing the shorter side to 256.
	var rw, rh int
	if sw < sh {
		rw = resizeShort
		rh = int(math.Round(float64(resizeShort) * float64(sh) / float64(sw)))
	} else {
		rh = resizeShort
		rw = int(math.Round(float64(resizeShort) * float64(sw) / float64(sh)))
	}
	resized := bilinearResize(src, rw, rh)

	// Step 2: center-crop offsets.
	cx := (rw - imageSize) / 2
	cy := (rh - imageSize) / 2

	// Steps 3+4: normalize into channel-planar NCHW layout.
	plane := imageSize * imageSize
	out := make([]float32, channels*plane)
	for y := 0; y < imageSize; y++ {
		for x := 0; x < imageSize; x++ {
			off := resized.PixOffset(cx+x, cy+y)
			r := float32(resized.Pix[off]) / 255.0
			g := float32(resized.Pix[off+1]) / 255.0
			b := float32(resized.Pix[off+2]) / 255.0
			i := y*imageSize + x
			out[0*plane+i] = (r - imagenetMean[0]) / imagenetStd[0]
			out[1*plane+i] = (g - imagenetMean[1]) / imagenetStd[1]
			out[2*plane+i] = (b - imagenetMean[2]) / imagenetStd[2]
		}
	}
	return out
}

// toRGBA returns img as *image.RGBA, copying only if it isn't already one.
func toRGBA(img image.Image) *image.RGBA {
	if rgba, ok := img.(*image.RGBA); ok && rgba.Rect.Min == image.Pt(0, 0) {
		return rgba
	}
	b := img.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(dst, dst.Bounds(), img, b.Min, draw.Src)
	return dst
}

// bilinearResize resamples src to dstW x dstH using bilinear interpolation with
// the half-pixel convention: srcCoord = (dstCoord + 0.5) * scale - 0.5.
func bilinearResize(src *image.RGBA, dstW, dstH int) *image.RGBA {
	sw, sh := src.Rect.Dx(), src.Rect.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, dstW, dstH))
	scaleX := float64(sw) / float64(dstW)
	scaleY := float64(sh) / float64(dstH)

	for dy := 0; dy < dstH; dy++ {
		fy := (float64(dy)+0.5)*scaleY - 0.5
		y0 := int(math.Floor(fy))
		wy := fy - float64(y0)
		y1 := clamp(y0+1, 0, sh-1)
		y0 = clamp(y0, 0, sh-1)

		for dx := 0; dx < dstW; dx++ {
			fx := (float64(dx)+0.5)*scaleX - 0.5
			x0 := int(math.Floor(fx))
			wx := fx - float64(x0)
			x1 := clamp(x0+1, 0, sw-1)
			x0 = clamp(x0, 0, sw-1)

			o00 := src.PixOffset(x0, y0)
			o10 := src.PixOffset(x1, y0)
			o01 := src.PixOffset(x0, y1)
			o11 := src.PixOffset(x1, y1)
			do := dst.PixOffset(dx, dy)

			for c := 0; c < 4; c++ { // R,G,B,A
				top := lerp(float64(src.Pix[o00+c]), float64(src.Pix[o10+c]), wx)
				bot := lerp(float64(src.Pix[o01+c]), float64(src.Pix[o11+c]), wx)
				dst.Pix[do+c] = uint8(math.Round(lerp(top, bot, wy)))
			}
		}
	}
	return dst
}

func lerp(a, b, t float64) float64 { return a + (b-a)*t }

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
