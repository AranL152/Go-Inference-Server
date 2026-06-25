// Phase 6 Path A: CPU JPEG preprocessing via libjpeg-turbo (TurboJPEG API).
//
// The nvJPEG scale-factor gate proved GPU downscale decode is impossible on this
// card (hardware-decoder-only; no NVJPG engine), and measurement showed CPU
// scaled decode aggregates ~3,200/s across 16 cores — beating the ~2,100/s GPU
// decode-only ceiling AND freeing the GPU to run inference only (5,885 ceiling).
//
// The win is SIMD libjpeg-turbo (~7ms full-res) vs Go's pure-Go image/jpeg
// (~38ms); the 1/4 DCT prescale adds a smaller gain (decode is Huffman-bound) but
// shrinks the downstream CPU resize + memory traffic, so we keep it. We decode
// straight to RGBA and reuse the proven CPU resizeForModel + normalizeCHW tail —
// Path A is, by design, just "swap the decoder."
package main

/*
#cgo LDFLAGS: -lturbojpeg

#include <turbojpeg.h>
#include <stdlib.h>

static void* tj_make(void)    { return (void*)tjInitDecompress(); }
static void  tj_free(void* h) { if (h) tjDestroy((tjhandle)h); }

// tj_dims reads the JPEG header and returns the scaled output dims for
// snum/sdenom (e.g. 1/4). Cheap; lets Go size the destination buffer before the
// full decode. Returns 0 on success.
static int tj_dims(void* h, const unsigned char* data, unsigned long len,
                   int snum, int sdenom, int* sw, int* sh) {
    tjhandle handle = (tjhandle)h;
    int w, ht, subsamp, cs;
    if (tjDecompressHeader3(handle, data, len, &w, &ht, &subsamp, &cs) != 0) return -1;
    *sw = (w * snum + sdenom - 1) / sdenom;   // == TJSCALED(w, {snum,sdenom})
    *sh = (ht * snum + sdenom - 1) / sdenom;
    return 0;
}

// tj_decode_rgba decodes into dst as interleaved RGBA (A=255), DCT-downscaled to
// sw x sh via the SIMD fast path. dst must hold >= sw*sh*4 bytes. Passing the
// exact 1/4 scaled dims selects the 1/4 scaling factor.
static int tj_decode_rgba(void* h, const unsigned char* data, unsigned long len,
                          int sw, int sh, unsigned char* dst) {
    tjhandle handle = (tjhandle)h;
    if (tjDecompress2(handle, data, len, dst, sw, 0, sh, TJPF_RGBA, TJFLAG_FASTDCT) != 0)
        return -2;
    return 0;
}
*/
import "C"

import (
	"fmt"
	"image"
	"sync/atomic"
	"time"
	"unsafe"
)

// preprocessTurbo fuses resize-shorter-256 + center-crop-224 + per-channel
// normalize into a single pass over the 224×224 output, sampling the decoded
// RGBA directly with bilinear interpolation (half-pixel convention, matching
// resizeForModel/normalizeCHW). It avoids the separate path's full-frame
// intermediate image, alpha channel, float64 math, and second crop pass — the
// resize tail was the co-bottleneck once decode moved to the CPU. Output is the
// model's NCHW float tensor (length channels*224*224).
func preprocessTurbo(img *image.RGBA) []float32 {
	dw, dh := img.Rect.Dx(), img.Rect.Dy()

	// Size after resizing the shorter side to 256 (preserve aspect ratio).
	var rw, rh int
	if dw < dh {
		rw = resizeShort
		rh = int((int64(resizeShort)*int64(dh) + int64(dw)/2) / int64(dw)) // round
	} else {
		rh = resizeShort
		rw = int((int64(resizeShort)*int64(dw) + int64(dh)/2) / int64(dh))
	}
	cx := (rw - imageSize) / 2
	cy := (rh - imageSize) / 2
	scaleX := float32(dw) / float32(rw)
	scaleY := float32(dh) / float32(rh)

	pix := img.Pix
	stride := img.Stride
	plane := imageSize * imageSize
	out := make([]float32, channels*plane)

	for y := 0; y < imageSize; y++ {
		fy := (float32(cy+y)+0.5)*scaleY - 0.5
		y0 := floor32(fy)
		wy := fy - float32(y0)
		y1 := clamp(y0+1, 0, dh-1)
		y0 = clamp(y0, 0, dh-1)
		row0, row1 := y0*stride, y1*stride

		for x := 0; x < imageSize; x++ {
			fx := (float32(cx+x)+0.5)*scaleX - 0.5
			x0 := floor32(fx)
			wx := fx - float32(x0)
			x1 := clamp(x0+1, 0, dw-1)
			x0 = clamp(x0, 0, dw-1)

			o00, o10 := row0+x0*4, row0+x1*4
			o01, o11 := row1+x0*4, row1+x1*4
			i := y*imageSize + x

			for c := 0; c < channels; c++ {
				top := float32(pix[o00+c])*(1-wx) + float32(pix[o10+c])*wx
				bot := float32(pix[o01+c])*(1-wx) + float32(pix[o11+c])*wx
				v := (top*(1-wy) + bot*wy) / 255.0
				out[c*plane+i] = (v - imagenetMean[c]) / imagenetStd[c]
			}
		}
	}
	return out
}

// floor32 returns the floor of f as an int (math.Floor without the float64 trip).
func floor32(f float32) int {
	i := int(f)
	if float32(i) > f {
		i--
	}
	return i
}

// jpegDecoder wraps a TurboJPEG decompressor (one per goroutine — tjhandle is
// NOT thread-safe) plus a reusable RGBA scratch buffer.
type jpegDecoder struct {
	h   unsafe.Pointer
	buf []byte
}

func newJPEGDecoder() (*jpegDecoder, error) {
	h := C.tj_make()
	if h == nil {
		return nil, fmt.Errorf("tjInitDecompress returned NULL")
	}
	return &jpegDecoder{h: h}, nil
}

func (d *jpegDecoder) Close() {
	C.tj_free(d.h)
	d.h = nil
	d.buf = nil
}

// scaleFactors are libjpeg-turbo DCT downscale factors, most-aggressive first.
// All are supported by tjGetScalingFactors on libjpeg-turbo.
var scaleFactors = [...]struct{ num, den int }{
	{1, 8}, {1, 4}, {3, 8}, {1, 2}, {5, 8}, {3, 4}, {7, 8}, {1, 1},
}

// chooseScale picks the most-aggressive DCT downscale whose scaled shorter side
// still ≥ target, so we cut decode work without ever upscaling (which would
// destroy detail for small source images — the recipe downscales to 256). Falls
// back to 1/1 for images already smaller than target.
func chooseScale(w, h, target int) (num, den int) {
	short := w
	if h < w {
		short = h
	}
	for _, f := range scaleFactors {
		if (short*f.num+f.den-1)/f.den >= target {
			return f.num, f.den
		}
	}
	return 1, 1
}

// DecodeRGBA decodes jpeg into the decoder's reusable buffer at an adaptively
// chosen DCT scale (most downscale that keeps the shorter side ≥ 256) and returns
// it wrapped as an *image.RGBA. The returned image aliases that buffer, so it is
// valid only until this decoder's next decode — callers must finish reading it
// (e.g. resize) before returning the decoder to a pool.
func (d *jpegDecoder) DecodeRGBA(jpeg []byte) (*image.RGBA, error) {
	if len(jpeg) == 0 {
		return nil, fmt.Errorf("empty jpeg input")
	}
	src := (*C.uchar)(unsafe.Pointer(&jpeg[0]))
	n := C.ulong(len(jpeg))

	var fw, fh C.int
	if rc := C.tj_dims(d.h, src, n, 1, 1, &fw, &fh); rc != 0 {
		return nil, fmt.Errorf("tj header read failed (rc=%d)", int(rc))
	}
	num, den := chooseScale(int(fw), int(fh), resizeShort)
	return d.decodeScaled(src, n, num, den)
}

// decodeScaled decodes at an explicit DCT scale num/den into the reusable buffer.
func (d *jpegDecoder) decodeScaled(src *C.uchar, n C.ulong, num, den int) (*image.RGBA, error) {
	var sw, sh C.int
	if rc := C.tj_dims(d.h, src, n, C.int(num), C.int(den), &sw, &sh); rc != 0 {
		return nil, fmt.Errorf("tj header read failed (rc=%d)", int(rc))
	}
	need := int(sw) * int(sh) * 4
	if cap(d.buf) < need {
		d.buf = make([]byte, need)
	} else {
		d.buf = d.buf[:need]
	}
	if rc := C.tj_decode_rgba(d.h, src, n, sw, sh, (*C.uchar)(unsafe.Pointer(&d.buf[0]))); rc != 0 {
		return nil, fmt.Errorf("tj decode failed (rc=%d, scale=%d/%d)", int(rc), num, den)
	}
	return &image.RGBA{
		Pix:    d.buf[:need],
		Stride: int(sw) * 4,
		Rect:   image.Rect(0, 0, int(sw), int(sh)),
	}, nil
}

// DecodeRGBAScale decodes at an explicit DCT scale (for benchmarks/diagnostics).
func (d *jpegDecoder) DecodeRGBAScale(jpeg []byte, num, den int) (*image.RGBA, error) {
	if len(jpeg) == 0 {
		return nil, fmt.Errorf("empty jpeg input")
	}
	return d.decodeScaled((*C.uchar)(unsafe.Pointer(&jpeg[0])), C.ulong(len(jpeg)), num, den)
}

// cpuDecoderPool is a fixed set of libjpeg-turbo decoders. Handlers borrow one,
// preprocess on it, and return it; the pool size bounds concurrent CPU decodes
// (the new concurrency knob for the CPU path, mirroring gpuDecoderPool).
type cpuDecoderPool struct {
	free chan *jpegDecoder
	size int

	// Timing observability (atomic ns): SIMD decode vs CPU resize+normalize, count.
	statDecodeNs int64
	statNormNs   int64
	statCount    int64
}

// newCPUDecoderPool creates `size` libjpeg-turbo decoders. Each picks its DCT
// downscale per image (most aggressive that keeps the shorter side ≥ 256).
func newCPUDecoderPool(size int) (*cpuDecoderPool, error) {
	if size < 1 {
		size = 1
	}
	p := &cpuDecoderPool{free: make(chan *jpegDecoder, size), size: size}
	for i := 0; i < size; i++ {
		d, err := newJPEGDecoder()
		if err != nil {
			p.Close()
			return nil, fmt.Errorf("create cpu decoder %d/%d: %w", i+1, size, err)
		}
		p.free <- d
	}
	return p, nil
}

// Preprocess decodes a JPEG and returns the model's NCHW normalized tensor,
// entirely on the CPU (libjpeg-turbo decode@1/4 + the existing resize/normalize).
// Drop-in replacement for gpuDecoderPool.Preprocess / preprocessReader's output.
func (p *cpuDecoderPool) Preprocess(jpeg []byte) ([]float32, error) {
	d := <-p.free
	defer func() { p.free <- d }()

	tDecode := time.Now()
	img, err := d.DecodeRGBA(jpeg)
	if err != nil {
		return nil, err
	}
	decodeNs := time.Since(tDecode)

	tNorm := time.Now()
	out := preprocessTurbo(img)

	atomic.AddInt64(&p.statDecodeNs, int64(decodeNs))
	atomic.AddInt64(&p.statNormNs, int64(time.Since(tNorm)))
	atomic.AddInt64(&p.statCount, 1)
	return out, nil
}

// PreprocStats returns avg CPU decode ms, avg CPU resize+normalize ms, and count.
func (p *cpuDecoderPool) PreprocStats() (avgDecodeMs, avgNormMs float64, n int64) {
	n = atomic.LoadInt64(&p.statCount)
	if n > 0 {
		avgDecodeMs = float64(atomic.LoadInt64(&p.statDecodeNs)) / float64(n) / 1e6
		avgNormMs = float64(atomic.LoadInt64(&p.statNormNs)) / float64(n) / 1e6
	}
	return
}

// ResetStats zeroes the preprocessing timers.
func (p *cpuDecoderPool) ResetStats() {
	atomic.StoreInt64(&p.statDecodeNs, 0)
	atomic.StoreInt64(&p.statNormNs, 0)
	atomic.StoreInt64(&p.statCount, 0)
}

// Close destroys all decoders. Call only after no Preprocess calls are in flight.
func (p *cpuDecoderPool) Close() {
	for i := 0; i < p.size; i++ {
		select {
		case d := <-p.free:
			d.Close()
		default:
		}
	}
}
