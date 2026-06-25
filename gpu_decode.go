// GPU JPEG decode via nvJPEG (Phase 4, Task 1 — standalone gate).
//
// This is the first, deliberately isolated step of moving image preprocessing
// onto the GPU. It does ONE thing: decode a JPEG to interleaved RGB on the GPU
// using nvJPEG and copy the pixels back to host memory. It does not touch the
// server, the engine, or the batching collector. The point is to prove that:
//
//  1. nvJPEG + cudart link cleanly against the protected Phase-0 ORT/cuDNN/CUDA
//     stack (CUDA 12.6), and
//  2. GPU-decoded pixels match Go's image/jpeg decode within tolerance, and
//  3. which nvJPEG backend this card actually provides — the dedicated hardware
//     NVJPG engine (decode on separate silicon, no contention with inference) or
//     the GPU-hybrid path (CUDA cores, shares SMs with inference).
//
// Resize/crop/normalize and the decoder pool come in later tasks.
package main

/*
#cgo CFLAGS: -I/usr/local/cuda-12.6/include
#cgo LDFLAGS: -L/usr/local/cuda-12.6/lib64 -lnvjpeg -lnppig -lnppc -lcudart -lm

#include <nvjpeg.h>
#include <npp.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// gpu_try_backend creates and immediately destroys an nvJPEG handle for the
// given backend. Returns 0 if the backend is usable on this GPU, else the
// negative nvjpegStatus_t so the caller can tell "unsupported" from "broken".
static int gpu_try_backend(int backend) {
    nvjpegHandle_t h;
    nvjpegStatus_t s = nvjpegCreateEx((nvjpegBackend_t)backend, NULL, NULL, 0, &h);
    if (s != NVJPEG_STATUS_SUCCESS) {
        return -(int)s;
    }
    nvjpegDestroy(h);
    return 0;
}

// gpu_decode_rgb decodes one JPEG to an interleaved RGB (w*h*3) host buffer.
// On success returns 0 and sets *out (malloc'd, caller frees), *width, *height.
// Negative return codes encode which stage failed plus the nvjpeg/cuda status.
static int gpu_decode_rgb(const unsigned char* data, size_t len, int backend,
                          unsigned char** out, int* width, int* height) {
    nvjpegHandle_t handle = NULL;
    nvjpegJpegState_t state = NULL;
    cudaStream_t stream = NULL;
    unsigned char* dev = NULL;
    unsigned char* host = NULL;
    int rc = 0;
    nvjpegStatus_t s;

    s = nvjpegCreateEx((nvjpegBackend_t)backend, NULL, NULL, 0, &handle);
    if (s != NVJPEG_STATUS_SUCCESS) { return -100 - (int)s; }

    s = nvjpegJpegStateCreate(handle, &state);
    if (s != NVJPEG_STATUS_SUCCESS) { rc = -200 - (int)s; goto cleanup; }

    if (cudaStreamCreate(&stream) != cudaSuccess) { rc = -300; goto cleanup; }

    int nComp;
    nvjpegChromaSubsampling_t subs;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    s = nvjpegGetImageInfo(handle, data, len, &nComp, &subs, widths, heights);
    if (s != NVJPEG_STATUS_SUCCESS) { rc = -400 - (int)s; goto cleanup; }

    int w = widths[0];
    int h = heights[0];
    size_t nbytes = (size_t)w * (size_t)h * 3;

    if (cudaMalloc((void**)&dev, nbytes) != cudaSuccess) { rc = -500; goto cleanup; }

    nvjpegImage_t outImg;
    memset(&outImg, 0, sizeof(outImg));
    outImg.channel[0] = dev;
    outImg.pitch[0]   = (size_t)w * 3;

    s = nvjpegDecode(handle, state, data, len, NVJPEG_OUTPUT_RGBI, &outImg, stream);
    if (s != NVJPEG_STATUS_SUCCESS) { rc = -600 - (int)s; goto cleanup; }

    if (cudaStreamSynchronize(stream) != cudaSuccess) { rc = -700; goto cleanup; }

    host = (unsigned char*)malloc(nbytes);
    if (!host) { rc = -800; goto cleanup; }
    if (cudaMemcpy(host, dev, nbytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        free(host); host = NULL; rc = -900; goto cleanup;
    }

    *out = host;
    *width = w;
    *height = h;

cleanup:
    if (dev)    cudaFree(dev);
    if (stream) cudaStreamDestroy(stream);
    if (state)  nvjpegJpegStateDestroy(state);
    if (handle) nvjpegDestroy(handle);
    return rc;
}

// gpu_decode_resize_rgb decodes a JPEG, resizes its shorter side to 256 px
// (bilinear, preserving aspect ratio) and center-crops 224x224 — all on the GPU
// via nvJPEG + NPP — copying the final 224*224*3 interleaved-RGB crop back to
// the caller-provided host buffer (must be >= 224*224*3 bytes). This mirrors the
// CPU resizeForModel + center-crop. Creates/destroys nvJPEG state per call;
// handle reuse arrives with the decoder pool (Task 3).
static int gpu_decode_resize_rgb(const unsigned char* data, size_t len, int backend,
                                 unsigned char* host_out, int crop) {
    nvjpegHandle_t handle = NULL;
    nvjpegJpegState_t state = NULL;
    cudaStream_t stream = NULL;
    unsigned char* devSrc = NULL;  // decoded full image (RGBI)
    unsigned char* devDst = NULL;  // resized image (RGBI)
    int rc = 0;
    nvjpegStatus_t s;

    s = nvjpegCreateEx((nvjpegBackend_t)backend, NULL, NULL, 0, &handle);
    if (s != NVJPEG_STATUS_SUCCESS) { return -100 - (int)s; }
    s = nvjpegJpegStateCreate(handle, &state);
    if (s != NVJPEG_STATUS_SUCCESS) { rc = -200 - (int)s; goto cleanup; }
    if (cudaStreamCreate(&stream) != cudaSuccess) { rc = -300; goto cleanup; }

    int nComp;
    nvjpegChromaSubsampling_t subs;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    s = nvjpegGetImageInfo(handle, data, len, &nComp, &subs, widths, heights);
    if (s != NVJPEG_STATUS_SUCCESS) { rc = -400 - (int)s; goto cleanup; }

    int w = widths[0];
    int h = heights[0];
    if (cudaMalloc((void**)&devSrc, (size_t)w * h * 3) != cudaSuccess) { rc = -500; goto cleanup; }

    nvjpegImage_t outImg;
    memset(&outImg, 0, sizeof(outImg));
    outImg.channel[0] = devSrc;
    outImg.pitch[0]   = (size_t)w * 3;

    s = nvjpegDecode(handle, state, data, len, NVJPEG_OUTPUT_RGBI, &outImg, stream);
    if (s != NVJPEG_STATUS_SUCCESS) { rc = -600 - (int)s; goto cleanup; }
    if (cudaStreamSynchronize(stream) != cudaSuccess) { rc = -700; goto cleanup; }

    // Resize target: shorter side -> 256, preserving aspect ratio (matches CPU).
    const int kShort = 256;
    int rw, rh;
    if (w < h) { rw = kShort; rh = (int)lround((double)kShort * h / w); }
    else       { rh = kShort; rw = (int)lround((double)kShort * w / h); }

    if (cudaMalloc((void**)&devDst, (size_t)rw * rh * 3) != cudaSuccess) { rc = -510; goto cleanup; }

    NppiSize srcSize = { w, h };
    NppiRect srcROI  = { 0, 0, w, h };
    NppiSize dstSize = { rw, rh };
    NppiRect dstROI  = { 0, 0, rw, rh };
    NppStatus ns = nppiResize_8u_C3R(
        devSrc, w * 3, srcSize, srcROI,
        devDst, rw * 3, dstSize, dstROI,
        NPPI_INTER_LINEAR);
    if (ns != NPP_SUCCESS) { rc = -1000 - (int)ns; goto cleanup; }
    if (cudaDeviceSynchronize() != cudaSuccess) { rc = -1100; goto cleanup; }

    // Center-crop `crop` x `crop` and copy that window back to host (tight rows).
    int cx = (rw - crop) / 2;
    int cy = (rh - crop) / 2;
    const unsigned char* cropStart = devDst + ((size_t)cy * rw + cx) * 3;
    if (cudaMemcpy2D(host_out, (size_t)crop * 3,
                     cropStart, (size_t)rw * 3,
                     (size_t)crop * 3, (size_t)crop,
                     cudaMemcpyDeviceToHost) != cudaSuccess) { rc = -1200; goto cleanup; }

cleanup:
    if (devSrc) cudaFree(devSrc);
    if (devDst) cudaFree(devDst);
    if (stream) cudaStreamDestroy(stream);
    if (state)  nvjpegJpegStateDestroy(state);
    if (handle) nvjpegDestroy(handle);
    return rc;
}

// ---- Decoder-context pool (Task 3) ---------------------------------------
//
// A gpu_ctx holds a persistent nvJPEG handle/state, a CUDA stream, an NPP stream
// context bound to that stream, and grow-on-demand device buffers. Reusing these
// across calls removes the per-call setup cost (the naive path's ~4ms floor) and
// keeps each context's work on its own stream so concurrent contexts don't
// serialize on a global device sync.
typedef struct {
    nvjpegHandle_t    handle;
    nvjpegJpegState_t state;
    cudaStream_t      stream;
    NppStreamContext  npp;
    unsigned char*    devSrc; size_t devSrcCap;
    unsigned char*    devDst; size_t devDstCap;
} gpu_ctx;

// gpu_ensure (re)allocates *buf to at least `need` bytes, growing as required.
static int gpu_ensure(unsigned char** buf, size_t* cap, size_t need) {
    if (*cap >= need) return 0;
    if (*buf) cudaFree(*buf);
    *buf = NULL; *cap = 0;
    if (cudaMalloc((void**)buf, need) != cudaSuccess) return -1;
    *cap = need;
    return 0;
}

static int gpu_ctx_create(int backend, gpu_ctx** out) {
    gpu_ctx* c = (gpu_ctx*)calloc(1, sizeof(gpu_ctx));
    if (!c) return -1;
    nvjpegStatus_t s = nvjpegCreateEx((nvjpegBackend_t)backend, NULL, NULL, 0, &c->handle);
    if (s != NVJPEG_STATUS_SUCCESS) { free(c); return -100 - (int)s; }
    s = nvjpegJpegStateCreate(c->handle, &c->state);
    if (s != NVJPEG_STATUS_SUCCESS) { nvjpegDestroy(c->handle); free(c); return -200 - (int)s; }
    if (cudaStreamCreate(&c->stream) != cudaSuccess) {
        nvjpegJpegStateDestroy(c->state); nvjpegDestroy(c->handle); free(c); return -300;
    }
    // Fill NPP device fields from the current context, then bind our stream.
    if (nppGetStreamContext(&c->npp) != NPP_SUCCESS) {
        cudaStreamDestroy(c->stream); nvjpegJpegStateDestroy(c->state); nvjpegDestroy(c->handle); free(c); return -310;
    }
    c->npp.hStream = c->stream;
    *out = c;
    return 0;
}

static void gpu_ctx_destroy(gpu_ctx* c) {
    if (!c) return;
    if (c->devSrc) cudaFree(c->devSrc);
    if (c->devDst) cudaFree(c->devDst);
    if (c->stream) cudaStreamDestroy(c->stream);
    if (c->state)  nvjpegJpegStateDestroy(c->state);
    if (c->handle) nvjpegDestroy(c->handle);
    free(c);
}

// gpu_ctx_decode_resize: decode + resize-shorter-256 + center-crop on the
// context's own stream, copying the crop*crop*3 RGB window to host_out. All work
// is ordered on c->stream (decode -> resize), so a single stream sync suffices.
static int gpu_ctx_decode_resize(gpu_ctx* c, const unsigned char* data, size_t len,
                                 unsigned char* host_out, int crop) {
    nvjpegStatus_t s;
    int nComp;
    nvjpegChromaSubsampling_t subs;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    s = nvjpegGetImageInfo(c->handle, data, len, &nComp, &subs, widths, heights);
    if (s != NVJPEG_STATUS_SUCCESS) return -400 - (int)s;

    int w = widths[0], h = heights[0];
    if (gpu_ensure(&c->devSrc, &c->devSrcCap, (size_t)w * h * 3) != 0) return -500;

    nvjpegImage_t outImg;
    memset(&outImg, 0, sizeof(outImg));
    outImg.channel[0] = c->devSrc;
    outImg.pitch[0]   = (size_t)w * 3;
    s = nvjpegDecode(c->handle, c->state, data, len, NVJPEG_OUTPUT_RGBI, &outImg, c->stream);
    if (s != NVJPEG_STATUS_SUCCESS) return -600 - (int)s;

    const int kShort = 256;
    int rw, rh;
    if (w < h) { rw = kShort; rh = (int)lround((double)kShort * h / w); }
    else       { rh = kShort; rw = (int)lround((double)kShort * w / h); }
    if (gpu_ensure(&c->devDst, &c->devDstCap, (size_t)rw * rh * 3) != 0) return -510;

    NppiSize srcSize = { w, h };
    NppiRect srcROI  = { 0, 0, w, h };
    NppiSize dstSize = { rw, rh };
    NppiRect dstROI  = { 0, 0, rw, rh };
    NppStatus ns = nppiResize_8u_C3R_Ctx(
        c->devSrc, w * 3, srcSize, srcROI,
        c->devDst, rw * 3, dstSize, dstROI,
        NPPI_INTER_LINEAR, c->npp);
    if (ns != NPP_SUCCESS) return -1000 - (int)ns;

    if (cudaStreamSynchronize(c->stream) != cudaSuccess) return -1100;

    int cx = (rw - crop) / 2, cy = (rh - crop) / 2;
    const unsigned char* cropStart = c->devDst + ((size_t)cy * rw + cx) * 3;
    if (cudaMemcpy2D(host_out, (size_t)crop * 3,
                     cropStart, (size_t)rw * 3,
                     (size_t)crop * 3, (size_t)crop,
                     cudaMemcpyDeviceToHost) != cudaSuccess) return -1200;
    return 0;
}
*/
import "C"

import (
	"fmt"
	"sync/atomic"
	"time"
	"unsafe"
)

// nvjpegBackend identifies a decode backend (mirrors nvjpegBackend_t).
type nvjpegBackend int

const (
	backendDefault   nvjpegBackend = 0
	backendHybrid    nvjpegBackend = 1 // CPU Huffman
	backendGPUHybrid nvjpegBackend = 2 // GPU-assisted Huffman, CUDA cores
	backendHardware  nvjpegBackend = 3 // dedicated NVJPG engine, separate silicon
)

func (b nvjpegBackend) String() string {
	switch b {
	case backendDefault:
		return "DEFAULT"
	case backendHybrid:
		return "HYBRID(cpu-huffman)"
	case backendGPUHybrid:
		return "GPU_HYBRID(cuda-cores)"
	case backendHardware:
		return "HARDWARE(nvjpg-engine)"
	default:
		return fmt.Sprintf("backend(%d)", int(b))
	}
}

// gpuBackendAvailable reports whether nvJPEG can create a handle for the given
// backend on this GPU. The key question for Phase 4 is whether backendHardware
// is available (dedicated decode silicon) or we fall back to GPU_HYBRID.
func gpuBackendAvailable(b nvjpegBackend) (bool, int) {
	rc := int(C.gpu_try_backend(C.int(b)))
	return rc == 0, rc
}

// gpuDecodeRGB decodes a JPEG to interleaved RGB (length w*h*3) on the GPU and
// returns the pixels in host memory. backend selects the nvJPEG path.
func gpuDecodeRGB(jpeg []byte, backend nvjpegBackend) (pix []byte, w, h int, err error) {
	if len(jpeg) == 0 {
		return nil, 0, 0, fmt.Errorf("empty jpeg input")
	}
	var out *C.uchar
	var cw, ch C.int
	rc := C.gpu_decode_rgb(
		(*C.uchar)(unsafe.Pointer(&jpeg[0])),
		C.size_t(len(jpeg)),
		C.int(backend),
		&out, &cw, &ch,
	)
	if rc != 0 {
		return nil, 0, 0, fmt.Errorf("nvjpeg decode failed (rc=%d, backend=%s)", int(rc), backend)
	}
	defer C.free(unsafe.Pointer(out))
	w, h = int(cw), int(ch)
	pix = C.GoBytes(unsafe.Pointer(out), C.int(w*h*3))
	return pix, w, h, nil
}

// gpuDecodeResizeRGB decodes a JPEG and produces the model's 224x224x3
// interleaved-RGB center crop entirely on the GPU (nvJPEG decode + NPP resize),
// mirroring the CPU resizeForModel + center-crop. Returns imageSize*imageSize*3
// bytes of uint8 RGB, ready for normalizeRGBI.
func gpuDecodeResizeRGB(jpeg []byte, backend nvjpegBackend) ([]byte, error) {
	if len(jpeg) == 0 {
		return nil, fmt.Errorf("empty jpeg input")
	}
	out := make([]byte, imageSize*imageSize*3)
	rc := C.gpu_decode_resize_rgb(
		(*C.uchar)(unsafe.Pointer(&jpeg[0])),
		C.size_t(len(jpeg)),
		C.int(backend),
		(*C.uchar)(unsafe.Pointer(&out[0])),
		C.int(imageSize),
	)
	if rc != 0 {
		return nil, fmt.Errorf("nvjpeg decode+resize failed (rc=%d, backend=%s)", int(rc), backend)
	}
	return out, nil
}

// gpuDecoderPool is a fixed set of reusable nvJPEG decoder contexts. Handlers
// borrow a context, decode+resize on it, and return it. The pool size bounds
// concurrent GPU decodes (and thus GPU memory) and is the new concurrency knob
// that replaces "however many CPU cores happen to be free."
type gpuDecoderPool struct {
	free    chan *gpuDecoder
	backend nvjpegBackend
	size    int

	// Timing observability (atomic, nanoseconds): GPU decode+resize cgo call vs
	// CPU normalize, plus a request count. Splits where preprocessing time goes.
	statDecodeNs int64
	statNormNs   int64
	statCount    int64
}

// PreprocStats returns avg GPU decode+resize ms, avg CPU normalize ms, and count.
func (p *gpuDecoderPool) PreprocStats() (avgDecodeMs, avgNormMs float64, n int64) {
	n = atomic.LoadInt64(&p.statCount)
	if n > 0 {
		avgDecodeMs = float64(atomic.LoadInt64(&p.statDecodeNs)) / float64(n) / 1e6
		avgNormMs = float64(atomic.LoadInt64(&p.statNormNs)) / float64(n) / 1e6
	}
	return
}

// ResetStats zeroes the preprocessing timers.
func (p *gpuDecoderPool) ResetStats() {
	atomic.StoreInt64(&p.statDecodeNs, 0)
	atomic.StoreInt64(&p.statNormNs, 0)
	atomic.StoreInt64(&p.statCount, 0)
}

type gpuDecoder struct {
	ctx *C.gpu_ctx
}

// newGPUDecoderPool creates `size` decoder contexts using the best available
// backend (hardware NVJPG if present, else GPU-hybrid).
func newGPUDecoderPool(size int) (*gpuDecoderPool, error) {
	if size < 1 {
		size = 1
	}
	backend := backendGPUHybrid
	if ok, _ := gpuBackendAvailable(backendHardware); ok {
		backend = backendHardware
	}
	p := &gpuDecoderPool{free: make(chan *gpuDecoder, size), backend: backend, size: size}
	for i := 0; i < size; i++ {
		var ctx *C.gpu_ctx
		if rc := C.gpu_ctx_create(C.int(backend), &ctx); rc != 0 {
			p.Close()
			return nil, fmt.Errorf("create gpu decoder context %d/%d: rc=%d (backend=%s)", i+1, size, int(rc), backend)
		}
		p.free <- &gpuDecoder{ctx: ctx}
	}
	return p, nil
}

// Backend reports the nvJPEG backend the pool's contexts use.
func (p *gpuDecoderPool) Backend() nvjpegBackend { return p.backend }

// Preprocess decodes a JPEG and returns the model's NCHW normalized tensor,
// entirely via the GPU (nvJPEG decode + NPP resize) plus a cheap CPU normalize.
// Drop-in replacement for the CPU preprocessReader's output.
func (p *gpuDecoderPool) Preprocess(jpeg []byte) ([]float32, error) {
	if len(jpeg) == 0 {
		return nil, fmt.Errorf("empty jpeg input")
	}
	d := <-p.free
	defer func() { p.free <- d }()

	rgb := make([]byte, imageSize*imageSize*3)
	tDecode := time.Now()
	rc := C.gpu_ctx_decode_resize(
		d.ctx,
		(*C.uchar)(unsafe.Pointer(&jpeg[0])),
		C.size_t(len(jpeg)),
		(*C.uchar)(unsafe.Pointer(&rgb[0])),
		C.int(imageSize),
	)
	decodeNs := time.Since(tDecode)
	if rc != 0 {
		return nil, fmt.Errorf("gpu decode+resize failed (rc=%d)", int(rc))
	}
	tNorm := time.Now()
	out := normalizeRGBI(rgb)
	atomic.AddInt64(&p.statDecodeNs, int64(decodeNs))
	atomic.AddInt64(&p.statNormNs, int64(time.Since(tNorm)))
	atomic.AddInt64(&p.statCount, 1)
	return out, nil
}

// Close destroys all decoder contexts. Call only after no Preprocess calls are
// in flight (e.g. after the HTTP server has stopped accepting requests).
func (p *gpuDecoderPool) Close() {
	for i := 0; i < p.size; i++ {
		select {
		case d := <-p.free:
			C.gpu_ctx_destroy(d.ctx)
		default:
			// A context is checked out (shouldn't happen at clean shutdown);
			// skip rather than block forever.
		}
	}
}

// normalizeRGBI turns a 224x224x3 interleaved-RGB uint8 buffer into the model's
// per-channel-normalized, channel-planar NCHW float tensor — the same output as
// normalizeCHW, but reading interleaved RGB instead of an *image.RGBA. This is
// the cheap (~0.26ms) tail of preprocessing kept on the CPU.
func normalizeRGBI(rgb []byte) []float32 {
	plane := imageSize * imageSize
	out := make([]float32, channels*plane)
	for i := 0; i < plane; i++ {
		r := float32(rgb[i*3+0]) / 255.0
		g := float32(rgb[i*3+1]) / 255.0
		b := float32(rgb[i*3+2]) / 255.0
		out[0*plane+i] = (r - imagenetMean[0]) / imagenetStd[0]
		out[1*plane+i] = (g - imagenetMean[1]) / imagenetStd[1]
		out[2*plane+i] = (b - imagenetMean[2]) / imagenetStd[2]
	}
	return out
}
