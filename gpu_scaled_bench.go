// Phase 6 Task 2 GATE: does nvJPEG scale-factor downscale decode actually make
// decoding cheaper, or is decode Huffman-bound (entropy decode happens before
// any scaling, so 1/4 ≈ NONE)?
//
// We need only ~256px out of a ~1546px source. If decoding at NVJPEG_SCALE_1_BY_4
// is materially faster than NVJPEG_SCALE_NONE, scaled GPU decode raises the
// ~2,100 req/s decode-only ceiling and we build it into the pool. If 1/4 ≈ NONE,
// decode is Huffman-bound and we pivot to a different lever.
//
// The single-phase nvjpegDecode does NOT support scale factors — only the
// decoupled API (nvjpegDecodeJpegHost/TransferToDevice/Device with a
// nvjpegDecodeParams carrying the scale factor) does. This file is a standalone
// gate: it touches nothing in the server.
package main

/*
#cgo CFLAGS: -I/usr/local/cuda-12.6/include
#cgo LDFLAGS: -L/usr/local/cuda-12.6/lib64 -lnvjpeg -lnppig -lnppc -lcudart -lm

#include <nvjpeg.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// gpu_scaled_bench decodes `data` `iters` times at the given scale factor
// (0=NONE, 1=1/2, 2=1/4, 3=1/8) using the decoupled nvJPEG API, syncing each
// iteration (decode-only — no resize/normalize/D2H of the full image, just the
// device decode + stream sync, which is what the decode-only ceiling measures).
// All nvJPEG objects are created once and reused across iters so we measure the
// steady-state decode cost, not setup. On success returns 0 and sets *avg_ns
// (avg per decode), *out_w/*out_h (scaled output dims). Negative rc on failure.
static int gpu_scaled_bench(const unsigned char* data, size_t len, int backend,
                            int scale, int iters,
                            double* avg_ns, int* out_w, int* out_h) {
    nvjpegHandle_t handle = NULL;
    nvjpegJpegState_t dstate = NULL;
    nvjpegJpegDecoder_t decoder = NULL;
    nvjpegJpegStream_t jstream = NULL;
    nvjpegDecodeParams_t params = NULL;
    nvjpegBufferDevice_t devbuf = NULL;
    nvjpegBufferPinned_t pinbuf = NULL;
    cudaStream_t stream = NULL;
    unsigned char* dev = NULL;
    int rc = 0;
    nvjpegStatus_t s;

    s = nvjpegCreateEx((nvjpegBackend_t)backend, NULL, NULL, 0, &handle);
    if (s != NVJPEG_STATUS_SUCCESS) { return -100 - (int)s; }
    if (cudaStreamCreate(&stream) != cudaSuccess) { rc = -110; goto cleanup; }

    s = nvjpegDecoderCreate(handle, (nvjpegBackend_t)backend, &decoder);
    if (s != NVJPEG_STATUS_SUCCESS) { rc = -120 - (int)s; goto cleanup; }
    s = nvjpegDecoderStateCreate(handle, decoder, &dstate);
    if (s != NVJPEG_STATUS_SUCCESS) { rc = -130 - (int)s; goto cleanup; }

    s = nvjpegBufferDeviceCreate(handle, NULL, &devbuf);
    if (s != NVJPEG_STATUS_SUCCESS) { rc = -140 - (int)s; goto cleanup; }
    s = nvjpegBufferPinnedCreate(handle, NULL, &pinbuf);
    if (s != NVJPEG_STATUS_SUCCESS) { rc = -150 - (int)s; goto cleanup; }
    s = nvjpegStateAttachDeviceBuffer(dstate, devbuf);
    if (s != NVJPEG_STATUS_SUCCESS) { rc = -160 - (int)s; goto cleanup; }
    s = nvjpegStateAttachPinnedBuffer(dstate, pinbuf);
    if (s != NVJPEG_STATUS_SUCCESS) { rc = -170 - (int)s; goto cleanup; }

    s = nvjpegJpegStreamCreate(handle, &jstream);
    if (s != NVJPEG_STATUS_SUCCESS) { rc = -180 - (int)s; goto cleanup; }
    s = nvjpegDecodeParamsCreate(handle, &params);
    if (s != NVJPEG_STATUS_SUCCESS) { rc = -190 - (int)s; goto cleanup; }
    s = nvjpegDecodeParamsSetOutputFormat(params, NVJPEG_OUTPUT_RGBI);
    if (s != NVJPEG_STATUS_SUCCESS) { rc = -200 - (int)s; goto cleanup; }
    s = nvjpegDecodeParamsSetScaleFactor(params, (nvjpegScaleFactor_t)scale);
    if (s != NVJPEG_STATUS_SUCCESS) { rc = -210 - (int)s; goto cleanup; }

    // Parse once to learn full dimensions, then derive scaled output size.
    s = nvjpegJpegStreamParse(handle, data, len, 0, 0, jstream);
    if (s != NVJPEG_STATUS_SUCCESS) { rc = -220 - (int)s; goto cleanup; }
    unsigned int fw, fh;
    s = nvjpegJpegStreamGetFrameDimensions(jstream, &fw, &fh);
    if (s != NVJPEG_STATUS_SUCCESS) { rc = -230 - (int)s; goto cleanup; }

    int div = 1 << scale; // scale 0..3 -> 1,2,4,8
    int sw = ((int)fw + div - 1) / div;
    int sh = ((int)fh + div - 1) / div;

    size_t cap = (size_t)sw * sh * 3 + 64;
    if (cudaMalloc((void**)&dev, cap) != cudaSuccess) { rc = -240; goto cleanup; }

    nvjpegImage_t outImg;
    memset(&outImg, 0, sizeof(outImg));
    outImg.channel[0] = dev;
    outImg.pitch[0]   = (size_t)sw * 3;

    // Warm up once (first decode allocates internal buffers / JIT).
    s = nvjpegJpegStreamParse(handle, data, len, 0, 0, jstream);
    if (s != NVJPEG_STATUS_SUCCESS) { rc = -250 - (int)s; goto cleanup; }
    s = nvjpegDecodeJpegHost(handle, decoder, dstate, params, jstream);
    if (s != NVJPEG_STATUS_SUCCESS) { rc = -260 - (int)s; goto cleanup; }
    s = nvjpegDecodeJpegTransferToDevice(handle, decoder, dstate, jstream, stream);
    if (s != NVJPEG_STATUS_SUCCESS) { rc = -270 - (int)s; goto cleanup; }
    s = nvjpegDecodeJpegDevice(handle, decoder, dstate, &outImg, stream);
    if (s != NVJPEG_STATUS_SUCCESS) { rc = -280 - (int)s; goto cleanup; }
    if (cudaStreamSynchronize(stream) != cudaSuccess) { rc = -290; goto cleanup; }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++) {
        s = nvjpegJpegStreamParse(handle, data, len, 0, 0, jstream);
        if (s != NVJPEG_STATUS_SUCCESS) { rc = -300 - (int)s; goto cleanup; }
        s = nvjpegDecodeJpegHost(handle, decoder, dstate, params, jstream);
        if (s != NVJPEG_STATUS_SUCCESS) { rc = -310 - (int)s; goto cleanup; }
        s = nvjpegDecodeJpegTransferToDevice(handle, decoder, dstate, jstream, stream);
        if (s != NVJPEG_STATUS_SUCCESS) { rc = -320 - (int)s; goto cleanup; }
        s = nvjpegDecodeJpegDevice(handle, decoder, dstate, &outImg, stream);
        if (s != NVJPEG_STATUS_SUCCESS) { rc = -330 - (int)s; goto cleanup; }
        if (cudaStreamSynchronize(stream) != cudaSuccess) { rc = -340; goto cleanup; }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    {
        double total = (double)(t1.tv_sec - t0.tv_sec) * 1e9 +
                       (double)(t1.tv_nsec - t0.tv_nsec);
        *avg_ns = total / (double)iters;
        *out_w = sw;
        *out_h = sh;
    }

cleanup:
    if (dev)     cudaFree(dev);
    if (params)  nvjpegDecodeParamsDestroy(params);
    if (jstream) nvjpegJpegStreamDestroy(jstream);
    if (pinbuf)  nvjpegBufferPinnedDestroy(pinbuf);
    if (devbuf)  nvjpegBufferDeviceDestroy(devbuf);
    if (dstate)  nvjpegJpegStateDestroy(dstate);
    if (decoder) nvjpegDecoderDestroy(decoder);
    if (stream)  cudaStreamDestroy(stream);
    if (handle)  nvjpegDestroy(handle);
    return rc;
}
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// scaledDecodeBench decodes jpeg `iters` times at the given scale (0=NONE,
// 1=1/2, 2=1/4, 3=1/8) on `backend`, returning avg ms/decode and scaled dims.
func scaledDecodeBench(jpeg []byte, backend nvjpegBackend, scale, iters int) (avgMs float64, w, h int, err error) {
	if len(jpeg) == 0 {
		return 0, 0, 0, fmt.Errorf("empty jpeg input")
	}
	var avgNs C.double
	var cw, ch C.int
	rc := C.gpu_scaled_bench(
		(*C.uchar)(unsafe.Pointer(&jpeg[0])),
		C.size_t(len(jpeg)),
		C.int(backend),
		C.int(scale),
		C.int(iters),
		&avgNs, &cw, &ch,
	)
	if rc != 0 {
		return 0, 0, 0, fmt.Errorf("gpu_scaled_bench failed (rc=%d, scale=%d, backend=%s)", int(rc), scale, backend)
	}
	return float64(avgNs) / 1e6, int(cw), int(ch), nil
}
