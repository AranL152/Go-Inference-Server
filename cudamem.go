// Thin cgo wrappers over the CUDA runtime for the -device-binding path: the
// engine owns device-resident I/O buffers and pinned host staging, so ORT's
// IoBinding uses those buffers in place and the H2D copy happens from
// page-locked memory under engine control instead of inside Run.
//
// Links libcudart from the same CUDA 12.6 install run.sh already requires.
package main

/*
#cgo CFLAGS: -I/usr/local/cuda-12.6/include
#cgo LDFLAGS: -L/usr/local/cuda-12.6/lib64 -lcudart
#include <cuda_runtime.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

func cudaErr(op string, code C.cudaError_t) error {
	if code == C.cudaSuccess {
		return nil
	}
	return fmt.Errorf("%s: %s", op, C.GoString(C.cudaGetErrorString(code)))
}

// cudaMalloc allocates n bytes of device memory.
func cudaMalloc(n int) (unsafe.Pointer, error) {
	var p unsafe.Pointer
	return p, cudaErr("cudaMalloc", C.cudaMalloc(&p, C.size_t(n)))
}

func cudaFree(p unsafe.Pointer) { C.cudaFree(p) }

// cudaHostAlloc allocates n bytes of page-locked (pinned) host memory:
// DMA-transferable at full PCIe rate, unlike pageable Go memory.
func cudaHostAlloc(n int) (unsafe.Pointer, error) {
	var p unsafe.Pointer
	return p, cudaErr("cudaHostAlloc",
		C.cudaHostAlloc(&p, C.size_t(n), C.cudaHostAllocDefault))
}

func cudaFreeHost(p unsafe.Pointer) { C.cudaFreeHost(p) }

// cudaMemcpyH2D copies n bytes host -> device (synchronous).
func cudaMemcpyH2D(dst, src unsafe.Pointer, n int) error {
	return cudaErr("cudaMemcpy H2D",
		C.cudaMemcpy(dst, src, C.size_t(n), C.cudaMemcpyHostToDevice))
}

// cudaMemcpyD2H copies n bytes device -> host (synchronous).
func cudaMemcpyD2H(dst, src unsafe.Pointer, n int) error {
	return cudaErr("cudaMemcpy D2H",
		C.cudaMemcpy(dst, src, C.size_t(n), C.cudaMemcpyDeviceToHost))
}

// cudaStream is a non-blocking CUDA stream. Copies issued on it do not wait
// for work on other streams, so one lane's copy can run under another lane's
// kernels instead of queueing behind them on the legacy default stream.
type cudaStream struct{ s C.cudaStream_t }

func newCudaStream() (*cudaStream, error) {
	st := &cudaStream{}
	return st, cudaErr("cudaStreamCreate", C.cudaStreamCreateWithFlags(&st.s, C.cudaStreamNonBlocking))
}

func (st *cudaStream) destroy() { C.cudaStreamDestroy(st.s) }

// copyH2D copies n bytes host -> device on this stream and waits for it.
func (st *cudaStream) copyH2D(dst, src unsafe.Pointer, n int) error {
	if err := cudaErr("cudaMemcpyAsync H2D", C.cudaMemcpyAsync(dst, src, C.size_t(n), C.cudaMemcpyHostToDevice, st.s)); err != nil {
		return err
	}
	return cudaErr("cudaStreamSynchronize", C.cudaStreamSynchronize(st.s))
}

// copyD2H copies n bytes device -> host on this stream and waits for it.
func (st *cudaStream) copyD2H(dst, src unsafe.Pointer, n int) error {
	if err := cudaErr("cudaMemcpyAsync D2H", C.cudaMemcpyAsync(dst, src, C.size_t(n), C.cudaMemcpyDeviceToHost, st.s)); err != nil {
		return err
	}
	return cudaErr("cudaStreamSynchronize", C.cudaStreamSynchronize(st.s))
}
