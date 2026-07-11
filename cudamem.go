// Thin cgo wrappers over the CUDA runtime for the -device-binding path
// (Phase 9 Task 4): the engine owns device-resident I/O buffers and pinned
// host staging, so ORT's IoBinding uses our buffers in place and the H2D copy
// happens from page-locked memory under our control instead of inside Run.
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
