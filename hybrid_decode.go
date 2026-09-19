// Hybrid CPU+GPU decode routing.
//
// With libjpeg-turbo doing all decoding, the CPU saturates 13-14 of 16 cores at
// ~1,900 req/s while the GPU sits ~65% idle. The GPU has a working nvJPEG
// decode path (~2,100/s standalone), but GPU decode contends with inference for
// SMs: carrying every decode tripled per-batch inference time. So the router
// treats the GPU as overflow capacity only:
//
//  1. If a CPU decoder is free, use it (never touch the GPU while the CPU has
//     slack).
//  2. Only when all CPU decoders are busy, wait on both pools and take
//     whichever frees a decoder first.
//
// The GPU pool size (-gpu-decoders) caps how much decode contention inference
// can suffer; tune it while watching avg_infer_run_ms. It climbing back toward
// 24ms means the spill is too aggressive.
package main

import "sync/atomic"

// hybridPreprocessor routes decodes to the CPU pool first, spilling to the GPU
// pool only when every CPU decoder is checked out.
type hybridPreprocessor struct {
	cpu *cpuDecoderPool
	gpu *gpuDecoderPool

	// Routing counters (atomic): how many decodes each side served.
	statCPU int64
	statGPU int64
}

// Preprocess decodes a JPEG into the model's NCHW tensor via whichever pool has
// capacity, preferring the CPU.
func (h *hybridPreprocessor) Preprocess(jpeg []byte) ([]float32, error) {
	// Fast path: a CPU decoder is free right now.
	select {
	case d := <-h.cpu.free:
		atomic.AddInt64(&h.statCPU, 1)
		return h.cpu.preprocessWith(d, jpeg)
	default:
	}
	// CPU saturated: block on both pools, take the first decoder to free up.
	select {
	case d := <-h.cpu.free:
		atomic.AddInt64(&h.statCPU, 1)
		return h.cpu.preprocessWith(d, jpeg)
	case g := <-h.gpu.free:
		atomic.AddInt64(&h.statGPU, 1)
		return h.gpu.preprocessWith(g, jpeg)
	}
}

// Split returns how many decodes went to each pool.
func (h *hybridPreprocessor) Split() (cpu, gpu int64) {
	return atomic.LoadInt64(&h.statCPU), atomic.LoadInt64(&h.statGPU)
}

// ResetStats zeroes the routing counters and both pools' timing stats.
func (h *hybridPreprocessor) ResetStats() {
	atomic.StoreInt64(&h.statCPU, 0)
	atomic.StoreInt64(&h.statGPU, 0)
	h.cpu.ResetStats()
	h.gpu.ResetStats()
}
