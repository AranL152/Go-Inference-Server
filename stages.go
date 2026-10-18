// Pipeline stage labels for CPU profiles and regions for execution traces
package main

import (
	"context"
	"runtime/pprof"
	"runtime/trace"
)

const (
	stageRequest     = "request"      // HTTP handler or job worker outside of decode
	stageDecoderWait = "decoder wait" // waiting for a free decoder from the pool
	stageDecode      = "decode"
	stageBatcher     = "batcher"
	stageInfer       = "infer"
	stageUnpack      = "unpack"
)

var stageNames = []string{stageRequest, stageDecoderWait, stageDecode, stageBatcher, stageInfer, stageUnpack}

// stageCtx holds one labelled context per stage, built once so labelling a
// goroutine on the request path does not allocate
var stageCtx = func() map[string]context.Context {
	m := make(map[string]context.Context, len(stageNames))
	for _, s := range stageNames {
		m[s] = pprof.WithLabels(context.Background(), pprof.Labels("stage", s))
	}
	return m
}()

// setStage tags the calling goroutine so CPU profile samples carry its stage
func setStage(name string) { pprof.SetGoroutineLabels(stageCtx[name]) }

// region marks a span of the calling goroutine in execution traces and costs
// one flag check when no trace is running
func region(name string) *trace.Region { return trace.StartRegion(stageCtx[name], name) }
