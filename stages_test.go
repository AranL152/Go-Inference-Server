package main

import "testing"

// Stage labels and regions sit on the request path, so they must not allocate
// when no trace is running
func TestStageMarkersDoNotAllocate(t *testing.T) {
	allocs := testing.AllocsPerRun(1000, func() {
		setStage(stageDecode)
		region(stageDecode).End()
		setStage(stageRequest)
	})
	if allocs != 0 {
		t.Errorf("stage markers allocate %.1f times per call", allocs)
	}
}
