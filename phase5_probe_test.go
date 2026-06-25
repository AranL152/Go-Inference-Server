package main

import (
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

// TestORTConcurrencyProbe is the Phase 5 Task-1 gate. It bypasses the batching
// collector and hammers session.Run() directly with a fixed batch, to answer two
// questions that decide the Phase 5 pipeline design:
//
//	A) Does ONE session serialize concurrent Run() calls, or overlap them?
//	B) Do N separate sessions give true compute interleaving on this one GPU?
//
// If one session already overlaps (B>A), a single infer lane suffices and we just
// overlap pack/unpack. If it serializes, we need multiple sessions (lanes) to push
// past the ~1,450 ceiling — or accept that Run() is the hard floor.
func TestORTConcurrencyProbe(t *testing.T) {
	if testing.Short() {
		t.Skip("skip GPU probe in -short")
	}
	// This probe manages its OWN ORT environment (Initialize/DestroyEnvironment),
	// which conflicts with the other GPU tests' lifecycle when run in the same test
	// binary. It is a standalone one-off diagnostic — run it explicitly with
	// ORT_PROBE=1 (its findings are recorded in PHASE5.md / Task 1).
	if os.Getenv("ORT_PROBE") == "" {
		t.Skip("standalone ORT probe; run with ORT_PROBE=1 (owns its ORT env)")
	}
	if _, err := os.Stat("resnet18.onnx"); err != nil {
		t.Skipf("model missing: %v", err)
	}

	const batch = 32
	const libPath = "onnxruntime/lib/libonnxruntime.so"

	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		t.Fatalf("init ORT: %v", err)
	}
	defer ort.DestroyEnvironment()

	ins, outs, err := ort.GetInputOutputInfo("resnet18.onnx")
	if err != nil {
		t.Fatalf("io info: %v", err)
	}
	inName, outName := ins[0].Name, outs[0].Name

	// makeSession builds one CUDA DynamicAdvancedSession and a destroyer.
	makeSession := func() (*ort.DynamicAdvancedSession, func()) {
		opts, err := ort.NewSessionOptions()
		if err != nil {
			t.Fatalf("session opts: %v", err)
		}
		cuda, err := ort.NewCUDAProviderOptions()
		if err != nil {
			t.Fatalf("cuda opts: %v", err)
		}
		if err := cuda.Update(map[string]string{"device_id": "0"}); err != nil {
			t.Fatalf("cuda update: %v", err)
		}
		if err := opts.AppendExecutionProviderCUDA(cuda); err != nil {
			t.Fatalf("append cuda: %v", err)
		}
		sess, err := ort.NewDynamicAdvancedSession("resnet18.onnx",
			[]string{inName}, []string{outName}, opts)
		if err != nil {
			t.Fatalf("new session: %v", err)
		}
		return sess, func() { sess.Destroy(); cuda.Destroy(); opts.Destroy() }
	}

	// oneRun does exactly what runBatch's GPU part does: fresh [batch,3,224,224]
	// input tensor (H2D), Run with ORT-allocated output (D2H), then free both.
	oneRun := func(sess *ort.DynamicAdvancedSession, data []float32) error {
		inT, err := ort.NewTensor(ort.NewShape(int64(batch), channels, imageSize, imageSize), data)
		if err != nil {
			return err
		}
		outputs := []ort.Value{nil}
		runErr := sess.Run([]ort.Value{inT}, outputs)
		inT.Destroy()
		if runErr != nil {
			return runErr
		}
		if outputs[0] != nil {
			outputs[0].Destroy()
		}
		return nil
	}

	// measure runs `totalBatches` Run()s spread over `lanes` goroutines, each lane
	// round-robining over `sessions` (len==1 => shared single session; len==lanes
	// => one session per lane). Returns batches/sec.
	measure := func(name string, sessions []*ort.DynamicAdvancedSession, lanes, totalBatches int) {
		// Each lane has its own input buffer (concurrent Run needs distinct tensors).
		bufs := make([][]float32, lanes)
		for i := range bufs {
			bufs[i] = make([]float32, batch*inputElems)
		}
		var done int64
		var wg sync.WaitGroup
		start := time.Now()
		for l := 0; l < lanes; l++ {
			wg.Add(1)
			go func(l int) {
				defer wg.Done()
				sess := sessions[l%len(sessions)]
				for atomic.AddInt64(&done, 1) <= int64(totalBatches) {
					if err := oneRun(sess, bufs[l]); err != nil {
						t.Errorf("%s: run: %v", name, err)
						return
					}
				}
			}(l)
		}
		wg.Wait()
		el := time.Since(start)
		bps := float64(totalBatches) / el.Seconds()
		t.Logf("%-28s lanes=%-2d sessions=%d -> %.0f batch/s  (%.0f req/s @ batch-%d)  %s",
			name, lanes, len(sessions), bps, bps*float64(batch), batch, el.Round(time.Millisecond))
	}

	const total = 2000

	// A) one session, single-threaded baseline.
	s0, d0 := makeSession()
	measure("A single-session serial", []*ort.DynamicAdvancedSession{s0}, 1, total)

	// B) one session, concurrent Run() from 2 and 4 lanes. Overlap => batch/s rises.
	measure("B single-session conc=2", []*ort.DynamicAdvancedSession{s0}, 2, total)
	measure("B single-session conc=4", []*ort.DynamicAdvancedSession{s0}, 4, total)
	d0()

	// C) N separate sessions, one lane each. True interleaving => batch/s rises with N.
	s2a, d2a := makeSession()
	s2b, d2b := makeSession()
	measure("C two-session conc=2", []*ort.DynamicAdvancedSession{s2a, s2b}, 2, total)
	d2a()
	d2b()

	s3a, d3a := makeSession()
	s3b, d3b := makeSession()
	s3c, d3c := makeSession()
	measure("C three-session conc=3", []*ort.DynamicAdvancedSession{s3a, s3b, s3c}, 3, total)
	d3a()
	d3b()
	d3c()
}
