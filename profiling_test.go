package main

import (
	"bytes"
	"runtime"
	"runtime/pprof"
	"runtime/trace"
	"strings"
	"sync"
	"testing"
	"time"
)

func spin(d time.Duration) {
	for start := time.Now(); time.Since(start) < d; {
	}
}

// traceOf records an execution trace of f and returns its summary by stage
func traceOf(t *testing.T, f func()) map[string]stageTime {
	t.Helper()
	var buf bytes.Buffer
	if err := trace.Start(&buf); err != nil {
		t.Skipf("tracing unavailable: %v", err)
	}
	f()
	trace.Stop()
	sum, err := analyzeTrace(&buf)
	if err != nil {
		t.Fatal(err)
	}
	out := map[string]stageTime{}
	for _, s := range sum.Stages {
		out[s.Stage] = s
	}
	return out
}

func TestAnalyzeTraceStates(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(2))
	stages := traceOf(t, func() {
		var wg sync.WaitGroup
		ch := make(chan int)
		wg.Add(2)
		go func() { // parked on a channel for most of its region
			defer wg.Done()
			defer region(stageInfer).End()
			<-ch
			spin(5 * time.Millisecond)
		}()
		go func() { // busy on a CPU with one to spare
			defer wg.Done()
			defer region(stageDecode).End()
			spin(150 * time.Millisecond)
		}()
		time.Sleep(150 * time.Millisecond)
		ch <- 1
		wg.Wait()
	})
	infer, decode := stages[stageInfer], stages[stageDecode]
	if infer.Blocked < 80 {
		t.Errorf("parked goroutine reads %.0f%% blocked, want most of it", infer.Blocked)
	}
	if len(infer.Waits) == 0 || !strings.Contains(infer.Waits[0].Reason, "chan receive") {
		t.Errorf("parked goroutine top wait is %+v, want chan receive", infer.Waits)
	}
	if decode.Running < 80 {
		t.Errorf("spinning goroutine reads %.0f%% running, want most of it", decode.Running)
	}
}

// More busy goroutines than CPUs leaves each one ready but waiting for a CPU
// much of the time, which is what CPU contention looks like in the summary
func TestAnalyzeTraceCPUContention(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(2))
	stages := traceOf(t, func() {
		var wg sync.WaitGroup
		for range 8 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				defer region(stageDecode).End()
				for range 20 {
					spin(5 * time.Millisecond)
					runtime.Gosched()
				}
			}()
		}
		wg.Wait()
	})
	d := stages[stageDecode]
	if d.Goroutines != 8 {
		t.Errorf("saw %d decode goroutines, want 8", d.Goroutines)
	}
	if d.Runnable < 40 {
		t.Errorf("8 busy goroutines on 2 CPUs read %.0f%% runnable, want a large share", d.Runnable)
	}
	if d.SchedP50Us <= 0 {
		t.Errorf("no scheduler delay measured")
	}
	t.Logf("running %.0f%%, runnable %.0f%%, sched delay p50 %.0f µs p99 %.0f µs", d.Running, d.Runnable, d.SchedP50Us, d.SchedP99Us)
}

var sink int

// burn keeps a CPU busy mostly in its own frame rather than in the clock
func burn(d time.Duration) {
	x := 0
	for start := time.Now(); time.Since(start) < d; {
		for i := range 1_000_000 {
			x += i * i
		}
	}
	sink = x
}

// A labelled busy goroutine should own the CPU samples and an idle one none
func TestAnalyzeCPUProfileByStage(t *testing.T) {
	var buf bytes.Buffer
	if err := pprof.StartCPUProfile(&buf); err != nil {
		t.Skipf("profiling unavailable: %v", err)
	}
	start := time.Now()
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		setStage(stageDecode)
		burn(600 * time.Millisecond)
	}()
	go func() {
		defer wg.Done()
		setStage(stageInfer)
		time.Sleep(600 * time.Millisecond)
	}()
	wg.Wait()
	pprof.StopCPUProfile()
	sum, err := analyzeCPUProfile(&buf, time.Since(start))
	if err != nil {
		t.Fatal(err)
	}
	if len(sum.Stages) == 0 || sum.Stages[0].Stage != stageDecode || sum.Stages[0].Pct < 80 {
		t.Fatalf("stages %+v, want decode to hold most samples", sum.Stages)
	}
	near(t, "decode cores", sum.Stages[0].Cores, 1, 0.3)
	if !strings.HasSuffix(sum.Top[0].Func, ".burn") {
		t.Errorf("hottest function %q, want burn", sum.Top[0].Func)
	}
	for _, s := range sum.Stages {
		if s.Stage == stageInfer && s.Pct > 5 {
			t.Errorf("idle infer goroutine got %.0f%% of samples", s.Pct)
		}
	}
}
