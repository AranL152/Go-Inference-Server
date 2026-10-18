package main

import (
	"encoding/json"
	"math"
	"net/http/httptest"
	"runtime"
	"slices"
	"testing"
	"time"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
)

func near(t *testing.T, name string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s = %v, want %v (±%v)", name, got, want, tol)
	}
}

// Guest time is already inside user and nice, so counting it again would
// inflate both busy and total.
func TestParseCPUTimes(t *testing.T) {
	stat := "cpu  100 10 50 800 40 5 5 0 30 0\n" +
		"cpu0 60 0 20 400 20 0 0 0 30 0\n" +
		"cpu1 40 10 30 400 20 5 5 0 0 0\n" +
		"intr 12345 0 0\n"
	got := parseCPUTimes(stat)
	want := []cpuTimes{{busy: 170, total: 1010}, {busy: 80, total: 500}, {busy: 90, total: 510}}
	if len(got) != len(want) {
		t.Fatalf("got %d rows, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("row %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestParseMemInfo(t *testing.T) {
	used, total := parseMemInfo("MemTotal:       16384000 kB\nMemFree:         1000000 kB\nMemAvailable:    4096000 kB\nBuffers: 1 kB\n")
	near(t, "total", total, 16000, 1e-9)
	near(t, "used", used, 12000, 1e-9)
}

// The process name can hold spaces and parens; utime and stime are fields 14
// and 15 counted after it.
func TestParseProcStat(t *testing.T) {
	stat := "1234 (my (odd) proc) S 1 1 1 0 -1 4194304 100 0 0 0 250 75 0 0 20 0 12 0 100\n"
	ticks, rss := parseProcStat(stat, "5000 2560 300 1 0 900 0\n", 4096)
	near(t, "ticks", ticks, 325, 0)
	near(t, "rss MB", rss, 10, 1e-9)
}

func TestLatencyWindowPercentiles(t *testing.T) {
	var l latencyWindow
	for i := 100; i >= 1; i-- {
		l.add(time.Duration(i) * time.Millisecond)
	}
	p50, p99 := l.drain()
	near(t, "p50", p50, 50, 1e-9)
	near(t, "p99", p99, 99, 1e-9)
	if p50, p99 = l.drain(); p50 != 0 || p99 != 0 {
		t.Errorf("drain did not reset the window: p50=%v p99=%v", p50, p99)
	}
	l.add(7 * time.Millisecond)
	p50, p99 = l.drain()
	near(t, "single p50", p50, 7, 1e-9)
	near(t, "single p99", p99, 7, 1e-9)
}

// NVML reports the reference clock in microseconds and the violation in
// nanoseconds: a quarter second capped in one second is 25 percent.
func TestViolationPctUnits(t *testing.T) {
	prev := nvml.ViolationTime{ReferenceTime: 1_000_000, ViolationTime: 0}
	near(t, "quarter", violationPct(&prev, nvml.ViolationTime{ReferenceTime: 2_000_000, ViolationTime: 250_000_000}), 25, 1e-9)
	near(t, "none", violationPct(&prev, nvml.ViolationTime{ReferenceTime: 3_000_000, ViolationTime: 250_000_000}), 0, 0)
	near(t, "all", violationPct(&prev, nvml.ViolationTime{ReferenceTime: 3_500_000, ViolationTime: 750_000_000}), 100, 1e-9)
	near(t, "counter reset", violationPct(&prev, nvml.ViolationTime{ReferenceTime: 4_000_000, ViolationTime: 10}), 0, 0)
	near(t, "clock stalled", violationPct(&prev, nvml.ViolationTime{ReferenceTime: 4_000_000, ViolationTime: 20}), 0, 0)
}

func TestFillPipeline(t *testing.T) {
	ms := int64(time.Millisecond)
	prev := pipelineCounters{batches: 5, reqs: 20, waitNs: 99 * ms, decoded: 20, decodeNs: 50 * ms}
	cur := prev
	cur.batches += 10
	cur.reqs += 80
	cur.waitNs += 80 * 2 * ms
	cur.runNs += 10 * 5 * ms
	cur.busyNs += 600 * ms
	cur.idleNs += 400 * ms
	cur.h2dNs += 10 * ms
	cur.computeNs += 10 * 3 * ms
	cur.d2hNs += 10 * ms / 2
	cur.postNs += 10 * ms / 5
	cur.decoded += 80
	cur.decodeNs += 80 * 4 * ms
	cur.letterboxNs += 80 * ms

	var s sysSample
	fillPipeline(&s, prev, cur, 2)
	for _, c := range []struct {
		name      string
		got, want float64
	}{
		{"req/s", s.ReqPerSec, 40},
		{"avg batch", s.AvgBatch, 8},
		{"queue ms", s.QueueMs, 2},
		{"run ms", s.RunMs, 5},
		{"lane busy", s.LaneBusy, 60},
		{"h2d ms", s.H2DMs, 1},
		{"compute ms", s.ComputeMs, 3},
		{"d2h ms", s.D2HMs, 0.5},
		{"post ms", s.PostMs, 0.2},
		{"decode ms", s.DecodeMs, 4},
		{"letterbox ms", s.LetterboxMs, 1},
	} {
		near(t, c.name, c.got, c.want, 1e-9)
	}

	// an idle interval and a counter that went backwards both read as zero, not NaN or negative
	for name, pair := range map[string][2]pipelineCounters{"idle": {cur, cur}, "reset": {cur, prev}} {
		var z sysSample
		fillPipeline(&z, pair[0], pair[1], 1)
		b, _ := json.Marshal(z)
		var back sysSample
		if err := json.Unmarshal(b, &back); err != nil {
			t.Fatalf("%s: sample does not round trip through JSON: %v", name, err)
		}
		if z.ReqPerSec != 0 || z.AvgBatch != 0 || z.QueueMs != 0 || z.LaneBusy != 0 || z.ComputeMs != 0 || z.DecodeMs != 0 {
			t.Errorf("%s interval should read zero, got %+v", name, z)
		}
	}
}

func TestBatchTracerRing(t *testing.T) {
	tr := newBatchTracer(4)
	if got := tr.since(-1); len(got) != 0 {
		t.Fatalf("empty tracer returned %d spans", len(got))
	}
	for i := range 6 {
		tr.add(batchSpan{N: i + 1})
	}
	seqs := func(s []batchSpan) (out []int64) {
		for _, x := range s {
			out = append(out, x.Seq)
		}
		return
	}
	all := tr.since(-1)
	if got := seqs(all); len(got) != 4 || got[0] != 2 || got[3] != 5 {
		t.Errorf("since(-1) = %v, want the four newest 2..5", got)
	}
	if all[0].N != 3 {
		t.Errorf("oldest kept span has N=%d, want 3", all[0].N)
	}
	if got := seqs(tr.since(3)); len(got) != 2 || got[0] != 4 || got[1] != 5 {
		t.Errorf("since(3) = %v, want [4 5]", got)
	}
	if got := tr.since(5); len(got) != 0 {
		t.Errorf("since(newest) returned %d spans", len(got))
	}
}

func TestHandleSystemSince(t *testing.T) {
	m := &sysMonitor{ring: []sysSample{{T: 1000}, {T: 2000}, {T: 3000}}}
	rec := httptest.NewRecorder()
	m.handleSystem(rec, httptest.NewRequest("GET", "/api/system?since=2000", nil))
	var body struct {
		Samples []sysSample `json:"samples"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if len(body.Samples) != 1 || body.Samples[0].T != 3000 {
		t.Errorf("since=2000 returned %+v, want only T=3000", body.Samples)
	}
}

// Spinning one goroutine for a second should read as about one core of this
// process, measured the same way the dashboard does.
func TestProcCPUMatchesKnownBurn(t *testing.T) {
	if testing.Short() {
		t.Skip("timing test")
	}
	cpu0 := readCPUTimes()
	ticks0, _ := readProc()
	start := time.Now()
	for time.Since(start) < time.Second {
	}
	dt := time.Since(start).Seconds()
	ticks1, _ := readProc()
	cpu1 := readCPUTimes()

	oneCore := 100 / float64(runtime.NumCPU())
	proc := pct((ticks1-ticks0)/clockTicks, dt*float64(runtime.NumCPU()))
	near(t, "process CPU %", proc, oneCore, oneCore*0.35)
	host := pct(cpu1[0].busy-cpu0[0].busy, cpu1[0].total-cpu0[0].total)
	if host < proc*0.6 {
		t.Errorf("host CPU %.1f%% is below this process's %.1f%%", host, proc)
	}
}

func TestSummarizeBatches(t *testing.T) {
	ms := int64(time.Millisecond)
	full := batchSpan{N: 4, Gather: 100 * ms, Pack: 102 * ms, Packed: 103 * ms, Infer: 103 * ms}
	partial := batchSpan{N: 1, Gather: 110 * ms, Pack: 115 * ms, Packed: 116 * ms, Infer: 117 * ms}
	old := batchSpan{N: 2, Gather: 0, Pack: 1 * ms, Packed: 1 * ms, Infer: 2 * ms}
	late := batchSpan{N: 3, Gather: 900 * ms, Pack: 901 * ms, Packed: 901 * ms, Infer: 902 * ms}
	st := summarizeBatches([]batchSpan{old, full, full, partial, late}, 4, 5*time.Millisecond, 50*ms, 500*ms)
	if st.Batches != 3 {
		t.Fatalf("counted %d batches, want the 3 inside the window", st.Batches)
	}
	if want := []int{1, 0, 0, 2}; !slices.Equal(st.Sizes, want) {
		t.Errorf("sizes %v, want %v", st.Sizes, want)
	}
	near(t, "avg size", st.AvgSize, 3, 1e-9)
	near(t, "full pct", st.FullPct, 200.0/3, 1e-9)
	near(t, "fill p50", st.FillP50Ms, 2, 1e-9)
	near(t, "fill p99", st.FillP99Ms, 5, 1e-9)
	near(t, "lane wait p99", st.LaneWaitP99, 1, 1e-9)
	near(t, "max wait", st.MaxWaitMs, 5, 1e-9)

	if empty := summarizeBatches(nil, 4, time.Millisecond, 0, 1); empty.Batches != 0 || empty.AvgSize != 0 {
		t.Errorf("no spans should give an empty summary, got %+v", empty)
	}
}
