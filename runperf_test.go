package main

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// A finished job keeps a record of its own run: samples from while it ran,
// batches that account for every image, and a CPU profile taken mid run
func TestJobRecordsItsRun(t *testing.T) {
	if testing.Short() {
		t.Skip("needs a GPU")
	}
	const model = "yolox_s_u8.onnx"
	if _, err := os.Stat(model); err != nil {
		t.Skipf("model missing: %v", err)
	}
	labels, _ := loadLabels("coco_classes.txt")
	e, err := NewEngine(model, "onnxruntime/lib/libonnxruntime.so", labels, 16, 5*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	pool, err := newCPUDecoderPool(16)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	dir := t.TempDir()
	const images = 3000
	srcs := []string{"dog.jpg", "cat.jpg", "street.jpg"}
	for i := range images {
		src, _ := filepath.Abs(filepath.Join("testdata", srcs[i%len(srcs)]))
		if err := os.Symlink(src, filepath.Join(dir, fmt.Sprintf("%05d.jpg", i))); err != nil {
			t.Fatal(err)
		}
	}

	srv := &server{engine: e, cpuPool: pool}
	st := &jobStore{s: srv, mon: startSysMonitor(e, pool)}
	j, err := st.start(dir, dir, "", nil)
	if err != nil {
		t.Fatal(err)
	}
	var p *runPerf
	for deadline := time.Now().Add(2 * time.Minute); p == nil && time.Now().Before(deadline); time.Sleep(200 * time.Millisecond) {
		st.mu.Lock()
		p = st.perf[j.ID]
		st.mu.Unlock()
	}
	if p == nil {
		t.Fatal("no run record after the job finished")
	}
	run := p.End.Sub(p.Start)
	t.Logf("run %.1f s, %d samples, %d batches avg %.1f, cpu %v, trace %v", run.Seconds(), len(p.Samples), p.Batching.Batches, p.Batching.AvgSize, p.CPU != nil, p.Trace != nil)

	if len(p.Samples) < int(run.Seconds()) {
		t.Errorf("%d samples for a %.1f s run", len(p.Samples), run.Seconds())
	}
	for _, s := range p.Samples {
		if s.T <= p.Start.UnixMilli() || s.T > p.End.Add(sampleEvery).UnixMilli() {
			t.Errorf("sample at %d is outside the run", s.T)
		}
	}
	if got := int(p.Batching.AvgSize*float64(p.Batching.Batches) + 0.5); got != images {
		t.Errorf("batches cover %d images, the job had %d", got, images)
	}
	if run > runWarmup+runCPUCapture && (p.CPU == nil || p.Trace == nil) {
		t.Errorf("run outlasted the captures but has CPU %v trace %v", p.CPU != nil, p.Trace != nil)
	}
	if p.CPU != nil && (len(p.CPU.Stages) == 0 || p.CPU.Stages[0].Stage != stageDecode) {
		t.Errorf("CPU profile stages %+v, want decode on top for a detection job", p.CPU.Stages)
	}
}
