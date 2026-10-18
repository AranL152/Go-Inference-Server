// Per run performance records, so a finished job keeps its dashboard
package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// runPerf is what the dashboard shows for one job: its per second samples,
// how its batches formed, and a CPU profile and trace taken while it ran
type runPerf struct {
	JobID    string        `json:"job_id"`
	Start    time.Time     `json:"start"`
	End      time.Time     `json:"end"`
	Samples  []sysSample   `json:"samples"`
	Batching batchingStats `json:"batching"`
	CPU      *cpuSummary   `json:"cpu"`
	Trace    *traceSummary `json:"trace"`
}

// run captures start this far in so they see steady state, not ramp up
const (
	runWarmup     = 500 * time.Millisecond
	runCPUCapture = 2 * time.Second
	runTraceSpan  = time.Second
)

// between returns the samples taken from a up to and including b
func (m *sysMonitor) between(a, b time.Time) []sysSample {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := []sysSample{}
	for _, s := range m.ring {
		if s.T > a.UnixMilli() && s.T <= b.UnixMilli() {
			out = append(out, s)
		}
	}
	return out
}

// recordRun profiles the server while a job runs and returns its record once
// done closes. Captures the run is too short for are skipped.
func (st *jobStore) recordRun(j *job, done <-chan struct{}) *runPerf {
	e := st.s.engine
	p := &runPerf{JobID: j.ID, Start: time.Now()}
	from := e.now()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	wait := func(d time.Duration) bool {
		select {
		case <-time.After(d):
			return true
		case <-done:
			return false
		}
	}
	if wait(runWarmup) {
		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			cpu, err := captureCPU(ctx, runCPUCapture)
			if err != nil {
				log.Printf("job %s: CPU profile skipped: %v", j.ID, err)
			}
			p.CPU = cpu
		}()
		go func() {
			defer wg.Done()
			tr, err := captureTrace(ctx, runTraceSpan)
			if err != nil {
				log.Printf("job %s: trace skipped: %v", j.ID, err)
			}
			p.Trace = tr
		}()
		wg.Wait()
	}
	<-done
	p.End = time.Now()
	p.Batching = summarizeBatches(e.tracer.since(-1), e.maxBatch, e.maxWait, from, e.now())
	// the sample covering the last partial second lands up to a second later
	time.Sleep(sampleEvery + 100*time.Millisecond)
	if st.mon != nil {
		p.Samples = st.mon.between(p.Start, p.End.Add(sampleEvery))
	}
	return p
}

func (st *jobStore) perfPath(id string) string { return filepath.Join(st.dir, id+".perf.json") }

func (st *jobStore) savePerf(p *runPerf) {
	st.mu.Lock()
	if st.perf == nil {
		st.perf = map[string]*runPerf{}
	}
	st.perf[p.JobID] = p
	st.mu.Unlock()
	if st.dir == "" {
		return
	}
	b, err := json.Marshal(p)
	if err == nil {
		err = os.WriteFile(st.perfPath(p.JobID), b, 0o644)
	}
	if err != nil {
		log.Printf("save run perf %s: %v", p.JobID, err)
	}
}

func (st *jobStore) handlePerf(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	st.mu.Lock()
	p := st.perf[id]
	st.mu.Unlock()
	if p != nil {
		writeJSON(w, http.StatusOK, p)
		return
	}
	if st.dir != "" && st.get(id) != nil {
		if b, err := os.ReadFile(st.perfPath(id)); err == nil && json.Valid(b) {
			w.Header().Set("Content-Type", "application/json")
			w.Write(b)
			return
		}
	}
	writeError(w, http.StatusNotFound, "no performance record for this job")
}
