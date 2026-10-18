// On demand execution trace captures of the live server, summarised per pipeline stage
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime/trace"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	xtrace "golang.org/x/exp/trace"
)

const (
	profileDir   = "profiles"
	keepCaptures = 5
)

// captureMu allows one trace or CPU profile at a time, as the runtime does
var captureMu sync.Mutex

var captureName = regexp.MustCompile(`^(trace|cpu)-\d{8}-\d{6}\.(out|pb\.gz)$`)

type waitReason struct {
	Reason string  `json:"reason"`
	Pct    float64 `json:"pct"`
}

// stageTime splits the goroutine time spent in one stage by scheduler state
type stageTime struct {
	Stage      string       `json:"stage"`
	Goroutines int          `json:"goroutines"`
	Seconds    float64      `json:"seconds"` // goroutine seconds, so it can exceed the capture
	Running    float64      `json:"running_pct"`
	Runnable   float64      `json:"runnable_pct"` // ready to run but waiting for a CPU
	Blocked    float64      `json:"blocked_pct"`
	Syscall    float64      `json:"syscall_pct"`
	Waits      []waitReason `json:"waits"`
	SchedP50Us float64      `json:"sched_p50_us"` // runnable to running delay
	SchedP99Us float64      `json:"sched_p99_us"`
}

type traceSummary struct {
	File      string      `json:"file"`
	Taken     time.Time   `json:"taken"`
	Seconds   float64     `json:"seconds"`
	Stages    []stageTime `json:"stages"`
	GCs       int         `json:"gcs"`
	GCMarkPct float64     `json:"gc_mark_pct"` // share of the capture with a GC mark phase running
	STWMs     float64     `json:"stw_ms"`      // total stop the world time
}

type goTrack struct {
	state    xtrace.GoState
	since    xtrace.Time
	readyAt  xtrace.Time
	reason   string
	regions  []string
	tracking bool
}

type stageAcc struct {
	goroutines map[xtrace.GoID]bool
	byState    [6]time.Duration
	waits      map[string]time.Duration
	sched      []time.Duration
}

// analyzeTrace reads an execution trace and splits the time goroutines spent
// inside each stage region into running, runnable, blocked and syscall, along
// with what they blocked on and how long they waited for a CPU
func analyzeTrace(r io.Reader) (*traceSummary, error) {
	rd, err := xtrace.NewReader(bufio.NewReader(r))
	if err != nil {
		return nil, err
	}
	gs := map[xtrace.GoID]*goTrack{}
	accs := map[string]*stageAcc{}
	acc := func(stage string) *stageAcc {
		a := accs[stage]
		if a == nil {
			a = &stageAcc{goroutines: map[xtrace.GoID]bool{}, waits: map[string]time.Duration{}}
			accs[stage] = a
		}
		return a
	}
	charge := func(id xtrace.GoID, g *goTrack, now xtrace.Time) {
		if g.tracking && len(g.regions) > 0 {
			a := acc(g.regions[len(g.regions)-1])
			d := now.Sub(g.since)
			a.byState[g.state] += d
			if g.state == xtrace.GoWaiting {
				a.waits[g.reason] += d
			}
			a.goroutines[id] = true
		}
		g.since = now
	}
	track := func(id xtrace.GoID, state xtrace.GoState, now xtrace.Time) *goTrack {
		g := gs[id]
		if g == nil {
			g = &goTrack{state: state, since: now, tracking: true}
			gs[id] = g
		}
		return g
	}

	var first, last xtrace.Time
	seen := false
	open := map[string]xtrace.Time{}
	var stw, mark time.Duration
	gcs := 0
	for {
		ev, err := rd.ReadEvent()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		t := ev.Time()
		if !seen {
			first, seen = t, true
		}
		last = t
		switch ev.Kind() {
		case xtrace.EventStateTransition:
			st := ev.StateTransition()
			if st.Resource.Kind != xtrace.ResourceGoroutine {
				continue
			}
			id := st.Resource.Goroutine()
			from, to := st.Goroutine()
			g := track(id, from, t)
			charge(id, g, t)
			if from == xtrace.GoRunnable && to == xtrace.GoRunning && len(g.regions) > 0 && g.readyAt != 0 {
				a := acc(g.regions[len(g.regions)-1])
				a.sched = append(a.sched, t.Sub(g.readyAt))
			}
			g.state = to
			switch to {
			case xtrace.GoRunnable:
				g.readyAt = t
			case xtrace.GoWaiting:
				g.reason = st.Reason
			case xtrace.GoNotExist:
				delete(gs, id)
			}
		case xtrace.EventRegionBegin:
			id := ev.Goroutine()
			g := track(id, xtrace.GoRunning, t)
			charge(id, g, t)
			g.regions = append(g.regions, ev.Region().Type)
		case xtrace.EventRegionEnd:
			id := ev.Goroutine()
			g := track(id, xtrace.GoRunning, t)
			charge(id, g, t)
			if n := len(g.regions); n > 0 && g.regions[n-1] == ev.Region().Type {
				g.regions = g.regions[:n-1]
			}
		case xtrace.EventRangeBegin, xtrace.EventRangeActive:
			rg := ev.Range()
			at := t
			if ev.Kind() == xtrace.EventRangeActive {
				at = first
			}
			open[rg.Name+rg.Scope.String()] = at
		case xtrace.EventRangeEnd:
			rg := ev.Range()
			key := rg.Name + rg.Scope.String()
			start, ok := open[key]
			if !ok {
				continue
			}
			delete(open, key)
			switch d := t.Sub(start); {
			case strings.HasPrefix(rg.Name, "stop-the-world"):
				stw += d
			case rg.Name == "GC concurrent mark phase":
				mark += d
				gcs++
			}
		}
	}
	for id, g := range gs {
		charge(id, g, last)
	}

	wall := last.Sub(first)
	sum := &traceSummary{Seconds: wall.Seconds(), GCs: gcs, STWMs: float64(stw) / 1e6}
	if wall > 0 {
		sum.GCMarkPct = 100 * float64(mark) / float64(wall)
	}
	for _, name := range stageNames {
		a := accs[name]
		if a == nil {
			continue
		}
		var total time.Duration
		for _, d := range a.byState {
			total += d
		}
		if total == 0 {
			continue
		}
		p := func(d time.Duration) float64 { return 100 * float64(d) / float64(total) }
		s := stageTime{
			Stage:      name,
			Goroutines: len(a.goroutines),
			Seconds:    total.Seconds(),
			Running:    p(a.byState[xtrace.GoRunning]),
			Runnable:   p(a.byState[xtrace.GoRunnable]),
			Blocked:    p(a.byState[xtrace.GoWaiting]),
			Syscall:    p(a.byState[xtrace.GoSyscall]),
		}
		for reason, d := range a.waits {
			if reason == "" {
				reason = "unknown"
			}
			s.Waits = append(s.Waits, waitReason{Reason: reason, Pct: p(d)})
		}
		sort.Slice(s.Waits, func(i, j int) bool { return s.Waits[i].Pct > s.Waits[j].Pct })
		s.Waits = s.Waits[:min(len(s.Waits), 3)]
		if n := len(a.sched); n > 0 {
			slices.Sort(a.sched)
			rank := func(q float64) float64 { return float64(a.sched[max(int(math.Ceil(q*float64(n)))-1, 0)]) / 1e3 }
			s.SchedP50Us, s.SchedP99Us = rank(0.50), rank(0.99)
		}
		sum.Stages = append(sum.Stages, s)
	}
	return sum, nil
}

func captureSeconds(r *http.Request, def, most int) time.Duration {
	n, err := strconv.Atoi(r.URL.Query().Get("seconds"))
	if err != nil || n < 1 {
		n = def
	}
	return time.Duration(min(n, most)) * time.Second
}

// pruneCaptures keeps the newest few raw captures of one kind
func pruneCaptures(prefix string) {
	files, _ := filepath.Glob(filepath.Join(profileDir, prefix+"-*"))
	var raw []string
	for _, f := range files {
		if captureName.MatchString(filepath.Base(f)) {
			raw = append(raw, f)
		}
	}
	sort.Strings(raw)
	for _, f := range raw[:max(len(raw)-keepCaptures, 0)] {
		os.Remove(f)
	}
}

func saveSummary(name string, v any) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err == nil {
		os.WriteFile(filepath.Join(profileDir, name), b, 0o644)
	}
}

// handleTraceCapture records an execution trace of the running server and
// returns its per stage summary
func handleTraceCapture(w http.ResponseWriter, r *http.Request) {
	d := captureSeconds(r, 2, 10)
	if !captureMu.TryLock() {
		writeError(w, http.StatusConflict, "a capture is already running")
		return
	}
	defer captureMu.Unlock()
	os.MkdirAll(profileDir, 0o755)
	now := time.Now()
	name := "trace-" + now.Format("20060102-150405") + ".out"
	f, err := os.Create(filepath.Join(profileDir, name))
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer f.Close()
	if err := trace.Start(f); err != nil {
		writeError(w, http.StatusConflict, "could not start trace: "+err.Error())
		return
	}
	select {
	case <-time.After(d):
	case <-r.Context().Done():
	}
	trace.Stop()
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	sum, err := analyzeTrace(f)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "could not read trace: "+err.Error())
		return
	}
	sum.File, sum.Taken = name, now
	saveSummary("trace-latest.json", sum)
	pruneCaptures("trace")
	writeJSON(w, http.StatusOK, sum)
}

// handleLatest returns the newest saved summary of one capture kind
func handleLatest(kind string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		b, err := os.ReadFile(filepath.Join(profileDir, kind+"-latest.json"))
		if err != nil {
			writeJSON(w, http.StatusOK, nil)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write(b)
	}
}

// handleCaptureFile serves a raw capture for go tool trace or go tool pprof
func handleCaptureFile(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if !captureName.MatchString(name) {
		writeError(w, http.StatusNotFound, "no such capture")
		return
	}
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", name))
	http.ServeFile(w, r, filepath.Join(profileDir, name))
}
