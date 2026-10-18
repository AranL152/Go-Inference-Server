// On demand execution trace and CPU profile captures of the live server, summarised per pipeline stage
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime/pprof"
	"runtime/trace"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/pprof/profile"
	xtrace "golang.org/x/exp/trace"
)

const (
	profileDir   = "profiles"
	keepCaptures = 20
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

var errCaptureBusy = errors.New("a capture is already running")

// captureTrace records an execution trace of the running server for d and
// returns its per stage summary
func captureTrace(ctx context.Context, d time.Duration) (*traceSummary, error) {
	if !captureMu.TryLock() {
		return nil, errCaptureBusy
	}
	defer captureMu.Unlock()
	os.MkdirAll(profileDir, 0o755)
	now := time.Now()
	name := "trace-" + now.Format("20060102-150405") + ".out"
	f, err := os.Create(filepath.Join(profileDir, name))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	if err := trace.Start(f); err != nil {
		return nil, err
	}
	select {
	case <-time.After(d):
	case <-ctx.Done():
	}
	trace.Stop()
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	sum, err := analyzeTrace(f)
	if err != nil {
		return nil, fmt.Errorf("could not read trace: %w", err)
	}
	sum.File, sum.Taken = name, now
	pruneCaptures("trace")
	return sum, nil
}

func handleTraceCapture(w http.ResponseWriter, r *http.Request) {
	sum, err := captureTrace(r.Context(), captureSeconds(r, 2, 10))
	if err != nil {
		writeError(w, captureStatus(err), err.Error())
		return
	}
	saveSummary("trace-latest.json", sum)
	writeJSON(w, http.StatusOK, sum)
}

func captureStatus(err error) int {
	if errors.Is(err, errCaptureBusy) {
		return http.StatusConflict
	}
	return http.StatusInternalServerError
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

type funcShare struct {
	Func string  `json:"func"`
	Pct  float64 `json:"pct"`
}

type stageCPU struct {
	Stage string      `json:"stage"`
	Pct   float64     `json:"pct"`   // share of all sampled CPU
	Cores float64     `json:"cores"` // average cores busy in this stage
	Top   []funcShare `json:"top"`   // hottest functions inside the stage
}

type cpuSummary struct {
	File    string      `json:"file"`
	Taken   time.Time   `json:"taken"`
	Seconds float64     `json:"seconds"`
	Cores   float64     `json:"cores"` // average cores busy across the process
	Stages  []stageCPU  `json:"stages"`
	Top     []funcShare `json:"top"`
}

// analyzeCPUProfile splits CPU samples by pipeline stage label and lists the
// functions where the samples landed
func analyzeCPUProfile(r io.Reader, wall time.Duration) (*cpuSummary, error) {
	p, err := profile.Parse(r)
	if err != nil {
		return nil, err
	}
	vi := len(p.SampleType) - 1 // cpu nanoseconds
	var total int64
	byStage := map[string]int64{}
	flat := map[string]int64{}
	stageFlat := map[string]map[string]int64{}
	for _, s := range p.Sample {
		v := s.Value[vi]
		total += v
		stage := "other"
		if l := s.Label["stage"]; len(l) > 0 {
			stage = l[0]
		}
		byStage[stage] += v
		fn := "unknown"
		if len(s.Location) > 0 && len(s.Location[0].Line) > 0 {
			fn = s.Location[0].Line[0].Function.Name
		}
		flat[fn] += v
		if stageFlat[stage] == nil {
			stageFlat[stage] = map[string]int64{}
		}
		stageFlat[stage][fn] += v
	}
	pct := func(v int64) float64 {
		if total == 0 {
			return 0
		}
		return 100 * float64(v) / float64(total)
	}
	top := func(m map[string]int64, n int) []funcShare {
		var out []funcShare
		for f, v := range m {
			out = append(out, funcShare{Func: f, Pct: pct(v)})
		}
		sort.Slice(out, func(i, j int) bool { return out[i].Pct > out[j].Pct })
		return out[:min(len(out), n)]
	}
	sum := &cpuSummary{Seconds: wall.Seconds(), Top: top(flat, 12)}
	if wall > 0 {
		sum.Cores = float64(total) / float64(wall)
	}
	for _, name := range append(slices.Clone(stageNames), "other") {
		v, ok := byStage[name]
		if !ok {
			continue
		}
		sc := stageCPU{Stage: name, Pct: pct(v), Top: top(stageFlat[name], 3)}
		if wall > 0 {
			sc.Cores = float64(v) / float64(wall)
		}
		sum.Stages = append(sum.Stages, sc)
	}
	sort.Slice(sum.Stages, func(i, j int) bool { return sum.Stages[i].Pct > sum.Stages[j].Pct })
	return sum, nil
}

// captureCPU records a CPU profile of the running server for d and returns
// where the CPU went by stage and function
func captureCPU(ctx context.Context, d time.Duration) (*cpuSummary, error) {
	if !captureMu.TryLock() {
		return nil, errCaptureBusy
	}
	defer captureMu.Unlock()
	os.MkdirAll(profileDir, 0o755)
	now := time.Now()
	name := "cpu-" + now.Format("20060102-150405") + ".pb.gz"
	f, err := os.Create(filepath.Join(profileDir, name))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	if err := pprof.StartCPUProfile(f); err != nil {
		return nil, err
	}
	start := time.Now()
	select {
	case <-time.After(d):
	case <-ctx.Done():
	}
	pprof.StopCPUProfile()
	wall := time.Since(start)
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	sum, err := analyzeCPUProfile(f, wall)
	if err != nil {
		return nil, fmt.Errorf("could not read profile: %w", err)
	}
	sum.File, sum.Taken = name, now
	pruneCaptures("cpu")
	return sum, nil
}

func handleCPUCapture(w http.ResponseWriter, r *http.Request) {
	sum, err := captureCPU(r.Context(), captureSeconds(r, 3, 30))
	if err != nil {
		writeError(w, captureStatus(err), err.Error())
		return
	}
	saveSummary("cpu-latest.json", sum)
	writeJSON(w, http.StatusOK, sum)
}
