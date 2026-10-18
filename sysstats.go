// Live GPU CPU memory and pipeline samples for the performance view
package main

import (
	"log"
	"math"
	"net/http"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	sampleEvery   = time.Second
	sampleHistory = 900
)

type gpuSample struct {
	Name       string  `json:"name"`
	Util       float64 `json:"util"`
	MemUtil    float64 `json:"mem_util"`
	MemUsedMB  float64 `json:"mem_used_mb"`
	MemTotalMB float64 `json:"mem_total_mb"`
	TempC      float64 `json:"temp_c"`
	PowerW     float64 `json:"power_w"`
	PowerMaxW  float64 `json:"power_limit_w"`
	SMClock    float64 `json:"sm_clock_mhz"`
	SMClockMax float64 `json:"sm_clock_max_mhz"`
	MemClock   float64 `json:"mem_clock_mhz"`
	Fan        float64 `json:"fan"`
	PCIeGen    float64 `json:"pcie_gen"`
	PCIeWidth  float64 `json:"pcie_width"`

	PCIeRxMBs      float64  `json:"pcie_rx_mbs"`
	PCIeTxMBs      float64  `json:"pcie_tx_mbs"`
	EncUtil        float64  `json:"enc_util"`
	DecUtil        float64  `json:"dec_util"`
	JpgUtil        float64  `json:"jpg_util"`
	Throttle       []string `json:"throttle"`
	PowerViolPct   float64  `json:"power_violation_pct"`
	ThermalViolPct float64  `json:"thermal_violation_pct"`
}

type sysSample struct {
	T          int64      `json:"t"`
	GPU        *gpuSample `json:"gpu"`
	CPU        float64    `json:"cpu"`
	Cores      []float64  `json:"cores"`
	MemUsedMB  float64    `json:"mem_used_mb"`
	MemTotalMB float64    `json:"mem_total_mb"`
	ProcCPU    float64    `json:"proc_cpu"`
	ProcRSSMB  float64    `json:"proc_rss_mb"`
	Goroutines int        `json:"goroutines"`
	ReqPerSec  float64    `json:"req_per_sec"`
	AvgBatch   float64    `json:"avg_batch"`
	QueueMs    float64    `json:"queue_ms"`
	RunMs      float64    `json:"run_ms"`
	LaneBusy   float64    `json:"lane_busy"`
	P50Ms      float64    `json:"p50_ms"`
	P99Ms      float64    `json:"p99_ms"`

	// where the time goes per image for decode and letterbox and per batch for the GPU stages
	DecodeMs    float64 `json:"decode_ms"`
	LetterboxMs float64 `json:"letterbox_ms"`
	H2DMs       float64 `json:"h2d_ms"`
	ComputeMs   float64 `json:"compute_ms"`
	D2HMs       float64 `json:"d2h_ms"`
	PostMs      float64 `json:"post_ms"`
}

// latencyWindow collects request latencies between samples
type latencyWindow struct {
	mu sync.Mutex
	ms []float64
}

var recentLatency latencyWindow

func (l *latencyWindow) add(d time.Duration) {
	l.mu.Lock()
	if len(l.ms) < 1<<16 {
		l.ms = append(l.ms, float64(d)/1e6)
	}
	l.mu.Unlock()
}

func (l *latencyWindow) drain() (p50, p99 float64) {
	l.mu.Lock()
	v := l.ms
	l.ms = make([]float64, 0, len(v))
	l.mu.Unlock()
	if len(v) == 0 {
		return 0, 0
	}
	slices.Sort(v)
	// nearest rank: the smallest value with at least p of the samples at or below it
	rank := func(p float64) float64 { return v[max(int(math.Ceil(p*float64(len(v))))-1, 0)] }
	return rank(0.50), rank(0.99)
}

type sysMonitor struct {
	engine *Engine
	pool   *cpuDecoderPool
	mu     sync.Mutex
	ring   []sysSample
}

func startSysMonitor(e *Engine, pool *cpuDecoderPool) *sysMonitor {
	m := &sysMonitor{engine: e, pool: pool}
	go m.sample()
	return m
}

type cpuTimes struct{ busy, total float64 }

// parseCPUTimes reads the aggregate and per core lines of /proc/stat. The
// columns are user nice system idle iowait irq softirq steal guest guest_nice;
// guest time is already counted inside user and nice so only the first eight
// go into the total, and idle plus iowait are the time not busy.
func parseCPUTimes(stat string) []cpuTimes {
	var out []cpuTimes
	for _, line := range strings.Split(stat, "\n") {
		if !strings.HasPrefix(line, "cpu") {
			break
		}
		var t cpuTimes
		for i, f := range strings.Fields(line)[1:] {
			if i >= 8 {
				break
			}
			v, _ := strconv.ParseFloat(f, 64)
			t.total += v
			if i != 3 && i != 4 {
				t.busy += v
			}
		}
		out = append(out, t)
	}
	return out
}

func readCPUTimes() []cpuTimes {
	b, err := os.ReadFile("/proc/stat")
	if err != nil {
		return nil
	}
	return parseCPUTimes(string(b))
}

// parseMemInfo returns used and total host memory in MB from /proc/meminfo,
// counting as used whatever the kernel says is not available
func parseMemInfo(meminfo string) (used, total float64) {
	var avail float64
	for _, line := range strings.Split(meminfo, "\n") {
		f := strings.Fields(line)
		if len(f) < 2 {
			continue
		}
		v, _ := strconv.ParseFloat(f[1], 64)
		switch f[0] {
		case "MemTotal:":
			total = v / 1024
		case "MemAvailable:":
			avail = v / 1024
		}
	}
	return total - avail, total
}

func readMemMB() (used, total float64) {
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return
	}
	return parseMemInfo(string(b))
}

// parseProcStat returns user plus system CPU ticks from /proc/self/stat and
// resident memory in MB from /proc/self/statm. The process name in stat can
// hold spaces and parens, so fields are counted from the last closing paren.
func parseProcStat(stat, statm string, pageSize int) (ticks, rssMB float64) {
	if i := strings.LastIndexByte(stat, ')'); i >= 0 && i+2 <= len(stat) {
		f := strings.Fields(stat[i+2:])
		if len(f) > 12 {
			u, _ := strconv.ParseFloat(f[11], 64) // utime, field 14
			k, _ := strconv.ParseFloat(f[12], 64) // stime, field 15
			ticks = u + k
		}
	}
	if f := strings.Fields(statm); len(f) > 1 {
		pages, _ := strconv.ParseFloat(f[1], 64)
		rssMB = pages * float64(pageSize) / (1 << 20)
	}
	return
}

func readProc() (ticks, rssMB float64) {
	stat, _ := os.ReadFile("/proc/self/stat")
	statm, _ := os.ReadFile("/proc/self/statm")
	return parseProcStat(string(stat), string(statm), os.Getpagesize())
}

// clockTicks is USER_HZ, the unit of /proc CPU times, which Linux fixes at 100
const clockTicks = 100

func pct(a, b float64) float64 {
	if b <= 0 {
		return 0
	}
	return 100 * a / b
}

// pipelineCounters is one snapshot of the engine and decoder running totals
type pipelineCounters struct {
	batches, reqs, waitNs, runNs, busyNs, idleNs int64
	h2dNs, computeNs, d2hNs, postNs              int64
	decodeNs, letterboxNs, decoded               int64
}

func (m *sysMonitor) counters() pipelineCounters {
	var c pipelineCounters
	c.batches, c.reqs, _, c.waitNs, c.runNs, _, c.busyNs, c.idleNs = m.engine.CumulativeStats()
	c.h2dNs, c.computeNs, c.d2hNs, c.postNs = m.engine.StageNs()
	if m.pool != nil {
		c.decodeNs = atomic.LoadInt64(&m.pool.statDecodeNs)
		c.letterboxNs = atomic.LoadInt64(&m.pool.statNormNs)
		c.decoded = atomic.LoadInt64(&m.pool.statCount)
	}
	return c
}

// fillPipeline sets the pipeline fields of s from two snapshots dt seconds
// apart. Throughput and queue wait are per request, batch size and the GPU
// stages are per batch, decode and letterbox are per image, and lane busy is
// the share of lane time spent on a batch rather than waiting for one.
func fillPipeline(s *sysSample, prev, cur pipelineCounters, dt float64) {
	d := func(a, b int64) float64 { return float64(max(b-a, 0)) }
	batches, reqs := d(prev.batches, cur.batches), d(prev.reqs, cur.reqs)
	if dt > 0 {
		s.ReqPerSec = reqs / dt
	}
	if batches > 0 {
		s.AvgBatch = reqs / batches
		s.RunMs = d(prev.runNs, cur.runNs) / batches / 1e6
		s.H2DMs = d(prev.h2dNs, cur.h2dNs) / batches / 1e6
		s.ComputeMs = d(prev.computeNs, cur.computeNs) / batches / 1e6
		s.D2HMs = d(prev.d2hNs, cur.d2hNs) / batches / 1e6
		s.PostMs = d(prev.postNs, cur.postNs) / batches / 1e6
	}
	if reqs > 0 {
		s.QueueMs = d(prev.waitNs, cur.waitNs) / reqs / 1e6
	}
	busy := d(prev.busyNs, cur.busyNs)
	s.LaneBusy = pct(busy, busy+d(prev.idleNs, cur.idleNs))
	if n := d(prev.decoded, cur.decoded); n > 0 {
		s.DecodeMs = d(prev.decodeNs, cur.decodeNs) / n / 1e6
		s.LetterboxMs = d(prev.letterboxNs, cur.letterboxNs) / n / 1e6
	}
}

// sampler turns running totals into one sample per interval
type sampler struct {
	m         *sysMonitor
	gpu       *nvmlGPU
	prevCPU   []cpuTimes
	prevTicks float64
	prev      pipelineCounters
	prevT     time.Time
}

func newSampler(m *sysMonitor, gpu *nvmlGPU) *sampler {
	sp := &sampler{m: m, gpu: gpu, prevCPU: readCPUTimes(), prev: m.counters(), prevT: time.Now()}
	sp.prevTicks, _ = readProc()
	return sp
}

func (sp *sampler) step(now time.Time) sysSample {
	dt := now.Sub(sp.prevT).Seconds()
	s := sysSample{T: now.UnixMilli(), Goroutines: runtime.NumGoroutine()}

	cpu := readCPUTimes()
	if len(cpu) == len(sp.prevCPU) && len(cpu) > 0 {
		d := func(i int) float64 { return pct(cpu[i].busy-sp.prevCPU[i].busy, cpu[i].total-sp.prevCPU[i].total) }
		s.CPU = d(0)
		for i := 1; i < len(cpu); i++ {
			s.Cores = append(s.Cores, d(i))
		}
	}
	sp.prevCPU = cpu
	s.MemUsedMB, s.MemTotalMB = readMemMB()
	ticks, rss := readProc()
	s.ProcCPU = pct((ticks-sp.prevTicks)/clockTicks, dt*float64(runtime.NumCPU()))
	s.ProcRSSMB = rss
	sp.prevTicks = ticks

	cur := sp.m.counters()
	fillPipeline(&s, sp.prev, cur, dt)
	sp.prev = cur
	s.P50Ms, s.P99Ms = recentLatency.drain()
	sp.prevT = now

	if sp.gpu != nil {
		s.GPU = sp.gpu.sample()
	}
	return s
}

func (m *sysMonitor) sample() {
	gpu, err := openGPU()
	if err != nil {
		log.Printf("GPU stats unavailable: %v", err)
	}
	sp := newSampler(m, gpu)
	for now := range time.Tick(sampleEvery) {
		s := sp.step(now)
		m.mu.Lock()
		m.ring = append(m.ring, s)
		if len(m.ring) > sampleHistory {
			m.ring = m.ring[len(m.ring)-sampleHistory:]
		}
		m.mu.Unlock()
	}
}

// handleSystem returns samples newer than the since query param in unix ms
func (m *sysMonitor) handleSystem(w http.ResponseWriter, r *http.Request) {
	since, _ := strconv.ParseInt(r.URL.Query().Get("since"), 10, 64)
	m.mu.Lock()
	out := []sysSample{}
	for _, s := range m.ring {
		if s.T > since {
			out = append(out, s)
		}
	}
	m.mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]any{"samples": out, "cores": runtime.NumCPU()})
}
