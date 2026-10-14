// Live GPU CPU memory and pipeline samples for the performance view
package main

import (
	"bufio"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	sampleEvery   = time.Second
	sampleHistory = 120
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
	return v[len(v)/2], v[min(len(v)*99/100, len(v)-1)]
}

type sysMonitor struct {
	engine *Engine
	mu     sync.Mutex
	gpu    *gpuSample
	ring   []sysSample
}

func startSysMonitor(e *Engine) *sysMonitor {
	m := &sysMonitor{engine: e}
	go m.readGPU()
	go m.sample()
	return m
}

var gpuFields = "name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu," +
	"power.draw,power.limit,clocks.sm,clocks.max.sm,clocks.mem,fan.speed,pcie.link.gen.current,pcie.link.width.current"

// readGPU streams NVML readings from nvidia-smi and restarts it if it exits
func (m *sysMonitor) readGPU() {
	for {
		cmd := exec.Command("nvidia-smi", "--query-gpu="+gpuFields, "--format=csv,noheader,nounits", "-lms", "1000")
		out, err := cmd.StdoutPipe()
		if err == nil && cmd.Start() == nil {
			sc := bufio.NewScanner(out)
			for sc.Scan() {
				f := strings.Split(sc.Text(), ", ")
				if len(f) < 14 {
					continue
				}
				num := func(i int) float64 { v, _ := strconv.ParseFloat(strings.TrimSpace(f[i]), 64); return v }
				g := &gpuSample{Name: f[0], Util: num(1), MemUtil: num(2), MemUsedMB: num(3), MemTotalMB: num(4),
					TempC: num(5), PowerW: num(6), PowerMaxW: num(7), SMClock: num(8), SMClockMax: num(9),
					MemClock: num(10), Fan: num(11), PCIeGen: num(12), PCIeWidth: num(13)}
				m.mu.Lock()
				m.gpu = g
				m.mu.Unlock()
			}
			cmd.Wait()
		}
		time.Sleep(5 * time.Second)
	}
}

type cpuTimes struct{ busy, total float64 }

func readCPUTimes() []cpuTimes {
	b, err := os.ReadFile("/proc/stat")
	if err != nil {
		return nil
	}
	var out []cpuTimes
	for _, line := range strings.Split(string(b), "\n") {
		if !strings.HasPrefix(line, "cpu") {
			break
		}
		var t cpuTimes
		for i, f := range strings.Fields(line)[1:] {
			v, _ := strconv.ParseFloat(f, 64)
			t.total += v
			if i != 3 && i != 4 { // idle and iowait
				t.busy += v
			}
		}
		out = append(out, t)
	}
	return out
}

func readMemMB() (used, total float64) {
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return
	}
	var avail float64
	for _, line := range strings.Split(string(b), "\n") {
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

// readProc returns this process CPU ticks and resident memory in MB
func readProc() (ticks, rssMB float64) {
	if b, err := os.ReadFile("/proc/self/stat"); err == nil {
		s := string(b)
		f := strings.Fields(s[strings.LastIndexByte(s, ')')+2:])
		u, _ := strconv.ParseFloat(f[11], 64)
		k, _ := strconv.ParseFloat(f[12], 64)
		ticks = u + k
	}
	if b, err := os.ReadFile("/proc/self/statm"); err == nil {
		f := strings.Fields(string(b))
		pages, _ := strconv.ParseFloat(f[1], 64)
		rssMB = pages * float64(os.Getpagesize()) / (1 << 20)
	}
	return
}

func pct(a, b float64) float64 {
	if b <= 0 {
		return 0
	}
	return 100 * a / b
}

func (m *sysMonitor) sample() {
	prevCPU := readCPUTimes()
	prevTicks, _ := readProc()
	pb, pr, _, pw, pn, _, pbusy, pidle := m.engine.CumulativeStats()
	prevT := time.Now()
	for range time.Tick(sampleEvery) {
		now := time.Now()
		dt := now.Sub(prevT).Seconds()
		s := sysSample{T: now.UnixMilli(), Goroutines: runtime.NumGoroutine()}

		cpu := readCPUTimes()
		if len(cpu) == len(prevCPU) && len(cpu) > 0 {
			d := func(i int) float64 { return pct(cpu[i].busy-prevCPU[i].busy, cpu[i].total-prevCPU[i].total) }
			s.CPU = d(0)
			for i := 1; i < len(cpu); i++ {
				s.Cores = append(s.Cores, d(i))
			}
		}
		prevCPU = cpu
		s.MemUsedMB, s.MemTotalMB = readMemMB()
		ticks, rss := readProc()
		s.ProcCPU = pct((ticks-prevTicks)/100, dt*float64(runtime.NumCPU()))
		s.ProcRSSMB = rss
		prevTicks = ticks

		b, r, _, w, n, _, busy, idle := m.engine.CumulativeStats()
		db, dr := max(b-pb, 0), max(r-pr, 0)
		s.ReqPerSec = float64(dr) / dt
		if db > 0 {
			s.AvgBatch = float64(dr) / float64(db)
			s.RunMs = float64(max(n-pn, 0)) / float64(db) / 1e6
		}
		if dr > 0 {
			s.QueueMs = float64(max(w-pw, 0)) / float64(dr) / 1e6
		}
		s.LaneBusy = pct(float64(max(busy-pbusy, 0)), float64(max(busy-pbusy, 0)+max(idle-pidle, 0)))
		pb, pr, pw, pn, pbusy, pidle = b, r, w, n, busy, idle
		s.P50Ms, s.P99Ms = recentLatency.drain()
		prevT = now

		m.mu.Lock()
		s.GPU = m.gpu
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
