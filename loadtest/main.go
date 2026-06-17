// Phase 2 — concurrent load baseline (measurement only).
//
// A closed-loop HTTP load generator for the Phase 1 inference server. It fires a
// fixed number of POST /predict requests using a configurable number of
// simultaneously in-flight requests (concurrency), records every request's
// latency, and reports throughput (req/s) and latency percentiles (p50/p95/p99).
//
// While each concurrency level runs, a background goroutine polls nvidia-smi so
// we can show whether the GPU is actually busy or sitting idle between the
// server's serialized (mutex-guarded) inferences.
//
// This tool does NOT modify or import the server — it only talks to it over HTTP.
package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

func main() {
	addr := flag.String("addr", "http://localhost:8080", "server base URL")
	levelsCSV := flag.String("levels", "1,4,16,64", "comma-separated concurrency levels to sweep")
	perLevel := flag.Int("n", 1500, "number of requests per concurrency level")
	warmup := flag.Int("warmup", 50, "warmup requests before each level (not measured)")
	imagesDir := flag.String("images", "testdata", "directory of test images to cycle through")
	imageNames := flag.String("imagefiles", "dog.jpg,cat.jpg", "comma-separated image filenames within -images")
	gpuInterval := flag.Duration("gpu-interval", 100*time.Millisecond, "nvidia-smi poll interval during a run")
	outPrefix := flag.String("out", "phase2_results", "prefix for raw results files (.json/.csv)")
	flag.Parse()

	levels := parseInts(*levelsCSV)
	if len(levels) == 0 {
		log.Fatal("no concurrency levels given")
	}

	images := loadImages(*imagesDir, strings.Split(*imageNames, ","))
	url := strings.TrimRight(*addr, "/") + "/predict"

	// One shared client. Crucially, allow enough idle keep-alive connections per
	// host so high concurrency reuses connections instead of exhausting ports.
	maxC := 0
	for _, c := range levels {
		if c > maxC {
			maxC = c
		}
	}
	client := &http.Client{
		Timeout: 60 * time.Second,
		Transport: &http.Transport{
			MaxIdleConns:        maxC * 2,
			MaxIdleConnsPerHost: maxC * 2,
			IdleConnTimeout:     60 * time.Second,
		},
	}

	if err := preflight(client, url, images[0]); err != nil {
		log.Fatalf("server not reachable at %s: %v", url, err)
	}
	log.Printf("server OK at %s — sweeping concurrency levels %v, %d requests each", url, levels, *perLevel)

	var results []Result
	for _, c := range levels {
		log.Printf("--- concurrency %d: warming up (%d) ---", c, *warmup)
		runLevel(client, url, images, c, *warmup) // discard warmup

		gpu := startGPUSampler(*gpuInterval)
		log.Printf("--- concurrency %d: measuring %d requests ---", c, *perLevel)
		lat, errCount, elapsed := runLevel(client, url, images, c, *perLevel)
		gpuSummary := gpu.stop()

		res := summarize(c, *perLevel, errCount, elapsed, lat, gpuSummary)
		results = append(results, res)
		log.Printf("    -> %.1f req/s | p50=%.1fms p95=%.1fms p99=%.1fms | GPU util avg=%.0f%% max=%.0f%% | errors=%d",
			res.Throughput, res.P50ms, res.P95ms, res.P99ms, res.GPUUtilAvg, res.GPUUtilMax, res.Errors)
	}

	printTable(results)
	writeJSON(*outPrefix+".json", results)
	writeCSV(*outPrefix+".csv", results)
	log.Printf("raw results written to %s.json and %s.csv", *outPrefix, *outPrefix)
}

// Result holds the measured stats for one concurrency level.
type Result struct {
	Concurrency  int     `json:"concurrency"`
	Requests     int     `json:"requests"`
	Errors       int     `json:"errors"`
	ElapsedSec   float64 `json:"elapsed_sec"`
	Throughput   float64 `json:"throughput_req_s"`
	Meanms       float64 `json:"mean_ms"`
	P50ms        float64 `json:"p50_ms"`
	P95ms        float64 `json:"p95_ms"`
	P99ms        float64 `json:"p99_ms"`
	Maxms        float64 `json:"max_ms"`
	GPUUtilAvg   float64 `json:"gpu_util_avg_pct"`
	GPUUtilMax   float64 `json:"gpu_util_max_pct"`
	GPUMemMaxMiB int     `json:"gpu_mem_max_mib"`
	GPUSamples   int     `json:"gpu_samples"`
}

// runLevel sends `total` requests using `concurrency` workers in a closed loop
// (each worker fires the next request the instant its previous one returns).
// Returns per-request latencies, error count, and wall-clock elapsed time.
func runLevel(client *http.Client, url string, images [][]byte, concurrency, total int) ([]time.Duration, int, time.Duration) {
	var remaining int64 = int64(total)
	var errCount int64
	var imgIdx uint64

	perWorker := make([][]time.Duration, concurrency)
	var wg sync.WaitGroup
	start := time.Now()
	for w := 0; w < concurrency; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			local := make([]time.Duration, 0, total/concurrency+1)
			for atomic.AddInt64(&remaining, -1) >= 0 {
				img := images[int(atomic.AddUint64(&imgIdx, 1))%len(images)]
				t0 := time.Now()
				if err := doRequest(client, url, img); err != nil {
					atomic.AddInt64(&errCount, 1)
				}
				local = append(local, time.Since(t0))
			}
			perWorker[w] = local
		}(w)
	}
	wg.Wait()
	elapsed := time.Since(start)

	var all []time.Duration
	for _, l := range perWorker {
		all = append(all, l...)
	}
	return all, int(errCount), elapsed
}

func doRequest(client *http.Client, url string, body []byte) error {
	resp, err := client.Post(url, "application/octet-stream", bytes.NewReader(body))
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body) // must drain to reuse the keep-alive connection
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("status %d", resp.StatusCode)
	}
	return nil
}

func summarize(c, requests, errs int, elapsed time.Duration, lat []time.Duration, g gpuSummary) Result {
	sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
	ok := requests - errs
	thru := 0.0
	if elapsed > 0 {
		thru = float64(ok) / elapsed.Seconds()
	}
	return Result{
		Concurrency:  c,
		Requests:     requests,
		Errors:       errs,
		ElapsedSec:   elapsed.Seconds(),
		Throughput:   thru,
		Meanms:       meanMs(lat),
		P50ms:        percentileMs(lat, 0.50),
		P95ms:        percentileMs(lat, 0.95),
		P99ms:        percentileMs(lat, 0.99),
		Maxms:        percentileMs(lat, 1.0),
		GPUUtilAvg:   g.utilAvg,
		GPUUtilMax:   g.utilMax,
		GPUMemMaxMiB: g.memMax,
		GPUSamples:   g.count,
	}
}

// percentileMs uses nearest-rank on the already-sorted slice. q in [0,1].
func percentileMs(sorted []time.Duration, q float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	rank := int(q * float64(len(sorted)))
	if rank >= len(sorted) {
		rank = len(sorted) - 1
	}
	return float64(sorted[rank].Microseconds()) / 1000.0
}

func meanMs(lat []time.Duration) float64 {
	if len(lat) == 0 {
		return 0
	}
	var sum time.Duration
	for _, d := range lat {
		sum += d
	}
	return float64((sum / time.Duration(len(lat))).Microseconds()) / 1000.0
}

// --- GPU sampling ---------------------------------------------------------

type gpuSummary struct {
	utilAvg float64
	utilMax float64
	memMax  int
	count   int
}

type gpuSampler struct {
	stopCh chan struct{}
	done   chan gpuSummary
}

func startGPUSampler(interval time.Duration) *gpuSampler {
	s := &gpuSampler{stopCh: make(chan struct{}), done: make(chan gpuSummary, 1)}
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		var utils []int
		memMax := 0
		for {
			select {
			case <-s.stopCh:
				sum := gpuSummary{count: len(utils), memMax: memMax}
				var total int
				for _, u := range utils {
					total += u
					if u > int(sum.utilMax) {
						sum.utilMax = float64(u)
					}
				}
				if len(utils) > 0 {
					sum.utilAvg = float64(total) / float64(len(utils))
				}
				s.done <- sum
				return
			case <-ticker.C:
				if util, mem, ok := sampleGPU(); ok {
					utils = append(utils, util)
					if mem > memMax {
						memMax = mem
					}
				}
			}
		}
	}()
	return s
}

func (s *gpuSampler) stop() gpuSummary {
	close(s.stopCh)
	return <-s.done
}

// sampleGPU returns (utilization%, memoryUsedMiB, ok).
func sampleGPU() (int, int, bool) {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=utilization.gpu,memory.used",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return 0, 0, false
	}
	parts := strings.Split(strings.TrimSpace(strings.Split(string(out), "\n")[0]), ",")
	if len(parts) < 2 {
		return 0, 0, false
	}
	util, err1 := strconv.Atoi(strings.TrimSpace(parts[0]))
	mem, err2 := strconv.Atoi(strings.TrimSpace(parts[1]))
	if err1 != nil || err2 != nil {
		return 0, 0, false
	}
	return util, mem, true
}

// --- helpers --------------------------------------------------------------

func parseInts(csv string) []int {
	var out []int
	for _, s := range strings.Split(csv, ",") {
		if n, err := strconv.Atoi(strings.TrimSpace(s)); err == nil {
			out = append(out, n)
		}
	}
	return out
}

func loadImages(dir string, names []string) [][]byte {
	var imgs [][]byte
	for _, name := range names {
		p := filepath.Join(dir, strings.TrimSpace(name))
		b, err := os.ReadFile(p)
		if err != nil {
			log.Fatalf("read image %s: %v", p, err)
		}
		imgs = append(imgs, b)
	}
	if len(imgs) == 0 {
		log.Fatal("no images loaded")
	}
	return imgs
}

func preflight(client *http.Client, url string, img []byte) error {
	return doRequest(client, url, img)
}

func printTable(results []Result) {
	fmt.Println()
	fmt.Println("| Concurrency | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) | max (ms) | GPU util avg | GPU util max | errors |")
	fmt.Println("|------------:|-------------------:|---------:|---------:|---------:|----------:|---------:|-------------:|-------------:|-------:|")
	for _, r := range results {
		fmt.Printf("| %11d | %18.1f | %8.1f | %8.1f | %8.1f | %9.1f | %8.1f | %11.0f%% | %11.0f%% | %6d |\n",
			r.Concurrency, r.Throughput, r.P50ms, r.P95ms, r.P99ms, r.Meanms, r.Maxms, r.GPUUtilAvg, r.GPUUtilMax, r.Errors)
	}
	fmt.Println()
}

func writeJSON(path string, results []Result) {
	f, err := os.Create(path)
	if err != nil {
		log.Printf("write json: %v", err)
		return
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	_ = enc.Encode(results)
}

func writeCSV(path string, results []Result) {
	f, err := os.Create(path)
	if err != nil {
		log.Printf("write csv: %v", err)
		return
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()
	_ = w.Write([]string{"concurrency", "throughput_req_s", "p50_ms", "p95_ms", "p99_ms", "mean_ms", "max_ms", "gpu_util_avg_pct", "gpu_util_max_pct", "gpu_mem_max_mib", "gpu_samples", "errors", "elapsed_sec"})
	for _, r := range results {
		_ = w.Write([]string{
			strconv.Itoa(r.Concurrency),
			fmt.Sprintf("%.2f", r.Throughput),
			fmt.Sprintf("%.2f", r.P50ms),
			fmt.Sprintf("%.2f", r.P95ms),
			fmt.Sprintf("%.2f", r.P99ms),
			fmt.Sprintf("%.2f", r.Meanms),
			fmt.Sprintf("%.2f", r.Maxms),
			fmt.Sprintf("%.1f", r.GPUUtilAvg),
			fmt.Sprintf("%.1f", r.GPUUtilMax),
			strconv.Itoa(r.GPUMemMaxMiB),
			strconv.Itoa(r.GPUSamples),
			strconv.Itoa(r.Errors),
			fmt.Sprintf("%.3f", r.ElapsedSec),
		})
	}
}
