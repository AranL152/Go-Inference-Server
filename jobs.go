// Dataset detection jobs for the web UI
package main

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"log"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// jobWorkers is how many images a job keeps in flight; set it with setJobWorkers
var jobWorkers = 256

// jobReadBufs holds the JPEG bytes of the images job workers are processing
var jobReadBufs = newFreeList(jobWorkers, func() *bytes.Buffer { return new(bytes.Buffer) })

// setJobWorkers sizes job concurrency and the buffer lists that follow it
func setJobWorkers(n int) {
	jobWorkers = max(n, 1)
	jobReadBufs = newFreeList(jobWorkers, func() *bytes.Buffer { return new(bytes.Buffer) })
	letterboxBufs = newFreeList(jobWorkers, func() []byte { return make([]byte, 3*letterboxSize*letterboxSize) })
}

// readFileInto replaces buf's contents with the file at path
func readFileInto(buf *bytes.Buffer, path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	buf.Reset()
	if fi, err := f.Stat(); err == nil {
		buf.Grow(int(fi.Size()) + bytes.MinRead)
	}
	_, err = buf.ReadFrom(f)
	return err
}

type imageResult struct {
	Path       string      `json:"path"`
	Width      int         `json:"width"`
	Height     int         `json:"height"`
	Detections []Detection `json:"detections"`
	Error      string      `json:"error,omitempty"`
}

type job struct {
	ID       string
	Source   string
	Classes  []string
	Status   string
	Total    int
	Created  time.Time
	Finished time.Time

	done    atomic.Int64
	failed  atomic.Int64
	objects atomic.Int64
	cancel  atomic.Bool

	root    string
	files   []string
	want    map[int]bool
	mu      sync.Mutex
	results []imageResult
	tmpDir  string
}

type jobView struct {
	ID       string    `json:"id"`
	Source   string    `json:"source"`
	Classes  []string  `json:"classes"`
	Status   string    `json:"status"`
	Total    int       `json:"total"`
	Done     int64     `json:"done"`
	Failed   int64     `json:"failed"`
	Objects  int64     `json:"objects"`
	Rate     float64   `json:"images_per_sec"`
	Created  time.Time `json:"created_at"`
	Finished time.Time `json:"finished_at,omitzero"`
}

func (j *job) view() jobView {
	j.mu.Lock()
	v := jobView{ID: j.ID, Source: j.Source, Classes: j.Classes, Status: j.Status, Total: j.Total,
		Created: j.Created, Finished: j.Finished}
	j.mu.Unlock()
	v.Done, v.Failed, v.Objects = j.done.Load(), j.failed.Load(), j.objects.Load()
	end := time.Now()
	if !v.Finished.IsZero() {
		end = v.Finished
	}
	if secs := end.Sub(j.Created).Seconds(); secs > 0 {
		v.Rate = float64(v.Done) / secs
	}
	return v
}

type jobStore struct {
	s    *server
	mon  *sysMonitor
	dir  string // where finished jobs and uploads are kept or empty for memory only
	mu   sync.Mutex
	jobs []*job
	perf map[string]*runPerf
}

func newID() string {
	b := make([]byte, 6)
	rand.Read(b)
	return hex.EncodeToString(b)
}

// resolveClasses maps typed class names to label indices and reports unknown names
func (st *jobStore) resolveClasses(names []string) (map[int]bool, []string, error) {
	idx := map[string]int{}
	for i, l := range st.s.engine.labels {
		idx[strings.ToLower(l)] = i
	}
	want := map[int]bool{}
	var clean, unknown []string
	for _, n := range names {
		n = strings.ToLower(strings.TrimSpace(n))
		if n == "" {
			continue
		}
		i, ok := idx[n]
		if !ok {
			unknown = append(unknown, n)
			continue
		}
		if !want[i] {
			want[i] = true
			clean = append(clean, n)
		}
	}
	if len(unknown) > 0 {
		return nil, nil, fmt.Errorf("unknown classes: %s", strings.Join(unknown, ", "))
	}
	return want, clean, nil
}

func listImages(dir string) ([]string, error) {
	var files []string
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		ext := strings.ToLower(filepath.Ext(p))
		if !d.IsDir() && (ext == ".jpg" || ext == ".jpeg") {
			files = append(files, p)
		}
		return nil
	})
	sort.Strings(files)
	return files, err
}

func (st *jobStore) start(source, dir, tmpDir string, classes []string) (*job, error) {
	want, clean, err := st.resolveClasses(classes)
	if err != nil {
		return nil, err
	}
	files, err := listImages(dir)
	if err != nil {
		return nil, fmt.Errorf("read dataset: %w", err)
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("no JPEG images found in %s", source)
	}
	j := &job{ID: newID(), Source: source, Classes: clean, Status: "running", Total: len(files),
		Created: time.Now(), root: dir, files: files, want: want, tmpDir: tmpDir,
		results: make([]imageResult, len(files))}
	if j.Classes == nil {
		j.Classes = []string{}
	}
	st.mu.Lock()
	st.jobs = append([]*job{j}, st.jobs...)
	st.mu.Unlock()
	log.Printf("job %s started: %d images from %s", j.ID, j.Total, source)
	go st.run(j)
	return j, nil
}

func (st *jobStore) run(j *job) {
	done := make(chan struct{})
	go func() { st.savePerf(st.recordRun(j, done)) }()
	defer close(done)
	next := make(chan int)
	var wg sync.WaitGroup
	for w := 0; w < jobWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			setStage(stageRequest)
			for i := range next {
				st.process(j, i)
			}
		}()
	}
	for i := range j.files {
		if j.cancel.Load() {
			break
		}
		next <- i
	}
	close(next)
	wg.Wait()
	j.mu.Lock()
	j.Finished = time.Now()
	j.Status = "completed"
	if j.cancel.Load() {
		j.Status = "cancelled"
	}
	j.mu.Unlock()
	v := j.view()
	log.Printf("job %s %s: %d images, %d objects, %.0f img/s", j.ID, v.Status, v.Done, v.Objects, v.Rate)
	st.save(j)
}

func (st *jobStore) process(j *job, i int) {
	rel, _ := filepath.Rel(j.root, j.files[i])
	r := imageResult{Path: filepath.ToSlash(rel), Detections: []Detection{}}
	buf := jobReadBufs.get()
	defer jobReadBufs.put(buf)
	err := readFileInto(buf, j.files[i])
	if err == nil {
		var res detectResponse
		t := time.Now()
		res, err = st.s.detect(buf.Bytes())
		recentLatency.add(time.Since(t))
		r.Width, r.Height = res.Width, res.Height
		for _, d := range res.Detections {
			if len(j.want) == 0 || j.want[d.ClassIndex] {
				r.Detections = append(r.Detections, d)
			}
		}
	}
	if err != nil {
		r.Error = err.Error()
		j.failed.Add(1)
	}
	j.objects.Add(int64(len(r.Detections)))
	j.mu.Lock()
	j.results[i] = r
	j.mu.Unlock()
	j.done.Add(1)
}

func (st *jobStore) get(id string) *job {
	st.mu.Lock()
	defer st.mu.Unlock()
	for _, j := range st.jobs {
		if j.ID == id {
			return j
		}
	}
	return nil
}

func (st *jobStore) routes(mux *http.ServeMux) {
	mux.HandleFunc("GET /api/config", st.handleConfig)
	mux.HandleFunc("GET /api/jobs", st.handleList)
	mux.HandleFunc("POST /api/jobs", st.handleCreate)
	mux.HandleFunc("POST /api/jobs/upload", st.handleUpload)
	mux.HandleFunc("GET /api/jobs/{id}", st.handleGet)
	mux.HandleFunc("POST /api/jobs/{id}/cancel", st.handleCancel)
	mux.HandleFunc("GET /api/jobs/{id}/results", st.handleResults)
	mux.HandleFunc("GET /api/jobs/{id}/perf", st.handlePerf)
	mux.HandleFunc("GET /api/jobs/{id}/images/{n}", st.handleImage)
}

func (st *jobStore) handleConfig(w http.ResponseWriter, r *http.Request) {
	e := st.s.engine
	decoders := 0
	if st.s.cpuPool != nil {
		decoders = st.s.cpuPool.size
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"model": st.s.modelName, "detector": e.Detector(), "labels": e.labels,
		"max_batch": e.maxBatch, "conf": detConfThresh, "iou": detIoUThresh,
		"max_wait_ms": float64(e.maxWait) / 1e6, "infer_lanes": max(numInferLanes, 1),
		"unpack_workers": numUnpackers, "cpu_decoders": decoders, "cores": runtime.NumCPU(),
	})
}

func (st *jobStore) handleList(w http.ResponseWriter, r *http.Request) {
	st.mu.Lock()
	views := make([]jobView, len(st.jobs))
	for i, j := range st.jobs {
		views[i] = j.view()
	}
	st.mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]any{"jobs": views})
}

func (st *jobStore) handleCreate(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Path    string   `json:"path"`
		Classes []string `json:"classes"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Path == "" {
		writeError(w, http.StatusBadRequest, "body must be JSON with a dataset path")
		return
	}
	if !st.s.engine.Detector() {
		writeError(w, http.StatusBadRequest, "dataset jobs need a detector model")
		return
	}
	j, err := st.start(req.Path, req.Path, "", req.Classes)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, j.view())
}

// handleUpload stores multipart images in a temp dir and starts a job over it
func (st *jobStore) handleUpload(w http.ResponseWriter, r *http.Request) {
	mr, err := r.MultipartReader()
	if err != nil {
		writeError(w, http.StatusBadRequest, "expected multipart form")
		return
	}
	dir, err := os.MkdirTemp(st.dir, "upload-")
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	var classes []string
	n := 0
	for {
		part, err := mr.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			os.RemoveAll(dir)
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		if part.FormName() == "classes" {
			b, _ := io.ReadAll(io.LimitReader(part, 1<<16))
			classes = splitClasses(string(b))
			continue
		}
		name := filepath.Base(part.FileName())
		if ext := strings.ToLower(filepath.Ext(name)); ext != ".jpg" && ext != ".jpeg" {
			continue
		}
		f, err := os.Create(filepath.Join(dir, fmt.Sprintf("%06d_%s", n, name)))
		if err == nil {
			_, err = io.Copy(f, io.LimitReader(part, maxImageBytes))
			f.Close()
		}
		if err != nil {
			os.RemoveAll(dir)
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		n++
	}
	j, err := st.start(fmt.Sprintf("upload (%d images)", n), dir, dir, classes)
	if err != nil {
		os.RemoveAll(dir)
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, j.view())
}

func splitClasses(s string) []string {
	return strings.FieldsFunc(s, func(r rune) bool { return r == ',' || r == '\n' })
}

func (st *jobStore) handleGet(w http.ResponseWriter, r *http.Request) {
	j := st.get(r.PathValue("id"))
	if j == nil {
		writeError(w, http.StatusNotFound, "no such job")
		return
	}
	// per class counts of images and objects
	type count struct {
		Class   string `json:"class"`
		Images  int    `json:"images"`
		Objects int    `json:"objects"`
	}
	counts := map[string]*count{}
	empty := 0
	j.mu.Lock()
	for i := range j.results {
		res := &j.results[i]
		if res.Path == "" {
			continue
		}
		if len(res.Detections) == 0 {
			empty++
		}
		seen := map[string]bool{}
		for _, d := range res.Detections {
			c := counts[d.ClassName]
			if c == nil {
				c = &count{Class: d.ClassName}
				counts[d.ClassName] = c
			}
			c.Objects++
			if !seen[d.ClassName] {
				seen[d.ClassName] = true
				c.Images++
			}
		}
	}
	j.mu.Unlock()
	list := make([]*count, 0, len(counts))
	for _, c := range counts {
		list = append(list, c)
	}
	sort.Slice(list, func(a, b int) bool { return list[a].Images > list[b].Images })
	writeJSON(w, http.StatusOK, map[string]any{"job": j.view(), "categories": list, "no_objects": empty})
}

func (st *jobStore) handleCancel(w http.ResponseWriter, r *http.Request) {
	j := st.get(r.PathValue("id"))
	if j == nil {
		writeError(w, http.StatusNotFound, "no such job")
		return
	}
	j.cancel.Store(true)
	writeJSON(w, http.StatusOK, j.view())
}

// handleResults pages through finished images optionally filtered to one class
func (st *jobStore) handleResults(w http.ResponseWriter, r *http.Request) {
	j := st.get(r.PathValue("id"))
	if j == nil {
		writeError(w, http.StatusNotFound, "no such job")
		return
	}
	q := r.URL.Query()
	class := q.Get("class")
	offset, _ := strconv.Atoi(q.Get("offset"))
	limit, err := strconv.Atoi(q.Get("limit"))
	if err != nil || limit <= 0 || limit > 500 {
		limit = 60
	}
	type item struct {
		Index int `json:"index"`
		imageResult
	}
	var items []item
	matched := 0
	j.mu.Lock()
	for i := range j.results {
		res := j.results[i]
		if res.Path == "" {
			continue
		}
		ok := class == ""
		if class == "_none" {
			ok = len(res.Detections) == 0
		}
		for _, d := range res.Detections {
			if d.ClassName == class {
				ok = true
				break
			}
		}
		if !ok {
			continue
		}
		if matched >= offset && len(items) < limit {
			items = append(items, item{i, res})
		}
		matched++
	}
	j.mu.Unlock()
	if items == nil {
		items = []item{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"total": matched, "items": items})
}

func (st *jobStore) handleImage(w http.ResponseWriter, r *http.Request) {
	j := st.get(r.PathValue("id"))
	n, err := strconv.Atoi(r.PathValue("n"))
	if j == nil || err != nil || n < 0 || n >= len(j.files) {
		writeError(w, http.StatusNotFound, "no such image")
		return
	}
	w.Header().Set("Content-Type", mime.TypeByExtension(".jpg"))
	w.Header().Set("Cache-Control", "max-age=3600")
	http.ServeFile(w, r, j.files[n])
}
