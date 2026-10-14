package main

import (
	"compress/gzip"
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"sort"
)

type jobFile struct {
	View    jobView       `json:"job"`
	Root    string        `json:"root"`
	Files   []string      `json:"files"`
	Want    []int         `json:"want"`
	Results []imageResult `json:"results"`
}

// save writes a finished job to dir as gzipped JSON
func (st *jobStore) save(j *job) {
	if st.dir == "" {
		return
	}
	f := jobFile{View: j.view(), Root: j.root, Files: j.files}
	for c := range j.want {
		f.Want = append(f.Want, c)
	}
	j.mu.Lock()
	f.Results = j.results
	err := writeGzipJSON(filepath.Join(st.dir, j.ID+".json.gz"), f)
	j.mu.Unlock()
	if err != nil {
		log.Printf("save job %s: %v", j.ID, err)
	}
}

func writeGzipJSON(path string, v any) error {
	tmp := path + ".tmp"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	zw := gzip.NewWriter(out)
	err = json.NewEncoder(zw).Encode(v)
	if cerr := zw.Close(); err == nil {
		err = cerr
	}
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

// load restores saved jobs newest first
func (st *jobStore) load() {
	paths, _ := filepath.Glob(filepath.Join(st.dir, "*.json.gz"))
	for _, p := range paths {
		in, err := os.Open(p)
		if err != nil {
			continue
		}
		var f jobFile
		zr, err := gzip.NewReader(in)
		if err == nil {
			err = json.NewDecoder(zr).Decode(&f)
		}
		in.Close()
		if err != nil {
			log.Printf("skip saved job %s: %v", p, err)
			continue
		}
		v := f.View
		j := &job{ID: v.ID, Source: v.Source, Classes: v.Classes, Status: v.Status, Total: v.Total,
			Created: v.Created, Finished: v.Finished, root: f.Root, files: f.Files,
			want: map[int]bool{}, results: f.Results}
		for _, c := range f.Want {
			j.want[c] = true
		}
		j.done.Store(v.Done)
		j.failed.Store(v.Failed)
		j.objects.Store(v.Objects)
		st.jobs = append(st.jobs, j)
	}
	sort.Slice(st.jobs, func(a, b int) bool { return st.jobs[a].Created.After(st.jobs[b].Created) })
	if len(st.jobs) > 0 {
		log.Printf("loaded %d saved jobs from %s", len(st.jobs), st.dir)
	}
}
