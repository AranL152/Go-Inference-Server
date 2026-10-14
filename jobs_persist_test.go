package main

import (
	"testing"
	"time"
)

func TestJobSaveLoadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	j := &job{ID: "abc", Source: "/data", Classes: []string{"dog"}, Status: "completed", Total: 2,
		Created: time.Now().Add(-time.Minute), Finished: time.Now(), root: "/data",
		files: []string{"/data/a.jpg", "/data/b.jpg"}, want: map[int]bool{16: true},
		results: []imageResult{
			{Path: "a.jpg", Width: 640, Height: 480, Detections: []Detection{{ClassIndex: 16, ClassName: "dog", Confidence: 0.9, Box: [4]float32{1, 2, 3, 4}}}},
			{Path: "b.jpg", Width: 320, Height: 240, Detections: []Detection{}},
		}}
	j.done.Store(2)
	j.objects.Store(1)
	(&jobStore{dir: dir}).save(j)

	st := &jobStore{dir: dir}
	st.load()
	if len(st.jobs) != 1 {
		t.Fatalf("loaded %d jobs", len(st.jobs))
	}
	got := st.jobs[0]
	v := got.view()
	if v.ID != "abc" || v.Done != 2 || v.Objects != 1 || v.Status != "completed" || !got.want[16] {
		t.Errorf("view %+v want %v", v, got.want)
	}
	if got.files[1] != "/data/b.jpg" || got.results[0].Detections[0].Box != [4]float32{1, 2, 3, 4} {
		t.Errorf("files %v results %+v", got.files, got.results)
	}
}
