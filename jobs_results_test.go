package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestResultsClassFilters(t *testing.T) {
	dog := Detection{ClassName: "dog"}
	cat := Detection{ClassName: "cat"}
	j := &job{ID: "j1", results: []imageResult{
		{Path: "a.jpg", Detections: []Detection{dog}},
		{Path: "b.jpg", Detections: []Detection{}},
		{Path: "c.jpg", Detections: []Detection{cat, dog}},
		{Path: "d.jpg", Detections: []Detection{}},
		{Path: ""}, // not processed yet
	}}
	st := &jobStore{jobs: []*job{j}}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/jobs/{id}/results", st.handleResults)

	for class, want := range map[string][]string{
		"":      {"a.jpg", "b.jpg", "c.jpg", "d.jpg"},
		"_some": {"a.jpg", "c.jpg"},
		"_none": {"b.jpg", "d.jpg"},
		"dog":   {"a.jpg", "c.jpg"},
		"cat":   {"c.jpg"},
	} {
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest("GET", "/api/jobs/j1/results?class="+class, nil))
		var body struct {
			Total int `json:"total"`
			Items []struct {
				Path string `json:"path"`
			} `json:"items"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
			t.Fatalf("class %q: %v", class, err)
		}
		var got []string
		for _, it := range body.Items {
			got = append(got, it.Path)
		}
		if body.Total != len(want) || len(got) != len(want) {
			t.Errorf("class %q: got %v (total %d), want %v", class, got, body.Total, want)
			continue
		}
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("class %q: got %v, want %v", class, got, want)
				break
			}
		}
	}
}
