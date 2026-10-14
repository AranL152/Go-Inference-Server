// Start and stop hooks for Nsight Compute so profile_kernels.sh captures only steady state batches
package main

/*
#cgo CFLAGS: -I/usr/local/cuda-12.6/include
#cgo LDFLAGS: -L/usr/local/cuda-12.6/lib64 -lcudart
#include <cuda_profiler_api.h>
*/
import "C"
import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
)

// handleCudaProfiler is a no op unless the server runs under ncu or nsys with profiling off at start
func handleCudaProfiler(w http.ResponseWriter, r *http.Request) {
	var code C.cudaError_t
	switch r.PathValue("action") {
	case "start":
		code = C.cudaProfilerStart()
	case "stop":
		code = C.cudaProfilerStop()
	default:
		writeError(w, http.StatusNotFound, "use start or stop")
		return
	}
	if err := cudaErr("cuda profiler", code); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"profiler": r.PathValue("action")})
}

// handleProfiles serves the kernel summaries written by profile_kernels.sh
func handleProfiles(w http.ResponseWriter, r *http.Request) {
	paths, _ := filepath.Glob("profiles/*.json")
	out := []json.RawMessage{}
	for _, p := range paths {
		if b, err := os.ReadFile(p); err == nil && json.Valid(b) {
			out = append(out, b)
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"profiles": out})
}
