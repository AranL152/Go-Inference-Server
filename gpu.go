// GPU inspection helpers, carried over from Phase 0.
//
// WSL2's nvidia-smi reports per-process memory as [N/A], so the primary proof
// that work landed on the GPU is the total GPU-memory delta around session
// creation; the PID-in-compute-apps list is a secondary signal.
package main

import (
	"os"
	"os/exec"
	"strconv"
	"strings"
)

// gpuMemUsedMiB returns total GPU memory in use via nvidia-smi, or -1 on error.
func gpuMemUsedMiB() int {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=memory.used", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return -1
	}
	n, _ := strconv.Atoi(strings.TrimSpace(strings.Split(string(out), "\n")[0]))
	return n
}

// processOnGPU reports whether this PID appears in nvidia-smi's compute-app list.
// Often empty under WSL2, so it's a bonus check, not the primary one.
func processOnGPU() bool {
	pid := strconv.Itoa(os.Getpid())
	out, err := exec.Command("nvidia-smi",
		"--query-compute-apps=pid,used_memory", "--format=csv,noheader").Output()
	if err != nil {
		return false
	}
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), pid+",") {
			return true
		}
	}
	return false
}
