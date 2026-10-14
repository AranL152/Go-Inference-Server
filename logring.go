// Recent server log lines for the dashboard log widget
package main

import (
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

type logLine struct {
	Seq  int64  `json:"seq"`
	T    int64  `json:"t"`
	Text string `json:"text"`
}

type logRing struct {
	mu    sync.Mutex
	seq   int64
	lines []logLine
}

var serverLogs = &logRing{}

// captureLogs tees the standard logger into the ring
func captureLogs() {
	log.SetOutput(io.MultiWriter(os.Stderr, serverLogs))
}

func (l *logRing) Write(p []byte) (int, error) {
	text := strings.TrimRight(string(p), "\n")
	if i := strings.Index(text, " "); i > 0 {
		if j := strings.Index(text[i+1:], " "); j > 0 {
			text = text[i+j+2:] // drop the date and time the logger adds
		}
	}
	l.mu.Lock()
	l.seq++
	l.lines = append(l.lines, logLine{Seq: l.seq, T: time.Now().UnixMilli(), Text: text})
	if len(l.lines) > 500 {
		l.lines = l.lines[len(l.lines)-500:]
	}
	l.mu.Unlock()
	return len(p), nil
}

// handleLogs returns lines after the since sequence number
func (l *logRing) handleLogs(w http.ResponseWriter, r *http.Request) {
	since, _ := strconv.ParseInt(r.URL.Query().Get("since"), 10, 64)
	l.mu.Lock()
	out := []logLine{}
	for _, ln := range l.lines {
		if ln.Seq > since {
			out = append(out, ln)
		}
	}
	l.mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]any{"lines": out})
}
