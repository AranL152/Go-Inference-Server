package main

import (
	"io"
	"log"
	"net/http"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var letterboxF32Pool = sync.Pool{New: func() any { return make([]float32, 3*letterboxSize*letterboxSize) }}

type detectResponse struct {
	Width      int         `json:"width"`
	Height     int         `json:"height"`
	Detections []Detection `json:"detections"`
}

// detect runs one JPEG through letterbox preprocessing and the detector
func (s *server) detect(jpeg []byte) (detectResponse, error) {
	in, lb, err := s.cpuPool.PreprocessLetterbox(jpeg)
	if err != nil {
		return detectResponse{}, err
	}
	var pred Prediction
	if s.engine.InputU8() {
		pred, err = s.engine.PredictBytes(in)
	} else {
		f := letterboxF32Pool.Get().([]float32)
		for i, v := range in {
			f[i] = float32(v)
		}
		pred, err = s.engine.Predict(f)
		letterboxF32Pool.Put(f)
	}
	letterboxBufs.put(in)
	if err != nil {
		return detectResponse{}, err
	}
	lb.toOriginal(pred.Detections)
	if pred.Detections == nil {
		pred.Detections = []Detection{}
	}
	return detectResponse{Width: lb.W, Height: lb.H, Detections: pred.Detections}, nil
}

func (s *server) handleDetect(w http.ResponseWriter, body io.Reader) {
	buf, err := readBodyPooled(body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "could not read image body: "+err.Error())
		return
	}
	res, err := s.detect(buf.Bytes())
	bodyBufPool.Put(buf)
	if err != nil {
		writeError(w, http.StatusBadRequest, "could not detect (expected JPEG): "+err.Error())
		return
	}
	if !s.quiet {
		log.Printf("detect: %dx%d -> %d objects", res.Width, res.Height, len(res.Detections))
	}
	writeJSON(w, http.StatusOK, res)
}

// modelIsDetector peeks at the output rank so labels can default before the engine starts
func modelIsDetector(modelPath, libPath string) bool {
	if ensureORT(libPath) != nil {
		return false
	}
	_, outs, err := ort.GetInputOutputInfo(modelPath)
	return err == nil && len(outs) == 1 && len(outs[0].Dimensions) == 3
}
