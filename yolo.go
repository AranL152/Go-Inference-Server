// YOLOX letterbox preprocessing and detection decoding
package main

import (
	"image"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

const (
	letterboxSize = 640
	letterboxPad  = 114
	maxDetections = 300
)

// Letterbox maps model input pixels back to the original image
type Letterbox struct {
	Scale float32 // model pixels per original pixel
	W, H  int     // original image size
}

// Detection is one box in original image pixels as x1 y1 x2 y2
type Detection struct {
	ClassIndex int        `json:"class_index"`
	ClassName  string     `json:"class_name"`
	Confidence float64    `json:"confidence"`
	Box        [4]float32 `json:"box"`
}

var letterboxPool = sync.Pool{New: func() any { return make([]float32, 3*letterboxSize*letterboxSize) }}

// chooseScaleLong picks the strongest DCT downscale that keeps the longer side at or above target
func chooseScaleLong(w, h, target int) (num, den int) {
	long := max(w, h)
	for _, f := range scaleFactors {
		if (long*f.num+f.den-1)/f.den >= target {
			return f.num, f.den
		}
	}
	return 1, 1
}

// DecodeLetterbox decodes jpeg into out as a 3x640x640 BGR float tensor in the 0 to 255 range
func (d *jpegDecoder) DecodeLetterbox(jpeg []byte, out []float32) (Letterbox, error) {
	img, fw, fh, err := d.decodeFitLong(jpeg, letterboxSize)
	if err != nil {
		return Letterbox{}, err
	}
	r := letterboxInto(img, out)
	return Letterbox{Scale: r * float32(img.Rect.Dx()) / float32(fw), W: fw, H: fh}, nil
}

// letterboxInto resizes img to fit 640x640 at the top left and pads the rest
func letterboxInto(img *image.RGBA, out []float32) float32 {
	sw, sh := img.Rect.Dx(), img.Rect.Dy()
	r := min(float32(letterboxSize)/float32(sw), float32(letterboxSize)/float32(sh))
	rw, rh := int(float32(sw)*r), int(float32(sh)*r)
	const plane = letterboxSize * letterboxSize
	b, g, rd := out[:plane], out[plane:2*plane], out[2*plane:3*plane]

	// source column offsets and weights are the same for every row
	var x0s, x1s [letterboxSize]int
	var wxs [letterboxSize]float32
	inv := 1 / r
	for x := 0; x < rw; x++ {
		fx := (float32(x)+0.5)*inv - 0.5
		x0 := max(floor32(fx), 0)
		x0s[x], x1s[x] = x0*4, min(x0+1, sw-1)*4
		wxs[x] = max(fx-float32(x0), 0)
	}
	for y := 0; y < letterboxSize; y++ {
		i := y * letterboxSize
		if y >= rh {
			fill(b[i:i+letterboxSize], g[i:i+letterboxSize], rd[i:i+letterboxSize])
			continue
		}
		fill(b[i+rw:i+letterboxSize], g[i+rw:i+letterboxSize], rd[i+rw:i+letterboxSize])
		if rw == sw && rh == sh {
			row := img.Pix[y*img.Stride:]
			for x := 0; x < rw; x++ {
				p := row[x*4 : x*4+3]
				rd[i+x], g[i+x], b[i+x] = float32(p[0]), float32(p[1]), float32(p[2])
			}
			continue
		}
		fy := (float32(y)+0.5)*inv - 0.5
		y0 := max(floor32(fy), 0)
		wy := max(fy-float32(y0), 0)
		row0 := img.Pix[y0*img.Stride:]
		row1 := img.Pix[min(y0+1, sh-1)*img.Stride:]
		for x := 0; x < rw; x++ {
			a0, a1, wx := x0s[x], x1s[x], wxs[x]
			for c, dst := range [3][]float32{rd, g, b} {
				top := float32(row0[a0+c]) + (float32(row0[a1+c])-float32(row0[a0+c]))*wx
				bot := float32(row1[a0+c]) + (float32(row1[a1+c])-float32(row1[a0+c]))*wx
				dst[i+x] = top + (bot-top)*wy
			}
		}
	}
	return r
}

func fill(planes ...[]float32) {
	for _, p := range planes {
		for i := range p {
			p[i] = letterboxPad
		}
	}
}

// PreprocessLetterbox decodes on a pooled decoder and returns a pooled input tensor
func (p *cpuDecoderPool) PreprocessLetterbox(jpeg []byte) ([]float32, Letterbox, error) {
	d := <-p.free
	defer func() { p.free <- d }()
	t := time.Now()
	out := letterboxPool.Get().([]float32)
	lb, err := d.DecodeLetterbox(jpeg, out)
	if err != nil {
		letterboxPool.Put(out)
		return nil, Letterbox{}, err
	}
	atomic.AddInt64(&p.statDecodeNs, int64(time.Since(t)))
	atomic.AddInt64(&p.statCount, 1)
	return out, lb, nil
}

// decodeDetections turns one 8400 x (5+classes) YOLOX row into NMS filtered boxes in model pixels
func decodeDetections(row []float32, numCls int, labels []string, confThresh, iouThresh float32) []Detection {
	stride := 5 + numCls
	type cand struct {
		box   [4]float32
		score float32
		cls   int
	}
	var cands []cand
	for off := 0; off+stride <= len(row); off += stride {
		obj := row[off+4]
		if obj < confThresh {
			continue
		}
		best, cls := float32(0), 0
		for c, s := range row[off+5 : off+stride] {
			if s > best {
				best, cls = s, c
			}
		}
		score := obj * best
		if score < confThresh {
			continue
		}
		cx, cy, w, h := row[off], row[off+1], row[off+2], row[off+3]
		cands = append(cands, cand{[4]float32{cx - w/2, cy - h/2, cx + w/2, cy + h/2}, score, cls})
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].score > cands[j].score })

	var dets []Detection
	kept := make([]bool, len(cands))
	for i, c := range cands {
		suppressed := false
		for j := 0; j < i; j++ {
			if kept[j] && cands[j].cls == c.cls && iou(cands[j].box, c.box) > iouThresh {
				suppressed = true
				break
			}
		}
		if suppressed {
			continue
		}
		kept[i] = true
		d := Detection{ClassIndex: c.cls, Confidence: float64(c.score), Box: c.box}
		if c.cls < len(labels) {
			d.ClassName = labels[c.cls]
		}
		dets = append(dets, d)
		if len(dets) == maxDetections {
			break
		}
	}
	return dets
}

func iou(a, b [4]float32) float32 {
	iw := min(a[2], b[2]) - max(a[0], b[0])
	ih := min(a[3], b[3]) - max(a[1], b[1])
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := iw * ih
	return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
}

// toOriginal rescales boxes from model pixels to original image pixels and clips them
func (lb Letterbox) toOriginal(dets []Detection) {
	for i := range dets {
		b := &dets[i].Box
		for k := range b {
			b[k] /= lb.Scale
		}
		b[0], b[2] = clipBox(b[0], float32(lb.W)), clipBox(b[2], float32(lb.W))
		b[1], b[3] = clipBox(b[1], float32(lb.H)), clipBox(b[3], float32(lb.H))
	}
}

func clipBox(v, hi float32) float32 { return min(max(v, 0), hi) }
