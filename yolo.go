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

var letterboxPool = sync.Pool{New: func() any { return make([]byte, 3*letterboxSize*letterboxSize) }}

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

// letterboxInto resizes img to fit 640x640 at the top left and pads the rest.
// Bilinear in 8 bit fixed point: each source row is resampled across once into
// a cached row, and output rows blend the two cached rows they fall between.
func letterboxInto(img *image.RGBA, out []byte) float32 {
	sw, sh := img.Rect.Dx(), img.Rect.Dy()
	r := min(float32(letterboxSize)/float32(sw), float32(letterboxSize)/float32(sh))
	rw, rh := int(float32(sw)*r), int(float32(sh)*r)
	const plane = letterboxSize * letterboxSize
	b, g, rd := out[:plane], out[plane:2*plane], out[2*plane:3*plane]
	for y := rh; y < letterboxSize; y++ {
		i := y * letterboxSize
		fill(b[i:i+letterboxSize], g[i:i+letterboxSize], rd[i:i+letterboxSize])
	}
	if rw == sw && rh == sh {
		for y := 0; y < rh; y++ {
			i := y * letterboxSize
			fill(b[i+rw:i+letterboxSize], g[i+rw:i+letterboxSize], rd[i+rw:i+letterboxSize])
			row := img.Pix[y*img.Stride : y*img.Stride+rw*4]
			for x := 0; x < rw; x++ {
				rd[i+x], g[i+x], b[i+x] = row[x*4], row[x*4+1], row[x*4+2]
			}
		}
		return r
	}

	// source columns and weights out of 256 are the same for every row
	var x0s, x1s [letterboxSize]int
	var wxs [letterboxSize]int
	inv := 1 / r
	for x := 0; x < rw; x++ {
		fx := (float32(x)+0.5)*inv - 0.5
		x0 := max(floor32(fx), 0)
		x0s[x], x1s[x] = x0*4, min(x0+1, sw-1)*4
		wxs[x] = min(int(max(fx-float32(x0), 0)*256+0.5), 256)
	}

	// two cached source rows resampled across, as RGB planes of value*256
	var cache [2][3][letterboxSize]uint16
	cached := [2]int{-1, -1}
	load := func(slot, y int) {
		row := img.Pix[y*img.Stride:]
		cr, cg, cb := cache[slot][0][:rw], cache[slot][1][:rw], cache[slot][2][:rw]
		xs0, xs1, ws := x0s[:len(cr)], x1s[:len(cr)], wxs[:len(cr)]
		for x := range cr {
			p, q, w := row[xs0[x]:xs0[x]+3], row[xs1[x]:xs1[x]+3], ws[x]
			w0 := 256 - w
			cr[x] = uint16(int(p[0])*w0 + int(q[0])*w)
			cg[x] = uint16(int(p[1])*w0 + int(q[1])*w)
			cb[x] = uint16(int(p[2])*w0 + int(q[2])*w)
		}
		cached[slot] = y
	}
	slotOf := func(y int) int {
		switch y {
		case cached[0]:
			return 0
		case cached[1]:
			return 1
		}
		return -1
	}

	for y := 0; y < rh; y++ {
		i := y * letterboxSize
		fill(b[i+rw:i+letterboxSize], g[i+rw:i+letterboxSize], rd[i+rw:i+letterboxSize])
		fy := (float32(y)+0.5)*inv - 0.5
		y0 := max(floor32(fy), 0)
		y1 := min(y0+1, sh-1)
		wy := min(int(max(fy-float32(y0), 0)*256+0.5), 256)
		s0 := slotOf(y0)
		if s0 < 0 {
			s0 = 0
			if cached[0] == y1 {
				s0 = 1
			}
			load(s0, y0)
		}
		s1 := slotOf(y1)
		if s1 < 0 {
			s1 = 1 - s0
			load(s1, y1)
		}
		wy0 := 256 - wy
		dr := rd[i : i+rw]
		dg, db := g[i:i+len(dr)], b[i:i+len(dr)]
		tr, tg, tb := cache[s0][0][:len(dr)], cache[s0][1][:len(dr)], cache[s0][2][:len(dr)]
		br, bg, bb := cache[s1][0][:len(dr)], cache[s1][1][:len(dr)], cache[s1][2][:len(dr)]
		for x := range dr {
			dr[x] = uint8((int(tr[x])*wy0 + int(br[x])*wy + 1<<15) >> 16)
			dg[x] = uint8((int(tg[x])*wy0 + int(bg[x])*wy + 1<<15) >> 16)
			db[x] = uint8((int(tb[x])*wy0 + int(bb[x])*wy + 1<<15) >> 16)
		}
	}
	return r
}

var padRow = func() (p [letterboxSize]byte) {
	for i := range p {
		p[i] = letterboxPad
	}
	return
}()

func fill(planes ...[]byte) {
	for _, p := range planes {
		copy(p, padRow[:len(p)])
	}
}

// PreprocessLetterbox decodes on a pooled decoder and returns a pooled uint8 input
func (p *cpuDecoderPool) PreprocessLetterbox(jpeg []byte) ([]byte, Letterbox, error) {
	w := region(stageDecoderWait)
	d := <-p.free
	w.End()
	setStage(stageDecode)
	defer setStage(stageRequest)
	defer region(stageDecode).End()
	defer func() { p.free <- d }()
	t := time.Now()
	img, fw, fh, err := d.decodeFitLong(jpeg, letterboxSize)
	if err != nil {
		return nil, Letterbox{}, err
	}
	t2 := time.Now()
	out := letterboxPool.Get().([]byte)
	r := letterboxInto(img, out)
	atomic.AddInt64(&p.statDecodeNs, int64(t2.Sub(t)))
	atomic.AddInt64(&p.statNormNs, int64(time.Since(t2)))
	atomic.AddInt64(&p.statCount, 1)
	return out, Letterbox{Scale: r * float32(img.Rect.Dx()) / float32(fw), W: fw, H: fh}, nil
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
