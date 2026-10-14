package main

import (
	"image"
	"testing"
)

func yoloRow(boxes [][6]float32, numCls int) []float32 {
	stride := 5 + numCls
	row := make([]float32, 8400*stride)
	for i, b := range boxes {
		off := i * stride
		copy(row[off:], b[:5])
		row[off+5+int(b[5])] = 1
	}
	return row
}

func TestDecodeDetectionsNMS(t *testing.T) {
	labels := []string{"cat", "dog"}
	// cx cy w h obj class
	row := yoloRow([][6]float32{
		{100, 100, 50, 50, 0.9, 0},
		{102, 101, 50, 50, 0.8, 0}, // overlaps the first same class so suppressed
		{102, 101, 50, 50, 0.7, 1}, // same box other class so kept
		{400, 400, 20, 20, 0.2, 1}, // below threshold
	}, 2)
	dets := decodeDetections(row, 2, labels, 0.3, 0.45)
	if len(dets) != 2 {
		t.Fatalf("got %d detections %+v want 2", len(dets), dets)
	}
	if dets[0].ClassName != "cat" || dets[0].Box != [4]float32{75, 75, 125, 125} {
		t.Errorf("first detection %+v", dets[0])
	}
	if dets[1].ClassName != "dog" {
		t.Errorf("second detection %+v", dets[1])
	}
}

func TestLetterboxToOriginal(t *testing.T) {
	dets := []Detection{{Box: [4]float32{-10, 20, 320, 700}}}
	Letterbox{Scale: 0.5, W: 600, H: 1000}.toOriginal(dets)
	if dets[0].Box != [4]float32{0, 40, 600, 1000} {
		t.Errorf("box %v", dets[0].Box)
	}
}

func TestLetterboxIntoPadsAndSwapsChannels(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 320, 160))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = 10, 20, 30, 255
	}
	out := make([]float32, 3*letterboxSize*letterboxSize)
	if r := letterboxInto(img, out); r != 2 {
		t.Fatalf("scale %v want 2", r)
	}
	const plane = letterboxSize * letterboxSize
	if out[0] != 30 || out[plane] != 20 || out[2*plane] != 10 {
		t.Errorf("pixel 0 BGR %v %v %v", out[0], out[plane], out[2*plane])
	}
	last := plane - 1
	if out[last] != letterboxPad {
		t.Errorf("padding %v", out[last])
	}
}
