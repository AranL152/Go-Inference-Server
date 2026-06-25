package main

import "testing"

// TestSplitBatchLogits_Routing verifies that splitBatchLogits keeps strict row
// correspondence: row i of the [N,1000] output must become preds[i], never
// swapped with another row. This is the result-routing invariant the whole
// batching design depends on, tested here without a GPU.
func TestSplitBatchLogits_Routing(t *testing.T) {
	labels := make([]string, numClass)
	labels[258] = "Samoyed"
	labels[285] = "Egyptian cat"

	// Build a 3-row batch with a distinct, unambiguous argmax per row.
	wantIdx := []int{258, 285, 7}
	logits := make([]float32, len(wantIdx)*numClass)
	for i, idx := range wantIdx {
		logits[i*numClass+idx] = 10.0 // dominate the row
	}

	preds, err := splitBatchLogits(logits, len(wantIdx), labels)
	if err != nil {
		t.Fatalf("splitBatchLogits: %v", err)
	}
	if len(preds) != len(wantIdx) {
		t.Fatalf("got %d preds, want %d", len(preds), len(wantIdx))
	}
	for i, idx := range wantIdx {
		if preds[i].ClassIndex != idx {
			t.Errorf("row %d: got class %d, want %d (rows swapped/corrupted)", i, preds[i].ClassIndex, idx)
		}
	}
	if preds[0].ClassName != "Samoyed" || preds[1].ClassName != "Egyptian cat" {
		t.Errorf("label routing wrong: row0=%q row1=%q", preds[0].ClassName, preds[1].ClassName)
	}
}

// TestSplitBatchLogits_OrderSwaps confirms that reordering the rows reorders the
// results identically — i.e. results follow their row, proving there is no hidden
// fixed mapping.
func TestSplitBatchLogits_OrderSwaps(t *testing.T) {
	mk := func(order []int) []Prediction {
		logits := make([]float32, len(order)*numClass)
		for i, idx := range order {
			logits[i*numClass+idx] = 5.0
		}
		preds, err := splitBatchLogits(logits, len(order), nil)
		if err != nil {
			t.Fatalf("splitBatchLogits: %v", err)
		}
		return preds
	}
	a := mk([]int{258, 285})
	b := mk([]int{285, 258})
	if a[0].ClassIndex != 258 || a[1].ClassIndex != 285 {
		t.Fatalf("order A wrong: %d,%d", a[0].ClassIndex, a[1].ClassIndex)
	}
	if b[0].ClassIndex != 285 || b[1].ClassIndex != 258 {
		t.Fatalf("order B wrong: %d,%d (results did not follow rows)", b[0].ClassIndex, b[1].ClassIndex)
	}
}

func TestSplitBatchLogits_LengthMismatch(t *testing.T) {
	if _, err := splitBatchLogits(make([]float32, numClass+1), 1, nil); err == nil {
		t.Fatal("expected error on mismatched logits length, got nil")
	}
}
