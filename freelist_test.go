package main

import "testing"

func TestFreeListRecyclesUpToSize(t *testing.T) {
	made := 0
	f := newFreeList(2, func() []byte { made++; return make([]byte, 8) })
	a, b, c := f.get(), f.get(), f.get()
	if made != 3 {
		t.Fatalf("made %d buffers for 3 gets on an empty list, want 3", made)
	}
	f.put(a)
	f.put(b)
	f.put(c) // over capacity, dropped
	f.get()
	f.get()
	if made != 3 {
		t.Errorf("made %d buffers, want the two kept ones reused", made)
	}
	f.get()
	if made != 4 {
		t.Errorf("made %d buffers, want a new one once the list is empty", made)
	}
}
