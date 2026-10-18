// Bounded free lists for the engine's large per-batch buffers
package main

// freeList recycles buffers through a channel. sync.Pool drops its contents at
// every GC, which for 20 to 50 MB batch buffers means a fresh allocation almost
// every batch, and those allocations are what trigger the next GC.
type freeList[T any] struct {
	ch    chan T
	fresh func() T
}

func newFreeList[T any](size int, fresh func() T) *freeList[T] {
	return &freeList[T]{ch: make(chan T, size), fresh: fresh}
}

// get returns a recycled buffer, or a new one when all are in flight
func (f *freeList[T]) get() T {
	select {
	case v := <-f.ch:
		return v
	default:
		return f.fresh()
	}
}

// put keeps the buffer for reuse unless the list is already full
func (f *freeList[T]) put(v T) {
	select {
	case f.ch <- v:
	default:
	}
}
