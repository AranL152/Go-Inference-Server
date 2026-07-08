# PHASE 7 — Close the In-Server Gap (results)

Goal (see `PHASE7_SPEC.md`): after Phase 6 the live server did ~1,790 req/s while
isolated CPU preprocess measured ~2,450/s and the GPU sat ~65% idle. Pin the gap
by measurement, fix what the numbers justify, and try (gated) spilling decode to
the idle GPU. Target ≥2,100 req/s.

**Outcome: ~2,050–2,130 req/s sustained (peak 2,131 @224), 0 errors — target met.**
Allocation churn cut ~45×, per-batch inference halved (9.2 → 4.7 ms), p99 tail cut
~2.5×. The hybrid GPU-decode spill was built, measured, and **rejected** (negative
result — it reduces throughput at every pool size tested).

All numbers below are same-day A/B runs on the usual box (16-core, RTX 4080, WSL2),
`loadtest -n 12000 -warmup 200` closed-loop, after a 4,000-request warm pass (the
first level after server start otherwise eats a cold-start artifact: ~990 req/s
with a 1.1s max latency).

## Task 1 — pin the gap (measurement only)

Same-day baseline re-measure (unmodified Phase 6 code): **1,848–2,033 req/s**
across c=160–224 — already above the recorded 1,790; call it ~1,930 typical.

| Suspect | Verdict | Evidence |
|---|---|---|
| GC pressure | real, ~6% | `GOGC=1000`, zero code change: 2,017 @192 / 2,061 @224 vs 1,943/1,928. NumGC under load: ~30/s (996 GCs in a few minutes) vs 71 total for the whole GOGC=1000 run |
| Loadtest client stealing cores | **ruled out** | `top` during load: server ~13.0 cores, client ~0.4 core, ~10% machine idle |
| Allocation sites | confirmed, profile ranks them | `pprof -alloc_space`, 99.5% of 109 GB sampled: `io.ReadAll` **40.6%**, `preprocessTurbo` 29.6%, `packAndSend` 29.3% |

Alloc rate before fixes: TotalAlloc grew ~1.9 MB per request (~3.7 GB/s at load) —
matching the spec's 2.5–3 GB/s churn estimate. The spec guessed the tensor alloc
was #1; the profile says the body read (`io.ReadAll` doubling growth) was.

## Task 2 — allocation pooling (all three kept)

1. **`inputPool`** (`engine.go`): per-request NCHW tensors recycled; all three
   producers (`preprocessTurbo`, `normalizeCHW`, `normalizeRGBI`) Get from it.
   **Deviation from spec:** the spec had `packAndSend` Put the slice after copying.
   That is unsafe — the throughput/routing tests share ONE tensor across thousands
   of concurrent `Predict` calls, so pooling it while other calls still read it
   corrupts batches. Instead the **handler** Puts after `Predict` returns (it is
   the sole owner on the server paths); `Predict` documents the ownership rule.
2. **`batchBufPool`** (`engine.go`): the ~14 MB batch staging buffer is pooled at
   full `maxBatch` capacity and sliced to the realized batch size; returned in
   `inferLoop` right after `inT.Destroy()` (ORT pins the Go slice while the
   OrtValue lives). The 14 MB memcpy itself remains — it is the H2D staging.
3. **`bodyBufPool`** (`main.go`): `io.ReadAll` replaced by a pooled `bytes.Buffer`
   held until the cgo decode has consumed the raw bytes.

Step 1 alone measured flat (kept: strictly less garbage, no complexity added) —
GC frequency tracks *total* alloc rate, so the win only lands once all three
sites are fixed.

| Metric (c=224 unless noted) | Before (Phase 6) | After (all pools) |
|---|---|---|
| Throughput | 1,928–2,033 | **2,053–2,131** (+~8% same-day; +19% vs recorded 1,790) |
| Alloc per request | ~1.9 MB | **~42 KB (45× less)** |
| GCs per 48k-request sweep | ~1,000 | **9–10** |
| avg_infer_run_ms | 9.2 | **4.5–4.7** |
| avg_queue_wait_ms | 35 | **8–11** |
| p99 @224 | 141–154 ms | **126–135 ms** |

`/stats` now also reports `total_alloc_mb` and `num_gc`.

## Task 3 — hybrid CPU+GPU decode: built, measured, REJECTED

`-cpu-preprocess -gpu-preprocess` together now enables hybrid routing
(`hybrid_decode.go`): non-blocking CPU checkout first, GPU strictly as overflow,
`/stats` reports the `decode_cpu`/`decode_gpu` split. Gate conditions held after
Task 2 (CPU saturated, GPU ~38%, throughput <2,400), so it was measured:

| Config | Peak req/s | avg_infer_run_ms | GPU share of decodes |
|---|---|---|---|
| CPU-only (Task 2) | **2,131** | 4.5 | — |
| hybrid, 8 GPU decoders | 2,011 | 17.0 | 18% |
| hybrid, 4 GPU decoders | 2,035 | 17.5 | 14% |
| hybrid, 2 GPU decoders | 2,056 | 12.0 | 9% |

The Phase-4 failure mode is fully reproduced: this nvJPEG backend is
GPU_HYBRID(cuda-cores), so every GPU decode steals SMs from inference — per-batch
inference inflates 2.7–3.8×, costing more capacity than the extra decoders add.
Monotone: fewer GPU decoders → less bad. Zero is best. **Do not use hybrid mode
for throughput on this hardware.** The mode remains available (opt-in, both flags;
default behavior unchanged) for hardware where nvJPEG runs on the dedicated
engine instead of CUDA cores.

## Task 4 — correctness

- dog → Samoyed 0.9015, cat → Egyptian cat 0.6856 (unchanged).
- `TestPipelineRoutingStress` PASS (0 swaps), full `go test .` green,
  `go vet .` and `go vet -tags cpujpeg .` clean, 0 errors / 0 rejects under load.
- Default (no-flag) build and behavior unchanged.

## Where the ceiling is now

~2,100–2,130 req/s with the CPU at ~13 cores of Huffman-bound libjpeg-turbo
decode (6.3 ms/decode) plus HTTP stack, GPU still ~62% idle. The remaining gap to
the ~2,450/s isolated preprocess number is the HTTP/handler overhead sharing the
same cores — not GC (gone) and not the client (measured ~0.4 core). Next real
lever would be cheaper decode (e.g. lower-quality JPEGs, hardware decode engine,
or more cores), not further Go-level tuning.
