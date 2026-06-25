# Phase 5 — Pipeline the Inference Stage (Honest Negative Throughput Result)

**Headline (honest): pipelining the inference stage was built correctly and did
exactly what it was designed to do at the goroutine level — it freed the inference
lane (busy-fraction 0.99 → 0.10) and moved result-splitting off the critical path
— but end-to-end throughput barely moved (~1,462 → ~1,500 req/s, +3%). The reason
is the headline finding: the ~1,500 ceiling was never the single-collector
serialization that Phase 4 blamed. Freeing that lane didn't help because the real
wall is GPU-side contention between JPEG decode and inference sharing one GPU. This
corrects Phase 4's conclusion with a direct, instrumented disproof.**

Two things did land: a measured re-diagnosis of the true bottleneck, and working
backpressure that turns the Phase 4 overload *collapse* (latency → seconds) into
graceful load-shedding (bounded latency + HTTP 503).

## What was built

- **3-stage pipeline** replacing the single collector goroutine
  (`engine.go`): `gather+pack → infer (1 lane) → unpack+reply (worker pool)`,
  connected by channels that carry the ordered result-channel slice (so routing
  stays correct). The gatherer assembles the next batch's input tensor while the
  lane runs the previous one; 4 unpack workers split logits + reply off the lane.
- **Single inference lane / single session**, per the Task-1 probe (below).
- **Backpressure** (`-max-inflight`): an admission gate at *request entry* (before
  the GPU decode) that sheds excess with HTTP 503 instead of queueing unboundedly.
- **A latent bug fixed:** ORT's environment must be initialized once per process.
  The suite had been doing `InitializeEnvironment`/`DestroyEnvironment` per engine;
  the Nth cycle segfaults. Now `ensureORT` (a `sync.Once`) initializes once and
  `Close()` no longer tears the environment down. This also hardens the server.

## Task 1 — ORT concurrency probe (the gate that shaped the design)

`TestORTConcurrencyProbe` (run with `ORT_PROBE=1`; skipped in the normal suite
because it manages its own ORT env) hammered `session.Run()` directly at batch-32:

| Config | req/s @ batch-32 |
|---|---:|
| 1 session, serial | 5,885 |
| 1 session, 2 concurrent Run | 6,793 |
| 1 session, 4 concurrent Run | 6,840 |
| 2 sessions | 5,597 |
| 3 sessions | 5,017 |

**Decisions it forced:** (a) raw inference does ~5,900 req/s — so the served
~1,450 is nowhere near inference compute; the gap must be non-Run work or GPU
contention. (b) **Multiple sessions HURT** — so a single lane, not multi-lane.

## Correctness — preserved (Task 3)

- **`TestPipelineRoutingStress`** (new): 1,000 concurrent mixed dog/cat requests
  through the pipeline → **0 swaps**. ✅
- Phase 3 `TestMixedBatchNoSwap`, `TestKnownAnswersConcurrent`,
  `TestSplitBatchLogits_*` all pass through the pipeline; routing units pass under
  `-race`. ✅ Live HTTP: dog → Samoyed, cat → Egyptian cat. ✅

## Throughput — before/after (same machine, same tooling)

| | Phase 4 (single collector) | Phase 5 (pipelined) |
|---|---:|---:|
| closed-loop c=128 | 1,462 | 1,485 |
| closed-loop c=256 | 1,449 | 1,505 |
| **peak** | **~1,460** | **~1,500 (+3%)** |

No meaningful gain. The pipeline is correct and slightly faster, but it did not
break the ceiling.

## Why pipelining didn't help — measured, and it corrects Phase 4

Phase 4 concluded the bottleneck was the single collector's serialized per-batch
cycle (it was 99% busy). Phase 5 **freed** that lane and measured the result with
`/stats` under open-loop saturation:

- **Infer-lane busy-fraction dropped 0.99 → 0.10.** The lane is now 90% *idle* —
  it is emphatically no longer the bottleneck.
- **`avg_runbatch_ms` ≈ `avg_infer_run_ms`** (5.08 ≈ 5.06) — the unpack/split work
  is fully off the lane now.
- **Yet throughput stayed ~1,500.** So the inference-lane serialization was *not*
  the binding constraint. Phase 4's "collector 99% busy" was a **symptom** of the
  lane waiting on a contended GPU, not the cause.

The real wall is **GPU-side decode↔inference contention**:
- `avg_gpu_decode_ms` ≈ 5 ms; the GPU does both decode (nvJPEG/NPP on the CUDA
  cores — this card has no dedicated NVJPG engine) and inference.
- **More decoders don't help:** 8 / 16 / 24 / 32 contexts all plateau at ~1,400–
  1,500 with GPU util stuck at ~63–66%.
- The GPU is never driven past ~65% util, yet nothing upstream or downstream is
  the limit — the signature of **non-overlapping GPU work**: per-decode
  `cudaStreamSynchronize`, synchronous ORT `Run`, and nvJPEG/NPP/ORT not sharing
  streams mean decode and inference kernels serialize with bubbles between them.

Breaking this needs **GPU-level** overlap, not Go-level: decode and inference on
separate non-blocking CUDA streams, removing the per-decode sync, CUDA graphs, or
moving decode off the inference GPU entirely (a second GPU, or the dedicated NVJPG
engine consumer Ada lacks). That is Phase 6 territory.

## Backpressure — works (Task 4)

Under an open-loop overload (3,000 req/s offered, capacity ~1,500):

| | latency p50 | behavior |
|---|---:|---|
| Phase 4 (none) | ~2,000 ms, climbing | unbounded queue, collapse |
| gate **inside** Predict (after decode) | 3,700 ms | wrong: every request decodes before being shed |
| gate at **entry**, `-max-inflight 96` | **60 ms** | excess shed as 503 at entry; admitted requests bounded |

The key lesson: the admission gate must sit **before** the expensive GPU decode,
or shed requests still pay the decode cost. With it in the right place, overload
degrades gracefully (10,061 × 503, bounded p50) instead of collapsing.

## Conclusion

- ✅ Pipeline built and correct; inference lane freed (busy-frac 0.99→0.10),
  routing preserved (0 swaps), latent ORT-init bug fixed.
- ✅ Backpressure converts overload collapse into graceful 503 shedding.
- ❌ **No throughput gain (~1,460 → ~1,500, +3%).** Honestly reported.
- 🔎 **Measured cause, correcting Phase 4:** the ceiling is GPU decode↔inference
  contention on one shared GPU (no NVJPG engine; non-overlapping kernels; ~63%
  util wall), *not* CPU-side collector serialization. Freeing the collector
  proved this by disproving it.
- ➡️ **Phase 6:** GPU-level decode/inference overlap (separate CUDA streams, drop
  per-decode sync, CUDA graphs) or a second GPU — the only paths left to the
  ~5,900 inference ceiling.

## Files

- `engine.go` — 3-stage pipeline (`collect`/`packAndSend`/`inferLoop`/`unpackLoop`),
  `Acquire`/`Release` backpressure, `ensureORT` (once-per-process), timing stats.
- `phase5_probe_test.go` — ORT concurrency probe (run with `ORT_PROBE=1`).
- `pipeline_test.go` — 1,000-request mixed routing stress (0 swaps).
- `main.go` — `-max-inflight` flag; entry-time admission gate; `/stats` rejects.
- `loadtest/` — open-loop `-rates`/`-rate-dur`/`-max-inflight` mode (Phase 4).

## Reproduce

```bash
./run.sh -gpu-preprocess -gpu-decoders 8 -max-inflight 0 &
# throughput (no gain over Phase 4):
./loadtest/loadtest -levels 64,128,256 -n 8000 -warmup 300
# bottleneck proof: infer lane idle, decode-bound:
curl -s 'localhost:8080/stats?reset=1'; ./loadtest/loadtest -rates 3000 -rate-dur 12s -max-inflight 512; curl -s localhost:8080/stats
# backpressure: restart with -max-inflight 96, offer 3000 -> bounded p50 + 503s
# ORT probe:
ORT_PROBE=1 ./test_phase3.sh -run TestORTConcurrencyProbe
```
