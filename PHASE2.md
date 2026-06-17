# Phase 2 — Concurrent Load Baseline (measurement only)

**Headline: baseline ≈ 345 req/s, and it plateaus there from concurrency 16 onward
— while the GPU sits ~62% idle (avg compute util ~38%, never sustained above ~50%).
Past concurrency 16, extra load only inflates latency (p99 96 ms → 835 ms at c=128),
not throughput.** This is the number Phase 3's dynamic batching will be measured against.

The server was **not modified** in this phase. This measures the Phase 1
one-request-at-a-time design (inference serialized by a mutex) to quantify the
bottleneck before Phase 3 fixes it.

## Method

- **Load tool:** `loadtest/` — a closed-loop Go HTTP driver (separate binary, no
  cgo, does not import the server). `C` workers each fire the next `POST /predict`
  the instant the previous response returns; it records every request's latency
  and computes throughput + p50/p95/p99. A background goroutine polls
  `nvidia-smi --query-gpu=utilization.gpu,memory.used` every 100 ms during each level.
- **Server:** unmodified Phase 1 `./server` (ResNet-18, CUDA EP), started via `./run.sh`.
- **Workload:** `testdata/dog.jpg` and `testdata/cat.jpg`, cycled round-robin.
- **Per level:** 50 warmup requests (discarded) + 1500 measured requests.
- **Run:** `./loadtest/loadtest -levels 1,2,4,8,16,32,64,128 -n 1500 -warmup 50`
- Raw data: **`phase2_results.csv`** / **`phase2_results.json`** (committed alongside this file).

## Results

| Concurrency | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) | max (ms) | GPU util avg | GPU util max |
|------------:|-------------------:|---------:|---------:|---------:|----------:|---------:|-------------:|-------------:|
|           1 |               28.7 |     48.5 |     52.4 |     53.5 |      34.8 |     59.0 |          35% |          36% |
|           2 |               62.7 |     44.4 |     53.2 |     56.1 |      31.9 |     62.0 |          46% |          58% |
|           4 |              133.8 |     42.3 |     51.3 |     54.5 |      29.8 |     59.5 |          51% |          59% |
|           8 |              251.0 |     40.9 |     59.9 |     66.3 |      31.8 |     82.2 |          38% |          74% |
|      **16** |          **342.2** |     48.7 |     86.0 |     96.0 |      46.5 |    111.1 |          39% |          50% |
|          32 |              350.4 |     92.5 |    146.8 |    177.1 |      90.7 |    229.8 |          38% |          43% |
|          64 |              342.7 |    171.6 |    263.3 |    430.9 |     183.8 |    596.5 |          36% |          43% |
|         128 |              330.8 |    364.3 |    563.1 |    834.8 |     374.8 |    964.6 |          35% |          52% |

(The four spec-requested levels are **1, 16, 64** plus **4**: 28.7 / 133.8 / 342.2 / 342.7 req/s.)

## Does throughput plateau? Yes — quantified.

```
req/s  350 |                         ● ─ ● ─ ● ─ ●   <- plateau ~345 req/s (c=16..128)
       300 |                   ●
       250 |              ●
       200 |
       150 |         ●
       100 |
        50 |    ●
         0 |  ●
           +--------------------------------------------
             1    2    4    8   16   32   64  128   concurrency
```

- Throughput rises while concurrency is below ~16 (28.7 → 342 req/s), then **flattens
  at ~345 req/s** and even dips slightly at 128. The plateau begins at **concurrency 16**.
- Past the plateau, latency grows ~linearly with concurrency (requests queue on the
  mutex): p50 ≈ 49 → 93 → 172 → 364 ms and p99 ≈ 96 → 177 → 431 → 835 ms as
  concurrency goes 16 → 32 → 64 → 128. Classic closed-loop saturation — you pay in
  latency for offered load the server can't actually parallelize.
- **Why ~345 and why c≈16:** the plateau throughput implies the serialized
  (mutex-guarded) inference critical section is ≈ 1000/345 ≈ **2.9 ms** per request.
  Single-request latency is ≈ 35 ms (dominated by CPU JPEG-decode + resize, which
  *do* parallelize across cores). Concurrency saturates the one serial resource at
  about latency/critical-section ≈ 35/2.9 ≈ **12**, matching the observed knee at 16.

## GPU underutilization (the evidence for Phase 3)

Two independent measurements agree the GPU is idle most of the time, even at max load:

1. **Embedded sampler (`--query-gpu=utilization.gpu`)**, averaged over each level:
   GPU compute util stays **35–51%**, and at the plateau (c≥16) it is only **~36–39% avg**,
   max ~50%. GPU memory is flat at ~1358 MiB throughout (one model, one image at a time).

2. **`nvidia-smi dmon` during a sustained c=64 burst** (independent of the sampler):
   ```
   #    sm    mem    pwr   pclk
   #     %      %      W    MHz
        25      4     32   2595
        28      5     67   2595
        36      5     61   1980
        38      5     58   1920
   ```
   Streaming-multiprocessor (`sm`) utilization 25–38% and power 32–67 W on a ~320 W
   card — the RTX 4080 is loafing while clients queue.

**Interpretation:** because the server processes exactly one image per GPU call and
serializes those calls, the card spends the majority of wall-clock time idle between
tiny batch-1 inferences (kernel-launch + host↔device copy overhead dominate a single
ResNet-18 forward pass). The GPU has ample headroom (~60%+ idle, ~14 GB free memory) to
process many images per call. That is exactly what Phase 3's dynamic batching will exploit.

## Baseline to beat (for Phase 4)

| Metric | Baseline (Phase 1 server, this phase) |
|---|---|
| **Peak throughput** | **~345 req/s** (plateau, concurrency ≥ 16) |
| Plateau onset | concurrency ≈ 16 |
| p50 / p99 at c=64 | 172 ms / 431 ms |
| GPU compute util at plateau | ~38% avg (≈ 62% idle), max ~50% |
| Serialized inference critical section | ≈ 2.9 ms/req |

**One-line headline:** *baseline ≈ 345 req/s peak (plateaus at concurrency 16); GPU
compute idle ≈ 62% of the time even at concurrency 64.*

## Files

- `loadtest/main.go` — the load generator + GPU sampler.
- `loadtest/loadtest` — built binary (`CGO_ENABLED=0 go build -o loadtest/loadtest ./loadtest`).
- `phase2_results.csv`, `phase2_results.json` — raw per-level numbers (the dataset Phase 4 compares against).

## Reproduce

```bash
./run.sh &                 # start the unmodified Phase 1 server on :8080
CGO_ENABLED=0 ~/sdk/go/bin/go build -o loadtest/loadtest ./loadtest
./loadtest/loadtest -levels 1,2,4,8,16,32,64,128 -n 1500 -warmup 50 -out phase2_results
```
