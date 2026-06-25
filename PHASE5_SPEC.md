# PHASE 5 SPEC — Pipeline the Inference Stage (Break the Single-Collector Ceiling)

> **Lineage.** Phase 4 moved preprocessing onto the GPU and hit ~1,450 req/s
> (~4.2× over baseline). Instrumentation (`/stats` per-stage timing + the
> open-loop load mode) proved the new ceiling is **not** CPU or GPU saturation —
> it is the single collector goroutine doing its per-batch cycle in lockstep:
> *gather → pack (15 MB CPU copy + H2D) → `session.Run()` → split results → reply*,
> with nothing overlapping. The collector is busy 0.97–0.99 of the time, yet both
> CPU and GPU sit ~40–55 % idle because each waits for the other's turn. Phase 5
> makes those stages overlap — "prep the next tray while the oven cooks."

## 1. Context — the measured bottleneck

From Phase 4 `/stats` at a stable operating point (batch ≈ 10) and at saturation
(batch ≈ 25–32):

| per-batch stage (single collector) | time | overlap-able? |
|---|---:|---|
| gather wait (`maxWait`) | ≤ 5 ms | yes (prep next while infer runs) |
| pack: assemble `[N,3,224,224]` (≈15 MB memcpy) + H2D copy | ~1–3 ms | yes (CPU + DMA, not compute) |
| `session.Run()` (the only true GPU-compute step) | ~7 ms @ batch-32 | the floor |
| split logits + reply to callers | ~1–4 ms | yes (pure CPU, off critical path) |

Everything except `Run()` is gather-wait, CPU work, or DMA — none of it needs to
block the GPU, yet today it all runs in series on one goroutine. The isolation
test (pre-packed tensor, GPU to itself) sustains ~4,200 req/s, so the inference
*compute* has large headroom over the served ~1,450.

Two further measured facts that constrain the design:
- **Overload collapse.** Past the knee, throughput *drops* (1,442 → 1,191) and
  latency explodes (33 ms → ~2 s) because decode and inference contend for the GPU
  (no NVJPG engine on this RTX 4080 SUPER — decode uses the CUDA cores). Pushing
  more concurrency without bound makes it worse. ⇒ Phase 5 must add **bounded
  in-flight / backpressure**, not just more parallelism.
- **`Run` wall-time is CPU-scheduling sensitive.** The collector goroutine
  competing for cores inflates measured `Run`. ⇒ keep the GPU-owning path lean and
  off the contended cores where possible.

## 2. Goal & success criteria

Raise sustained throughput meaningfully above ~1,450 req/s by overlapping the
non-compute stages with `Run()`, **without** breaking batching correctness.

- **Primary:** sustained throughput materially above 1,450 req/s (target: approach
  the inference-compute ceiling, ~2,500–4,000 depending on decode contention).
  Report the honest number, measured with the existing open-loop mode, and where
  the bottleneck moves next.
- **Correctness preserved (non-negotiable):** results must still route to the
  correct caller with zero swaps under the new concurrency — the Phase 3
  mixed-batch / no-swap / known-answer tests must pass unchanged. Parallelizing
  the result path is the main correctness risk.
- **Stable under overload:** with bounded in-flight + backpressure, offered load
  beyond capacity should degrade gracefully (flat throughput, bounded latency,
  explicit rejections) instead of the Phase 4 latency collapse.
- **No regressions:** GPU placement, predictions (dog → Samoyed, cat → Egyptian
  cat), and the Phase-0 build all unchanged. A measured partial result is still
  valid (Phase 3/4 precedent).

## 3. Architecture — keep one gather point, overlap the rest

The single rendezvous stays (batching *requires* one gather point, and there is
one GPU). What changes: the collector stops doing pack/infer/unpack in lockstep
and instead hands work down a pipeline so stage *k+1* of batch *i* overlaps stage
*k* of batch *i+1*.

```
handlers ─▶ [gather+pack] ─chan─▶ [infer lane(s)] ─chan─▶ [unpack+reply workers]
            (1 goroutine)          (GPU-owning)            (N goroutines)
```

- **Gather+pack goroutine.** Forms a batch, assembles the input tensor, and sends
  the *packed* batch (tensor + the ordered list of result channels) to the infer
  channel — then immediately loops to gather the next batch. The 15 MB copy and
  H2D now happen *while the previous batch is inferring*.
- **Infer lane(s).** Own the ORT session; pull a packed batch, run inference, send
  `(output, batch)` to the unpack channel. Start with **one** lane (simplest
  correct overlap); evaluate 2–3 lanes only if measurement shows the single lane
  is the limit (see Decisions).
- **Unpack+reply workers.** A small pool that splits `[N,1000]` logits and delivers
  each row to its caller's channel. Pure CPU, fully parallel, off the GPU path.

Result routing correctness is preserved by carrying the **ordered result-channel
slice with the batch** through every stage (same index correspondence Phase 3
unit-tested), never by any shared/aliased mapping.

## 4. Key decisions to resolve (measure, don't assume)

1. **Can one ORT session overlap H2D of batch i+1 with compute of batch i?**
   `DynamicAdvancedSession.Run` is synchronous and manages its own stream.
   Investigate whether decoupling pack from Run actually overlaps, or whether ORT
   serializes internally. If it serializes, the win comes from overlapping
   *gather+pack+unpack* with Run (still substantial), not from H2D/compute overlap.
2. **One infer lane vs N lanes (multiple ORT sessions).** Multiple sessions =
   true compute interleaving but N× weights in GPU memory (ResNet-18 ≈ 76 MiB, so
   2–3 is cheap) and N× decode/infer GPU contention. Measure 1 vs 2 vs 3; stop
   when throughput stops improving or latency degrades.
3. **Separate CUDA streams / priorities for decode vs inference.** The decode pool
   and inference share the GPU; giving inference stream priority may reduce the
   overload-collapse contention. Measure.
4. **Backpressure policy.** Bound total in-flight (queue depth). On exceed: block
   briefly then 503/429, or shed. Pick a default that keeps latency bounded under
   the open-loop overload test.

## 5. Scope

**In:** decouple gather/pack, infer, and unpack/reply into a pipeline; ordered
result routing preserved; bounded in-flight + backpressure (reject when full);
re-measure with the existing `/stats` + open-loop tooling; `PHASE5.md`.

**Out:** multi-GPU (Phase 6); full production observability/metrics endpoint,
tracing, structured logging, TLS/auth (a later hardening phase); GPU-side
normalize; re-export of the model. PNG support stays on the CPU fallback.

## 6. Tasks (ordered to de-risk early)

1. **Concurrency probe (gate).** Before refactoring, micro-measure whether two
   `Run()` calls (one session vs two sessions) overlap on this ORT build, and
   whether H2D overlaps compute. Decides one-lane vs N-lane and sets the realistic
   target. Reuse `engine_throughput_test.go`.
2. **Split the collector into the 3-stage pipeline** with channels, **one** infer
   lane first. Carry the ordered result-channel slice with each batch.
3. **Re-run the Phase 3 correctness suite** (mixed-batch no-swap, known-answers
   concurrent, routing units) — must pass before any perf claim.
4. **Move unpack+reply to a worker pool**; confirm the collector no longer blocks
   on result delivery (`/stats` busy-fraction of the gather stage should drop).
5. **Add bounded in-flight + backpressure**; verify the open-loop overload test no
   longer collapses (flat throughput, bounded p99, explicit rejects).
6. **Evaluate N infer lanes and decode/infer stream separation** per the probe;
   keep only what measurably helps.
7. **Re-measure** (open-loop sweep + `/stats`): throughput, per-stage times,
   busy-fractions, the new bottleneck. Before/after vs the ~1,450 Phase 4 number.
8. **`PHASE5.md`** — honest report: what overlapped, the new throughput and
   ceiling, correctness evidence, and backpressure behavior.

## 7. Correctness requirements

- Every result returns to its originating caller — **zero swaps** — verified by the
  existing `TestMixedBatchNoSwap`, `TestKnownAnswersConcurrent`, and
  `TestSplitBatchLogits_*` tests, plus a new test that runs many *mixed* concurrent
  requests through the pipelined path and checks each answer.
- No data races: run the correctness tests under `go test -race` (CPU-only paths)
  where feasible.

## 8. Risks & mitigations

- **Result mis-routing under concurrency (highest).** *Mitigation:* the ordered
  result-channel slice travels with the batch through every stage; no shared map;
  Phase 3 routing tests are the gate; add a mixed-load pipeline test.
- **ORT session not concurrency-friendly.** *Mitigation:* Task 1 probe decides
  single vs multi-session before committing; fall back to single-lane overlap of
  the CPU/DMA stages (still a real win).
- **GPU OOM from multiple in-flight batches / sessions.** *Mitigation:* bounded
  in-flight; ResNet-18 sessions are tiny; measure GPU mem.
- **Overload still collapses** if backpressure is wrong. *Mitigation:* tune
  against the open-loop overload test; prefer bounded queue + reject.
- **No throughput gain if ORT fully serializes the GPU.** *Mitigation:* a negative
  result with the measured reason (overlap of CPU stages didn't move the floor
  because Run dominates) is a valid outcome — report it.

## 9. Deliverables

- Pipelined engine (gather/pack → infer → unpack/reply) with bounded in-flight.
- New mixed-load pipeline correctness test; Phase 3 suite still green (+ `-race`).
- Reused `/stats` + open-loop measurements; before/after throughput + per-stage table.
- `PHASE5.md` honest report including the new bottleneck location.
