# Summarize an Nsight Systems SQLite export of the server into the JSON the dashboard shows
# python summarize_nsys.py report.sqlite model.onnx > report.nsys.json
import json
import re
import sqlite3
import sys

COPY_H2D, COPY_D2H = 1, 2
PINNED = 1


def union(intervals):
    """Merge overlapping (start, end) intervals."""
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def length(merged):
    return sum(e - s for s, e in merged)


def intersect(a, b):
    """Total overlap between two merged interval lists."""
    i = j = total = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def pct(part, whole):
    return round(100 * part / whole, 1) if whole else 0.0


def rank(values, q):
    if not values:
        return 0.0
    v = sorted(values)
    return v[max(int(-(-q * len(v) // 1)) - 1, 0)]


def short(name):
    name = re.sub(r"^void ", "", name)
    return re.split(r"[<(]", name)[0][:72]


def table(db, name):
    return db.execute("select 1 from sqlite_master where type='table' and name=?", (name,)).fetchone()


def summarize(db, model):
    strings = dict(db.execute("select id, value from StringIds"))
    if not table(db, "CUPTI_ACTIVITY_KIND_KERNEL"):
        raise SystemExit("no GPU kernel records in the report; this Nsight Systems may not support the driver")
    kernels = db.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    copies = []
    if table(db, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        copies = db.execute("select start, end, bytes, copyKind, srcKind, dstKind from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
    api = db.execute("select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME").fetchall()

    starts = [r[0] for r in kernels + copies + api]
    ends = [r[1] for r in kernels + copies + api]
    window = max(ends) - min(starts)

    k = union((s, e) for s, e, _ in kernels)
    c = union((s, e) for s, e, *_ in copies)
    busy = union([tuple(x) for x in k + c])
    gaps = [b[0] - a[1] for a, b in zip(busy, busy[1:])]

    def copy_stats(kind):
        rows = [r for r in copies if r[3] == kind]
        total_bytes = sum(r[2] for r in rows)
        total_ns = sum(r[1] - r[0] for r in rows)
        pinned = sum(r[2] for r in rows if (r[4] if kind == COPY_H2D else r[5]) == PINNED)
        return {
            "count": len(rows),
            "mb": round(total_bytes / 1e6, 1),
            "avg_kb": round(total_bytes / len(rows) / 1e3, 1) if rows else 0,
            "gb_per_s": round(total_bytes / total_ns, 2) if total_ns else 0,
            "pinned_pct": pct(pinned, total_bytes),
        }

    per_kernel = {}
    for s, e, name in kernels:
        n = short(strings.get(name, str(name)))
        calls, ns = per_kernel.get(n, (0, 0))
        per_kernel[n] = (calls + 1, ns + e - s)
    kernel_ns = sum(ns for _, ns in per_kernel.values())
    top = sorted(per_kernel.items(), key=lambda kv: -kv[1][1])[:8]

    def api_time(pattern):
        rows = [r for r in api if re.search(pattern, strings.get(r[2], ""))]
        return len(rows), sum(r[1] - r[0] for r in rows)

    launches, launch_ns = api_time(r"Launch")
    syncs, sync_ns = api_time(r"Synchronize|cudaMemcpy(?!Async)")
    durations = [e - s for s, e, _ in kernels]

    return {
        "model": model,
        "seconds": round(window / 1e9, 2),
        "gpu_busy_pct": pct(length(busy), window),
        "compute_pct": pct(length(k), window),
        "copy_pct": pct(length(c), window),
        "copy_hidden_pct": pct(intersect(c, k), length(c)),
        "idle_gaps": len(gaps),
        "idle_gap_p50_us": round(rank(gaps, 0.5) / 1e3, 1),
        "idle_gap_p99_us": round(rank(gaps, 0.99) / 1e3, 1),
        "kernels_per_s": round(len(kernels) / (window / 1e9), 0),
        "kernel_avg_us": round(sum(durations) / len(durations) / 1e3, 1) if durations else 0,
        "short_kernel_pct": pct(sum(1 for d in durations if d < 10_000), len(durations)),
        "launch_avg_us": round(launch_ns / launches / 1e3, 1) if launches else 0,
        "threads_blocked_on_gpu": round(sync_ns / window, 2),
        "h2d": copy_stats(COPY_H2D),
        "d2h": copy_stats(COPY_D2H),
        "top_kernels": [
            {"name": n, "calls": calls, "total_ms": round(ns / 1e6, 2), "avg_us": round(ns / calls / 1e3, 1), "share_pct": pct(ns, kernel_ns)}
            for n, (calls, ns) in top
        ],
    }


if __name__ == "__main__":
    print(json.dumps(summarize(sqlite3.connect(sys.argv[1]), sys.argv[2]), indent=1))
