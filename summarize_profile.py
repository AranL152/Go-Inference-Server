# Summarize an Nsight Compute raw CSV export into the JSON the dashboard shows
# python summarize_profile.py report.csv model.onnx > report.json
import csv
import json
import re
import sys
from collections import defaultdict

rows = list(csv.reader(open(sys.argv[1])))
head, units, data = rows[0], rows[1], rows[2:]
col = {name: i for i, name in enumerate(head)}
SCALE = {"ns": 1e-6, "nsecond": 1e-6, "us": 1e-3, "usecond": 1e-3, "ms": 1, "msecond": 1}


def num(row, name):
    i = col.get(name)
    if i is None or row[i] in ("", "n/a"):
        return 0.0
    return float(row[i].replace(",", ""))


def short(name):
    name = re.sub(r"^void ", "", name)
    return re.split(r"[<(]", name)[0][:72]


dur_scale = SCALE[units[col["gpu__time_duration.sum"]]]
kernels = []
for r in data:
    kernels.append({
        "name": short(r[col["Kernel Name"]]),
        "ms": num(r, "gpu__time_duration.sum") * dur_scale,
        "sm": num(r, "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        "mem": num(r, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
        "occupancy": num(r, "sm__warps_active.avg.pct_of_peak_sustained_active"),
        "tensor": num(r, "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active")
        + num(r, "sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active"),
    })

# each inference starts with the same first kernel so count batches by it
batches = max(1, sum(k["name"] == kernels[0]["name"] for k in kernels))
total = sum(k["ms"] for k in kernels)


def weighted(key, ks):
    t = sum(k["ms"] for k in ks)
    return sum(k[key] * k["ms"] for k in ks) / t if t else 0


groups = defaultdict(list)
for k in kernels:
    groups[k["name"]].append(k)
top = sorted(groups.items(), key=lambda g: -sum(k["ms"] for k in g[1]))
print(json.dumps({
    "model": sys.argv[2],
    "batches": batches,
    "kernels_per_batch": round(len(kernels) / batches),
    "gpu_ms_per_batch": round(total / batches, 3),
    "sm_throughput_pct": round(weighted("sm", kernels), 1),
    "memory_throughput_pct": round(weighted("mem", kernels), 1),
    "occupancy_pct": round(weighted("occupancy", kernels), 1),
    "tensor_active_pct": round(weighted("tensor", kernels), 1),
    "top_kernels": [{
        "name": name,
        "calls": len(ks) // batches or 1,
        "ms_per_batch": round(sum(k["ms"] for k in ks) / batches, 3),
        "share_pct": round(100 * sum(k["ms"] for k in ks) / total, 1),
        "sm_pct": round(weighted("sm", ks), 1),
        "memory_pct": round(weighted("mem", ks), 1),
        "occupancy_pct": round(weighted("occupancy", ks), 1),
        "tensor_pct": round(weighted("tensor", ks), 1),
    } for name, ks in top[:12]],
}, indent=1))
