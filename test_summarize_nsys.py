# python3 -m unittest test_summarize_nsys
import sqlite3
import unittest

import summarize_nsys as sn


def report(kernels, copies, api):
    db = sqlite3.connect(":memory:")
    db.executescript("""
        create table StringIds (id integer, value text);
        create table CUPTI_ACTIVITY_KIND_KERNEL (start integer, end integer, shortName integer);
        create table CUPTI_ACTIVITY_KIND_MEMCPY (start integer, end integer, bytes integer, copyKind integer, srcKind integer, dstKind integer);
        create table CUPTI_ACTIVITY_KIND_RUNTIME (start integer, end integer, nameId integer);
    """)
    db.executemany("insert into StringIds values (?, ?)", [(1, "void conv_kernel<float>(int)"), (2, "relu"), (3, "cudaLaunchKernel"), (4, "cudaStreamSynchronize")])
    db.executemany("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?)", kernels)
    db.executemany("insert into CUPTI_ACTIVITY_KIND_MEMCPY values (?, ?, ?, ?, ?, ?)", copies)
    db.executemany("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?, ?, ?)", api)
    return db


class IntervalTest(unittest.TestCase):
    def test_union_and_intersect(self):
        u = sn.union([(5, 8), (0, 2), (1, 3), (8, 9)])
        self.assertEqual(u, [[0, 3], [5, 9]])
        self.assertEqual(sn.length(u), 7)
        self.assertEqual(sn.intersect(u, sn.union([(2, 6)])), 2)


class SummaryTest(unittest.TestCase):
    # a 1 ms window: kernels busy 0-400 and 600-800 µs, an H2D copy at 300-500 µs
    # half hidden behind the first kernel, and a D2H copy at 800-900 µs
    def setUp(self):
        us = 1000
        self.s = sn.summarize(report(
            kernels=[(0, 400 * us, 1), (600 * us, 800 * us, 2)],
            copies=[(300 * us, 500 * us, 4_000_000, sn.COPY_H2D, sn.PINNED, 2), (800 * us, 900 * us, 1_000_000, sn.COPY_D2H, 2, 0)],
            api=[(0, 10 * us, 3), (590 * us, 600 * us, 3), (900 * us, 1000 * us, 4)],
        ), "m.onnx")

    def test_busy_and_overlap(self):
        self.assertEqual(self.s["gpu_busy_pct"], 80.0)  # 0-500 and 600-900
        self.assertEqual(self.s["compute_pct"], 60.0)
        self.assertEqual(self.s["copy_pct"], 30.0)
        self.assertEqual(self.s["copy_hidden_pct"], 33.3)  # 100 of 300 µs of copies overlap kernels
        self.assertEqual(self.s["idle_gaps"], 1)
        self.assertEqual(self.s["idle_gap_p50_us"], 100.0)

    def test_copies(self):
        h2d, d2h = self.s["h2d"], self.s["d2h"]
        self.assertEqual(h2d["gb_per_s"], 20.0)  # 4 MB in 200 µs
        self.assertEqual(h2d["pinned_pct"], 100.0)
        self.assertEqual(d2h["pinned_pct"], 0.0)
        self.assertEqual(d2h["gb_per_s"], 10.0)

    def test_kernels_and_api(self):
        self.assertEqual([k["name"] for k in self.s["top_kernels"]], ["conv_kernel", "relu"])
        self.assertEqual(self.s["top_kernels"][0]["share_pct"], 66.7)
        self.assertEqual(self.s["launch_avg_us"], 10.0)
        self.assertEqual(self.s["threads_blocked_on_gpu"], 0.1)

    def test_missing_kernels(self):
        db = sqlite3.connect(":memory:")
        db.execute("create table StringIds (id integer, value text)")
        with self.assertRaises(SystemExit):
            sn.summarize(db, "m.onnx")


if __name__ == "__main__":
    unittest.main()
