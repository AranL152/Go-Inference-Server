# Score the running detector on COCO val2017 with the official COCO metric
#
# Start the server with the standard eval thresholds first
# ./demo.sh -conf 0.001 -iou 0.65
# python eval_coco.py ~/datasets/val2017 ~/datasets/annotations/instances_val2017.json
#
# Requires pip install pycocotools
import json
import os
import sys
import time
import urllib.request

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

images, ann_file = os.path.expanduser(sys.argv[1]), os.path.expanduser(sys.argv[2])
server = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8080"


def call(path, body=None):
    req = urllib.request.Request(server + path, data=json.dumps(body).encode() if body else None)
    return json.load(urllib.request.urlopen(req))


coco = COCO(ann_file)
cat_ids = sorted(coco.getCatIds())  # model class i is the i th COCO category

job = call("/api/jobs", {"path": images, "classes": []})
while (j := call(f"/api/jobs/{job['id']}")["job"])["status"] == "running":
    time.sleep(1)
print(f"{j['done']} images at {j['images_per_sec']:.0f} images per second")

dets, offset = [], 0
while True:
    page = call(f"/api/jobs/{job['id']}/results?offset={offset}&limit=500")
    for item in page["items"]:
        image_id = int(os.path.splitext(os.path.basename(item["path"]))[0])
        for d in item["detections"]:
            x1, y1, x2, y2 = d["box"]
            dets.append({"image_id": image_id, "category_id": cat_ids[d["class_index"]],
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "score": d["confidence"]})
    offset += len(page["items"])
    if offset >= page["total"]:
        break

ev = COCOeval(coco, coco.loadRes(dets), "bbox")
ev.evaluate()
ev.accumulate()
ev.summarize()
print(f"mAP 0.5:0.95 = {ev.stats[0] * 100:.1f}  mAP 0.5 = {ev.stats[1] * 100:.1f}")
