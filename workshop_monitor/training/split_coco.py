"""Делит COCO-экспорт CVAT на train/valid в формате, который ждёт RF-DETR.
python training/split_coco.py instances.json images_dir out_dir [--valid 0.15]"""
import argparse
import json
import random
import shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("coco_json")
ap.add_argument("images")
ap.add_argument("out")
ap.add_argument("--valid", type=float, default=0.15)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

coco = json.loads(Path(a.coco_json).read_text(encoding="utf-8"))
imgs = coco["images"][:]
random.Random(a.seed).shuffle(imgs)
n_val = max(1, int(len(imgs) * a.valid))
parts = {"valid": imgs[:n_val], "train": imgs[n_val:]}

for name, items in parts.items():
    d = Path(a.out) / name
    d.mkdir(parents=True, exist_ok=True)
    ids = {i["id"] for i in items}
    for im in items:
        src = Path(a.images) / Path(im["file_name"]).name
        shutil.copy(src, d / Path(im["file_name"]).name)
        im["file_name"] = Path(im["file_name"]).name
    out = {**coco, "images": items, "annotations": [x for x in coco["annotations"] if x["image_id"] in ids]}
    (d / "_annotations.coco.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(name, len(items))
