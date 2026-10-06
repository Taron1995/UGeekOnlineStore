"""Дообучение RF-DETR (PyTorch) на разметке из CVAT: person, phone, workpiece.

1) В CVAT: Export task dataset -> "COCO 1.0" (+ images).
2) python training/split_coco.py export/annotations/instances_default.json export/images data/ds
3) python training/train.py --dataset data/ds --out runs/rfdetr_small
Результат: runs/.../checkpoint_best_total.pth -> detector.weights в config.yaml.
"""
import argparse

from rfdetr import RFDETRMedium, RFDETRSmall

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--out", default="runs/rfdetr")
ap.add_argument("--size", choices=["small", "medium"], default="small")
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-4)
a = ap.parse_args()

model = (RFDETRSmall if a.size == "small" else RFDETRMedium)()
model.train(dataset_dir=a.dataset, epochs=a.epochs, batch_size=a.batch, grad_accum_steps=16 // a.batch,
            lr=a.lr, output_dir=a.out)
