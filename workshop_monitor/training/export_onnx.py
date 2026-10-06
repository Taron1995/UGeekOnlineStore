"""Экспорт обученной модели в ONNX для ONNX Runtime (GPU) / OpenVINO (Intel CPU).
python training/export_onnx.py --weights runs/rfdetr_small/checkpoint_best_total.pth --out models"""
import argparse

from rfdetr import RFDETRMedium, RFDETRSmall

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--size", choices=["small", "medium"], default="small")
ap.add_argument("--out", default="models")
a = ap.parse_args()

model = (RFDETRSmall if a.size == "small" else RFDETRMedium)(pretrain_weights=a.weights)
model.export(output_dir=a.out)
print("готово; укажите detector.onnx_path и backend: onnx в config.yaml")
