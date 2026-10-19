"""Setup helpers: snapshots with the configured zones, model download, frames for annotation, COCO split, activity probe."""
from __future__ import annotations

import json
import random
import shutil
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .config import normalize_camera
from .lamp import LampReader
from .video import active_intervals, fmt_ts, open_capture
from .zones import Zone

MODEL_URLS = {
    "yolox_tiny.onnx": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx",
    "yolox_s.onnx": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
}


def fetch_models(names, dest: str = "models"):
    """Downloads into dest (skips files that exist). The RF-DETR weights are fetched by the rfdetr package itself on first use."""
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    for n in names:
        target = out / n
        if target.exists() and target.stat().st_size > 1_000_000:
            print(f"[have] {target}")
            continue
        tmp = target.with_name(n + ".part")
        print(f"[get ] {n} ...", flush=True)
        urllib.request.urlretrieve(MODEL_URLS[n], tmp)
        if tmp.stat().st_size < 1_000_000:
            tmp.unlink()
            raise RuntimeError(f"download of {n} looks wrong (file too small)")
        tmp.replace(target)
        print(f"[ok  ] {target} ({target.stat().st_size / 1e6:.1f} MB)")


def read_frame(video: str, at_s: float):
    cap = open_capture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, at_s * 1000)
    ok, f = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_MSEC, 0)
        ok, f = cap.read()
    cap.release()
    if not ok:
        raise OSError(f"cannot read {video}")
    return f


def snapshot(video: str, out_path: Path, cam_cfg: dict, at_s: float = 30.0):
    """One frame with a 10 % coordinate grid, the configured zones / lines / machines, and for lamp machines the colour the
    reader sees right now with its per-colour scores (use it to tune sat_min / val_min / min_frac)."""
    cam_cfg = normalize_camera(cam_cfg)
    f = read_frame(video, at_s)
    h, w = f.shape[:2]
    raw = f.copy()
    for i in range(1, 10):
        x, y = int(w * i / 10), int(h * i / 10)
        cv2.line(f, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.line(f, (0, y), (w, y), (255, 255, 255), 1)
        cv2.putText(f, f"{i / 10:.1f}", (x + 3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(f, f"{i / 10:.1f}", (3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    def poly(pts, color, label):
        """label: text or list of lines; drawn above the polygon and kept inside the image."""
        p = (np.array(pts) * [w, h]).astype(np.int32)
        cv2.polylines(f, [p], True, color, 2)
        lines = [label] if isinstance(label, str) else label
        y0 = int(p[:, 1].min())
        for i, text in enumerate(reversed(lines)):  # last line closest to the polygon
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            x = int(min(max(3, p[:, 0].min()), w - tw - 4))
            y = int(max(th + 4, y0 - 6 - (th + base + 6) * i))
            cv2.rectangle(f, (x - 2, y - th - 2), (x + tw + 2, y + base + 1), (0, 0, 0), -1)  # solid box: readable on any frame
            cv2.putText(f, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def arrow(ln, color):
        a, b = (int(ln["p1"][0] * w), int(ln["p1"][1] * h)), (int(ln["p2"][0] * w), int(ln["p2"][1] * h))
        cv2.arrowedLine(f, a, b, color, 2, tipLength=0.05)
        cv2.putText(f, ln.get("name", "line"), a, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for a in cam_cfg.get("areas") or []:  # where people are followed (blue) / ignored (red)
        watch = a.get("mode", "watch") == "watch"
        poly(a["polygon"], (255, 140, 0) if watch else (60, 60, 255), f"{'follow' if watch else 'ignore'}: {a['name']}")
    for z in cam_cfg.get("zones") or []:
        poly(z["polygon"], (0, 200, 0) if z.get("kind", "workstation") == "workstation" else (200, 200, 0), f"{z['name']} ({z.get('kind', 'workstation')})")
    for m in cam_cfg["machines"]:
        label = [m["name"]]
        if m["source"] == "lamp":
            s = LampReader(Zone(m["name"], m["roi"]), m.get("hues"), m.get("sat_min", 90), m.get("val_min", 140), m.get("min_frac", 0.08)).scores(raw)
            best = max(s, key=s.get)
            label = [f"{m['name']}: reads {best if s[best] >= m.get('min_frac', 0.08) else 'off'}",
                     "lit share  " + "  ".join(f"{c}={v:.2f}" for c, v in s.items())]
        poly(m["roi"], (0, 140, 255), label)
    for ln in cam_cfg.get("lines") or []:
        arrow(ln, (255, 0, 255))
    if cam_cfg.get("workpieces"):
        arrow(cam_cfg["workpieces"]["line"], (255, 128, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), f)


def extract_frames(srcs, out_dir: str, count: int = 300, hours=None, seed: int = 0, min_diff: float = 4.0, min_sharp: float = 30.0) -> int:
    """Frames for annotation: one random frame per equal slice of the footage (so all days / hours are represented), skipping
    blurry frames and near-duplicates of the previous frame."""
    rng = random.Random(seed)
    spans = [(s, a, b) for s in srcs for a, b in active_intervals(s["start_ts"], s["duration"], hours)]
    total = sum(b - a for _, a, b in spans)
    if total <= 0 or count <= 0:
        return 0
    targets = sorted((i + rng.random()) * total / count for i in range(count))
    picks, acc, k = [], 0.0, 0
    for s, a, b in spans:  # map each target (seconds into the concatenated footage) to (source, file-relative time)
        while k < len(targets) and targets[k] < acc + (b - a):
            picks.append((s, a + targets[k] - acc))
            k += 1
        acc += b - a
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved, last_small, cur, cap = 0, None, None, None
    for s, t in picks:
        if cur is not s["path"]:
            if cap:
                cap.release()
            cap, cur = open_capture(s["path"]), s["path"]
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        if cv2.Laplacian(g, cv2.CV_64F).var() < min_sharp or (last_small is not None and float(np.abs(small - last_small).mean()) < min_diff):
            continue
        last_small = small
        cv2.imwrite(str(out / f"{s['camera']}_{fmt_ts(s['start_ts'] + t)}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
    if cap:
        cap.release()
    return saved


def split_coco(images_dir: str, ann_json: str, out_dir: str, valid: float = 0.15, test: float = 0.05, seed: int = 0) -> dict:
    """One COCO export (CVAT, Label Studio, ...) -> the train/valid/test folders RF-DETR expects (each with _annotations.coco.json)."""
    data = json.loads(Path(ann_json).read_text(encoding="utf-8"))
    imgs = list(data["images"])
    random.Random(seed).shuffle(imgs)
    n = len(imgs)
    n_test = int(round(n * test)) if n >= 20 else 0
    n_valid = max(1, int(round(n * valid))) if n >= 3 else 0
    splits = {"test": imgs[:n_test], "valid": imgs[n_test:n_test + n_valid], "train": imgs[n_test + n_valid:]}
    by_img: dict = {}
    for a in data.get("annotations", []):
        by_img.setdefault(a["image_id"], []).append(a)
    src_dir, out = Path(images_dir), Path(out_dir)
    for name, items in splits.items():
        if not items:
            continue
        (out / name).mkdir(parents=True, exist_ok=True)
        for im in items:
            p = src_dir / im["file_name"]
            if not p.exists():
                p = src_dir / Path(im["file_name"]).name
            if not p.exists():
                raise FileNotFoundError(f"image {im['file_name']} of the annotation file not found in {src_dir}")
            shutil.copy2(p, out / name / Path(im["file_name"]).name)
            im["file_name"] = Path(im["file_name"]).name
        ann = [a for im in items for a in by_img.get(im["id"], [])]
        (out / name / "_annotations.coco.json").write_text(
            json.dumps({**{k: v for k, v in data.items() if k not in ("images", "annotations")}, "images": items, "annotations": ann}), encoding="utf-8")
    return {k: len(v) for k, v in splits.items()}


def probe_activity(src: dict, cam_cfg: dict, detect, method: str, pose, start_s: float, seconds: float, thr: float | None = None):
    """Prints, per person track, how much 'activity' the inactivity method measures. Use it on a stretch where someone sits still
    and on one where they work, then set inactivity.motion_thr between the two."""
    from .analyzers import Ctx
    from .inactivity import InactivityAnalyzer, _Person
    from .tracker import Tracker
    cam = normalize_camera(cam_cfg)
    az = InactivityAnalyzer(dict(method=method), [], pose)
    thr = az.thr if thr is None else thr
    cap = open_capture(src["path"])
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps / 1.0))
    ctx = Ctx(src, cam, {}, None)
    tracker, people, acts, posts, i = Tracker(0.2, 5, 2), {}, {}, {}, 0
    while i < seconds * fps:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if (i - 1) % step:
            continue
        ctx.frame, ctx.h, ctx.w, ctx.t = frame, *frame.shape[:2], start_s + i / fps
        for tr in tracker.update(detect(frame)):
            p = people.setdefault(tr.id, _Person(1, 1))
            a, post = az._measure(ctx, tr, p)
            if a is not None:
                acts.setdefault(tr.id, []).append(a)
            if post:
                posts.setdefault(tr.id, Counter())[post] += 1
    cap.release()
    print(f"method={method}, threshold in use={thr} (a person counts as still while activity < threshold)")
    print(f"{'track':>6} {'samples':>8} {'median':>8} {'p90':>8} {'max':>8} {'% below thr':>12}  posture")
    for tid, v in sorted(acts.items()):
        if len(v) >= 3:
            v = np.array(v)
            print(f"{tid:>6} {len(v):>8} {np.median(v):>8.3f} {np.percentile(v, 90):>8.3f} {v.max():>8.3f} {100 * (v < thr).mean():>11.0f}%  "
                  f"{posts[tid].most_common(1)[0][0] if tid in posts else '-'}")
    if not acts:
        print("no person tracks with enough samples in that window")
