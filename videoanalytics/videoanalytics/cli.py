"""python -m videoanalytics scan | snap | run | report | models | frames | split-coco | train | probe  -  see README.md"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

from . import pipeline
from .analyze import analyze, ensure_model, run_analysis
from .config import DEFAULTS, normalize_camera, read_config
from .paths import models_dir
from .report import write_reports
from .sources import discover
from .storage import merge_shards
from .tools import extract_frames, fetch_models, probe_activity, snapshot, split_coco
from .video import active_intervals, fmt_ts


def parse_hours(s: str):
    return [[int(a), int(b)] for a, b in (r.split("-") for r in s.split(","))]


def load(a):
    cfg = read_config(a.config)
    opts = {**DEFAULTS, **(cfg.get("options") or {})}
    for k in ("fps", "conf"):
        if getattr(a, k, None) is not None:
            opts[k] = getattr(a, k)
    if getattr(a, "hours", None):
        opts["active_hours"] = parse_hours(a.hours)
    if getattr(a, "no_gate", False):
        opts["gate_motion"] = -1.0
    if getattr(a, "no_clips", False):
        opts["clips"] = False
    if getattr(a, "hw_decode", False):
        opts["hw_decode"] = True
    return cfg, opts, {k: normalize_camera(v) for k, v in cfg["cameras"].items()}


def sources(a, cams):
    srcs, issues = discover(a.videos, cams, a.time_regex)
    if getattr(a, "camera", None):
        srcs = [s for s in srcs if s["camera"] in a.camera]
    return srcs, issues


def print_issues(issues):
    if issues["unreadable"]:
        print(f"! {len(issues['unreadable'])} file(s) cannot be opened/decoded and are skipped, e.g. {issues['unreadable'][:3]} "
              f"(corrupt or partial? non-ASCII path on Windows? unsupported codec?)")
    if issues["unmatched"]:
        print(f"! {len(issues['unmatched'])} video file(s) match no camera in the config, e.g. {issues['unmatched'][:3]}")
    if issues["no_timestamp"]:
        print(f"! {len(issues['no_timestamp'])} file(s) have no date/time in their path, file modification time was used, "
              f"e.g. {issues['no_timestamp'][:3]}  (use --time-regex to fix)")


def features(cam: dict) -> str:
    f = []
    if cam.get("zones"):
        f.append(f"{len(cam['zones'])} zone(s)")
    if cam.get("lines"):
        f.append("lines")
    f += [f"machine:{m['name']}({m['source']})" for m in cam["machines"]]
    f += [k for k in ("gathering", "inactivity", "workpieces") if k in cam]
    return ", ".join(f) or "people only"


def cmd_scan(a):
    cfg, opts, cams = load(a)
    srcs, issues = sources(a, cams)
    per = defaultdict(lambda: [0, 0.0, 1e18, 0.0, 0.0])
    for s in srcs:
        p = per[s["camera"]]
        p[0] += 1
        p[1] += s["duration"]
        p[2], p[3] = min(p[2], s["start_ts"]), max(p[3], s["start_ts"] + s["duration"])
        p[4] += sum(y - x for x, y in active_intervals(s["start_ts"], s["duration"], opts["active_hours"]))
    print(f"{'camera':10} {'files':>6} {'video h':>9} {'to process h':>13}  period / features")
    for cam in cams:
        if cam in per:
            n, d, t0, t1, act = per[cam]
            print(f"{cam:10} {n:6d} {d / 3600:9.1f} {act / 3600:13.1f}  {fmt_ts(t0, '%Y-%m-%d %H:%M')} -> {fmt_ts(t1, '%Y-%m-%d %H:%M')}"
                  f"\n{'':44}{features(cams[cam])}")
        else:
            print(f"{cam:10} {'-':>6}  (no files found)")
    print_issues(issues)
    if a.model and srcs:
        from .detector import PersonDetector
        det = PersonDetector(a.model, conf=opts["conf"], threads=1)
        s = srcs[len(srcs) // 2]
        bench_opts = {**opts, "gate_motion": -1.0, "clips": False}
        pose, wp = pipeline._extras(cams[s["camera"]])
        _, _, meta = pipeline.process_video(s, cams[s["camera"]], bench_opts, det.detect, None, limit_s=a.bench_seconds, pose=pose, workpiece_detect=wp)
        cost = meta["elapsed_s"] / max(1e-6, meta["processed_s"])
        total = sum(p[4] for p in per.values())
        w = a.workers or os.cpu_count() or 1
        print(f"\nBenchmark on {s['rel']}: {meta['processed_s']:.0f} s of video in {meta['elapsed_s']:.1f} s "
              f"= {cost:.2f} CPU-s per video-s (decode + detector every frame, no motion gate, 1 thread, all analyzers of that camera)")
        print(f"Upper-bound estimate for {total / 3600:.0f} h of video with {w} worker(s): ~{total * cost / w / 3600:.1f} h wall-clock "
              f"(cameras with workpiece counting / pose are slower than the one measured if it had none; the motion gate usually brings this down)")


def cmd_run(a):
    cfg, opts, cams = load(a)
    srcs, issues = sources(a, cams)
    print_issues(issues)
    run_analysis(cams, opts, srcs, Path(a.out), ensure_model(a.model), a.workers, limit_files=a.limit_files, chunk_s=a.chunk_min * 60.0)
    if not a.no_report:
        cmd_report(a)


def cmd_analyze(a):
    analyze(a.paths, out=a.out, config=a.config, model=a.model, workers=a.workers, open_report=not a.no_open, auto=not a.no_auto,
            rediscover=a.rediscover, learn_minutes=a.learn_minutes, chunk_min=a.chunk_min, hours=parse_hours(a.hours) if a.hours else None,
            fps=a.fps, conf=a.conf, clips=not a.no_clips, hw_decode=a.hw_decode, no_gate=a.no_gate, time_regex=a.time_regex)


def cmd_gui(a):
    from .gui import main as gui_main
    gui_main(a.paths)


def cmd_report(a):
    out = Path(a.out)
    merge_shards(out)
    cfg = read_config(a.config) if getattr(a, "config", None) else None
    p = write_reports(out, a.idle_min * 60, cfg)
    print(f"Report: {p}  (+ hourly.csv, machines_hourly.csv, stations_daily.csv, absences.csv, idle_periods.csv, events.csv, analytics.db)")
    if cfg is None:
        print("note: pass --config to include the work schedule (absence outside breaks) and zone kinds")


def cmd_snap(a):
    cfg, opts, cams = load(a)
    srcs, issues = sources(a, cams)
    print_issues(issues)
    seen = set()
    for s in srcs:
        if s["camera"] in seen:
            continue
        seen.add(s["camera"])
        p = Path(a.out) / "snap" / f"{s['camera']}.jpg"
        snapshot(s["path"], p, cams[s["camera"]], a.at)
        print(f"{p}  <- {s['rel']}")


def cmd_models(a):
    names = ["yolox_tiny.onnx"] + (["yolox_s.onnx"] if a.yolox_s else []) + (["pose_landmarker_lite.task"] if a.pose else []) \
        + (["pose_landmarker_full.task"] if a.pose_full else [])
    fetch_models(names, a.dest or str(models_dir()))
    print("RF-DETR weights are downloaded by the rfdetr package on first use (~/.roboflow/models); copy that folder for offline machines.")


def cmd_frames(a):
    cfg, opts, cams = load(a)
    srcs, issues = sources(a, cams)
    print_issues(issues)
    n = extract_frames(srcs, a.out, a.count, parse_hours(a.hours) if a.hours else opts["active_hours"], a.seed)
    print(f"{n} frame(s) saved to {a.out}  (annotate them in CVAT / Label Studio / Roboflow, export COCO, then: split-coco)")


def cmd_split(a):
    print(split_coco(a.images, a.annotations, a.out, a.valid, a.test, a.seed), f"-> {a.out}")


def cmd_train(a):
    try:
        import rfdetr
    except ImportError:
        raise SystemExit('training needs: pip install "rfdetr[train]"  (PyTorch; a GPU is strongly recommended)')
    from .workpieces import SIZES
    model = getattr(rfdetr, SIZES[a.model])(**({"device": a.device} if a.device else {}))
    model.train(dataset_dir=a.dataset, epochs=a.epochs, batch_size=a.batch_size, grad_accum_steps=a.grad_accum, lr=a.lr,
                output_dir=a.out, num_workers=a.workers)
    out = Path(a.out)
    print(f"checkpoints written: {sorted(c.name for c in out.glob('*.pth')) + sorted(c.name for c in out.glob('*.ckpt'))}")
    cands = [out / n for n in ("checkpoint_best_total.pth", "last_ema.pth") if (out / n).exists()]
    best, best_f1 = None, -1.0
    try:
        from .workpieces import RFDETRDetector, evaluate_detector
        print(f"\nchecking the candidates on your validation images (confidence >= 0.5, IoU >= 0.5):")
        for c in cands:
            det = RFDETRDetector(str(c), threshold=0.5, device=a.device)
            r = evaluate_detector(det.detect, a.dataset)
            print(f"  {c.name:28} recall {100 * r['recall']:3.0f}%  precision {100 * r['precision']:3.0f}%  ({r['parts']} annotated parts in {r['images']} images)")
            if r["f1"] > best_f1 + 1e-9:
                best, best_f1 = c, r["f1"]
    except Exception as e:  # noqa: BLE001  the check is a safety net, never a reason to lose a finished training
        print(f"(checkpoint check skipped: {e!r})")
        best = cands[0] if cands else None
    if best is not None:
        print(f"\nrecommended: {best.name}  ->  cameras.yaml:  workpieces: {{weights: {best}, classes: [<your class>], line: ...}}")
        if best.name != "checkpoint_best_total.pth":
            print("note: the 'best' checkpoint is chosen by validation mAP, which can saturate early on a small dataset and pick a barely trained epoch")


def cmd_probe(a):
    from .detector import PersonDetector
    cfg, opts, cams = load(a)
    srcs, _ = sources(a, cams)
    srcs = [s for s in srcs if s["camera"] == a.camera[0]] if a.camera else srcs
    if not srcs:
        raise SystemExit("no matching video")
    pose = None
    if a.method == "pose":
        from .inactivity import PoseBackend
        pose = PoseBackend(a.pose_model)
    probe_activity(srcs[0], cams[srcs[0]["camera"]], PersonDetector(a.model, conf=opts["conf"]).detect, a.method, pose, a.start, a.seconds)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="videoanalytics", description="Offline video analytics for CCTV footage")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, videos=True):
        p.add_argument("--config", required=True)
        if videos:
            p.add_argument("--videos", required=True, help="root folder, searched recursively")
            p.add_argument("--time-regex", help="regex with 6 groups (Y,m,d,H,M,S) if file names are unusual")
            p.add_argument("--camera", nargs="*", help="only these cameras")
        p.add_argument("--out", default="out")

    def tuning(p):
        p.add_argument("--fps", type=float, help="processed frames per second (default 3)")
        p.add_argument("--conf", type=float, help="person confidence threshold")
        p.add_argument("--hours", help="only process these hours of the day, e.g. 8-20 or 6-14,14-22")
        p.add_argument("--no-gate", action="store_true", help="run the detector on every sampled frame")
        p.add_argument("--no-clips", action="store_true")
        p.add_argument("--hw-decode", action="store_true", help="try hardware video decoding")

    p = sub.add_parser("scan", help="list files per camera, check names/timestamps, estimate run time")
    common(p), tuning(p)
    p.add_argument("--model")
    p.add_argument("--workers", type=int)
    p.add_argument("--bench-seconds", type=float, default=60)
    p.set_defaults(f=cmd_scan)

    p = sub.add_parser("run", help="process all videos (resumable) and build the report")
    common(p), tuning(p)
    p.add_argument("--model", help="person detector .onnx (default: models/yolox_tiny.onnx, downloaded on first use)")
    p.add_argument("--workers", type=int)
    p.add_argument("--chunk-min", type=float, default=0, help="cut recordings longer than this into chunks for parallel workers (0 = off)")
    p.add_argument("--limit-files", type=int, help="trial run on the first N files")
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--idle-min", type=float, default=5, help="minimum idle period for the report, minutes")
    p.set_defaults(f=cmd_run)

    p = sub.add_parser("analyze", help="ONE STEP: give a video / folder, get the report (learns the scene, analyses, opens the page)")
    p.add_argument("paths", nargs="+", help="video file(s) and/or folder(s); drag them onto analyze_video.bat")
    p.add_argument("--out", help="results folder (default: <video>_analysis next to a single file, <folder>/_analysis otherwise)")
    p.add_argument("--config", help="use this cameras.yaml instead of learning the scene")
    p.add_argument("--model")
    p.add_argument("--workers", type=int)
    p.add_argument("--no-open", action="store_true", help="do not open the report in the browser")
    p.add_argument("--no-auto", action="store_true", help="skip scene learning: people counting only")
    p.add_argument("--rediscover", action="store_true", help="learn the scene again even if <out>/cameras.auto.yaml exists")
    p.add_argument("--learn-minutes", type=float, default=6, help="footage sampled per camera for scene learning")
    p.add_argument("--chunk-min", type=float, default=20, help="cut recordings longer than this into chunks so all cores are used (0 = off)")
    p.add_argument("--time-regex", help="regex with 6 groups (Y,m,d,H,M,S) if file names are unusual")
    tuning(p)
    p.set_defaults(f=cmd_analyze)

    p = sub.add_parser("gui", help="a window: choose videos / a folder, press Start, read the report (no command line needed)")
    p.add_argument("paths", nargs="*", help="videos / folders to put in the list at once")
    p.set_defaults(f=cmd_gui)

    p = sub.add_parser("report", help="rebuild the report from finished files")
    p.add_argument("--out", default="out")
    p.add_argument("--config", help="adds work schedule / zone kinds (absence outside breaks)")
    p.add_argument("--idle-min", type=float, default=5)
    p.set_defaults(f=cmd_report)

    p = sub.add_parser("snap", help="one snapshot per camera with a coordinate grid, the configured zones and lamp readout")
    common(p)
    p.add_argument("--at", type=float, default=30, help="second of the file to grab")
    p.set_defaults(f=cmd_snap)

    p = sub.add_parser("models", help="download the ONNX / pose models into models/")
    p.add_argument("--dest", help="default: the project's models/ folder (or $VIDAN_MODELS)")
    p.add_argument("--pose", action="store_true", help="MediaPipe pose_landmarker_lite.task (inactivity.method: pose)")
    p.add_argument("--pose-full", action="store_true")
    p.add_argument("--yolox-s", action="store_true", help="larger, more accurate person detector (3.5x slower)")
    p.set_defaults(f=cmd_models)

    p = sub.add_parser("frames", help="extract frames spread over all footage for annotating (workpiece detector training)")
    common(p)
    p.add_argument("--count", type=int, default=300)
    p.add_argument("--hours")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(f=cmd_frames, out="dataset/raw")

    p = sub.add_parser("split-coco", help="split one COCO export into train/valid/test folders for RF-DETR")
    p.add_argument("--images", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--out", default="dataset")
    p.add_argument("--valid", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(f=cmd_split)

    p = sub.add_parser("train", help="fine-tune RF-DETR on your annotated dataset (needs rfdetr[train]; use a GPU)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", choices=["nano", "small", "medium", "large"], default="nano")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device")
    p.add_argument("--out", default="output")
    p.set_defaults(f=cmd_train)

    p = sub.add_parser("probe", help="print the activity the inactivity method measures for each person (to choose motion_thr)")
    common(p)
    p.add_argument("--model", required=True, help="YOLOX onnx for the person detector")
    p.add_argument("--method", choices=["motion", "pose"], default="motion")
    p.add_argument("--pose-model")
    p.add_argument("--start", type=float, default=60, help="seconds into the first file")
    p.add_argument("--seconds", type=float, default=120)
    p.set_defaults(f=cmd_probe)

    for stream in (sys.stdout, sys.stderr):  # a Windows console may not encode Armenian / Russian paths: never crash on printing
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    a = ap.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
