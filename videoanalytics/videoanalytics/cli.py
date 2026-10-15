"""vidan scan | run | report | snap  -  see README.md"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from collections import defaultdict
from pathlib import Path

import yaml

from . import pipeline
from .pipeline import DEFAULTS, active_intervals, fmt_ts
from .sources import discover
from .storage import clean_stale, merge_shards, shard_path
from .report import write_reports
from .tools import snapshot


def parse_hours(s: str):
    return [[int(a), int(b)] for a, b in (r.split("-") for r in s.split(","))]


def load(a):
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
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
    cams = cfg["cameras"]
    return cfg, opts, cams


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
    print(f"{'camera':10} {'files':>6} {'video h':>9} {'to process h':>13}  period")
    for cam in cams:
        if cam in per:
            n, d, t0, t1, act = per[cam]
            print(f"{cam:10} {n:6d} {d / 3600:9.1f} {act / 3600:13.1f}  {fmt_ts(t0, '%Y-%m-%d %H:%M')} -> {fmt_ts(t1, '%Y-%m-%d %H:%M')}")
        else:
            print(f"{cam:10} {'-':>6}  (no files found)")
    print_issues(issues)
    if a.model and srcs:
        from .detector import PersonDetector
        det = PersonDetector(a.model, conf=opts["conf"], threads=1)
        s = srcs[len(srcs) // 2]
        bench_opts = {**opts, "gate_motion": -1.0, "clips": False}
        _, _, meta = pipeline.process_video(s, cams[s["camera"]], bench_opts, det.detect, None, limit_s=a.bench_seconds)
        cost = meta["elapsed_s"] / max(1e-6, meta["processed_s"])
        total = sum(p[4] for p in per.values())
        w = a.workers or os.cpu_count() or 1
        print(f"\nBenchmark on {s['rel']}: {meta['processed_s']:.0f} s of video in {meta['elapsed_s']:.1f} s "
              f"= {cost:.2f} CPU-s per video-s (decode + detector every frame, no motion gate, 1 thread)")
        print(f"Upper-bound estimate for {total / 3600:.0f} h of video with {w} worker(s): ~{total * cost / w / 3600:.1f} h wall-clock "
              f"(the motion gate usually brings this down; real speed-up depends on how empty the scenes are)")


def cmd_run(a):
    cfg, opts, cams = load(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clean_stale(out)
    srcs, issues = sources(a, cams)
    print_issues(issues)
    tasks, done = [], 0
    for s in srcs:
        if not active_intervals(s["start_ts"], s["duration"], opts["active_hours"]):
            continue
        sp = shard_path(out, s["camera"], s["rel"])
        if sp.exists():
            done += 1
            continue
        tasks.append(dict(src=s, cam=cams[s["camera"]], opts=opts, out=str(out), shard=str(sp)))
    if a.limit_files:
        tasks = tasks[: a.limit_files]
    w = min(a.workers or os.cpu_count() or 1, max(1, len(tasks)))
    print(f"{len(tasks)} file(s) to process ({done} already done, skipped), {w} worker(s), {opts['fps']} fps")
    errors, t0 = [], time.time()
    if tasks:
        with mp.get_context("spawn").Pool(w, pipeline.init_worker, (a.model, opts["conf"])) as pool:
            for i, r in enumerate(pool.imap_unordered(pipeline.run_task, tasks), 1):
                if r["ok"]:
                    print(f"[{i}/{len(tasks)}] {r['source']}: {r['processed_s']:.0f}s video, {r['det_frames']}/{r['frames']} "
                          f"frames detected, {r['elapsed_s']:.0f}s", flush=True)
                else:
                    errors.append(r)
                    print(f"[{i}/{len(tasks)}] ERROR {r['source']}: {r['error']}", flush=True)
    print(f"Processing finished in {(time.time() - t0) / 60:.1f} min, {len(errors)} error(s)")
    if errors:
        (out / "errors.log").write_text("\n".join(f"{e['source']}\t{e['error']}" for e in errors), encoding="utf-8")
        print(f"errors listed in {out / 'errors.log'}; run again to retry only the failed files")
    if not a.no_report:
        cmd_report(a)


def cmd_report(a):
    out = Path(a.out)
    merge_shards(out)
    p = write_reports(out, a.idle_min * 60)
    print(f"Report: {p}  (+ hourly.csv, events.csv, idle_periods.csv, analytics.db)")


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


def main():
    ap = argparse.ArgumentParser(prog="vidan", description="Offline video analytics (people + machine activity)")
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
    p.add_argument("--model", required=True)
    p.add_argument("--workers", type=int)
    p.add_argument("--limit-files", type=int, help="trial run on the first N files")
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--idle-min", type=float, default=5, help="minimum idle period for the report, minutes")
    p.set_defaults(f=cmd_run)

    p = sub.add_parser("report", help="rebuild the report from finished files")
    p.add_argument("--out", default="out")
    p.add_argument("--idle-min", type=float, default=5)
    p.set_defaults(f=cmd_report)

    p = sub.add_parser("snap", help="one snapshot per camera with a coordinate grid and the configured zones")
    common(p)
    p.add_argument("--at", type=float, default=30, help="second of the file to grab")
    p.set_defaults(f=cmd_snap)

    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
