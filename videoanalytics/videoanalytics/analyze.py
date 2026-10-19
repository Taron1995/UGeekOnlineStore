"""One step: videos in -> finished report out.  models -> scene learning -> analysis -> report -> browser."""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
import types
import webbrowser
from collections import Counter
from pathlib import Path

from . import pipeline
from .autoconfig import build_camera_config, learn_scene
from .config import DEFAULTS, normalize_camera, read_config
from .diagnostics import doctor_text, inspect_samples, notes, result_text
from .paths import find_model, models_dir
from .report import _dur, write_reports
from .setupfile import camera_picture, setup_file, write_info, write_setup
from .sources import VIDEO_EXT, discover, discover_auto, parse_start_arg
from .storage import SCHEMA_VERSION, clean_stale, merge_shards, shard_glob, shard_meta, shard_path
from .tools import MODEL_URLS, fetch_models
from .video import active_intervals, fmt_ts

MODEL = "yolox_tiny.onnx"
FINGERPRINT_OPTS = ("fps", "conf", "min_hits", "iou_thr", "max_lost_s", "gate_motion", "keepalive_s", "clips", "max_clips_per_file",
                    "clip_pad_s", "active_hours")


# ---- inputs -------------------------------------------------------------------------------------------------------------
def resolve_inputs(paths):
    """Files and folders (dropped or typed) -> (common root folder, video files)."""
    files, roots = [], []
    for raw in paths:
        p = Path(str(raw).strip().strip('"')).expanduser()
        if not p.exists():
            raise SystemExit(f"not found: {p}")
        if p.is_dir():
            vids = [f for f in sorted(p.rglob("*")) if f.suffix.lower() in VIDEO_EXT and f.is_file()]
            if not vids:
                kinds = Counter(f.suffix.lower() or "(no extension)" for f in p.rglob("*") if f.is_file()).most_common(6)
                raise SystemExit(f"no video files in {p}" + (f"; the files there are: {', '.join(f'{e} ({n})' for e, n in kinds)}" if kinds else " (it is empty)")
                                 + "\nVideo formats read: " + " ".join(sorted(VIDEO_EXT)))
            files += vids
            roots.append(p.resolve())
        else:
            files.append(p)
            roots.append(p.resolve().parent)
    try:
        root = Path(os.path.commonpath([str(r) for r in roots]))
    except ValueError:  # different drives
        root = roots[0]
    return root, sorted({f.resolve() for f in files})


def default_out(root: Path, paths) -> Path:
    if len(paths) == 1 and Path(str(paths[0]).strip('"')).expanduser().is_file():
        f = Path(str(paths[0]).strip('"')).expanduser().resolve()
        return f.parent / f"{f.stem}_analysis"
    return root / "_analysis"


def ensure_model(path: str | None) -> str:
    if path:
        if not Path(path).exists():
            raise SystemExit(f"model not found: {path}")
        return str(path)
    found = find_model(MODEL)
    if found:
        return str(found)
    dest = models_dir()
    print(f"First run: downloading the person detector ({MODEL}, 20 MB) to {dest} ...", flush=True)
    try:
        fetch_models([MODEL], str(dest))
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"could not download the model ({e!r}).\n  No internet on this machine? Download\n  {MODEL_URLS[MODEL]}\n"
                         f"  once on another computer, put it into {dest} (or pass --model) and run again.")
    return str(dest / MODEL)


# ---- planning ---------------------------------------------------------------------------------------------------------
def split_segments(duration: float, chunk_s: float):
    """[None] for a normal file; [(a, b), ...] when a long recording is cut into chunks so that several workers share it."""
    if not chunk_s or duration <= 1.5 * chunk_s:
        return [None]
    n = max(2, round(duration / chunk_s))
    step = duration / n
    return [(i * step, duration if i == n - 1 else (i + 1) * step) for i in range(n)]


def fingerprint(cam: dict, opts: dict, chunk_s: float) -> str:
    """Identifies everything that changes a shard's content, so edited setups recompute instead of reusing stale results."""
    blob = json.dumps({"cam": {k: v for k, v in normalize_camera(cam).items() if k != "match"}, "opts": {k: opts.get(k) for k in FINGERPRINT_OPTS},
                       "chunk": chunk_s, "schema": SCHEMA_VERSION}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def plan_tasks(srcs, cams, opts, out: Path, chunk_s: float = 0.0):
    """-> (tasks still to run, finished shards reused, shards recomputed/dropped because the setup or the chunking changed)."""
    tasks, done, stale = [], 0, 0
    for s in srcs:
        if not active_intervals(s["start_ts"], s["duration"], opts["active_hours"]):
            continue
        fp = fingerprint(cams[s["camera"]], opts, chunk_s)
        # a file that does not say how long it is is read once from its start to its end: chunks of a guessed hour would read the same frames again
        wanted = {shard_path(out, s["camera"], s["rel"], seg): seg for seg in split_segments(s["duration"] if s.get("duration_known", True) else 0.0, chunk_s)}
        for old in shard_glob(out, s["camera"], s["rel"]):  # shards of another chunking would be counted twice
            if old not in wanted:
                old.unlink(missing_ok=True)
                stale += 1
        for sp, seg in wanted.items():
            if sp.exists() and shard_meta(sp).get("config_hash") == fp:
                done += 1
                continue
            stale += sp.exists()
            tasks.append(dict(src={**s, "seg": seg} if seg else s, cam=cams[s["camera"]], opts=opts, out=str(out), shard=str(sp), config_hash=fp))
    return tasks, done, stale


class Runner:
    """A pool of worker processes that each hold the person detector; or, given `detect`, everything in this process (tests)."""

    def __init__(self, model, conf, workers, detect=None):
        self.model, self.conf, self.workers, self.detect, self.pool = model, conf, workers, detect, None

    def __enter__(self):
        if self.detect is not None:
            pipeline._DET = types.SimpleNamespace(detect=self.detect)
        else:
            self.pool = mp.get_context("spawn").Pool(self.workers, pipeline.init_worker, (self.model, self.conf))
        return self

    def imap(self, fn, items):
        return map(fn, items) if self.pool is None else self.pool.imap_unordered(fn, items)

    def __exit__(self, *exc):
        if self.pool:
            self.pool.terminate()
            self.pool.join()


def _eta(done: int, total: int, t0: float) -> str:
    if done < 1 or done >= total:
        return ""
    left = (time.time() - t0) / done * (total - done)
    return f", about {left / 60:.0f} min left" if left >= 60 else f", about {left:.0f} s left"


def run_analysis(cams, opts, srcs, out: Path, model, workers, *, limit_files=None, chunk_s: float = 0.0, detect=None):
    """Processes every recording not yet done (resumable). Returns the list of errors."""
    out.mkdir(parents=True, exist_ok=True)
    clean_stale(out)
    tasks, done, stale = plan_tasks(srcs, cams, opts, out, chunk_s)
    if limit_files:
        tasks = tasks[:limit_files]
    w = min(workers or os.cpu_count() or 1, max(1, len(tasks)))
    note = f", {stale} recomputed because the setup changed" if stale else ""
    print(f"{len(tasks)} job(s) to process ({done} already done{note}), {w} worker(s), {opts['fps']} fps"
          + (f", long files cut into {chunk_s / 60:.0f} min chunks" if chunk_s else ""), flush=True)
    if w > 2 and any("workpieces" in t["cam"] for t in tasks):
        print("! workpiece counting runs RF-DETR on PyTorch (all cores per process, ~1 GB RAM each): for those cameras use "
              "--camera camN --workers 1 (or 2) and run the other cameras separately")
    errors, t0 = [], time.time()
    if tasks:
        with Runner(model, opts["conf"], w, detect) as rn:
            for i, r in enumerate(rn.imap(pipeline.run_task, tasks), 1):
                if r["ok"]:
                    print(f"[{i}/{len(tasks)}] {r['source']}: {r['processed_s']:.0f}s video, {r['det_frames']}/{r['frames']} frames detected, "
                          f"{r['elapsed_s']:.0f}s{_eta(i, len(tasks), t0)}", flush=True)
                else:
                    errors.append(r)
                    print(f"[{i}/{len(tasks)}] ERROR {r['source']}: {r['error']}", flush=True)
    print(f"Processing finished in {(time.time() - t0) / 60:.1f} min, {len(errors)} error(s)", flush=True)
    if errors:
        (out / "errors.log").write_text("\n".join(f"{e['source']}\t{e['error']}" for e in errors), encoding="utf-8")
        print(f"errors listed in {out / 'errors.log'}; run again to retry only the failed files")
    return errors


def _load_detector(model_path: str, conf: float):
    from .detector import PersonDetector
    return PersonDetector(model_path, conf=conf, threads=1).detect


def cameras_summary(srcs, hours=None) -> str:
    """One line per camera: how many files, how much video, which period. Shown before anything is computed, so wrong camera names
    (a file naming the program cannot tell apart) are noticed in the first second."""
    per: dict = {}
    for s in srcs:
        p = per.setdefault(s["camera"], [0, 0.0, 1e18, 0.0])
        p[0] += 1
        p[1] += sum(b - a for a, b in active_intervals(s["start_ts"], s["duration"], hours))
        p[2], p[3] = min(p[2], s["start_ts"]), max(p[3], s["start_ts"] + s["duration"])
    lines = [f"  {cam}: {n} file(s), {_dur(t)} of video, {fmt_ts(t0, '%Y-%m-%d %H:%M')} -> {fmt_ts(t1, '%Y-%m-%d %H:%M')}"
             for cam, (n, t, t0, t1) in sorted(per.items())[:12]]
    if len(per) > 12:
        lines.append(f"  ... and {len(per) - 12} more camera(s)")
    return "\n".join(lines)


def benchmark(srcs, cams, opts, make_detect, bench_s: float = 15.0):
    """CPU-seconds needed per second of video (decoder + detector on every sampled frame, one thread, no motion gate), measured on a short
    piece of one recording; None when it cannot be measured. make_detect() returns the detector function."""
    s = srcs[len(srcs) // 2]
    try:
        _, _, meta = pipeline.process_video({k: v for k, v in s.items() if k != "seg"}, cams[s["camera"]], {**opts, "gate_motion": -1.0, "clips": False},
                                            make_detect(), None, limit_s=bench_s)
    except Exception:  # noqa: BLE001 - a measurement must never stop the real work
        return None
    return meta["elapsed_s"] / meta["processed_s"] if meta["processed_s"] >= 3 else None


def estimate(srcs, cams, opts, make_detect, workers, bench_s: float = 15.0):
    """A line with the wall-clock upper bound for the whole job, from a short test of one recording. None when the job is short or the
    test could not run."""
    total = sum(b - a for s in srcs for a, b in active_intervals(s["start_ts"], s["duration"], opts["active_hours"]))
    if total < 2 * 3600:
        return None
    cost = benchmark(srcs, cams, opts, make_detect, bench_s)
    if cost is None:
        return None
    w = workers or os.cpu_count() or 1
    wall = total * cost / w
    return (f"Estimate: about {_dur(wall)} for {_dur(total)} of video on {w} worker(s) - an upper bound from a quick test (the motion gate usually makes it "
            f"faster; cameras with workpiece counting are slower). To shorten it: --hours 8-20 (working hours only), --fps 2, or do one camera / one day first.")


# ---- scene learning ---------------------------------------------------------------------------------------------------
def learn_task(task: dict) -> dict:
    """Worker side: learn one camera's setup. Never raises."""
    try:
        cfg, info, (psrc, at) = learn_scene(task["srcs"], pipeline._DET.detect, minutes=task["minutes"], hours=task["hours"], lamps=task.get("lamps", False))
        return dict(ok=True, camera=task["camera"], cfg=cfg, info=info, preview=(psrc["path"], at))
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, camera=task["camera"], error=repr(e))


def learn_cameras(srcs, match, opts, out: Path, model, workers, minutes: float, detect=None, lamps: bool = False):
    """Learns every camera; returns the full config dict (and writes cameras.auto.yaml, snapshots and setup/setup.json)."""
    by_cam: dict = {}
    for s in srcs:
        by_cam.setdefault(s["camera"], []).append(s)
    tasks = [dict(camera=c, srcs=v, minutes=minutes, hours=opts["active_hours"], lamps=lamps) for c, v in by_cam.items()]
    w = min(workers or os.cpu_count() or 1, len(tasks))
    print(f"Learning the scene of {len(tasks)} camera(s) from about {minutes:.0f} min of footage each "
          f"(where people stay, what keeps moving{', which lights switch' if lamps else ''}) ...", flush=True)
    results = {}
    with Runner(model, opts["conf"], w, detect) as rn:
        for r in rn.imap(learn_task, tasks):
            results[r["camera"]] = r
            print(f"  {r['camera']}: " + (_describe(r["info"]) if r["ok"] else f"could not learn ({r['error']}), people counting only"), flush=True)
    cameras, setup = {}, {}
    (out / "setup").mkdir(parents=True, exist_ok=True)
    for cam in by_cam:
        r = results[cam]
        cfg = r["cfg"] if r["ok"] else build_camera_config([], [], [])
        cameras[cam] = {"match": match[cam]["match"], **cfg}
        if r["ok"]:
            picture = camera_picture(out, cam, r["preview"][0], r["preview"][1], cfg)
            setup[cam] = {**r["info"], **({"image": picture} if picture else {})}
    options = {"fps": opts["fps"], "conf": opts["conf"], "clips": opts["clips"]}
    if opts["active_hours"]:
        options["active_hours"] = opts["active_hours"]
    cfg_all = {"options": options, "cameras": cameras}
    path = write_setup(setup_file(out), cfg_all)
    write_info(out, setup)
    print(f"Setup written to {path} (edit it, or draw the zones by hand with  python -m videoanalytics zones ..., and run again to correct anything)", flush=True)
    return cfg_all


def _describe(info: dict) -> str:
    parts = [f"{len(info['spots'])} station(s)", f"{len(info['motion'])} moving machine zone(s)"]
    if info.get("lamps_looked_for"):
        parts.append(f"{len(info['lamps'])} signal lamp(s)")
    return ", ".join(parts) + f"  ({info['person_tracks']} person track(s) seen in {info['sampled_s']} s)"


# ---- the whole thing --------------------------------------------------------------------------------------------------
def _start_ts(start):
    return parse_start_arg(start) if isinstance(start, str) else start


def _inspect(srcs):
    try:
        return inspect_samples(srcs)
    except Exception as e:  # noqa: BLE001 - looking closely must never stop the work
        print(f"(could not look closely at the recordings: {e!r})", flush=True)
        return []


def write_diagnostics(out: Path, srcs, issues, samples, paths, opts, workers, finished: bool = False):
    """diagnostics.txt next to the results: written right after the recordings were read (so it exists even if a later step fails)
    and again at the end, with the outcome."""
    try:
        text = doctor_text(srcs, issues, paths=paths, samples=samples, hours=opts["active_hours"], workers=workers)
        (out / "diagnostics.txt").write_text(text + (result_text(out) if finished else ""), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"(diagnostics.txt could not be written: {e!r})", flush=True)


def analyze(paths, *, out=None, config=None, model=None, workers=None, open_report=True, auto=True, rediscover=False,
            learn_minutes: float = 6.0, chunk_min: float = 20.0, hours=None, fps=None, conf=None, clips=True, hw_decode=False,
            no_gate=False, time_regex=None, detect=None, limit_files=None, start=None, one_camera=None, lamps=False) -> Path:
    """Analyse the given videos / folders end to end and return the path of report.html.
    start: '2026-10-01 08:00', when files that carry no time in their name were recorded (they follow each other from there);
    one_camera: all files are parts of one camera (its name; '' = the folder name);
    lamps: also look for signal lamps (stack lights) while learning - only for machines that have them."""
    t_all = time.time()
    root, files = resolve_inputs(paths)
    out = Path(out) if out else default_out(root, paths)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Input: {len(files)} video file(s) under {root}\nResults: {out}", flush=True)
    start_ts = _start_ts(start)

    model_path = None if detect is not None else ensure_model(model)
    overrides = {k: v for k, v in dict(fps=fps, conf=conf, active_hours=hours, hw_decode=hw_decode or None, clips=None if clips else False,
                                       gate_motion=-1.0 if no_gate else None).items() if v is not None}
    chunk_s = chunk_min * 60.0

    auto_yaml = setup_file(out)
    cfg_path = config or (auto_yaml if auto_yaml.exists() and not rediscover else None)
    if cfg_path:
        cfg = read_config(cfg_path)
        print(f"Using your setup: {cfg_path}", flush=True)
        opts = {**DEFAULTS, **(cfg.get("options") or {}), **overrides}
        cams = {k: normalize_camera(v) for k, v in cfg["cameras"].items()}
        srcs, issues = discover(str(root), cams, time_regex, only=files, exclude=[out], start=start_ts)
    else:
        opts = {**DEFAULTS, **overrides}
        srcs, issues, match = discover_auto(str(root), files, time_regex, exclude=[out], start=start_ts, one_camera=one_camera)
        cfg = None
    if not srcs:
        for n in notes([], issues, [], paths):
            print(f"! {n}", flush=True)
        raise SystemExit("no readable video files")
    hours_total = sum(s["duration"] for s in srcs) / 3600
    print(f"{len(srcs)} recording(s) of {len({s['camera'] for s in srcs})} camera(s), {hours_total:.1f} h of video", flush=True)
    print(cameras_summary(srcs, opts["active_hours"]), flush=True)
    samples = _inspect(srcs)
    for n in notes(srcs, issues, samples, paths):
        print(f"! {n}", flush=True)
    write_diagnostics(out, srcs, issues, samples, paths, opts, workers)

    if cfg is None:
        if auto:
            cfg = learn_cameras(srcs, match, opts, out, model_path, workers, learn_minutes, detect, lamps)
        else:
            cfg = {"cameras": {c: {"match": m["match"], **build_camera_config([], [], [])} for c, m in match.items()}}
        cams = {k: normalize_camera(v) for k, v in cfg["cameras"].items()}

    line = estimate(srcs, cams, opts, (lambda: detect) if detect is not None else (lambda: _load_detector(model_path, opts["conf"])), workers)
    if line:
        print(line, flush=True)
    print("Analysing ...", flush=True)
    run_analysis(cams, opts, srcs, out, model_path, workers, limit_files=limit_files, chunk_s=chunk_s, detect=detect)
    merge_shards(out)
    report = write_reports(out, 300.0, cfg)
    write_diagnostics(out, srcs, issues, samples, paths, opts, workers, finished=True)
    print(f"Done in {(time.time() - t_all) / 60:.1f} min.", flush=True)
    print(f"Diagnostics (send this file if something looks wrong): {out / 'diagnostics.txt'}", flush=True)
    print(f"REPORT: {report}", flush=True)  # the GUI reads this line
    if open_report:
        webbrowser.open(report.resolve().as_uri())
    return report


def doctor(paths, *, out=None, model=None, workers=None, hours=None, time_regex=None, start=None, one_camera=None, bench=True,
           detect=None) -> Path:
    """The pre-flight check, 10-60 seconds even for days of footage: reads every recording, looks closely at a few, measures the speed
    of this computer, prints one text and saves it as doctor.txt next to where the results would go."""
    root, files = resolve_inputs(paths)
    out = Path(out) if out else default_out(root, paths)
    srcs, issues, _ = discover_auto(str(root), files, time_regex, exclude=[out], start=_start_ts(start), one_camera=one_camera)
    samples = _inspect(srcs)
    cost = None
    if bench and srcs:
        opts = {**DEFAULTS, **({"active_hours": hours} if hours else {})}
        cams = {c: normalize_camera({}) for c in {s["camera"] for s in srcs}}
        cost = benchmark(srcs, cams, opts, (lambda: detect) if detect is not None else (lambda: _load_detector(ensure_model(model), opts["conf"])))
    text = doctor_text(srcs, issues, paths=paths, samples=samples, hours=hours, cost=cost, workers=workers)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "doctor.txt"
    path.write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"Saved as {path}", flush=True)
    return path
