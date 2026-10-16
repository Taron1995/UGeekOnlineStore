import shutil
import sqlite3

import pytest
import yaml

from tests.test_autoconfig import draw_lamps
from tests.test_features import make_video
from videoanalytics import analyze as an
from videoanalytics.analyze import analyze, cameras_summary, ensure_model, estimate, fingerprint, plan_tasks, resolve_inputs, split_segments
from videoanalytics.config import DEFAULTS, check_config, normalize_camera
from videoanalytics.sources import auto_camera_name, discover_auto
from videoanalytics.storage import shard_path, write_shard

STILL_A, STILL_B = (40, 90, 70, 170, 0.9), (230, 90, 260, 170, 0.9)
two_people = lambda frame: [STILL_A, STILL_B]


def scene(path, seconds=80):
    make_video(path, seconds, draw_lamps)  # a stack light + decoys; two people stand still (stub detector)


def run(tmp_path, *paths, **kw):
    return analyze([str(p) for p in paths], detect=two_people, open_report=False, workers=1, **kw)


def test_one_video_in_report_out(tmp_path, capsys):
    v = tmp_path / "cam1_20261001_080000.mp4"
    scene(v)
    report = run(tmp_path, v)
    out = tmp_path / "cam1_20261001_080000_analysis"
    assert report == out / "report.html" and report.stat().st_size > 5000
    cfg = yaml.safe_load((out / "cameras.auto.yaml").read_text(encoding="utf-8"))
    check_config(cfg)  # the learned setup is a valid, editable config
    cam = cfg["cameras"]["cam1"]
    assert len(cam["zones"]) == 2 and [m["source"] for m in cam["machines"]] == ["lamp"] and cam["machines"][0]["work_colors"] == ["green"]
    assert (out / "setup" / "cam1.jpg").exists() and (out / "setup" / "setup.json").exists()
    text = capsys.readouterr().out
    assert "Learning the scene" in text and "2 station(s)" in text and f"REPORT: {report}" in text
    html = report.read_text(encoding="utf-8")
    assert "cam1" in html and "lamp1" in html and "spot1" in html
    diag = (out / "diagnostics.txt").read_text(encoding="utf-8")  # what to send if something looks wrong
    assert "== Computer" in diag and "== Samples" in diag and "== Result" in diag and "learned 2 station(s)" in diag and "speed of this run" in diag
    assert f"Diagnostics (send this file if something looks wrong): {out / 'diagnostics.txt'}" in text


def test_second_run_reuses_the_setup_and_finished_work(tmp_path, capsys):
    v = tmp_path / "cam1_20261001_080000.mp4"
    scene(v)
    run(tmp_path, v)
    capsys.readouterr()
    run(tmp_path, v)
    text = capsys.readouterr().out
    assert "Using your setup" in text and "Learning the scene" not in text and "0 job(s) to process (1 already done" in text


def test_editing_the_setup_recomputes_the_results(tmp_path, capsys):
    v = tmp_path / "cam1_20261001_080000.mp4"
    scene(v)
    run(tmp_path, v)
    path = tmp_path / "cam1_20261001_080000_analysis" / "cameras.auto.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["cameras"]["cam1"]["zones"], cfg["cameras"]["cam1"]["machines"] = [], []  # the user deletes what was wrong
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    capsys.readouterr()
    run(tmp_path, v)
    text = capsys.readouterr().out
    assert "1 job(s) to process (0 already done, 1 recomputed because the setup changed)" in text
    con = sqlite3.connect(tmp_path / "cam1_20261001_080000_analysis" / "analytics.db")
    assert con.execute("SELECT COUNT(*) FROM zone_minutes").fetchone()[0] == 0  # the old results are gone, not merged in


def test_folder_with_two_cameras_and_results_inside_it(tmp_path, capsys):
    for n in ("cam1", "cam2"):
        scene(tmp_path / f"{n}_20261001_080000.mp4", 40)
    run(tmp_path, tmp_path)
    out = tmp_path / "_analysis"
    cfg = yaml.safe_load((out / "cameras.auto.yaml").read_text(encoding="utf-8"))
    assert sorted(cfg["cameras"]) == ["cam1", "cam2"] and all("match" in c for c in cfg["cameras"].values())
    assert list(out.rglob("*.mp4")), "the lamp changes colour, so clips were written inside the results folder"
    capsys.readouterr()
    run(tmp_path, tmp_path)  # the clips under _analysis must not be picked up as new recordings
    assert "2 recording(s) of 2 camera(s)" in capsys.readouterr().out


def test_clips_of_an_earlier_run_are_not_inputs_even_if_the_results_folder_is_elsewhere(tmp_path, capsys):
    for n in ("cam1", "cam2"):
        scene(tmp_path / f"{n}_20261001_080000.mp4", 40)
    run(tmp_path, tmp_path, out=tmp_path / "first")
    assert list((tmp_path / "first").rglob("*.mp4"))  # evidence clips of the first run sit inside the input folder
    capsys.readouterr()
    run(tmp_path, tmp_path, out=tmp_path / "second")
    assert "2 recording(s) of 2 camera(s)" in capsys.readouterr().out


@pytest.mark.parametrize("rel, name", [
    ("cam1_20261001_080000.mp4", "cam1"),
    ("Cam 1/2026-10-01/08-00-00.mp4", "Cam_1"),
    ("2026-10-01/cam2/08-00-00.mp4", "cam2"),
    ("ch01_main_20261001080000.mp4", "ch01_main"),
    ("ch01_20261001080000[R][0@0][0].mp4", "ch01"),  # recorder tags are not part of the name
    ("garage_20261001_080000_001.mp4", "garage"),  # chunk counter
    ("Hall\\20261001_080000.mp4", "Hall"),  # Windows separators
    ("Камера 1/2026-10-01/08-00-00.mp4", "Камера_1"),
    ("08-00-00.mp4", None), ("20261001_080000.mp4", None), ("1.mp4", None), ("2026-10-01/5.mp4", None),
])
def test_camera_names_from_typical_layouts(rel, name):
    assert auto_camera_name(rel) == name


def test_date_folder_layout_and_no_auto_mode(tmp_path):
    v = tmp_path / "2026-10-02" / "cam3" / "09-30-00.mp4"
    v.parent.mkdir(parents=True)
    scene(v, 20)
    srcs, issues, cams = discover_auto(str(tmp_path))
    assert [s["camera"] for s in srcs] == ["cam3"] and srcs[0]["start_ts"] == 1790933400.0 and issues["no_timestamp"] == []
    run(tmp_path, tmp_path, auto=False)
    assert not (tmp_path / "_analysis" / "cameras.auto.yaml").exists() and (tmp_path / "_analysis" / "report.html").exists()


def test_nameless_files_get_the_folder_name(tmp_path):
    d = tmp_path / "hall"
    d.mkdir()
    scene(d / "20261001_080000.mp4", 10)
    srcs, _, cams = discover_auto(str(d))
    assert [s["camera"] for s in srcs] == ["hall"] and cams["hall"]["match"] == ".*"


def test_mixed_named_and_nameless_files_are_kept_apart(tmp_path):
    scene(tmp_path / "cam1_20261001_080000.mp4", 10)
    scene(tmp_path / "20261001_090000.mp4", 10)
    srcs, _, cams = discover_auto(str(tmp_path))
    assert {s["rel"]: s["camera"] for s in srcs} == {"cam1_20261001_080000.mp4": "cam1", "20261001_090000.mp4": tmp_path.name}
    from videoanalytics.sources import camera_for
    assert camera_for("cam1_20261001_080000.mp4", cams) == "cam1" and camera_for("20261001_090000.mp4", cams) == tmp_path.name  # survives a re-read


# ---------- chunks, fingerprints, inputs, model ----------
def test_long_recordings_are_cut_into_chunks():
    assert split_segments(1500, 1200) == [None] and split_segments(3000, 0) == [None]
    segs = split_segments(3600, 1200)
    assert len(segs) == 3 and segs[0] == (0, 1200) and segs[-1][1] == 3600 and all(a < b for a, b in segs)
    assert len(split_segments(8 * 3600, 1200)) == 24


def fake_src(rel="cam1/a.mp4", duration=3600.0):
    return dict(camera="cam1", path="/x/" + rel, rel=rel, start_ts=1790841600.0, duration=duration, fps=25.0)


def test_plan_tasks_chunks_and_drops_shards_of_another_chunking(tmp_path):
    from videoanalytics.config import DEFAULTS
    cams, opts = {"cam1": {}}, dict(DEFAULTS)
    tasks, done, stale = plan_tasks([fake_src()], cams, opts, tmp_path, 1200.0)
    assert len(tasks) == 3 and done == 0 and len({t["shard"] for t in tasks}) == 3 and tasks[0]["src"]["seg"] == (0, 1200)
    meta = dict(camera="cam1", source="x", start_ts=0, video_s=1, processed_s=1, frames=1, det_frames=0, elapsed_s=1, config_hash=tasks[0]["config_hash"])
    write_shard(shard_path(tmp_path, "cam1", "cam1/a.mp4"), [], {}, meta)  # an unchunked shard of the same file from an earlier run
    write_shard(__import__("pathlib").Path(tasks[0]["shard"]), [], {}, meta)  # the first chunk is already finished
    tasks2, done2, stale2 = plan_tasks([fake_src()], cams, opts, tmp_path, 1200.0)
    assert len(tasks2) == 2 and done2 == 1 and stale2 == 1 and not shard_path(tmp_path, "cam1", "cam1/a.mp4").exists()


def test_fingerprint_follows_the_setup_but_not_the_match_pattern():
    from videoanalytics.config import DEFAULTS
    cam = {"zones": [{"name": "a", "polygon": [[0, 0], [1, 0], [1, 1]]}], "match": "x"}
    base = fingerprint(cam, dict(DEFAULTS), 0)
    assert fingerprint({**cam, "match": "other"}, dict(DEFAULTS), 0) == base
    assert fingerprint({**cam, "zones": [{"name": "a", "polygon": [[0, 0], [0.5, 0], [1, 1]]}]}, dict(DEFAULTS), 0) != base
    assert fingerprint(cam, {**DEFAULTS, "fps": 5.0}, 0) != base and fingerprint(cam, dict(DEFAULTS), 1200.0) != base


def test_chunked_analysis_matches_the_unchunked_totals(tmp_path):
    v = tmp_path / "cam1_20261001_080000.mp4"
    scene(v, 80)
    totals = []
    for chunk in (0.0, 0.5):  # 0.5 min = 30 s chunks -> the 80 s file is cut into 3
        out = tmp_path / f"o{chunk}"
        analyze([str(v)], out=str(out), detect=two_people, open_report=False, workers=1, chunk_min=chunk, auto=False)
        con = sqlite3.connect(out / "analytics.db")
        totals.append(con.execute("SELECT SUM(covered_s), SUM(people_s) FROM minutes").fetchone())
    assert totals[0][0] == pytest.approx(totals[1][0], abs=1.0) and totals[0][1] == pytest.approx(totals[1][1], rel=0.08)
    assert len(list((tmp_path / "o0.5" / "shards").rglob("*.db"))) == 3


def test_inputs_and_errors(tmp_path):
    scene(tmp_path / "a" / "cam1_20261001_080000.mp4", 5) if (tmp_path / "a").mkdir() is None else None
    (tmp_path / "b").mkdir()
    shutil.copy(tmp_path / "a" / "cam1_20261001_080000.mp4", tmp_path / "b" / "cam2_20261001_080000.mp4")
    root, files = resolve_inputs([tmp_path / "a", str(tmp_path / "b" / "cam2_20261001_080000.mp4")])
    assert root == tmp_path.resolve() and len(files) == 2
    with pytest.raises(SystemExit, match="not found"):
        resolve_inputs([tmp_path / "nope.mp4"])
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no video files"):
        resolve_inputs([tmp_path / "empty"])


def test_model_lookup_and_offline_message(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDAN_MODELS", str(tmp_path / "m"))
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / an.MODEL).write_bytes(b"x" * 1_100_000)
    assert ensure_model(None) == str(tmp_path / "m" / an.MODEL)  # found: nothing is downloaded
    (tmp_path / "m" / an.MODEL).unlink()
    monkeypatch.chdir(tmp_path)  # no ./models either
    monkeypatch.setattr(an, "fetch_models", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(SystemExit) as e:
        ensure_model(None)
    assert "github.com/Megvii-BaseDetection" in str(e.value) and str(tmp_path / "m") in str(e.value)
    with pytest.raises(SystemExit, match="model not found"):
        ensure_model(str(tmp_path / "missing.onnx"))


# ---------- what the user sees before the long part ----------
def test_cameras_are_listed_before_anything_is_computed(tmp_path, capsys):
    for n in ("cam1", "cam2"):
        scene(tmp_path / f"{n}_20261001_080000.mp4", 20)
    run(tmp_path, tmp_path)
    text = capsys.readouterr().out
    assert "  cam1: 1 file(s), 20 s of video, 2026-10-01 08:00 -> 2026-10-01 08:00" in text and "  cam2: 1 file(s)" in text
    srcs, _, _ = discover_auto(str(tmp_path), None, None, [tmp_path / "_analysis"])
    many = cameras_summary(srcs + [{**srcs[0], "camera": f"x{i}"} for i in range(11)])  # 13 cameras: 12 lines and "... and 1 more"
    assert many.count("\n") == 12 and many.endswith("... and 1 more camera(s)")


def test_time_estimate_is_given_for_long_jobs_only(tmp_path, monkeypatch):
    v = tmp_path / "cam1_20261001_080000.mp4"
    scene(v, 20)
    srcs, _, _ = discover_auto(str(tmp_path), None, None, [])
    cams = {"cam1": normalize_camera({})}
    opts = {**DEFAULTS, "clips": False}
    assert estimate(srcs, cams, opts, lambda: two_people, 4) is None  # 20 s of video: no estimate needed
    long_src = [{**srcs[0], "duration": 20 * 3600.0}]  # pretend the file is 20 h long; the test only reads its first seconds
    line = estimate(long_src, cams, opts, lambda: two_people, 4, bench_s=10)
    assert line and line.startswith("Estimate: about ") and "20.0 h of video on 4 worker(s)" in line and "--hours" in line
    # the arithmetic, with a known cost: 10 video-s took 5 s -> 0.5 CPU-s per video-s; 72000 s of video on 4 workers = 9000 s
    monkeypatch.setattr(an.pipeline, "process_video", lambda *a, **k: ([], {}, dict(processed_s=10.0, elapsed_s=5.0)))
    assert "about 2.5 h for 20.0 h of video on 4 worker(s)" in estimate(long_src, cams, opts, lambda: two_people, 4)
    assert estimate(long_src, cams, opts, lambda: two_people, 1).startswith("Estimate: about 10.0 h")
    monkeypatch.undo()

    def no_model():
        raise RuntimeError("model cannot be loaded")

    assert estimate(long_src, cams, opts, no_model, 4) is None  # a failing test run must not stop the real work


def test_diagnostics_exist_even_when_a_later_step_fails(tmp_path, monkeypatch):
    v = tmp_path / "cam1_20261001_080000.mp4"
    scene(v, 10)

    def boom(*a, **k):
        raise RuntimeError("learning failed")

    monkeypatch.setattr(an, "learn_cameras", boom)
    with pytest.raises(RuntimeError, match="learning failed"):
        run(tmp_path, v)
    text = (tmp_path / "cam1_20261001_080000_analysis" / "diagnostics.txt").read_text(encoding="utf-8")
    assert "== Computer" in text and "== Samples" in text and "== Result" not in text  # the check came first, the outcome did not happen


def test_problems_are_printed_before_the_long_part(tmp_path, capsys):
    scene(tmp_path / "video1.mp4", 10)
    run(tmp_path, tmp_path / "video1.mp4", auto=False)
    out = capsys.readouterr().out
    assert "! 1 file(s) have no date/time in their name" in out and "--start" in out


def test_parts_of_one_camera_downloaded_without_times(tmp_path, capsys):
    for n in ("video1", "video2", "video10"):
        scene(tmp_path / f"{n}.mp4", 12)
    run(tmp_path, tmp_path, one_camera="Lathe 3", start="2026-10-01 08:00", auto=False)
    out = capsys.readouterr().out
    assert "3 recording(s) of 1 camera(s)" in out and "Lathe_3: 3 file(s)" in out and "(--start) is used" in out
    con = sqlite3.connect(tmp_path / "_analysis" / "analytics.db")
    starts = [r[0] for r in con.execute("SELECT start_ts FROM files ORDER BY start_ts")]
    t0 = 1790841600.0  # 2026-10-01 08:00 as wall-clock seconds
    assert starts == pytest.approx([t0, t0 + 12, t0 + 24], abs=0.5)  # one after the other, video10 after video2
    assert con.execute("SELECT COUNT(DISTINCT camera) FROM minutes").fetchone() == (1,)


def test_a_wrong_start_is_refused_with_the_format(tmp_path):
    scene(tmp_path / "a.mp4", 5)
    with pytest.raises(ValueError, match="2026-10-01 08:30"):
        run(tmp_path, tmp_path / "a.mp4", start="tomorrow")
