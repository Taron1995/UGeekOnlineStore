import math
from datetime import datetime

import cv2
import numpy as np
import pytest

from videoanalytics.detector import _grids, postprocess
from videoanalytics.pipeline import DEFAULTS, active_intervals, process_video
from videoanalytics.sources import camera_for, discover, naive_epoch, parse_start
from videoanalytics.storage import shard_path
from videoanalytics.tracker import Tracker
from videoanalytics.zones import Line, MachineState, Motion, Zone


# ---------- tracker / zones ----------
def test_tracker_keeps_id_for_fast_walker():
    tr = Tracker(min_hits=2)
    ids = set()
    for i in range(10):  # 40 px wide box moving 30 px per update: plain IoU would lose it
        ids |= {t.id for t in tr.update([(10 + 30 * i, 10, 50 + 30 * i, 90, 0.9)])}
    assert ids == {1}


def test_tracker_separates_two_people():
    tr = Tracker(min_hits=2)
    ids = set()
    for i in range(6):
        ids |= {t.id for t in tr.update([(10 + 5 * i, 10, 50 + 5 * i, 90, 0.9), (300, 10, 340, 90, 0.9)])}
    assert len(ids) == 2


def test_tracker_expires_lost_tracks():
    tr = Tracker(max_missed=3)
    tr.update([(0, 0, 10, 20, 0.9)])
    for _ in range(5):
        tr.update([])
    assert tr.tracks == []


def test_zone_contains():
    z = Zone("a", [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]])
    assert z.contains(10, 10, 100, 100) and not z.contains(90, 90, 100, 100)


def test_line_counts_clear_crossings_only():
    ln = Line("door", [0.5, 0], [0.5, 1])
    assert ln.update(1, (10, 50), 100, 100) is None  # first sighting: only remembers the side
    assert ln.update(1, (90, 50), 100, 100) == "in"
    assert ln.update(1, (10, 50), 100, 100) == "out"
    assert ln.update(2, (50.4, 50), 100, 100) is None  # on the line (inside the dead-band): ignored
    assert ln.update(2, (49.6, 50), 100, 100) is None
    assert ln.update(3, (30, 50), 100, 100) is None and ln.update(3, (40, 50), 100, 100) is None  # same side


# ---------- motion / machine ----------
def _frame(box=None):
    f = np.full((90, 160, 3), 90, np.uint8)
    if box:
        f[box[1]:box[3], box[0]:box[2]] = 255
    return f


def test_person_boxes_are_excluded_from_machine_motion():
    m, roi = Motion(), Zone("m", [[0, 0], [1, 0], [1, 1], [0, 1]])
    m.step(_frame())
    m.step(_frame((60, 20, 100, 80)))
    assert m.roi_ratio(roi) > 0.1
    assert m.roi_ratio(roi, [(58, 18, 102, 82)], (90, 160)) < 0.01


def test_machine_state_hysteresis():
    ms = MachineState(motion_thr=0.02, window_s=10, on_frac=0.3, off_frac=0.1)
    seq = [(t, 0.0) for t in range(0, 20)] + [(t, 0.2) for t in range(20, 40)] + [(t, 0.0) for t in range(40, 70)]
    changes = []
    for t, r in seq:
        state, changed = ms.update(r, t)
        if changed:
            changes.append((t, state))
    assert [s for _, s in changes] == ["work", "idle"]
    assert 20 <= changes[0][0] <= 24 and 49 <= changes[1][0] <= 52


# ---------- detector post-processing ----------
def _raw_with_person(cls=0):
    grid, stride = _grids(416)
    raw = np.zeros((grid.shape[0], 85), np.float32)
    i = 20 * 52 + 20  # an anchor of the stride-8 level (52x52 grid), away from the image edge
    raw[i, 0:2] = 0.0
    raw[i, 2:4] = [math.log(32 / stride[i, 0]), math.log(64 / stride[i, 0])]
    raw[i, 4], raw[i, 5 + cls] = 0.9, 0.9
    return raw, grid, stride, i


def test_postprocess_decodes_person_box():
    raw, grid, stride, i = _raw_with_person()
    (x1, y1, x2, y2, s), = postprocess(raw, grid, stride, 0.5, (832, 832), 0.35, 0.45)
    cx, cy = grid[i] * stride[i]
    assert (x2 - x1, y2 - y1) == pytest.approx((64, 128), abs=0.01)  # 1/0.5 scale back to frame pixels
    assert ((x1 + x2) / 2, (y1 + y2) / 2) == pytest.approx((cx * 2, cy * 2), abs=0.01)
    assert s == pytest.approx(0.81, abs=0.01)


def test_postprocess_ignores_other_classes():
    raw, grid, stride, _ = _raw_with_person(cls=2)
    assert postprocess(raw, grid, stride, 1.0, (416, 416), 0.35, 0.45) == []


# ---------- sources ----------
@pytest.mark.parametrize("rel", ["cam1/cam1_20261001_080000.mp4", "2026-10-01/cam1/08-00-00.mp4",
                                 "cam1_2026-10-01_08.00.00.avi", "ch01/20261001080000.mkv"])
def test_parse_start_formats(rel):
    assert parse_start(rel) == naive_epoch(datetime(2026, 10, 1, 8, 0, 0))


def test_parse_start_rejects_garbage():
    assert parse_start("cam1/video_000123.mp4") is None
    assert parse_start("cam1/20261301_080000.mp4") is None  # month 13


def test_camera_matching_is_whole_word():
    cams = {"cam1": {}, "cam10": {}, "gate": {"match": "entrance|gate"}}
    assert camera_for("cam1/a.mp4", cams) == "cam1"
    assert camera_for("cam10/a.mp4", cams) == "cam10"
    assert camera_for("webcam1/a.mp4", cams) is None
    assert camera_for("Entrance_20261001.mp4", cams) == "gate"


def test_active_intervals():
    start = naive_epoch(datetime(2026, 10, 1, 7, 30))
    assert active_intervals(start, 3600, [[8, 20]]) == [(1800.0, 3600.0)]
    assert active_intervals(start, 3600, None) == [(0.0, 3600)]
    night = naive_epoch(datetime(2026, 10, 1, 23, 0))
    assert active_intervals(night, 3 * 3600, [[22, 6]]) == [(0.0, 3 * 3600)]
    assert active_intervals(naive_epoch(datetime(2026, 10, 1, 12, 0)), 3600, [[22, 6]]) == []


# ---------- pipeline end to end (stub detector, synthetic video) ----------
def _video(path, person: bool, machine_motion: bool, seconds=20, fps=25, size=(320, 180)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(seconds * fps):
        f = np.full((size[1], size[0], 3), 90, np.uint8)
        t = i / fps
        if person:  # 30x80 white "person" walking left -> right at 17 px/s, crossing x = 160
            x = int(20 + 17 * t)
            f[100:180, x:x + 30] = 255
        if machine_motion and 5 <= t < 15:
            f[20:60, 200 + (i % 20) * 3:240 + (i % 20) * 3] = 255
        w.write(f)
    w.release()


def _stub_detect(frame):
    ys, xs = np.where((frame[:, :, 0] > 200) & (np.arange(frame.shape[0])[:, None] > 90))
    return [(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()), 0.9)] if len(xs) else []


CAM = dict(zones=[dict(name="left", polygon=[[0, 0], [0.5, 0], [0.5, 1], [0, 1]])],
           lines=[dict(name="mid", p1=[0.5, 0], p2=[0.5, 1])],
           machine=dict(polygon=[[0.55, 0.0], [1, 0.0], [1, 0.4], [0.55, 0.4]], window_s=4))


def _src(path):
    return dict(camera="cam1", path=str(path), rel="cam1/cam1_20261001_080000.mp4",
                start_ts=naive_epoch(datetime(2026, 10, 1, 8, 0)), duration=20.0)


def test_pipeline_counts_person_and_ignores_it_for_machine(tmp_path):
    v = tmp_path / "a.mp4"
    _video(v, person=True, machine_motion=False)
    events, tables, meta = process_video(_src(v), CAM, {**DEFAULTS, "fps": 5, "clips": False}, _stub_detect)
    types = [e[1] for e in events]
    assert types.count("person_appeared") == 1
    assert ("line_cross", "mid:in") in [(e[1], e[3]) for e in events]
    assert "zone_enter" in types and "zone_exit" in types
    assert "machine_state" not in types
    (row,) = tables["minutes"].values()
    assert row[0] == pytest.approx(20, abs=1) and row[2] == 1 and row[1] > 10  # covered_s, person-seconds, people_max


def test_pipeline_machine_work_and_gate(tmp_path):
    v = tmp_path / "b.mp4"
    _video(v, person=False, machine_motion=True)
    calls = []
    events, tables, meta = process_video(_src(v), CAM, {**DEFAULTS, "fps": 5, "clips": False, "keepalive_s": 5},
                                         lambda f: calls.append(1) or [])
    states = [e[3] for e in events if e[1] == "machine_state"]
    assert states == ["work", "idle"]
    (row,) = tables["machine_minutes"].values()
    assert 8 <= row[0] <= 14 and 1 <= row[1] <= 6  # work_s, idle_s (first 4 s = warm-up, not counted)
    assert 0 < meta["det_frames"] < meta["frames"]  # motion gate skipped the detector in the still parts


def test_pipeline_clips_written(tmp_path):
    v = tmp_path / "c.mp4"
    _video(v, person=False, machine_motion=True)
    out = tmp_path / "out"
    events, *_ = process_video(_src(v), CAM, {**DEFAULTS, "fps": 5, "clip_pad_s": 2}, lambda f: [], out)
    clips = [e for e in events if e[1] == "machine_state"]
    assert clips and all((out / c[4]).stat().st_size > 0 and (out / c[5]).exists() for c in clips)


def test_active_hours_skip_everything_outside(tmp_path):
    v = tmp_path / "d.mp4"
    _video(v, person=True, machine_motion=False)
    events, tables, meta = process_video(_src(v), CAM, {**DEFAULTS, "fps": 5, "clips": False, "active_hours": [[9, 10]]},
                                         _stub_detect)
    assert events == [] and tables["minutes"] == {} and meta["frames"] == 0


# ---------- storage ----------
def test_shard_paths_are_unique_and_short(tmp_path):
    a, b = shard_path(tmp_path, "cam1", "a/b.mp4"), shard_path(tmp_path, "cam1", "a__b.mp4")
    assert a != b and a == shard_path(tmp_path, "cam1", "a\\b.mp4")
    assert len(shard_path(tmp_path, "cam1", "x/" * 200 + "f.mp4").name) < 130


def test_discover_reports_unreadable_unmatched_and_untimed(tmp_path):
    good = tmp_path / "cam1" / "cam1_20261001_080000.mp4"
    good.parent.mkdir()
    _video(good, person=False, machine_motion=False, seconds=2)
    (tmp_path / "cam1" / "cam1_20261001_090000.mp4").write_text("not a video")
    (tmp_path / "other.mp4").write_bytes(good.read_bytes())
    (tmp_path / "cam1" / "cam1_plain.mp4").write_bytes(good.read_bytes())
    srcs, issues = discover(str(tmp_path), {"cam1": {}})
    assert sorted(s["rel"] for s in srcs) == ["cam1/cam1_20261001_080000.mp4", "cam1/cam1_plain.mp4"]
    assert issues["unreadable"] == ["cam1/cam1_20261001_090000.mp4"]
    assert issues["unmatched"] == ["other.mp4"] and issues["no_timestamp"] == ["cam1/cam1_plain.mp4"]
    assert srcs[0]["duration"] == pytest.approx(2, abs=0.2) and srcs[0]["fps"] == 25
