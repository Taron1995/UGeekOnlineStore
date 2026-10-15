"""Lamp colour, cycles, presence, gatherings, schedule and config validation."""
from datetime import datetime

import cv2
import numpy as np
import pytest

from videoanalytics.analyzers import CycleCounter, SegmentTracker, largest_group
from videoanalytics.config import ConfigError, check_config, normalize_camera
from videoanalytics.lamp import LampReader
from videoanalytics.pipeline import DEFAULTS, process_video
from videoanalytics.schedule import Schedule
from videoanalytics.sources import naive_epoch
from videoanalytics.tracker import Track
from videoanalytics.zones import Zone

SIZE, FPS = (320, 180), 25
GREEN, YELLOW, RED, DARK = (0, 200, 0), (0, 220, 255), (0, 0, 230), (50, 50, 50)
LAMP_ROI = [[0.45, 0.4], [0.55, 0.4], [0.55, 0.6], [0.45, 0.6]]  # 32 x 36 px around the centre


def make_video(path, seconds, draw):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    for i in range(int(seconds * FPS)):
        f = np.full((SIZE[1], SIZE[0], 3), 90, np.uint8)
        draw(f, i / FPS)
        w.write(f)
    w.release()


def src(path, seconds):
    return dict(camera="cam1", path=str(path), rel="cam1/cam1_20261001_080000.mp4",
                start_ts=naive_epoch(datetime(2026, 10, 1, 8, 0)), duration=float(seconds))


def opts(**kw):
    return {**DEFAULTS, "fps": 5, "clips": False, "gate_motion": -1.0, **kw}


def lamp(f, colour):
    cv2.rectangle(f, (144, 72), (176, 108), colour, -1)


# ---------- lamp ----------
def frame_with(colour):
    f = np.full((180, 320, 3), 90, np.uint8)
    lamp(f, colour)
    return f


@pytest.mark.parametrize("colour,label", [(GREEN, "green"), (YELLOW, "yellow"), (RED, "red"), (DARK, "off"), ((0, 60, 0), "off")])
def test_lamp_classifies_colours(colour, label):
    assert LampReader(Zone("l", LAMP_ROI)).classify(frame_with(colour)) == label


def test_lamp_scores_are_per_colour_shares():
    s = LampReader(Zone("l", LAMP_ROI)).scores(frame_with(GREEN))
    assert s["green"] > 0.9 and s["red"] == 0 and s["yellow"] == 0


def test_blinking_lamp_reads_as_its_colour():
    r = LampReader(Zone("l", LAMP_ROI), window_s=3.0)
    labels = [r.update(frame_with(RED if i % 2 == 0 else DARK), i * 0.5)[0] for i in range(20)]
    assert set(labels[6:]) == {"red"}


def test_lamp_machine_in_pipeline(tmp_path):
    v = tmp_path / "lamp.mp4"
    make_video(v, 20, lambda f, t: lamp(f, GREEN if t < 8 else RED if t < 14 else DARK))
    cam = dict(machines=[dict(name="press", source="lamp", roi=LAMP_ROI, window_s=2.0)])
    events, tables, _ = process_video(src(v, 20), cam, opts(), lambda f: [])
    assert [(e[1], e[3]) for e in events if e[1] in ("lamp_color", "machine_state")] == [
        ("lamp_color", "red"), ("machine_state", "idle"), ("lamp_color", "off")]
    (row,) = tables["machine_minutes"].values()
    work, idle, alarm, wait = row[:4]
    assert 5 <= work <= 8 and 10 <= idle <= 15 and 4 <= alarm <= 8 and wait == 0
    assert row[2] + 0 <= idle  # alarm time is part of idle time


# ---------- cycles ----------
def test_cycle_counter_merges_short_pauses_and_ignores_blips():
    c, done = CycleCounter(min_work_s=3, min_idle_s=1), []
    pattern = [(5, True), (3, False), (4, True), (0.5, False), (3, True), (5, False), (1, True), (4, False)]
    t = 0.0
    for dur, working in pattern:
        for _ in range(int(dur / 0.5)):
            done += c.update(t, working)
            t += 0.5
    done += c.flush(t)
    assert [(round(s), round(e)) for s, e in done] == [(0, 5), (8, 16)]  # second cycle = 4 s + 0.5 s pause + 3 s


def test_motion_machine_cycles_in_pipeline(tmp_path):
    v = tmp_path / "cyc.mp4"
    on = [(6, 10), (13, 17), (20, 24), (27, 31)]
    make_video(v, 34, lambda f, t: any(a <= t < b for a, b in on) and f.__setitem__(
        (slice(20, 60), slice(200 + int(t * 40) % 60, 240 + int(t * 40) % 60)), 255))
    cam = dict(machine=dict(polygon=[[0.55, 0.0], [1, 0.0], [1, 0.45], [0.55, 0.45]], window_s=4,
                            cycle=dict(window_s=1, on_frac=0.5, off_frac=0.2, min_work_s=2, min_idle_s=1)))
    events, tables, _ = process_video(src(v, 34), cam, opts(), lambda f: [])
    cycles = [e for e in events if e[1] == "cycle"]
    assert len(cycles) == 4 and sum(r[6] for r in tables["machine_minutes"].values()) == 4
    assert all(2 <= float(e[3].rstrip("s")) <= 6 for e in cycles)


# ---------- presence / time at the machine ----------
DESK = dict(name="desk", polygon=[[0, 0], [0.5, 0], [0.5, 1], [0, 1]], hold_s=2.0)


def person_for(seconds, box):
    n = [0]

    def detect(frame):
        n[0] += 1
        return [(*box, 0.9)] if n[0] / 5 < seconds else []
    return detect


def test_zone_occupancy_with_hold(tmp_path):
    v = tmp_path / "still.mp4"
    make_video(v, 20, lambda f, t: None)
    _, tables, _ = process_video(src(v, 20), dict(zones=[DESK]), opts(), person_for(10, (40, 60, 70, 140)))
    (row,) = tables["zone_minutes"].values()
    covered, occupied, people_s = row
    assert covered == pytest.approx(20, abs=0.5)
    assert 10 <= occupied <= 13  # 10 s of presence + 2 s hold, minus the unconfirmed first frame
    assert 8.5 <= people_s <= 10.5


def test_attended_vs_unattended_machine_time(tmp_path):
    v = tmp_path / "att.mp4"
    make_video(v, 30, lambda f, t: 5 <= t < 20 and f.__setitem__((slice(20, 60), slice(200 + int(t * 40) % 60, 240 + int(t * 40) % 60)), 255))
    cam = dict(zones=[DESK], machines=[dict(name="m", roi=[[0.55, 0.0], [1, 0.0], [1, 0.45], [0.55, 0.45]], window_s=4, zone="desk")])
    _, tables, _ = process_video(src(v, 30), cam, opts(), person_for(10, (40, 60, 70, 140)))
    (row,) = tables["machine_minutes"].values()
    work, work_att = row[0], row[4]
    assert work > 10 and 3 <= work_att <= 7 and work_att < work  # operator stays 10 s, machine runs ~15 s


# ---------- gatherings ----------
TWO_CLOSE = [(100, 100, 130, 180, 0.9), (150, 100, 180, 180, 0.9)]  # feet 50 px apart, 80 px tall
TWO_FAR = [(20, 100, 50, 180, 0.9), (250, 100, 280, 180, 0.9)]


def people_for(seconds, boxes):
    n = [0]

    def detect(frame):
        n[0] += 1
        return list(boxes) if n[0] / 5 < seconds else []
    return detect


def gather_events(tmp_path, boxes, zones=(), seconds=12, clips=False, out=None, **cfg):
    v = tmp_path / "g.mp4"
    make_video(v, 20, lambda f, t: None)
    cam = dict(zones=list(zones), gathering={"min_duration_s": 5, "gap_s": 2, **cfg})
    events, _, _ = process_video(src(v, 20), cam, opts(clips=clips), people_for(seconds, boxes), out)
    return [e for e in events if e[1] == "gathering"]


def test_gathering_reported_with_duration_and_people(tmp_path):
    (g,) = gather_events(tmp_path, TWO_CLOSE)
    assert g[3].startswith("people=2 duration=") and 8 <= int(g[3].split("duration=")[1].rstrip("s")) <= 11
    assert g[0] < src(tmp_path / "x", 20)["start_ts"] + 2  # timestamped at the START of the gathering


def test_gathering_needs_proximity_and_duration(tmp_path):
    assert gather_events(tmp_path, TWO_FAR) == []
    assert gather_events(tmp_path, TWO_CLOSE, seconds=3) == []
    assert gather_events(tmp_path, TWO_CLOSE, dist_k=0.3) == []


def test_no_gathering_inside_a_work_zone(tmp_path):
    zone = dict(name="bench", polygon=[[0.2, 0.3], [0.7, 0.3], [0.7, 1.0], [0.2, 1.0]])  # covers both people
    assert gather_events(tmp_path, TWO_CLOSE, zones=[zone]) == []
    assert len(gather_events(tmp_path, TWO_CLOSE, zones=[{**zone, "kind": "rest"}])) == 1  # non-work zones do not exclude


def test_gathering_evidence_files(tmp_path):
    out = tmp_path / "out"
    (g,) = gather_events(tmp_path, TWO_CLOSE, clips=True, out=out)
    assert (out / g[4]).stat().st_size > 0 and (out / g[5]).exists()


def test_largest_group_and_segments():
    mk = lambda i, x: Track(i, (x - 15, 100, x + 15, 180), pt=(x, 180.0))
    grp = largest_group([mk(1, 100), mk(2, 140), mk(3, 180), mk(4, 600)], 1.0)  # chain 1-2-3, 4 is far
    assert sorted(t.id for t in grp) == [1, 2, 3]
    assert largest_group([], 1.0) == []
    s = SegmentTracker(min_duration_s=3, gap_s=1)
    out = []
    for t, on in [(0, True), (1, True), (2, False), (2.5, True), (3.5, True), (4, False), (6, False)]:
        out += s.update(t, on, 2)
    assert [k for k, _ in out] == ["qualified", "ended"] and out[1][1]["start"] == 0 and out[1][1]["end"] == 3.5


# ---------- schedule ----------
def minute(y, mo, d, h, mi=0):
    return int(naive_epoch(datetime(y, mo, d, h, mi)) // 60)


def test_schedule_work_breaks_days():
    s = Schedule(dict(work=[["08:00", "12:00"], ["13:00", "17:00"]], breaks=[["10:00", "10:15"]], days=[1, 2, 3, 4, 5]))
    assert s.expected(minute(2026, 10, 1, 9))  # Thursday morning
    assert not s.expected(minute(2026, 10, 1, 10, 5))  # coffee break
    assert not s.expected(minute(2026, 10, 1, 12, 30))  # lunch (outside both work ranges)
    assert not s.expected(minute(2026, 10, 3, 9))  # Saturday
    night = Schedule(dict(work=[["22:00", "06:00"]]))
    assert night.expected(minute(2026, 10, 1, 23, 30)) and night.expected(minute(2026, 10, 2, 3)) and not night.expected(minute(2026, 10, 1, 12))
    assert Schedule().expected(minute(2026, 10, 4, 3))  # no schedule: always expected


# ---------- config ----------
def test_config_validation_reports_every_problem():
    with pytest.raises(ConfigError) as e:
        check_config({"options": {"fsp": 3}, "cameras": {"c": {"zones": [{"name": "a", "polygon": [[0, 0], [10, 0], [10, 10]]}],
                      "machines": [{"name": "m", "source": "lamp", "roi": [[0, 0], [1, 0], [1, 1]], "zone": "zz", "motion_trh": 1}],
                      "inactivity": {"method": "x"}, "workpieces": {"count": "up"}}}})
    msg = str(e.value)
    for part in ("did you mean 'fps'", "outside 0..1", "did you mean 'motion_thr'", "'zz' is not a zone", "must be 'pose' or 'motion'",
                 "workpieces.line: required", "must be 'in', 'out' or 'both'"):
        assert part in msg


def test_config_accepts_legacy_machine_and_flags():
    cfg = {"cameras": {"c": {"machine": {"polygon": [[0, 0], [1, 0], [1, 1]], "cycle": True}, "gathering": True, "inactivity": False}}}
    check_config(cfg)
    cam = normalize_camera(cfg["cameras"]["c"])
    assert cam["machines"][0]["name"] == "machine" and cam["machines"][0]["roi"] and cam["machines"][0]["cycle"] == {}
    assert cam["gathering"] == {} and "inactivity" not in cam


# ---------- report metrics ----------
def test_absence_outside_breaks_only():
    from videoanalytics.metrics import absence_stats
    s = Schedule(dict(work=[["08:00", "12:00"]], breaks=[["10:00", "10:15"]]))
    m0 = minute(2026, 10, 1, 8)
    rows = []
    for k in range(240):  # 08:00-12:00, one row per minute
        m = m0 + k
        occupied = 0.0 if (30 <= k < 45 or 120 <= k < 135 or 200 <= k < 202) else 60.0  # 08:30-08:45, 10:00-10:15 (break), 11:20-11:22
        rows.append((m, 60.0, occupied))
    daily, runs = absence_stats(rows, s)
    d = daily["2026-10-01"]
    assert d["expected_s"] == 225 * 60  # 240 minutes minus the 15 minute break
    assert d["absent_s"] == (15 + 2) * 60 and d["present_s"] == (225 - 17) * 60
    assert [(r[0] - m0, r[1]) for r in runs] == [(30, 15)]  # the 2 minute gap is below min_run, the break is not counted at all


def test_idle_runs_span_file_boundaries():
    from videoanalytics.metrics import idle_runs
    rows = [(10, 0.0, 60.0), (11, 0.0, 60.0), (12, 0.0, 60.0), (13, 0.0, 60.0), (14, 0.0, 60.0), (15, 30.0, 30.0),
            (16, 0.0, 60.0), (17, 0.0, 60.0)]
    assert idle_runs(rows, min_idle_s=300) == [(10, 300.0)]
    assert idle_runs(rows, min_idle_s=100) == [(10, 300.0), (16, 120.0)]


def test_example_config_is_valid():
    import pathlib
    import yaml
    path = pathlib.Path(__file__).resolve().parent.parent / "cameras.example.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    check_config(cfg)
    cams = {k: normalize_camera(v) for k, v in cfg["cameras"].items()}
    assert [m["source"] for m in cams["cam1"]["machines"]] == ["lamp"] and cams["cam2"]["machines"][0]["cycle"]["window_s"] == 2
    assert "workpieces" in cams["cam3"] and cams["cam4"]["machines"] == []
