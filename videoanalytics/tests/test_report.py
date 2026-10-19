import csv
import sqlite3
from datetime import datetime

import pytest

from videoanalytics.autoconfig import build_camera_config
from videoanalytics.report import TL_MAX_S, _dur, _Grid, _setup_section, write_reports
from videoanalytics.sources import naive_epoch
from videoanalytics.storage import FINE_S, merge_shards, write_shard

T0 = naive_epoch(datetime(2026, 10, 1, 8, 0))
M0 = int(T0 // 60)
B0 = int(T0 // FINE_S)


def shard(out, name, events=(), minutes=None, zones=None, machines=None, cam="cam1", fine=None, zone_fine=None, machine_fine=None):
    tables = dict(minutes=minutes or {}, zone_minutes=zones or {}, machine_minutes=machines or {})
    if fine is not None:  # shards of older versions have no 5 s tables
        tables.update(fine=fine, zone_fine=zone_fine or {}, machine_fine=machine_fine or {})
    write_shard(out / "shards" / cam / f"{name}.db", list(events), tables,
                dict(camera=cam, source=name, start_ts=T0, video_s=60, processed_s=60, frames=1, det_frames=0, elapsed_s=1))


def rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def test_merge_sums_minutes_that_span_two_files(tmp_path):
    for n in ("a", "b"):
        shard(tmp_path, n, minutes={M0: [30.0, 60.0, 3, 5]}, zones={("desk", M0): [30.0, 20.0, 40.0]},
              machines={("lathe", M0): [10.0, 20.0, 5.0, 0.0, 4.0, 0.0, 1]})
    con = sqlite3.connect(merge_shards(tmp_path))
    assert con.execute("SELECT covered_s, people_s, people_max, det_frames FROM minutes").fetchall() == [(60.0, 120.0, 3, 10)]
    assert con.execute("SELECT covered_s, occupied_s, people_s FROM zone_minutes").fetchall() == [(60.0, 40.0, 80.0)]
    assert con.execute("SELECT work_s, idle_s, alarm_s, work_att_s, cycles FROM machine_minutes").fetchall() == [(20.0, 40.0, 10.0, 8.0, 2)]


def test_old_shards_are_refused(tmp_path):
    shard(tmp_path, "a", minutes={M0: [1.0, 0.0, 0, 0]})
    p = next((tmp_path / "shards").rglob("*.db"))
    con = sqlite3.connect(p)
    con.execute("UPDATE meta SET v='1' WHERE k='schema'")
    con.commit()
    con.close()
    with pytest.raises(RuntimeError, match="older version"):
        merge_shards(tmp_path)


CFG = dict(schedule=dict(work=[["08:00", "12:00"]], breaks=[["10:00", "10:15"]]),
           cameras=dict(cam1=dict(zones=[dict(name="desk", polygon=[[0, 0], [1, 0], [1, 1]]), dict(name="lounge", kind="rest", polygon=[[0, 0], [1, 0], [1, 1]])],
                                  machines=[dict(name="lathe", source="lamp", roi=[[0, 0], [1, 0], [1, 1]], zone="desk")])))


def build_report(tmp_path, cfg=CFG):
    zones = {}
    for k in range(240):  # 08:00-12:00 minute by minute
        away = 30 <= k < 45 or 120 <= k < 135 or 200 <= k < 202  # 08:30-08:45, 10:00-10:15 (break), 11:20-11:22
        zones[("desk", M0 + k)] = [60.0, 0.0 if away else 60.0, 0.0 if away else 60.0]
        zones[("lounge", M0 + k)] = [60.0, 60.0 if 120 <= k < 135 else 0.0, 0.0]
    machines = {("lathe", M0 + k): [60.0 if k < 100 else 0.0, 0.0 if k < 100 else 60.0, 0.0, 0.0, 60.0 if k < 50 else 0.0, 0.0, 2 if k < 100 else 0]
                for k in range(240)}  # works for 100 min (operator there for the first 50), idle afterwards
    ev = [(T0 + 100 * 60 + 5, "machine_state", "lathe", "idle", "clips/cam1/x_idle.mp4", "clips/cam1/x_idle.jpg"),
          (T0 + 3600, "gathering", "g1", "people=3 duration=240s", "clips/cam1/g.mp4", "clips/cam1/g.jpg"),
          (T0 + 7200, "inactive", "t4", "duration=420s posture=seated zone=desk", "", ""),
          (T0 + 7300, "workpiece", "exit", "out", "", ""), (T0 + 7310, "workpiece", "exit", "out", "", "")]
    shard(tmp_path, "day", ev, {M0 + k: [60.0, 60.0, 1, 1] for k in range(240)}, zones, machines)
    merge_shards(tmp_path)
    return write_reports(tmp_path, idle_min_s=300, cfg=cfg).read_text(encoding="utf-8")


def test_stations_absence_follows_the_schedule(tmp_path):
    html = build_report(tmp_path)
    (d,) = [r for r in rows(tmp_path / "stations_daily.csv") if r["zone"] == "desk"]
    assert (d["at_station_min"], d["expected_min"], d["absent_min"], d["absence_pct"]) == ("208.0", "225.0", "17.0", "7.6")  # break not counted
    assert [(r["start"], r["duration_min"]) for r in rows(tmp_path / "absences.csv")] == [("2026-10-01 08:30:00", "15")]
    (lounge,) = [r for r in rows(tmp_path / "stations_daily.csv") if r["zone"] == "lounge"]
    assert lounge["at_station_min"] == "15.0"  # time spent in the rest zone is reported too
    assert "Longest absences from a station" in html and "Stations" in html


def test_without_config_there_is_no_absence_only_presence(tmp_path):
    build_report(tmp_path, cfg=None)
    d = [r for r in rows(tmp_path / "stations_daily.csv") if r["zone"] == "desk"][0]
    assert d["at_station_min"] == "208.0" and d["expected_min"] == "" and rows(tmp_path / "absences.csv") == []


def test_machines_lamp_cycles_and_unattended_time(tmp_path):
    html = build_report(tmp_path)
    hourly = rows(tmp_path / "machines_hourly.csv")
    assert sum(float(r["work_min"]) for r in hourly) == 100 and sum(int(r["cycles"]) for r in hourly) == 200
    assert sum(float(r["work_attended_min"]) for r in hourly) == 50  # operator present for half of the work
    assert "Work without operator" in html and ">50 min<" in html  # 100 - 50 minutes unattended
    assert "stayed at the machine's processing section while something moves there (activity source)" in html  # what "Working" means for a machine without a lamp
    (idle,) = rows(tmp_path / "idle_periods.csv")
    assert (idle["machine"], idle["start"], idle["duration_min"], idle["clip"]) == ("lathe", "2026-10-01 09:40:00", "140.0", "clips/cam1/x_idle.mp4")
    assert "completed cycles" in html and 'src="clips/cam1/x_idle.jpg"' in html


def test_incident_tables_and_people_counts(tmp_path):
    html = build_report(tmp_path)
    assert "<h2>Gatherings</h2>" in html and "4.0 min" in html and 'src="clips/cam1/g.jpg"' in html
    assert "<h2>People without movement</h2>" in html and "7.0 min" in html and "seated" in html
    h = {(r["date"], r["hour"]): r for r in rows(tmp_path / "hourly.csv")}
    assert h[("2026-10-01", "09")]["gatherings"] == "1" and h[("2026-10-01", "10")]["inactive_people"] == "1" and h[("2026-10-01", "10")]["workpieces"] == "2"


# ---- timelines ------------------------------------------------------------------------------------------------------------
def clip_80s(tmp_path, cam="cam1", n=16, **kw):
    """80 s of video at 5 s resolution: 2 people for the first 40 s, then 1; the desk is occupied all the time, the lathe works for 30 s."""
    fine = {B0 + k: [5.0, 5.0 * (2 if k < 8 else 1), 2 if k < 8 else 1] for k in range(n)}
    zf = {("desk", B0 + k): [5.0, 5.0] for k in range(n)}
    mf = {("lathe", B0 + k): [5.0, 0.0] if k < 6 else [0.0, 5.0] for k in range(n)}
    shard(tmp_path, "clip", minutes={M0: [60.0, 100.0, 2, 120], M0 + 1: [20.0, 20.0, 1, 40]}, cam=cam,
          zones={("desk", M0): [60.0, 60.0, 60.0], ("desk", M0 + 1): [20.0, 20.0, 20.0]},
          machines={("lathe", M0): [30.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0], ("lathe", M0 + 1): [0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0]},
          fine=fine, zone_fine=zf, machine_fine=mf, **kw)
    merge_shards(tmp_path)
    return write_reports(tmp_path, cfg=dict(cameras={cam: dict(zones=[dict(name="desk", polygon=[[0, 0], [1, 0], [1, 1]])],
                                                               machines=[dict(name="lathe", source="motion", roi=[[0, 0], [1, 0], [1, 1]])])})
                         ).read_text(encoding="utf-8")


def chart_values(html, nth=0):
    import html as h
    import json
    import re
    return json.loads(h.unescape(re.findall(r'<svg class="tl"[^>]*? data-v="([^"]*)"', html)[nth]))


def test_merge_sums_5s_bins_and_old_shards_still_merge(tmp_path):
    for n in ("a", "b"):
        shard(tmp_path, n, minutes={M0: [30.0, 60.0, 3, 5]}, fine={B0: [5.0, 10.0, 2]}, zone_fine={("desk", B0): [5.0, 4.0]},
              machine_fine={("lathe", B0): [3.0, 2.0]})
    shard(tmp_path, "old", minutes={M0: [30.0, 60.0, 3, 5]})  # written before the 5 s tables existed
    con = sqlite3.connect(merge_shards(tmp_path))
    assert con.execute("SELECT bin, covered_s, people_s, people_max FROM fine").fetchall() == [(B0, 10.0, 20.0, 2)]
    assert con.execute("SELECT zone, covered_s, occupied_s FROM zone_fine").fetchall() == [("desk", 10.0, 8.0)]
    assert con.execute("SELECT machine, work_s, idle_s FROM machine_fine").fetchall() == [("lathe", 6.0, 4.0)]
    assert con.execute("SELECT covered_s FROM minutes").fetchall() == [(90.0,)]


def test_short_clip_gets_a_5_second_timeline_and_no_heatmap(tmp_path):
    html = clip_80s(tmp_path)
    assert html.count('<svg class="tl"') == 3 and 'data-step="5"' in html  # people, station, machine
    assert '<table class="heat"' not in html  # the timeline shows it all
    assert chart_values(html, 0) == [2.0] * 8 + [1.0] * 8
    assert chart_values(html, 1) == [100.0] * 16  # the desk is occupied all the time
    assert chart_values(html, 2) == [100.0] * 6 + [0.0] * 10  # the lathe works for 30 s
    assert "80 s" in html and "0.0 h" not in html  # an 80 s clip is not "0.0 h"
    tl = rows(tmp_path / "timeline.csv")
    assert len(tl) == 16 and (tl[0]["time"], tl[0]["analysed_s"], tl[0]["avg_people"], tl[0]["peak_people"]) == ("2026-10-01 08:00:00", "5.0", "2.0", "2")
    assert sum(float(r["work_s"]) for r in rows(tmp_path / "machines_timeline.csv")) == 30
    assert sum(float(r["occupied_s"]) for r in rows(tmp_path / "stations_timeline.csv")) == 80


def test_timeline_csvs_of_an_earlier_run_do_not_linger(tmp_path):
    clip_80s(tmp_path)
    assert (tmp_path / "timeline.csv").exists()
    for p in (tmp_path / "shards").rglob("*.db"):
        p.unlink()
    # the same results folder now holds two days: too long for a timeline, so no (stale) timeline files may remain
    fine = {B0: [5.0, 5.0, 1], B0 + 30 * 3600 // FINE_S: [5.0, 5.0, 1]}
    shard(tmp_path, "days", minutes={M0: [5.0, 5.0, 1, 1]}, fine=fine)
    merge_shards(tmp_path)
    html = write_reports(tmp_path).read_text(encoding="utf-8")
    assert not (tmp_path / "timeline.csv").exists() and '<svg class="tl"' not in html and '<table class="heat"' in html


def test_old_shards_without_5s_tables_fall_back_to_minutes(tmp_path):
    html = build_report(tmp_path)  # four hours, per-minute rows only
    assert html.count('<svg class="tl"') == 4  # people, desk, lounge, lathe
    assert 'data-step="120"' in html and '<table class="heat"' in html  # four hours: a timeline and the heatmaps


def test_grid_step_follows_the_span():
    assert _Grid(5, [0, 75]).step == 5  # 80 s: every 5 s
    assert _Grid(5, [0, 3595]).step == 30  # one hour
    g = _Grid(5, [7, 43195])
    assert (g.step, g.t0, g.n) == (300, 0, 144)  # twelve hours: five-minute points, aligned to the clock
    assert _Grid(60, [0, 43140]).step == 300 and _Grid(60, [0, 600]).step == 60
    assert _Grid(5, [0, TL_MAX_S + 5]).span > TL_MAX_S


def test_durations_use_a_sensible_unit():
    assert [_dur(s) for s in (0, 45, 80, 100, 600, 3000, 5400, 12600, 86400)] == ["0 s", "45 s", "80 s", "1.7 min", "10 min", "50 min", "1.5 h", "3.5 h", "24.0 h"]


# ---- the setup block --------------------------------------------------------------------------------------------------------
SQUARE = lambda x, y, s=0.2: [[x, y], [x + s, y], [x + s, y + s], [x, y + s]]
LEARNED = dict(sampled_s=70, spots=[dict(total_s=164.0, polygon=SQUARE(0.1, 0.1)), dict(total_s=40.0, polygon=SQUARE(0.5, 0.5))],
               motion=[dict(freq=0.62, polygon=SQUARE(0.7, 0.1))], lamps=[dict(colours=["green", "red"], polygon=SQUARE(0.8, 0.8, 0.05))], image="setup/cam1.jpg")


def test_setup_block_of_a_pure_learning_result():
    block = _setup_section({"cam1": LEARNED})  # no config: rebuilt from what was learned
    assert "<h2>Learned setup <span class='tag'>automatic - please check</span></h2>" in block
    assert "station <b>spot1</b>: people stood here for 2.7 min in total" in block and "colours seen green, red" in block and "moving in 62% of the frames" in block
    assert "drawn by hand" not in block.split("<ul>")[1] and "learned from 70 s of footage</figcaption>" not in block  # the caption closes with the list
    assert 'src="setup/cam1.jpg"' in block and "python -m videoanalytics zones" in block
    assert "has no lamp is not found by itself: draw its <b>processing section</b>" in block  # a machine people work on has to be drawn


def test_setup_block_of_zones_drawn_by_hand_only():
    cfg = dict(cameras=dict(cam1=dict(zones=[dict(name="press", polygon=SQUARE(0.1, 0.1), kind="workstation"), dict(name="break", polygon=SQUARE(0.5, 0.5), kind="rest")],
                                      machines=[dict(name="m1", source="motion", roi=SQUARE(0.7, 0.1), zone="press"),
                                                dict(name="work1", source="activity", roi=SQUARE(0.4, 0.6, 0.15)), dict(name="l1", source="lamp", roi=SQUARE(0.9, 0.1, 0.05))],
                                      areas=[dict(name="floor", polygon=SQUARE(0, 0, 1), mode="watch"), dict(name="tv", polygon=SQUARE(0.8, 0.8, 0.1), mode="ignore")])))
    block = _setup_section({}, cfg)
    assert "<h2>Setup <span class='tag'>learned automatically and / or drawn by hand - please check</span></h2>" in block
    assert "<b>cam1</b> · drawn by hand" in block
    for fragment in ("station <b>press</b>", "rest zone <b>break</b>", "moving machine zone <b>m1</b> (linked to press)", "tracking area <b>floor</b>: people are followed only in here",
                     "ignored area <b>tv</b>: people are not analysed in here", "processing section (work by people) <b>work1</b>", "signal lamp <b>l1</b>"):
        assert fragment in block
    assert block.count("(drawn by hand)") == 7 and "a processing section where people work" in block  # the legend knows the new kind


def test_a_processing_section_is_never_taken_for_a_learned_moving_part():
    cam = dict(zones=[], machines=[dict(name="work1", source="activity", roi=LEARNED["motion"][0]["polygon"])])  # drawn over the very place where movement was learned
    block = _setup_section({"cam1": LEARNED}, dict(cameras=dict(cam1=cam)))
    assert "processing section (work by people) <b>work1</b> <span class=tag>(drawn by hand)</span>" in block and "moving in 62%" not in block


def test_setup_block_after_editing_what_was_learned():
    learned_cfg = build_camera_config(LEARNED["spots"], LEARNED["motion"], LEARNED["lamps"])
    cam = dict(learned_cfg)
    cam["zones"] = [learned_cfg["zones"][0], dict(name="bench", polygon=SQUARE(0.3, 0.7), kind="workstation")]  # spot2 deleted, a bench drawn
    cam["machines"] = [{k: v for k, v in m.items() if k != "zone"} for m in learned_cfg["machines"]]  # (they were linked to spot2)
    block = _setup_section({"cam1": LEARNED}, dict(cameras=dict(cam1=cam)))
    assert "learned from 70 s of footage, then changed by hand" in block
    assert "station <b>spot1</b>: people stood here for 2.7 min in total" in block  # still as it was found: the learning's numbers stay
    assert "station <b>bench</b> <span class=tag>(drawn by hand)</span>" in block
    assert "spot2" not in block  # the deleted one is gone from the list


def test_a_moved_zone_loses_the_learning_numbers():
    learned_cfg = build_camera_config(LEARNED["spots"], [], [])
    learned_cfg["zones"][0] = dict(learned_cfg["zones"][0], polygon=SQUARE(0.12, 0.1))  # moved by 2 % of the picture
    block = _setup_section({"cam1": LEARNED}, dict(cameras=dict(cam1=learned_cfg)))
    assert "people stood here for 2.7 min" not in block and "people stood here for 40 s" in block  # spot1 moved: its numbers no longer apply; spot2 is untouched
