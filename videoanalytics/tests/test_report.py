import csv
import sqlite3
from datetime import datetime

import pytest

from videoanalytics.report import write_reports
from videoanalytics.sources import naive_epoch
from videoanalytics.storage import merge_shards, write_shard

T0 = naive_epoch(datetime(2026, 10, 1, 8, 0))
M0 = int(T0 // 60)


def shard(out, name, events=(), minutes=None, zones=None, machines=None, cam="cam1"):
    write_shard(out / "shards" / cam / f"{name}.db", list(events),
                dict(minutes=minutes or {}, zone_minutes=zones or {}, machine_minutes=machines or {}),
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
    assert "Work without operator" in html and ">50<" in html.replace(" ", "")  # 100 - 50 minutes unattended
    (idle,) = rows(tmp_path / "idle_periods.csv")
    assert (idle["machine"], idle["start"], idle["duration_min"], idle["clip"]) == ("lathe", "2026-10-01 09:40:00", "140.0", "clips/cam1/x_idle.mp4")
    assert "completed cycles" in html and 'src="clips/cam1/x_idle.jpg"' in html


def test_incident_tables_and_people_counts(tmp_path):
    html = build_report(tmp_path)
    assert "<h2>Gatherings</h2>" in html and "4.0 min" in html and 'src="clips/cam1/g.jpg"' in html
    assert "<h2>People without movement</h2>" in html and "7.0 min" in html and "seated" in html
    h = {(r["date"], r["hour"]): r for r in rows(tmp_path / "hourly.csv")}
    assert h[("2026-10-01", "09")]["gatherings"] == "1" and h[("2026-10-01", "10")]["inactive_people"] == "1" and h[("2026-10-01", "10")]["workpieces"] == "2"
