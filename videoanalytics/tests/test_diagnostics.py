"""The pre-flight check and diagnostics.txt: what they report about recordings, and that they explain problems in plain words."""
import os
import sqlite3
from datetime import datetime

import cv2
import numpy as np
import pytest

from tests.test_features import make_video
from videoanalytics import diagnostics as dg
from videoanalytics.analyze import doctor
from videoanalytics.cli import main
from videoanalytics.sources import discover_auto, naive_epoch

STILL = lambda frame: [(40, 90, 70, 170, 0.9)]


def shapes_a(f, t):  # one scene ...
    cv2.rectangle(f, (30, 30), (120, 100), (200, 200, 200), -1)
    cv2.circle(f, (240, 120), 30, (60, 60, 60), -1)


def shapes_b(f, t):  # ... and a completely different one
    cv2.line(f, (0, 170), (319, 10), (255, 255, 255), 6)
    cv2.rectangle(f, (200, 20), (300, 60), (30, 30, 30), -1)


def grey_with_small_lamp(f, t):
    cv2.rectangle(f, (144, 72), (160, 96), (0, 200, 0), -1)


def clip(path, seconds=4, draw=shapes_a):
    path.parent.mkdir(parents=True, exist_ok=True)
    make_video(path, seconds, draw)
    return path


def srcs_of(root, **kw):
    srcs, issues, _ = discover_auto(str(root), **kw)
    return srcs, issues


# ---- one recording ------------------------------------------------------------------------------------------------------
def test_fourcc_is_readable():
    assert dg.fourcc_text(cv2.VideoWriter_fourcc(*"mp4v")) == "mp4v"
    assert dg.fourcc_text(0) == "?" and dg.fourcc_text(0x01020304).startswith("0x")


def test_inspect_file_reports_format_speed_and_picture(tmp_path):
    info = dg.inspect_file(str(clip(tmp_path / "a.mp4", 4)))
    assert info["opened"] and info["codec"] in ("mp4v", "FMP4") and (info["width"], info["height"]) == (320, 180)
    assert info["fps"] > 1 and info["decoded"] == 20 and info["decode_fps"] > 0 and info["frame_count"] > 60
    assert info["seek_ok"] and info["seek_s"] >= 0 and 0 < info["brightness"] < 255 and info["thumb"].shape == (36, 64)


def test_unreadable_recorder_file_is_explained_not_crashed(tmp_path):
    p = tmp_path / "ch01.dav"
    p.write_bytes(os.urandom(4000))
    info = dg.inspect_file(str(p))
    assert not info["opened"] and "cannot be opened" in info["error"]
    notes = dg.issue_hints({"unreadable": ["ch01.dav"]})
    assert len(notes) == 1 and "ffmpeg" in notes[0] and "ch01.dav" in notes[0]


def test_a_grey_scene_with_a_small_coloured_lamp_is_not_a_night_picture(tmp_path):
    lamp = dg.inspect_file(str(clip(tmp_path / "lamp.mp4", 4, grey_with_small_lamp)))
    grey = dg.inspect_file(str(clip(tmp_path / "grey.mp4", 4, lambda f, t: None)))
    assert lamp["coloured"] > 0.2 > grey["coloured"]  # the lamp keeps the colour readable; a flat grey picture does not
    notes = [n for n in dg.sample_hints([{**lamp, "camera": "a", "rel": "lamp.mp4"}, {**grey, "camera": "b", "rel": "grey.mp4"}]) if "no colour" in n]
    assert len(notes) == 1 and notes[0].startswith("b (grey.mp4)") and "infrared" in notes[0]


def test_sample_notes_for_odd_files():
    base = dict(opened=True, camera="c", rel="x.mp4", codec="hev1", width=1920, height=1080, fps=25.0, frame_count=1000, decode_fps=100.0,
                seek_ok=True, seek_s=0.1, coloured=20.0, brightness=100.0, decoded=20)
    assert dg.sample_hints([base]) == []
    text = " ".join(dg.sample_hints([{**base, "frame_count": 0, "fps": 0.0, "seek_ok": False, "decode_fps": 4.0, "width": 320}]))
    for fragment in ("read from its start to its end", "ffmpeg -i in -c copy out.mp4", "25 fps is assumed", "jumping into the middle", "only 4 frames per second",
                     "small picture"):
        assert fragment in text
    (raw,) = [n for n in dg.sample_hints([{**base, "rel": "cam/stream.h264", "frame_count": 0}])]
    assert "no frame rate" in raw and "ffmpeg -r 10 -i in.h264 -c copy out.mp4" in raw


def test_similarity_tells_views_apart(tmp_path):
    a1 = dg.inspect_file(str(clip(tmp_path / "a1.mp4", 3, shapes_a)))["thumb"]
    a2 = dg.inspect_file(str(clip(tmp_path / "a2.mp4", 3, shapes_a)))["thumb"]
    b = dg.inspect_file(str(clip(tmp_path / "b.mp4", 3, shapes_b)))["thumb"]
    assert dg.similarity(a1, a2) > 0.99 and dg.similarity(a1, b) < 0.5


# ---- the text -------------------------------------------------------------------------------------------------------------
def test_text_of_a_two_camera_folder(tmp_path):
    clip(tmp_path / "cam1_20261001_080000.mp4", 4, shapes_a)
    clip(tmp_path / "cam1_20261001_081000.mp4", 4, shapes_a)  # a gap of almost 10 minutes after the first file
    clip(tmp_path / "lathe_20261001_080000.mp4", 4, shapes_b)
    srcs, issues = srcs_of(tmp_path)
    text = dg.doctor_text(srcs, issues, paths=[tmp_path], cost=0.25, workers=2)
    for part in ("== Computer", "OpenCV", "== Recordings", "3 readable video file(s)", "== Samples", "== Speed on this computer", "0.25 CPU-seconds",
                 "== What to look at"):
        assert part in text
    listed = text.split("== What to look at")[1].strip().splitlines()
    assert listed and all("small picture" in n or "no colour" in n for n in listed)  # the test videos are small and grey; nothing else is worth a note
    assert "12 s of video on 2 worker(s): about 2 s" in text  # 12 s x 0.25 CPU-s per video-s / 2 workers = 1.5 s
    assert "cam1: 2 file(s)" in text and "lathe: 1 file(s)" in text and "1 gap(s) over 5 s" in text and "Time of the recordings: from the file name for 3" in text
    assert "no gaps" in text  # lathe
    assert str(tmp_path) not in text  # relative names only: no folders of the user's computer


def test_gaps_and_overlaps_between_recordings():
    f = lambda start, dur: dict(start_ts=float(start), duration=float(dur))
    assert dg._coverage([f(0, 100), f(100, 100), f(250, 50), f(290, 10)]) == (1, 50.0, 50.0, 1)  # 50 s gap, then a 10 s overlap
    assert dg._coverage([f(0, 100), f(103, 10)]) == (0, 0.0, 0.0, 0)  # 3 s is not a gap


def test_files_without_a_time_are_explained_and_numbered_names_questioned(tmp_path):
    for n in ("video1", "video2"):
        clip(tmp_path / f"{n}.mp4", 4, shapes_a)
    srcs, issues = srcs_of(tmp_path)
    text = " ".join(dg.notes(srcs, issues, dg.inspect_samples(srcs), [tmp_path]))
    assert "--start" in text and "when you saved them" in text  # the file time of a download is the download time
    assert "differ only by a number" in text and "--one-camera" in text
    assert "almost the same picture" in text  # video1 and video2 show the same view: probably parts of one camera
    srcs, issues = srcs_of(tmp_path, one_camera="", start=naive_epoch(datetime(2026, 10, 1, 8, 0)))
    text = " ".join(dg.notes(srcs, issues, dg.inspect_samples(srcs), [tmp_path]))
    assert "(--start) is used" in text and "differ only by a number" not in text and "when you saved them" not in text


def test_numbered_names_with_a_time_are_not_questioned(tmp_path):
    for n in (1, 2):
        clip(tmp_path / f"cam{n}_20261001_080000.mp4", 4, shapes_a if n == 1 else shapes_b)
    srcs, issues = srcs_of(tmp_path)
    assert dg.name_hints(srcs, dg.inspect_samples(srcs), issues) == []  # times in the names: the cameras are told apart reliably


def test_cameras_with_the_same_view_are_one_note_not_one_per_pair(tmp_path):
    for n in range(1, 5):
        clip(tmp_path / f"cam{n}_20261001_080000.mp4", 3, shapes_a)
    clip(tmp_path / "other_20261001_080000.mp4", 3, shapes_b)
    srcs, issues = srcs_of(tmp_path)
    notes = dg.name_hints(srcs, dg.inspect_samples(srcs), issues)
    assert len(notes) == 1 and "Cameras cam1, cam2, cam3, cam4 show almost the same picture" in notes[0] and "other" not in notes[0]


def test_the_ffmpeg_noise_is_switched_off_unless_the_user_chose_a_level():
    import subprocess
    import sys
    code = "import videoanalytics, os; print(os.environ['OPENCV_FFMPEG_LOGLEVEL'], os.environ['OPENCV_LOG_LEVEL'])"
    clean = {k: v for k, v in os.environ.items() if not k.startswith("OPENCV_")}
    run = lambda env: subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout.split()
    assert run(clean) == ["-8", "ERROR"]
    assert run({**clean, "OPENCV_FFMPEG_LOGLEVEL": "16"}) == ["16", "ERROR"]  # a level set by the user is kept


def test_different_views_in_one_camera_are_pointed_out(tmp_path):
    clip(tmp_path / "cam1_20261001_080000.mp4", 4, shapes_a)
    clip(tmp_path / "cam1_20261001_090000.mp4", 4, shapes_b)  # a different picture under the same name
    srcs, issues = srcs_of(tmp_path)
    notes = dg.name_hints(srcs, dg.inspect_samples(srcs), issues)
    assert len(notes) == 1 and "first and the last file of camera cam1" in notes[0]


def test_other_file_types_are_listed(tmp_path):
    (tmp_path / "a.dav").write_bytes(b"x" * 10)  # a recorder format that is tried as a video, so it is not "other"
    (tmp_path / "x.bin").write_bytes(b"x")
    (tmp_path / "y.rec").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("hi")
    (tmp_path / "b.txt").write_text("hi")
    clip(tmp_path / "ok.mp4", 3)
    seen = dg.other_files([tmp_path])
    assert dict(seen) == {".txt": 2, ".bin": 1, ".rec": 1}
    (note,) = dg.issue_hints({}, seen)
    assert "Recorder formats (.bin, .rec)" in note and ".txt (2)" in note


# ---- doctor, CLI, results -------------------------------------------------------------------------------------------------
def test_doctor_writes_doctor_txt_and_measures_speed(tmp_path, capsys):
    clip(tmp_path / "cam1_20261001_080000.mp4", 20, shapes_a)
    out = tmp_path / "res"
    path = doctor([str(tmp_path)], out=str(out), detect=STILL, workers=1)
    text = path.read_text(encoding="utf-8")
    assert path == out / "doctor.txt" and text == text.strip() + "\n" and "== Samples" in text and "== Speed on this computer" in text
    assert "Saved as" in capsys.readouterr().out


def test_doctor_command_without_the_speed_test(tmp_path, capsys):
    v = clip(tmp_path / "x_20261001_080000.mp4", 4)
    main(["doctor", str(v), "--no-bench", "--out", str(tmp_path / "d")])
    out = capsys.readouterr().out
    assert "== Samples" in out and "== Speed" not in out and (tmp_path / "d" / "doctor.txt").exists()


def test_doctor_on_a_folder_without_videos_says_what_is_there(tmp_path):
    (tmp_path / "readme.txt").write_text("x")
    with pytest.raises(SystemExit, match=r"no video files in .*\.txt \(1\)"):
        doctor([str(tmp_path)], bench=False)


def test_result_text_summarises_the_run(tmp_path):
    con = sqlite3.connect(tmp_path / "analytics.db")
    con.executescript("""
        CREATE TABLE minutes(camera TEXT, minute INTEGER, covered_s REAL, people_s REAL, people_max INTEGER, det_frames INTEGER);
        CREATE TABLE events(camera TEXT, ts REAL, type TEXT, subject TEXT, value TEXT, clip TEXT, thumb TEXT, source TEXT);
        CREATE TABLE files(source TEXT, camera TEXT, start_ts REAL, video_s REAL, processed_s REAL, det_frames INTEGER, elapsed_s REAL);
        CREATE TABLE machine_minutes(camera TEXT, machine TEXT, minute INTEGER, work_s REAL, idle_s REAL);
        CREATE TABLE zone_minutes(camera TEXT, zone TEXT, minute INTEGER, covered_s REAL, occupied_s REAL);
        INSERT INTO minutes VALUES('cam1', 1, 60, 120, 3, 10);
        INSERT INTO events VALUES('cam1', 1, 'gathering', 'g1', '', '', '', 's'), ('cam1', 2, 'gathering', 'g2', '', '', '', 's'), ('cam1', 3, 'cycle', 'm', '', '', '', 's');
        INSERT INTO files VALUES('a.mp4', 'cam1', 0, 60, 60, 10, 15);
        INSERT INTO machine_minutes VALUES('cam1', 'lathe', 1, 45, 15);
        INSERT INTO zone_minutes VALUES('cam1', 'desk', 1, 60, 30);
    """)
    con.commit()
    con.close()
    (tmp_path / "errors.log").write_text("bad.mp4\tOSError('cannot open')\n")
    text = dg.result_text(tmp_path)
    for fragment in ("cam1: analysed 60 s, 2.00 people in view on average, peak 3", "speed of this run: 0.25 CPU-seconds per second of video",
                     "events: gathering 2, cycle 1", "machine cam1/lathe: working 75%", "station cam1/desk: occupied 50%", "bad.mp4"):
        assert fragment in text
    assert dg.result_text(tmp_path / "nothing") == "\n== Result\n"  # a folder without results: just the heading
