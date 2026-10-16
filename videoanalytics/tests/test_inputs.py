"""Recordings as they come out of a camera: odd names, no time in the name, recorder formats, non-ASCII folders."""
import os
import shutil
import subprocess
from datetime import datetime

import cv2
import pytest

from tests.test_features import make_video
from videoanalytics import video
from videoanalytics import sources
from videoanalytics.analyze import plan_tasks
from videoanalytics.autoconfig import sample_windows
from videoanalytics.config import DEFAULTS, normalize_camera
from videoanalytics.sources import VIDEO_EXT, assign_names, discover, discover_auto, list_videos, naive_epoch, parse_start_arg, probe, probe_full


def clip(path, seconds=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    make_video(path, seconds, lambda f, t: None)
    return path


@pytest.mark.parametrize("text, expected", [
    ("2026-10-01 08:30", datetime(2026, 10, 1, 8, 30)),
    ("2026-10-01 08:30:15", datetime(2026, 10, 1, 8, 30, 15)),
    ("2026-10-01T08:30:15", datetime(2026, 10, 1, 8, 30, 15)),
    ("2026/10/1 8:05", datetime(2026, 10, 1, 8, 5)),
    ("2026-10-01", datetime(2026, 10, 1)),
    ("  2026.10.01_08-30-00 ", datetime(2026, 10, 1, 8, 30)),
])
def test_start_time_argument(text, expected):
    assert parse_start_arg(text) == naive_epoch(expected)


@pytest.mark.parametrize("text", ["tomorrow", "", "08:30", "2026-13-01 08:00", "2026-10-01 25:00"])
def test_start_time_argument_errors_say_what_to_write(text):
    with pytest.raises(ValueError, match="start time|valid date"):
        parse_start_arg(text)


def test_files_without_a_time_follow_each_other_from_the_given_start(tmp_path):
    for name, sec in (("part1", 4), ("part2", 6), ("part10", 5)):
        clip(tmp_path / "hall" / f"{name}.mp4", sec)
    clip(tmp_path / "hall" / "cam1_20261002_090000.mp4", 5)  # has its own time: --start must not touch it
    start = parse_start_arg("2026-10-01 08:00")
    srcs, issues, _ = discover_auto(str(tmp_path), None, None, [], start=start, one_camera="")
    by = {s["rel"].split(os.sep)[-1]: s["start_ts"] for s in srcs}
    assert by["part1.mp4"] == start
    assert by["part2.mp4"] == pytest.approx(start + 4, abs=0.01)  # abs: epoch seconds are ~1.8e9, a relative tolerance would be half an hour
    assert by["part10.mp4"] == pytest.approx(start + 10, abs=0.01)  # natural order: part2 comes before part10
    assert by["cam1_20261002_090000.mp4"] == naive_epoch(datetime(2026, 10, 2, 9, 0))
    assert sorted(issues["manual_time"]) == [os.path.join("hall", f"{n}.mp4") for n in ("part1", "part10", "part2")]
    assert issues["no_timestamp"] == []  # the file time is not used when a start was given


def test_each_camera_starts_from_the_given_time(tmp_path):
    clip(tmp_path / "cam1" / "video.mp4")
    clip(tmp_path / "cam2" / "video.mp4")
    start = parse_start_arg("2026-10-01 08:00")
    srcs, _, cams = discover_auto(str(tmp_path), start=start)
    assert sorted(cams) == ["cam1", "cam2"] and [s["start_ts"] for s in srcs] == [start, start]  # per camera, not one timeline for all
    srcs, _ = discover(str(tmp_path), {"cam1": {}, "cam2": {}}, start=start)
    assert [s["start_ts"] for s in srcs] == [start, start]


@pytest.mark.parametrize("rels, names", [
    # the folder is the camera when generic file names repeat in different camera folders
    (["cam1/video.mp4", "cam2/video.mp4"], {"cam1/video.mp4": "cam1", "cam2/video.mp4": "cam2"}),
    (["cam1/30.00.mp4", "cam1/31.00.mp4", "cam2/30.00.mp4"], {"cam1/30.00.mp4": "cam1", "cam1/31.00.mp4": "cam1", "cam2/30.00.mp4": "cam2"}),
    # dates are not camera folders: the same name on different days is one camera
    (["2026-10-01/a.mp4", "2026-10-02/a.mp4"], {"2026-10-01/a.mp4": "a", "2026-10-02/a.mp4": "a"}),
    # a name with a time in it is a real camera name, whatever the export folders are called
    (["export_A/cam1_20261001_080000.mp4", "export_B/cam1_20261001_090000.mp4"],
     {"export_A/cam1_20261001_080000.mp4": "cam1", "export_B/cam1_20261001_090000.mp4": "cam1"}),
    # numbered names stay separate (they may be 6 cameras); --one-camera is the answer when they are parts of one
    (["video1.mp4", "video2.mp4"], {"video1.mp4": "video1", "video2.mp4": "video2"}),
])
def test_names_of_cameras_in_typical_exports(rels, names):
    assert assign_names(rels) == names


def test_one_camera_collects_every_file(tmp_path):
    for n in ("video1", "video2", "clip_a"):
        clip(tmp_path / "Lathe 3 downloads" / f"{n}.mp4", 3)
    folder = str(tmp_path / "Lathe 3 downloads")
    srcs, issues, cams = discover_auto(folder, one_camera="")
    assert {s["camera"] for s in srcs} == {"Lathe_3_downloads"} and cams == {"Lathe_3_downloads": {"match": ".*"}}  # named after the folder
    srcs, _, cams = discover_auto(folder, one_camera="Lathe 3")
    assert {s["camera"] for s in srcs} == {"Lathe_3"} and list(cams) == ["Lathe_3"]
    assert [s["rel"].split(os.sep)[-1] for s in sorted(srcs, key=lambda s: s["rel"])] == ["clip_a.mp4", "video1.mp4", "video2.mp4"]


def test_files_are_listed_in_natural_order(tmp_path):
    for n in ("part10", "part2", "part1", "Part3"):
        clip(tmp_path / f"{n}.mp4", 2)
    assert [p.name for p in list_videos(tmp_path)] == ["part1.mp4", "part2.mp4", "Part3.mp4", "part10.mp4"]


def test_without_a_start_the_file_time_is_used_and_reported(tmp_path):
    clip(tmp_path / "plain.mp4", 4)
    srcs, issues, _ = discover_auto(str(tmp_path))
    assert issues["no_timestamp"] == ["plain.mp4"] and issues["manual_time"] == []
    assert srcs[0]["start_ts"] == pytest.approx(naive_epoch(datetime.fromtimestamp(os.path.getmtime(tmp_path / "plain.mp4"))) - srcs[0]["duration"], abs=0.01)


def test_recorder_formats_are_not_silently_ignored(tmp_path):
    (tmp_path / "ch01_20261001_080000.dav").write_bytes(os.urandom(5000))  # a format FFmpeg cannot read here
    clip(tmp_path / "ch02_20261001_080000.mp4")
    assert {".dav", ".h264", ".264", ".h265", ".265", ".webm", ".3gp"} <= VIDEO_EXT
    srcs, issues, _ = discover_auto(str(tmp_path))
    assert [s["camera"] for s in srcs] == ["ch02"] and issues["unreadable"] == ["ch01_20261001_080000.dav"]  # reported, so it can be explained


def test_non_ascii_folder_and_file_names_open(tmp_path):
    path = clip(tmp_path / "Տեսախցիկ 1" / "Камера_20261001_080000.mp4", 4)
    info = probe(str(path))
    assert info is not None and info[1] == pytest.approx(4, abs=0.5)
    srcs, issues, _ = discover_auto(str(tmp_path))
    assert srcs[0]["camera"] == "Камера" and issues["unreadable"] == []


def test_ascii_path_fallback_only_acts_on_windows(tmp_path, monkeypatch):
    p = str(tmp_path / "Камера.mp4")
    assert video._ascii_path(p) == p  # not Windows: unchanged
    monkeypatch.setattr(video.os, "name", "nt")
    assert video._ascii_path("plain.mp4") == "plain.mp4"  # ASCII names never need it
    assert video._ascii_path(p) == p  # no Windows API here: unchanged instead of raising


def test_open_capture_tries_the_short_path_when_the_first_attempt_fails(tmp_path, monkeypatch):
    real = clip(tmp_path / "x.mp4", 3)
    wanted = str(tmp_path / "Камера.mp4")  # does not exist under this name; "the short form" points to the real file
    monkeypatch.setattr(video, "_ascii_path", lambda path: str(real) if path == wanted else path)
    assert video.open_capture(wanted).isOpened()
    assert not video.open_capture(str(tmp_path / "missing.mp4")).isOpened()


# ---- files that do not say how long they are (raw streams) -----------------------------------------------------------------
class FakeCap:
    def __init__(self, frames, fps=25.0):
        self.values = {cv2.CAP_PROP_FRAME_COUNT: frames, cv2.CAP_PROP_FPS: fps}

    def isOpened(self):
        return True

    def grab(self):
        return True

    def get(self, prop):
        return self.values.get(prop, 0.0)

    def release(self):
        pass


def test_a_file_without_a_length_is_marked_and_given_an_hour(monkeypatch):
    monkeypatch.setattr(sources, "open_capture", lambda path: FakeCap(0))
    assert probe_full("raw.h264") == dict(fps=25.0, duration=3600.0, known=False)
    monkeypatch.setattr(sources, "open_capture", lambda path: FakeCap(500, 10.0))
    assert probe_full("clip.mp4") == dict(fps=10.0, duration=50.0, known=True) and probe("clip.mp4") == (10.0, 50.0)


def fake_src(known, rel="cam1/a.mp4"):
    return dict(camera="cam1", path=rel, rel=rel, start_ts=1790841600.0, duration=3600.0, fps=25.0, duration_known=known)


def test_a_file_of_unknown_length_is_not_cut_into_chunks(tmp_path):
    opts = {**DEFAULTS, "clips": False}
    cams = {"cam1": normalize_camera({})}
    assert len(plan_tasks([fake_src(True)], cams, opts, tmp_path, 1200.0)[0]) == 3  # a known hour: three chunks of 20 minutes
    (task,) = plan_tasks([fake_src(False)], cams, opts, tmp_path, 1200.0)[0]  # an unknown length: one pass from start to end, no repeated frames
    assert "seg" not in task["src"]


def test_scene_learning_trusts_an_unknown_length_for_its_first_minute_only():
    windows = sample_windows([fake_src(False)], total_s=360.0, k=6)
    assert windows and all(start + length <= 60.0 + 1e-6 for _, start, length in windows)
    known = sample_windows([fake_src(True)], total_s=360.0, k=6)
    assert max(start for _, start, _ in known) > 600  # a known hour is sampled all over


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg to write a raw stream")
def test_a_raw_h264_stream_is_read_once(tmp_path):
    src = clip(tmp_path / "in" / "walk.mp4", 20)
    raw = tmp_path / "cam" / "stream.h264"
    raw.parent.mkdir()
    done = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:v", "libx264", "-an", "-f", "h264", str(raw)], capture_output=True)
    if done.returncode != 0:
        pytest.skip("this ffmpeg has no H.264 encoder")
    (s,), issues, _ = discover_auto(str(raw.parent))
    assert s["duration_known"] is False and s["duration"] == 3600.0 and issues["unreadable"] == []
