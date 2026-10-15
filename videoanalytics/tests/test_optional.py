"""Inactivity (motion + pose helpers) and workpiece counting. Tests needing optional packages skip when they are missing."""
import numpy as np
import pytest

from videoanalytics.inactivity import InactivityAnalyzer, landmark_activity, posture
from videoanalytics.pipeline import process_video
from tests.test_features import make_video, opts, src, people_for

LEFT, RIGHT = (40, 100, 70, 180, 0.9), (230, 100, 260, 180, 0.9)
ZONES = [dict(name="left", polygon=[[0, 0], [0.5, 0], [0.5, 1], [0, 1]]), dict(name="right", polygon=[[0.5, 0], [1, 0], [1, 1], [0.5, 1]])]


def draw_active_right(f, t):
    y = 110 + int(t * 15) % 50  # a bright patch moving inside the right person's box
    f[y:y + 10, 238:252] = 255


def inactive_events(tmp_path, zones, **cfg):
    v = tmp_path / "i.mp4"
    make_video(v, 15, draw_active_right)
    cam = dict(zones=ZONES, inactivity=dict(method="motion", min_duration_s=5, gap_s=2) | cfg)
    if zones is not None:
        cam["inactivity"]["zones"] = zones
    events, _, _ = process_video(src(v, 15), cam, opts(), people_for(15, [LEFT, RIGHT]))
    return [e for e in events if e[1] == "inactive"]


def test_still_person_is_reported_active_person_is_not(tmp_path):
    (e,) = inactive_events(tmp_path, ["left"])
    assert e[3].startswith("duration=") and "zone=left" in e[3] and 9 <= int(e[3].split("=")[1].split("s")[0]) <= 14
    assert inactive_events(tmp_path, ["right"]) == []  # the person on the right keeps moving


def test_without_zones_every_person_is_watched(tmp_path):
    assert len(inactive_events(tmp_path, None)) == 1


def test_short_stillness_is_not_reported(tmp_path):
    assert inactive_events(tmp_path, ["left"], min_duration_s=60) == []


def test_inactivity_evidence_files(tmp_path):
    v = tmp_path / "e.mp4"
    make_video(v, 15, draw_active_right)
    out = tmp_path / "out"
    cam = dict(zones=ZONES, inactivity=dict(method="motion", min_duration_s=5, gap_s=2, zones=["left"]))
    events, _, _ = process_video(src(v, 15), cam, opts(clips=True), people_for(15, [LEFT, RIGHT]), out)
    (e,) = [e for e in events if e[1] == "inactive"]
    assert (out / e[4]).stat().st_size > 0 and (out / e[5]).exists()


def test_pose_method_needs_a_backend():
    with pytest.raises(RuntimeError, match="PoseBackend"):
        InactivityAnalyzer(dict(method="pose"), [], None)


def lm(**moves):
    a = np.zeros((33, 3), np.float32)
    a[:, 2] = 1.0
    for i, (dx, dy) in moves.items():
        a[int(i[1:])] += (dx, dy, 0)
    return a


def test_landmark_activity_uses_the_most_moving_points():
    still, typing = lm(), lm(L15=(10, 0), L16=(10, 0))  # both wrists move 10 px, person is 100 px tall
    assert landmark_activity(still, typing, 100.0) == pytest.approx(0.05)  # (0.1 + 0.1 + 0 + 0) / 4
    assert landmark_activity(still, still, 100.0) == 0.0
    hidden = still.copy()
    hidden[:, 2] = 0.1
    assert landmark_activity(still, hidden, 100.0) is None  # nothing visible in both frames


def test_posture_from_knee_and_hip_height():
    def body(knee_y, vis=1.0):
        a = np.zeros((33, 3), np.float32)
        a[[11, 12], 1], a[[23, 24], 1], a[[25, 26], 1] = 100, 200, knee_y
        a[:, 2] = 1.0
        a[[25, 26], 2] = vis
        return a
    assert posture(body(300)) == "standing" and posture(body(210)) == "seated" and posture(body(300, vis=0.1)) is None


def test_pose_backend_reports_missing_model(tmp_path):
    pytest.importorskip("mediapipe")
    from videoanalytics.inactivity import PoseBackend
    with pytest.raises(FileNotFoundError, match="models --pose"):
        PoseBackend(str(tmp_path / "missing.task"))


# ---------- workpieces ----------
sv = pytest.importorskip("supervision")
LINE = dict(p1=[0.5, 0.0], p2=[0.5, 1.0])  # vertical, top -> bottom: left -> right is "in"


def moving_part(direction):
    n = [0]

    def detect(frame):
        n[0] += 1
        x = 20 + 20 * n[0] / 5 if direction > 0 else 300 - 20 * n[0] / 5
        return [(x - 10, 90, x + 10, 110, 0.9)] if 10 < x < 310 else []
    return detect


def count_events(tmp_path, direction, count):
    v = tmp_path / "w.mp4"
    make_video(v, 14, lambda f, t: None)
    cam = dict(workpieces=dict(line=LINE, count=count), fps=5)
    events, _, _ = process_video(src(v, 14), cam, opts(), lambda f: [], workpiece_detect=moving_part(direction))
    return [(e[2], e[3]) for e in events if e[1] == "workpiece"]


def test_workpiece_crossing_directions(tmp_path):
    assert count_events(tmp_path, +1, "in") == [("workpiece_line", "in")]
    assert count_events(tmp_path, +1, "out") == []
    assert count_events(tmp_path, -1, "out") == [("workpiece_line", "out")]
    assert count_events(tmp_path, -1, "both") == [("workpiece_line", "out")]


def test_workpiece_needs_a_detector(tmp_path):
    v = tmp_path / "n.mp4"
    make_video(v, 3, lambda f, t: None)
    with pytest.raises(RuntimeError, match="no detector"):
        process_video(src(v, 3), dict(workpieces=dict(line=LINE)), opts(), lambda f: [])


# ---------- detector evaluation (guards the choice of checkpoint after training) ----------
def test_match_boxes_is_one_to_one_and_threshold_aware():
    from videoanalytics.workpieces import match_boxes
    gt = [(0, 0, 10, 10), (50, 50, 60, 60)]
    assert match_boxes(gt, [(0, 0, 10, 10, 0.9), (50, 50, 60, 60, 0.8)]) == (2, 0, 0)
    assert match_boxes(gt, [(0, 0, 10, 10, 0.9), (1, 1, 10, 10, 0.7)]) == (1, 1, 1)  # second box would reuse the same part
    assert match_boxes(gt, [(30, 30, 40, 40, 0.9)]) == (0, 1, 2)
    assert match_boxes(gt, [(0, 0, 4, 4, 0.9)]) == (0, 1, 2)  # IoU 0.16 < 0.5


def test_evaluate_detector_on_a_coco_split(tmp_path):
    import json
    import cv2
    from tests.test_tools import coco
    from videoanalytics.tools import split_coco
    from videoanalytics.workpieces import evaluate_detector
    split_coco(str(tmp_path / "img"), str(coco(tmp_path, 12)), str(tmp_path / "ds"), valid=0.5, test=0.0)
    truth = {im["file_name"]: [a["bbox"] for a in json.loads((tmp_path / "ds/valid/_annotations.coco.json").read_text())["annotations"] if a["image_id"] == im["id"]]
             for im in json.loads((tmp_path / "ds/valid/_annotations.coco.json").read_text())["images"]}
    perfect = lambda frame: [(1, 2, 11, 12, 0.9)]  # every annotated part in the fixture is at x=1,y=2,w=10,h=10
    nothing = lambda frame: []
    noisy = lambda frame: [(1, 2, 11, 12, 0.9), (30, 20, 40, 30, 0.8)]
    assert len(truth) == 6
    r = evaluate_detector(perfect, str(tmp_path / "ds"))
    assert (r["images"], r["parts"], r["recall"], r["precision"]) == (6, 6, 1.0, 1.0)
    assert evaluate_detector(nothing, str(tmp_path / "ds"))["recall"] == 0.0
    r = evaluate_detector(noisy, str(tmp_path / "ds"))
    assert r["recall"] == 1.0 and r["precision"] == 0.5
