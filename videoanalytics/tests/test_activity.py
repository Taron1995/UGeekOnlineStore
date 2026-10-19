"""Machines without a lamp: work = a person has been at the processing section for a while and something moves there."""
import cv2
import pytest

from tests.test_features import make_video, opts, src
from videoanalytics.config import ConfigError, check_config
from videoanalytics.pipeline import process_video

SECTION = [[0.55, 0.0], [1.0, 0.0], [1.0, 0.7], [0.55, 0.7]]  # x 176..320, y 0..126 of a 320x180 picture
AT_SECTION, ELSEWHERE = (240, 40, 270, 140), (20, 60, 50, 150)  # a person standing at the section / far from it


def moving(a, b):
    """Something in the section moves (a white block sliding to and fro) between a and b seconds."""
    def draw(f, t):
        if a <= t < b:
            x = 190 + int(t * 50) % 60
            cv2.rectangle(f, (x, 20), (x + 30, 70), (255, 255, 255), -1)
    return draw


def person(a, b, box):
    n = [0]

    def detect(frame):
        n[0] += 1
        return [(*box, 0.9)] if a <= n[0] / 5 < b else []  # opts() processes 5 frames per second
    return detect


def machine(tmp_path, seconds, draw, detect, source="activity", **kw):
    v = tmp_path / "m.mp4"
    make_video(v, seconds, draw)
    cam = dict(machines=[dict(name="press", source=source, roi=SECTION, **{"window_s": 4, **({"min_present_s": 5} if source == "activity" else {}), **kw})])
    events, tables, _ = process_video(src(v, seconds), cam, opts(), detect)
    (row,) = [r for (m, _), r in tables["machine_minutes"].items() if m == "press"]
    return dict(work=row[0], idle=row[1]), [e for e in events if e[1] == "machine_state"]


def test_a_person_who_stays_and_works_makes_the_machine_work(tmp_path):
    got, events = machine(tmp_path, 40, moving(5, 35), person(5, 35, AT_SECTION))
    assert 24 <= got["work"] <= 31  # 30 s at the section, minus the 5 s before it counts, plus the few seconds the window needs to empty
    assert [e[3] for e in events] == ["work", "idle"]  # started and stopped


def test_a_passer_by_is_not_work(tmp_path):
    got, events = machine(tmp_path, 30, moving(5, 25), person(5, 8, AT_SECTION))  # 3 s in the section, shorter than min_present_s
    assert got["work"] == pytest.approx(0, abs=1) and events == []


def test_a_person_standing_still_is_attended_but_not_working(tmp_path):
    got, _ = machine(tmp_path, 30, lambda f, t: None, person(2, 28, AT_SECTION))  # nothing moves in the section
    assert got["work"] == pytest.approx(0, abs=1)


def test_a_machine_that_runs_alone_is_not_work_by_people(tmp_path):
    got, _ = machine(tmp_path, 30, moving(2, 28), lambda frame: [])  # motion, but nobody there
    assert got["work"] == pytest.approx(0, abs=1)


def test_a_person_far_from_the_section_does_not_count(tmp_path):
    got, _ = machine(tmp_path, 30, moving(2, 28), person(2, 28, ELSEWHERE))
    assert got["work"] == pytest.approx(0, abs=1)


def test_the_time_at_the_section_survives_a_missed_detection(tmp_path):
    n = [0]

    def flickering(frame):  # detected, but not for one second in the middle (the tracker keeps the person alive for two)
        n[0] += 1
        t = n[0] / 5
        return [] if 14 <= t < 15 else [(*AT_SECTION, 0.9)] if 3 <= t < 30 else []

    got, _ = machine(tmp_path, 34, moving(3, 30), flickering)
    assert got["work"] >= 23  # the gap did not send the person back to "just arrived" (that would cost another 5 s of waiting: about 19 s of work)


def test_the_motion_source_masks_people_out_but_activity_counts_them(tmp_path):
    covering = (180, 10, 290, 140)  # the person stands in front of the whole moving block (x 190..280)
    by_people, _ = machine(tmp_path, 30, moving(3, 27), person(3, 27, covering), source="activity")
    by_itself, _ = machine(tmp_path, 30, moving(3, 27), person(3, 27, covering), source="motion")
    assert by_people["work"] > 12 and by_itself["work"] < 3  # a machine that runs by itself must not look busy just because someone walks past it


def test_thresholds_are_the_users_to_tune(tmp_path):
    shy, _ = machine(tmp_path, 40, moving(5, 35), person(5, 35, AT_SECTION), motion_thr=0.9)  # an impossible share of the section
    assert shy["work"] == pytest.approx(0, abs=1)
    far, _ = machine(tmp_path, 40, moving(5, 35), person(5, 35, AT_SECTION), min_present_s=60)  # nobody stays a minute
    assert far["work"] == pytest.approx(0, abs=1)
    strict, _ = machine(tmp_path, 40, moving(5, 35), person(5, 35, (240, 120, 270, 175)), min_overlap=0.9)  # the person's box must lie almost wholly in the section
    assert strict["work"] == pytest.approx(0, abs=1)


def test_the_config_accepts_the_activity_source_and_its_keys():
    ok = dict(cameras=dict(c=dict(machines=[dict(name="m", source="activity", roi=SECTION, min_present_s=6, min_overlap=0.2, motion_thr=0.03)])))
    check_config(ok)
    bad = dict(cameras=dict(c=dict(machines=[dict(name="m", source="human", roi=SECTION)])))
    with pytest.raises(ConfigError, match=r"must be one of \['activity', 'lamp', 'motion'\]"):
        check_config(bad)
