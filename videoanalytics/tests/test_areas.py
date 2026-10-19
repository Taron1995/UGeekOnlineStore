"""Tracking areas: 'watch' = only people standing there are analysed, 'ignore' = people standing there are not."""
import cv2
import numpy as np
import pytest

from tests.test_features import make_video, opts, src
from videoanalytics.analyze import fingerprint
from videoanalytics.config import ConfigError, check_config
from videoanalytics.pipeline import keep_in_areas, process_video
from videoanalytics.tools import snapshot
from videoanalytics.zones import Zone

LEFT_BOX, RIGHT_BOX = (40, 60, 70, 140), (240, 60, 270, 140)  # feet at x = 55 and 255 of 320
LEFT_HALF = [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]
RIGHT_HALF = [[0.5, 0], [1, 0], [1, 1], [0.5, 1]]


def two_people(frame):
    return [(*LEFT_BOX, 0.9), (*RIGHT_BOX, 0.9)]


def people_seconds(tmp_path, areas):
    v = tmp_path / "v.mp4"
    make_video(v, 20, lambda f, t: None)
    _, tables, _ = process_video(src(v, 20), dict(areas=areas), opts(), two_people)
    return sum(r[1] for r in tables["minutes"].values())  # person-seconds


def cfg(*areas):
    return dict(cameras=dict(cam1=dict(areas=list(areas))))


def test_areas_are_validated_like_zones():
    check_config(cfg(dict(name="floor", polygon=LEFT_HALF), dict(name="gate", polygon=RIGHT_HALF, mode="ignore")))
    with pytest.raises(ConfigError) as e:
        check_config(cfg(dict(name="a", polygon=LEFT_HALF, mode="follow"), dict(name="a", polygon=[[0, 0], [1, 1]]), dict(polygon=LEFT_HALF, colour="red")))
    text = str(e.value)
    for fragment in ("mode: 'follow' must be one of ['watch', 'ignore']", "duplicate area name 'a'", "needs at least 3", "'name' is required", "unknown key 'colour'"):
        assert fragment in text


def test_keep_in_areas_semantics():
    dets = [(*LEFT_BOX, 0.9), (*RIGHT_BOX, 0.8)]
    watch_left, ignore_left, ignore_right = (Zone("l", LEFT_HALF), "watch"), (Zone("l", LEFT_HALF), "ignore"), (Zone("r", RIGHT_HALF), "ignore")
    assert keep_in_areas(dets, [], 320, 180) is dets  # no areas: everyone
    assert keep_in_areas(dets, [watch_left], 320, 180) == [dets[0]]  # only the left
    assert keep_in_areas(dets, [ignore_left], 320, 180) == [dets[1]]  # everyone but the left
    assert keep_in_areas(dets, [(Zone("all", [[0, 0], [1, 0], [1, 1], [0, 1]]), "watch"), ignore_right], 320, 180) == [dets[0]]  # watch the whole picture, ignore the right
    assert keep_in_areas(dets, [ignore_left, ignore_right], 320, 180) == []  # ignoring everything


def test_the_foot_point_decides_not_the_body():
    top = [(Zone("top", [[0, 0], [1, 0], [1, 0.4], [0, 0.4]]), "watch")]  # the upper 40 % of a 180 px picture: y < 72
    assert keep_in_areas([(40, 10, 70, 90, 0.9)], top, 320, 180) == []  # the head is in it, the feet (y = 90) are not
    assert keep_in_areas([(40, 10, 70, 60, 0.9)], top, 320, 180) == [(40, 10, 70, 60, 0.9)]  # feet at y = 60


def test_watch_area_limits_what_is_counted(tmp_path):
    both = people_seconds(tmp_path, [])
    left = people_seconds(tmp_path, [dict(name="floor", polygon=LEFT_HALF, mode="watch")])
    assert both == pytest.approx(2 * left, rel=0.05) and left > 15  # two people everywhere, one in the left half


def test_ignore_area_removes_people_standing_in_it(tmp_path):
    ignoring_right = people_seconds(tmp_path, [dict(name="tv", polygon=RIGHT_HALF, mode="ignore")])
    ignoring_all = people_seconds(tmp_path, [dict(name="all", polygon=[[0, 0], [1, 0], [1, 1], [0, 1]], mode="ignore")])
    assert ignoring_right == pytest.approx(people_seconds(tmp_path, [dict(name="floor", polygon=LEFT_HALF, mode="watch")]), rel=0.02)
    assert ignoring_all < 1


def test_mode_defaults_to_watch(tmp_path):
    everywhere = [[0, 0], [1, 0], [1, 1], [0, 1]]
    # no mode = watch: a watched picture keeps both people (an ignored one would keep nobody)
    assert people_seconds(tmp_path, [dict(name="floor", polygon=everywhere)]) == pytest.approx(people_seconds(tmp_path, []))


def test_areas_belong_to_the_setup_fingerprint():
    base = dict(areas=[dict(name="floor", polygon=LEFT_HALF)])
    moved = dict(areas=[dict(name="floor", polygon=RIGHT_HALF)])
    o = {"fps": 3.0}
    assert fingerprint(base, o, 0) != fingerprint(moved, o, 0) != fingerprint({}, o, 0)  # a changed area recomputes that camera


def test_snapshot_draws_the_areas(tmp_path):
    v = tmp_path / "v.mp4"
    make_video(v, 3, lambda f, t: None)
    plain, drawn = tmp_path / "plain.jpg", tmp_path / "drawn.jpg"
    snapshot(str(v), plain, {}, 1.0)
    snapshot(str(v), drawn, dict(areas=[dict(name="floor", polygon=LEFT_HALF, mode="watch"), dict(name="tv", polygon=RIGHT_HALF, mode="ignore")]), 1.0)
    a, b = cv2.imread(str(plain)), cv2.imread(str(drawn))
    changed = np.abs(a.astype(int) - b.astype(int)).sum(axis=2) > 60
    assert changed.sum() > 400
    blue = (b[..., 0] > 200) & (b[..., 1] > 100) & (b[..., 2] < 80)  # BGR of the 'follow' colour
    red = (b[..., 2] > 200) & (b[..., 0] < 120) & (b[..., 1] < 120)  # BGR of the 'ignore' colour
    assert blue.sum() > 100 and red.sum() > 100
