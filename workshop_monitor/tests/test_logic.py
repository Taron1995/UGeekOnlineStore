from datetime import datetime, time

import numpy as np

from monitor.posture import is_seated, pose_motion
from monitor.schedule import Schedule, Shift
from monitor.timers import DebouncedState, TimerBank

TZ = "Asia/Yerevan"
BREAKS = [(time(9, 40), time(10, 0)), (time(13, 0), time(14, 0)), (time(15, 40), time(16, 0))]
SCHED = Schedule(TZ, BREAKS, [Shift("day", time(8, 0), time(17, 0)), Shift("night", time(20, 0), time(5, 0))])


def dt(h, m, d=6):
    return datetime(2026, 10, d, h, m, tzinfo=SCHED.tz)


def test_subtract_breaks_splits_interval():
    segs = SCHED.subtract_breaks(dt(9, 30), dt(10, 10))
    assert segs == [(dt(9, 30), dt(9, 40)), (dt(10, 0), dt(10, 10))]


def test_interval_inside_lunch_is_dropped():
    assert SCHED.subtract_breaks(dt(13, 10), dt(13, 50)) == []


def test_interval_without_breaks_unchanged():
    assert SCHED.subtract_breaks(dt(10, 30), dt(11, 0)) == [(dt(10, 30), dt(11, 0))]


def test_shift_at_day_and_night():
    assert SCHED.shift_at(dt(12, 0)).name == "day"
    assert SCHED.shift_at(dt(18, 0)) is None
    assert SCHED.shift_at(dt(22, 0)).name == "night"
    ref = SCHED.shift_at(dt(2, 0))
    assert ref.name == "night" and ref.shift_date.day == 5


def test_debounce_opens_and_closes_with_correct_bounds():
    s = DebouncedState(on_after=3, off_after=5)
    out = []
    for t in range(0, 30):
        active = 2 <= t < 15
        r = s.update(active, float(t))
        if r:
            out.append(r)
    assert out == [(2.0, 15.0)]


def test_debounce_ignores_short_blips():
    s = DebouncedState(on_after=3, off_after=5)
    for t, a in enumerate([1, 1, 0, 0, 1, 0, 0]):
        assert s.update(bool(a), float(t)) is None
    assert not s.on


def test_timerbank_closes_vanished_key():
    bank = TimerBank(on_after=1, off_after=2)
    closed = []
    for t in range(0, 10):
        keys = {"a"} if t < 4 else set()
        closed += list(bank.update(float(t), keys))
    assert closed == [("a", 0.0, 4.0)]


def _pose(knee_dy):
    pts = np.zeros((33, 2), dtype=np.float32)
    vis = np.ones(33, dtype=np.float32)
    pts[[11, 12], 1] = 0.2
    pts[[23, 24], 1] = 0.5
    pts[[25, 26], 1] = 0.5 + knee_dy
    return pts, vis


def test_seated_vs_standing():
    assert is_seated(*_pose(0.05)) is True
    assert is_seated(*_pose(0.3)) is False


def test_seated_unknown_when_knees_invisible():
    pts, vis = _pose(0.05)
    vis[[25, 26]] = 0.1
    assert is_seated(pts, vis) is None


def test_pose_motion():
    pts, vis = _pose(0.05)
    assert pose_motion(None, pts, None, vis) is None
    assert pose_motion(pts, pts, vis, vis) == 0.0
    assert pose_motion(pts, pts + 0.1, vis, vis) > 0.1
