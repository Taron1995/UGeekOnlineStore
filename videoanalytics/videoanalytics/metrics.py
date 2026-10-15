"""Pure functions behind the report: absence outside breaks, long idle stretches."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from .schedule import Schedule


def _date(minute: int) -> str:
    return datetime.fromtimestamp(minute * 60, timezone.utc).strftime("%Y-%m-%d")


def absence_stats(rows, schedule: Schedule, min_cover_s: float = 30.0, empty_frac: float = 0.2, min_run_min: int = 5):
    """rows: (minute, covered_s, occupied_s) of one zone, sorted by minute.

    Only minutes the schedule expects someone to be there count (work hours, no breaks, listed weekdays).
    Returns (daily, runs): daily {date: {expected_s, present_s, absent_s}}; runs [(start_minute, minutes, absent_s)] of
    consecutive expected minutes that were (almost) empty and lasted at least min_run_min minutes."""
    daily: dict = defaultdict(lambda: dict(expected_s=0.0, present_s=0.0, absent_s=0.0))
    runs, cur = [], None
    for minute, covered, occupied in rows:
        if covered < min_cover_s or not schedule.expected(minute):
            if cur:
                runs.append(cur)
            cur = None
            continue
        d = daily[_date(minute)]
        d["expected_s"] += covered
        d["present_s"] += occupied
        d["absent_s"] += max(0.0, covered - occupied)
        if occupied <= empty_frac * covered:
            if cur and cur[0] + cur[1] == minute:
                cur[1] += 1
                cur[2] += covered - occupied
            else:
                if cur:
                    runs.append(cur)
                cur = [minute, 1, covered - occupied]
        elif cur:
            runs.append(cur)
            cur = None
    if cur:
        runs.append(cur)
    return dict(daily), [tuple(r) for r in runs if r[1] >= min_run_min]


def idle_runs(rows, min_idle_s: float = 300.0, idle_frac: float = 0.9, min_cover_s: float = 30.0):
    """rows: (minute, work_s, idle_s) of one machine, sorted by minute -> [(start_minute, idle_seconds)] stretches of
    consecutive mostly-idle minutes whose idle time adds up to at least min_idle_s. Works across file boundaries."""
    out, run = [], None
    for minute, work, idle in rows:
        mostly_idle = work + idle >= min_cover_s and idle >= idle_frac * (work + idle)
        if run and run[0] + run[1] == minute and mostly_idle:
            run[1] += 1
            run[2] += idle
            continue
        if run:
            out.append(run)
        run = [minute, 1, idle] if mostly_idle else None
    if run:
        out.append(run)
    return [(m, idle) for m, _, idle in out if idle >= min_idle_s]
