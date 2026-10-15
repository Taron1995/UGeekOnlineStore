"""Work hours, breaks and weekdays: decides in which minutes a station is expected to be staffed."""
from __future__ import annotations

from datetime import datetime, timezone


def parse_hm(v) -> int:
    """'08:30' | 8 | 8.5 -> minutes after midnight ('24:00' = 1440)."""
    if isinstance(v, str):
        h, _, m = v.partition(":")
        return int(h) * 60 + int(m or 0)
    return int(round(float(v) * 60))


def _ranges(pairs):
    out = []
    for a, b in pairs or []:
        a, b = parse_hm(a), parse_hm(b)
        out += [(a, 1440), (0, b)] if b <= a else [(a, b)]  # overnight, e.g. 22:00-06:00
    return out


class Schedule:
    """cfg: {work: [["08:00","12:00"],["13:00","17:00"]], breaks: [["10:00","10:15"]], days: [1,2,3,4,5]}  (Mon=1)."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.work = _ranges(cfg.get("work")) or [(0, 1440)]
        self.breaks = _ranges(cfg.get("breaks"))
        self.days = set(cfg.get("days") or range(1, 8))

    def expected(self, minute: int) -> bool:
        """minute = wall-clock minutes since epoch (as stored in the *_minutes tables)."""
        dt = datetime.fromtimestamp(minute * 60, timezone.utc)
        if dt.isoweekday() not in self.days:
            return False
        m = dt.hour * 60 + dt.minute
        return any(a <= m < b for a, b in self.work) and not any(a <= m < b for a, b in self.breaks)
