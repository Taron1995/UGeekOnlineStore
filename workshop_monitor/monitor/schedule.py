"""Смены и установленные перерывы. Всё, что случилось в перерыв, нарушением не считается."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo


def parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


@dataclass(frozen=True)
class Shift:
    name: str
    start: time
    end: time  # если end <= start, смена переходит через полночь


@dataclass(frozen=True)
class ShiftRef:
    shift_date: date
    name: str


class Schedule:
    def __init__(self, tz: str, breaks: list[tuple[time, time]], shifts: list[Shift]):
        self.tz = ZoneInfo(tz)
        self.breaks = breaks
        self.shifts = shifts

    def to_dt(self, ts: float) -> datetime:
        return datetime.fromtimestamp(ts, self.tz)

    def shift_at(self, dt: datetime) -> ShiftRef | None:
        t = dt.timetz().replace(tzinfo=None)
        for s in self.shifts:
            if s.start < s.end:
                if s.start <= t < s.end:
                    return ShiftRef(dt.date(), s.name)
            else:  # ночная смена
                if t >= s.start:
                    return ShiftRef(dt.date(), s.name)
                if t < s.end:
                    return ShiftRef(dt.date() - timedelta(days=1), s.name)
        return None

    def in_break(self, dt: datetime) -> bool:
        t = dt.timetz().replace(tzinfo=None)
        return any(b0 <= t < b1 for b0, b1 in self.breaks)

    def subtract_breaks(self, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
        """Режет интервал [start, end) по перерывам; возвращает рабочие куски (>= 1 с)."""
        segs = [(start, end)]
        d = start.date()
        while d <= end.date():
            for b0, b1 in self.breaks:
                bs = datetime.combine(d, b0, tzinfo=self.tz)
                be = datetime.combine(d, b1, tzinfo=self.tz)
                out = []
                for s, e in segs:
                    if be <= s or bs >= e:
                        out.append((s, e))
                        continue
                    if s < bs:
                        out.append((s, bs))
                    if be < e:
                        out.append((be, e))
                segs = out
            d += timedelta(days=1)
        return [(s, e) for s, e in segs if (e - s).total_seconds() >= 1]
