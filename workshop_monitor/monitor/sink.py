"""Запись событий в PostgreSQL: привязка к смене и сотруднику, вычитание перерывов, нарезка клипа."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .clips import cut_clip
from .db import Event, Repo
from .schedule import Schedule

log = logging.getLogger(__name__)

# Эти типы не считаются в установленные перерывы
BREAK_EXEMPT = {"ABSENT", "MACHINE_IDLE", "PHONE", "MEETING", "SITTING_STILL"}
CLIP_TYPES = {"PHONE", "MEETING", "SITTING_STILL", "ABSENT"}


@dataclass
class EventRec:
    type: str
    start: float                 # epoch, секунды
    end: float
    camera_id: str
    workplace_id: str | None = None
    track_id: int | None = None
    meta: dict = field(default_factory=dict)
    video_src: str | None = None      # файл-источник, если клип можно вырезать
    video_offset_s: float | None = None


class EventSink:
    def __init__(self, repo: Repo, schedule: Schedule, clips_dir: str):
        self.repo, self.schedule = repo, schedule
        self.clips_dir = Path(clips_dir)
        self.lock = threading.Lock()

    def emit(self, rec: EventRec, employee_id: str | None = None) -> None:
        a, b = self.schedule.to_dt(rec.start), self.schedule.to_dt(rec.end)
        parts = self.schedule.subtract_breaks(a, b) if rec.type in BREAK_EXEMPT and b > a else [(a, b)]
        rows = []
        for s, e in parts:
            shift = self.schedule.shift_at(s)
            if rec.type in BREAK_EXEMPT and shift is None:
                continue  # вне смены нарушениями не считаем
            emp = employee_id
            if emp is None and rec.workplace_id and shift:
                emp = self.repo.employee_for(shift.shift_date, shift.name, rec.workplace_id)
            rows.append(Event(
                type=rec.type, ts_start=s, ts_end=e, duration_s=(e - s).total_seconds(),
                camera_id=rec.camera_id, workplace_id=rec.workplace_id, employee_id=emp,
                shift_date=shift.shift_date if shift else None, shift_name=shift.name if shift else None,
                track_id=rec.track_id, meta=rec.meta))
        if not rows:
            return
        with self.lock, self.repo.session() as s:
            s.add_all(rows)
            s.flush()
            if rec.type in CLIP_TYPES and rec.video_src and rec.video_offset_s is not None:
                for row in rows:
                    off = rec.video_offset_s + (row.ts_start.timestamp() - rec.start)
                    out = self.clips_dir / f"{row.type}_{row.id}.mp4"
                    if cut_clip(rec.video_src, off, min(row.duration_s, 60.0), out):
                        row.clip_path = str(out)
            s.commit()
        log.info("event %s wp=%s dur=%.0fs", rec.type, rec.workplace_id, rec.end - rec.start)
