from __future__ import annotations

import threading
from datetime import date, datetime

from sqlalchemy import (JSON, BigInteger, Date, DateTime, Float, ForeignKey, Integer, String,
                        UniqueConstraint, create_engine, select, text)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


PK = BigInteger().with_variant(Integer, "sqlite")  # автоинкремент и в SQLite (тесты)


class Base(DeclarativeBase):
    pass


class Employee(Base):
    __tablename__ = "employees"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    full_name: Mapped[str] = mapped_column(String(200))


class Workplace(Base):
    __tablename__ = "workplaces"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    camera_id: Mapped[str] = mapped_column(String(32), default="")


class Assignment(Base):
    """Кто на каком рабочем месте в какую смену: личность определяется графиком, а не лицом."""
    __tablename__ = "assignments"
    __table_args__ = (UniqueConstraint("shift_date", "shift_name", "workplace_id"),)
    id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    shift_date: Mapped[date] = mapped_column(Date)
    shift_name: Mapped[str] = mapped_column(String(32))
    workplace_id: Mapped[str] = mapped_column(ForeignKey("workplaces.id"))
    employee_id: Mapped[str] = mapped_column(ForeignKey("employees.id"))


class Event(Base):
    """AT_STATION, ABSENT, MACHINE_RUN, MACHINE_IDLE, PIECE, PHONE, MEETING, SITTING_STILL."""
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    type: Mapped[str] = mapped_column(String(24), index=True)
    ts_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    ts_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    camera_id: Mapped[str] = mapped_column(String(32))
    workplace_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    employee_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    shift_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    shift_name: Mapped[str | None] = mapped_column(String(32), nullable=True)
    track_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    meta: Mapped[dict] = mapped_column(JSON().with_variant(JSONB, "postgresql"), default=dict)
    clip_path: Mapped[str | None] = mapped_column(String(500), nullable=True)


# Сводка по смене: именно её читают API и Apache Superset
VIEW_SQL = """
CREATE OR REPLACE VIEW v_shift_metrics AS
SELECT e.shift_date, e.shift_name, e.workplace_id, e.employee_id, emp.full_name,
  COALESCE(SUM(duration_s) FILTER (WHERE type='AT_STATION'),0)   AS at_station_s,
  COALESCE(SUM(duration_s) FILTER (WHERE type='MACHINE_RUN'),0)  AS machine_run_s,
  COALESCE(SUM(duration_s) FILTER (WHERE type='MACHINE_IDLE'),0) AS machine_idle_s,
  COUNT(*) FILTER (WHERE type='PIECE')                           AS pieces,
  COALESCE(SUM(duration_s) FILTER (WHERE type='ABSENT'),0)       AS absent_s,
  COUNT(*) FILTER (WHERE type='PHONE')                           AS phone_events,
  COALESCE(SUM(duration_s) FILTER (WHERE type='PHONE'),0)        AS phone_s,
  COUNT(*) FILTER (WHERE type='MEETING')                         AS meeting_events,
  COALESCE(SUM(duration_s) FILTER (WHERE type='MEETING'),0)      AS meeting_s,
  COALESCE(SUM(duration_s) FILTER (WHERE type='SITTING_STILL'),0) AS sitting_still_s
FROM events e LEFT JOIN employees emp ON emp.id = e.employee_id
WHERE e.shift_date IS NOT NULL
GROUP BY e.shift_date, e.shift_name, e.workplace_id, e.employee_id, emp.full_name
"""


def make_engine(url: str):
    return create_engine(url, pool_pre_ping=True, future=True)


def init_db(engine) -> None:
    Base.metadata.create_all(engine)
    if engine.dialect.name == "postgresql":
        with engine.begin() as c:
            c.execute(text(VIEW_SQL))


class Repo:
    def __init__(self, engine):
        self.sf = sessionmaker(engine, expire_on_commit=False)
        self._cache: dict[tuple, str | None] = {}
        self._lock = threading.Lock()

    def session(self) -> Session:
        return self.sf()

    def employee_for(self, shift_date: date, shift_name: str, workplace_id: str) -> str | None:
        key = (shift_date, shift_name, workplace_id)
        with self._lock:
            if key in self._cache and self._cache[key] is not None:
                return self._cache[key]
        with self.sf() as s:
            emp = s.scalar(select(Assignment.employee_id).where(
                Assignment.shift_date == shift_date, Assignment.shift_name == shift_name,
                Assignment.workplace_id == workplace_id))
        with self._lock:
            self._cache[key] = emp
        return emp
