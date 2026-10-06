"""FastAPI: JSON API и веб-страница отчёта. Доступ только с паролем (HTTP Basic, пароль = API_KEY)."""
from __future__ import annotations

import html
import os
import secrets
from datetime import date

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from sqlalchemy import select, text

from .db import Assignment, Employee, Event, Repo, Workplace, init_db, make_engine

app = FastAPI(title="Workshop monitor")
security = HTTPBasic()
engine = make_engine(os.environ.get("DATABASE_URL", "postgresql+psycopg://monitor:monitor@localhost:5432/monitor"))
repo = Repo(engine)


def auth(cred: HTTPBasicCredentials = Depends(security)) -> str:
    key = os.environ.get("API_KEY", "")
    if not key or not secrets.compare_digest(cred.password.encode(), key.encode()):
        raise HTTPException(401, "Unauthorized", headers={"WWW-Authenticate": "Basic"})
    return cred.username


@app.on_event("startup")
def _startup():
    init_db(engine)


@app.get("/health")
def health():
    return {"ok": True}


class EmployeeIn(BaseModel):
    id: str
    full_name: str


class AssignmentIn(BaseModel):
    shift_date: date
    shift_name: str
    workplace_id: str
    employee_id: str


@app.post("/api/employees", dependencies=[Depends(auth)])
def put_employee(e: EmployeeIn):
    with repo.session() as s:
        s.merge(Employee(id=e.id, full_name=e.full_name))
        s.commit()
    return {"ok": True}


@app.post("/api/workplaces/{wp_id}", dependencies=[Depends(auth)])
def put_workplace(wp_id: str, name: str = "", camera_id: str = ""):
    with repo.session() as s:
        s.merge(Workplace(id=wp_id, name=name, camera_id=camera_id))
        s.commit()
    return {"ok": True}


@app.post("/api/assignments", dependencies=[Depends(auth)])
def put_assignment(a: AssignmentIn):
    """График: кто на каком рабочем месте в какую смену."""
    with repo.session() as s:
        row = s.scalar(select(Assignment).where(
            Assignment.shift_date == a.shift_date, Assignment.shift_name == a.shift_name,
            Assignment.workplace_id == a.workplace_id))
        if row:
            row.employee_id = a.employee_id
        else:
            s.add(Assignment(**a.model_dump()))
        s.commit()
    repo._cache.clear()
    return {"ok": True}


def _rows(shift_date: date, shift_name: str | None):
    q = "SELECT * FROM v_shift_metrics WHERE shift_date = :d" + (" AND shift_name = :s" if shift_name else "")
    with engine.connect() as c:
        return [dict(r._mapping) for r in c.execute(text(q), {"d": shift_date, "s": shift_name})]


@app.get("/api/shift-metrics", dependencies=[Depends(auth)])
def shift_metrics(shift_date: date | None = None, shift_name: str | None = None):
    return _rows(shift_date or date.today(), shift_name)


@app.get("/api/events", dependencies=[Depends(auth)])
def events(type: str | None = None, shift_date: date | None = None, employee_id: str | None = None,
           limit: int = 200):
    q = select(Event).order_by(Event.ts_start.desc()).limit(min(limit, 1000))
    if type:
        q = q.where(Event.type == type)
    if shift_date:
        q = q.where(Event.shift_date == shift_date)
    if employee_id:
        q = q.where(Event.employee_id == employee_id)
    with repo.session() as s:
        return [{"id": e.id, "type": e.type, "start": e.ts_start, "end": e.ts_end, "duration_s": e.duration_s,
                 "workplace_id": e.workplace_id, "employee_id": e.employee_id, "meta": e.meta,
                 "clip": f"/api/clips/{e.id}" if e.clip_path else None} for e in s.scalars(q)]


@app.get("/api/clips/{event_id}", dependencies=[Depends(auth)])
def clip(event_id: int):
    with repo.session() as s:
        e = s.get(Event, event_id)
    if not e or not e.clip_path or not os.path.isfile(e.clip_path):
        raise HTTPException(404, "Клип не найден")
    return FileResponse(e.clip_path, media_type="video/mp4")


COLS = [("workplace_id", "Место"), ("full_name", "Сотрудник"), ("at_station_s", "У станка, мин"),
        ("machine_run_s", "Станок работал, мин"), ("machine_idle_s", "Простой, мин"), ("pieces", "Заготовок"),
        ("absent_s", "Отсутствие, мин"), ("phone_s", "Телефон, мин"), ("meeting_s", "Собрания, мин"),
        ("sitting_still_s", "Сидит без движения, мин")]


@app.get("/", response_class=HTMLResponse, dependencies=[Depends(auth)])
def report(shift_date: date | None = None, shift_name: str | None = None):
    d = shift_date or date.today()
    head = "".join(f"<th>{c[1]}</th>" for c in COLS)
    body = ""
    for r in _rows(d, shift_name):
        cells = []
        for k, _ in COLS:
            v = r.get(k)
            if k.endswith("_s") and v is not None:
                v = f"{float(v) / 60:.1f}"
            cells.append(f"<td>{html.escape(str(v if v is not None else '—'))}</td>")
        body += "<tr>" + "".join(cells) + "</tr>"
    return (f"<html><head><meta charset='utf-8'><title>Смена {d}</title><style>"
            "body{font:14px sans-serif;margin:24px}table{border-collapse:collapse}"
            "td,th{border:1px solid #ccc;padding:6px 10px;text-align:right}th{background:#f3f3f3}"
            f"</style></head><body><h2>Показатели смены {d} {html.escape(shift_name or '')}</h2>"
            f"<table><tr>{head}</tr>{body}</table></body></html>")
