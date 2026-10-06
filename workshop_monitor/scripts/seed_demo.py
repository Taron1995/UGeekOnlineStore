"""Демо-график: кто на каком месте. Запуск: python scripts/seed_demo.py  (нужна работающая БД)."""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from monitor.config import load_config  # noqa: E402
from monitor.db import Assignment, Employee, Repo, Workplace, init_db, make_engine  # noqa: E402

cfg = load_config("config/config.yaml")
engine = make_engine(cfg.db_url)
init_db(engine)
with Repo(engine).session() as s:
    s.merge(Workplace(id="WP-01", name="Станок 1", camera_id="cam1"))
    s.merge(Workplace(id="WP-02", name="Станок 2", camera_id="cam1"))
    s.merge(Employee(id="E-001", full_name="Сотрудник 1"))
    s.merge(Employee(id="E-002", full_name="Сотрудник 2"))
    s.commit()  # сотрудники и места должны существовать до назначений
    for wp, emp in (("WP-01", "E-001"), ("WP-02", "E-002")):
        exists = s.scalar(select(Assignment.id).where(
            Assignment.shift_date == date.today(), Assignment.shift_name == "day", Assignment.workplace_id == wp))
        if not exists:
            s.add(Assignment(shift_date=date.today(), shift_name="day", workplace_id=wp, employee_id=emp))
    s.commit()
print("ok")
