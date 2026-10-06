"""Сквозной тест пайплайна на синтетических кадрах: подставной детектор, SQLite вместо PostgreSQL."""
from datetime import datetime, time

import numpy as np
import pytest

pytest.importorskip("cv2")
sv = pytest.importorskip("supervision")

from monitor.config import AppCfg, CameraCfg, DetectorCfg, Thresholds, WorkplaceCfg  # noqa: E402
from monitor.db import Assignment, Employee, Event, Repo, Workplace, init_db, make_engine  # noqa: E402
from monitor.pipeline import CameraWorker  # noqa: E402
from monitor.schedule import Schedule, Shift  # noqa: E402
from monitor.sink import EventSink  # noqa: E402

BASE = datetime.fromisoformat("2026-10-06T10:30:00+04:00").timestamp()
PERSON_BOX = [250, 400, 350, 650]      # низ рамки (300, 650) внутри station_zone
PHONE_BOX = [290, 450, 310, 480]


class FakeDetector:
    """t < 60: человек у станка; 60..120: никого; 120..150: человек с телефоном; 150..160: человек."""
    def __init__(self):
        self.t = 0.0

    def detect(self, frame):
        t = self.t
        boxes, cls = [], []
        if t < 60 or t >= 120:
            boxes.append(PERSON_BOX); cls.append(1)
        if 120 <= t < 150:
            boxes.append(PHONE_BOX); cls.append(77)
        if not boxes:
            return sv.Detections.empty()
        return sv.Detections(xyxy=np.array(boxes, np.float32), confidence=np.full(len(boxes), 0.9, np.float32),
                             class_id=np.array(cls))


def test_pipeline_emits_expected_events(tmp_path):
    sched = Schedule("Asia/Yerevan", [(time(13, 0), time(14, 0))], [Shift("day", time(8, 0), time(17, 0))])
    wp = WorkplaceCfg(id="WP-01", station_zone=[[100, 300], [500, 300], [500, 700], [100, 700]],
                      machine_roi=[[600, 300], [900, 300], [900, 500], [600, 500]])
    cam = CameraCfg(id="cam1", source="fake", workplaces=[wp])
    cfg = AppCfg(schedule=sched, db_url="", clips_dir=str(tmp_path), detector=DetectorCfg(),
                 thresholds=Thresholds(), cameras=[cam])
    engine = make_engine(f"sqlite:///{tmp_path}/t.db")
    init_db(engine)
    repo = Repo(engine)
    with repo.session() as s:
        s.add_all([Employee(id="E-001", full_name="Тест"), Workplace(id="WP-01")])
        s.commit()
        s.add(Assignment(shift_date=datetime(2026, 10, 6).date(), shift_name="day",
                         workplace_id="WP-01", employee_id="E-001"))
        s.commit()

    det = FakeDetector()
    worker = CameraWorker(cam, cfg, det, EventSink(repo, sched, str(tmp_path)), pose=None)
    rng = np.random.default_rng(0)
    for i in range(int(160 * 5)):
        t = i / 5
        det.t = t
        frame = np.zeros((800, 1000, 3), np.uint8)
        if t < 60:  # станок работает: в ROI меняется картинка
            frame[300:500, 600:900] = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
        worker.process(frame, BASE + t)
    worker.flush()

    with repo.session() as s:
        ev = list(s.query(Event).all())
    by = {}
    for e in ev:
        by.setdefault(e.type, []).append(e)

    def ts(d):  # SQLite теряет часовой пояс; в PostgreSQL (timestamptz) этого нет
        return (d if d.tzinfo else d.replace(tzinfo=sched.tz)).timestamp() - BASE

    def rel(e):
        return (ts(e.ts_start), ts(e.ts_end))

    absent = by["ABSENT"][0]
    assert 55 <= rel(absent)[0] <= 66 and 115 <= rel(absent)[1] <= 126
    assert absent.employee_id == "E-001" and absent.shift_name == "day"
    phone = by["PHONE"][0]
    assert 118 <= rel(phone)[0] <= 126 and 145 <= rel(phone)[1] <= 156
    run = by["MACHINE_RUN"][0]
    assert rel(run)[0] < 5 and 55 <= rel(run)[1] <= 75
    assert sum(e.duration_s for e in by["AT_STATION"]) > 80
    assert "MACHINE_IDLE" in by
