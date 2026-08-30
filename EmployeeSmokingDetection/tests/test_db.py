import importlib
import os


def test_employee_and_detection_roundtrip(tmp_path, monkeypatch):
    import config as config_module
    monkeypatch.setattr(config_module, "DB_PATH", str(tmp_path / "test.db"))

    from src import db as db_module
    importlib.reload(db_module)
    monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "test.db"))

    db_module.init_db()
    employee_id = db_module.upsert_employee(label_id=1, name="Test Employee", employee_code="E001")
    assert employee_id is not None

    db_module.log_detection(employee_id, "Test Employee", 0.9, None)
    detections = db_module.list_detections()
    assert len(detections) == 1
    assert detections[0]["employee_name"] == "Test Employee"

    summary = db_module.detections_per_employee()
    assert summary[0]["total"] == 1

    last_ts = db_module.last_detection_time("Test Employee")
    assert last_ts > 0
