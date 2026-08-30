"""SQLite persistence for employees and smoking-detection events."""
import sqlite3
import time
from contextlib import contextmanager

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label_id INTEGER UNIQUE NOT NULL,
    name TEXT NOT NULL,
    employee_code TEXT UNIQUE,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER,
    employee_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    snapshot_path TEXT,
    detected_at REAL NOT NULL,
    FOREIGN KEY (employee_id) REFERENCES employees(id)
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def upsert_employee(label_id: int, name: str, employee_code: str | None = None) -> int:
    with get_conn() as conn:
        cur = conn.execute("SELECT id FROM employees WHERE label_id = ?", (label_id,))
        row = cur.fetchone()
        if row:
            conn.execute(
                "UPDATE employees SET name = ?, employee_code = ? WHERE label_id = ?",
                (name, employee_code, label_id),
            )
            return row["id"]
        cur = conn.execute(
            "INSERT INTO employees (label_id, name, employee_code, created_at) VALUES (?, ?, ?, ?)",
            (label_id, name, employee_code, time.time()),
        )
        return cur.lastrowid


def get_employee_by_label(label_id: int):
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM employees WHERE label_id = ?", (label_id,))
        return cur.fetchone()


def list_employees():
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM employees ORDER BY name")
        return cur.fetchall()


def log_detection(employee_id, employee_name: str, confidence: float, snapshot_path: str | None):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO detections (employee_id, employee_name, confidence, snapshot_path, detected_at)
               VALUES (?, ?, ?, ?, ?)""",
            (employee_id, employee_name, confidence, snapshot_path, time.time()),
        )


def last_detection_time(employee_name: str):
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT MAX(detected_at) AS ts FROM detections WHERE employee_name = ?",
            (employee_name,),
        )
        row = cur.fetchone()
        return row["ts"] if row and row["ts"] else 0.0


def list_detections(limit: int = 200):
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM detections ORDER BY detected_at DESC LIMIT ?", (limit,)
        )
        return cur.fetchall()


def detections_per_employee():
    with get_conn() as conn:
        cur = conn.execute(
            """SELECT employee_name, COUNT(*) AS total, MAX(detected_at) AS last_seen
               FROM detections GROUP BY employee_name ORDER BY total DESC"""
        )
        return cur.fetchall()
