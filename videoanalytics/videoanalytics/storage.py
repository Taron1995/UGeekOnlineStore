"""Per-file SQLite shards (resumable, parallel-safe) merged into one analytics.db."""
from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path

SHARD_SCHEMA = """
CREATE TABLE events(ts REAL, type TEXT, subject TEXT, value TEXT, clip TEXT, thumb TEXT);
CREATE TABLE minutes(minute INTEGER PRIMARY KEY, covered_s REAL, people_s REAL, people_max INTEGER,
                     work_s REAL, idle_s REAL, det_frames INTEGER);
CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT);
"""
MAIN_SCHEMA = """
CREATE TABLE events(camera TEXT, ts REAL, type TEXT, subject TEXT, value TEXT, clip TEXT, thumb TEXT, source TEXT);
CREATE TABLE minutes(camera TEXT, minute INTEGER, covered_s REAL, people_s REAL, people_max INTEGER,
                     work_s REAL, idle_s REAL, det_frames INTEGER, PRIMARY KEY(camera, minute));
CREATE TABLE files(source TEXT PRIMARY KEY, camera TEXT, start_ts REAL, video_s REAL, processed_s REAL,
                   det_frames INTEGER, elapsed_s REAL);
CREATE INDEX events_ix ON events(camera, type, ts);
"""


def shard_path(out_dir: Path, camera: str, rel: str) -> Path:
    norm = rel.replace("\\", "/")
    safe = norm.replace("/", "__")[-100:]  # keep Windows paths short; the hash keeps names unique
    return out_dir / "shards" / camera / f"{safe}_{hashlib.md5(norm.encode(), usedforsecurity=False).hexdigest()[:8]}.db"


def write_shard(path: Path, events, minutes: dict, meta: dict):
    """Written to .tmp then renamed, so an interrupted run never leaves a half-finished shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.executescript(SHARD_SCHEMA)
    con.executemany("INSERT INTO events VALUES(?,?,?,?,?,?)", events)
    con.executemany("INSERT INTO minutes VALUES(?,?,?,?,?,?,?)", [(m, *row) for m, row in minutes.items()])
    con.executemany("INSERT INTO meta VALUES(?,?)", [(k, str(v)) for k, v in meta.items()])
    con.commit()
    con.close()
    os.replace(tmp, path)


def clean_stale(out_dir: Path):
    for t in (out_dir / "shards").rglob("*.tmp"):
        t.unlink(missing_ok=True)


def merge_shards(out_dir: Path) -> Path:
    db = out_dir / "analytics.db"
    db.unlink(missing_ok=True)
    con = sqlite3.connect(db, isolation_level=None)
    con.executescript(MAIN_SCHEMA)
    for sh in sorted((out_dir / "shards").rglob("*.db")):
        con.execute("ATTACH ? AS s", (str(sh),))
        meta = dict(con.execute("SELECT k, v FROM s.meta").fetchall())
        cam, src = meta["camera"], meta["source"]
        con.execute("BEGIN")
        con.execute("INSERT INTO events SELECT ?, ts, type, subject, value, clip, thumb, ? FROM s.events", (cam, src))
        con.execute("""INSERT INTO minutes SELECT ?, minute, covered_s, people_s, people_max, work_s, idle_s, det_frames
                       FROM s.minutes WHERE true
                       ON CONFLICT(camera, minute) DO UPDATE SET
                         covered_s = covered_s + excluded.covered_s, people_s = people_s + excluded.people_s,
                         people_max = max(people_max, excluded.people_max), work_s = work_s + excluded.work_s,
                         idle_s = idle_s + excluded.idle_s, det_frames = det_frames + excluded.det_frames""", (cam,))
        con.execute("INSERT OR REPLACE INTO files VALUES(?,?,?,?,?,?,?)",
                    (src, cam, float(meta["start_ts"]), float(meta["video_s"]), float(meta["processed_s"]),
                     int(meta["det_frames"]), float(meta["elapsed_s"])))
        con.execute("COMMIT")
        con.execute("DETACH s")
    con.close()
    return db
