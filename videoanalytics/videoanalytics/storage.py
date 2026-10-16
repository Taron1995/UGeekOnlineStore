"""Per-file SQLite shards (resumable, parallel-safe) merged into one analytics.db."""
from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path

SCHEMA_VERSION = "2"
FINE_S = 5  # resolution of the timelines drawn for short recordings (the minute tables are too coarse for a 90 s clip)
SHARD_SCHEMA = """
CREATE TABLE events(ts REAL, type TEXT, subject TEXT, value TEXT, clip TEXT, thumb TEXT);
CREATE TABLE minutes(minute INTEGER PRIMARY KEY, covered_s REAL, people_s REAL, people_max INTEGER, det_frames INTEGER);
CREATE TABLE zone_minutes(zone TEXT, minute INTEGER, covered_s REAL, occupied_s REAL, people_s REAL, PRIMARY KEY(zone, minute));
CREATE TABLE machine_minutes(machine TEXT, minute INTEGER, work_s REAL, idle_s REAL, alarm_s REAL, wait_s REAL,
                             work_att_s REAL, idle_att_s REAL, cycles INTEGER, PRIMARY KEY(machine, minute));
CREATE TABLE fine(bin INTEGER PRIMARY KEY, covered_s REAL, people_s REAL, people_max INTEGER);
CREATE TABLE zone_fine(zone TEXT, bin INTEGER, covered_s REAL, occupied_s REAL, PRIMARY KEY(zone, bin));
CREATE TABLE machine_fine(machine TEXT, bin INTEGER, work_s REAL, idle_s REAL, PRIMARY KEY(machine, bin));
CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT);
"""
MAIN_SCHEMA = """
CREATE TABLE events(camera TEXT, ts REAL, type TEXT, subject TEXT, value TEXT, clip TEXT, thumb TEXT, source TEXT);
CREATE TABLE minutes(camera TEXT, minute INTEGER, covered_s REAL, people_s REAL, people_max INTEGER, det_frames INTEGER,
                     PRIMARY KEY(camera, minute));
CREATE TABLE zone_minutes(camera TEXT, zone TEXT, minute INTEGER, covered_s REAL, occupied_s REAL, people_s REAL,
                          PRIMARY KEY(camera, zone, minute));
CREATE TABLE machine_minutes(camera TEXT, machine TEXT, minute INTEGER, work_s REAL, idle_s REAL, alarm_s REAL, wait_s REAL,
                             work_att_s REAL, idle_att_s REAL, cycles INTEGER, PRIMARY KEY(camera, machine, minute));
CREATE TABLE files(source TEXT PRIMARY KEY, camera TEXT, start_ts REAL, video_s REAL, processed_s REAL,
                   det_frames INTEGER, elapsed_s REAL);
CREATE TABLE fine(camera TEXT, bin INTEGER, covered_s REAL, people_s REAL, people_max INTEGER, PRIMARY KEY(camera, bin));
CREATE TABLE zone_fine(camera TEXT, zone TEXT, bin INTEGER, covered_s REAL, occupied_s REAL, PRIMARY KEY(camera, zone, bin));
CREATE TABLE machine_fine(camera TEXT, machine TEXT, bin INTEGER, work_s REAL, idle_s REAL, PRIMARY KEY(camera, machine, bin));
CREATE INDEX events_ix ON events(camera, type, ts);
"""


def _shard_stem(camera: str, rel: str):
    norm = rel.replace("\\", "/")
    safe = norm.replace("/", "__")[-100:]  # keep Windows paths short; the hash keeps names unique
    return f"{safe}_{hashlib.md5(norm.encode(), usedforsecurity=False).hexdigest()[:8]}"


def shard_path(out_dir: Path, camera: str, rel: str, seg=None) -> Path:
    """seg = (start_s, end_s) when a long file is analysed in chunks by several workers."""
    return out_dir / "shards" / camera / (_shard_stem(camera, rel) + (f"_s{int(seg[0])}-{int(seg[1])}" if seg else "") + ".db")


def shard_glob(out_dir: Path, camera: str, rel: str):
    """Every shard of one recording, whatever its chunking."""
    stem = _shard_stem(camera, rel)
    return [p for p in (out_dir / "shards" / camera).glob(stem + "*.db") if p.stem == stem or p.stem.startswith(stem + "_s")]


def shard_meta(path: Path) -> dict:
    """Meta table of a finished shard ({} if it cannot be read)."""
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return dict(con.execute("SELECT k, v FROM meta").fetchall())
        finally:
            con.close()
    except sqlite3.Error:
        return {}


def write_shard(path: Path, events, tables: dict, meta: dict):
    """Written to .tmp then renamed, so an interrupted run never leaves a half-finished shard.
    tables: minutes {minute: row}, zone_minutes {(zone, minute): row}, machine_minutes {(machine, minute): row}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.executescript(SHARD_SCHEMA)
    con.executemany("INSERT INTO events VALUES(?,?,?,?,?,?)", events)
    con.executemany("INSERT INTO minutes VALUES(?,?,?,?,?)", [(m, *r) for m, r in tables.get("minutes", {}).items()])
    con.executemany("INSERT INTO zone_minutes VALUES(?,?,?,?,?)", [(z, m, *r) for (z, m), r in tables.get("zone_minutes", {}).items()])
    con.executemany("INSERT INTO machine_minutes VALUES(?,?,?,?,?,?,?,?,?)",
                    [(n, m, *r) for (n, m), r in tables.get("machine_minutes", {}).items()])
    con.executemany("INSERT INTO fine VALUES(?,?,?,?)", [(b, *r) for b, r in tables.get("fine", {}).items()])
    con.executemany("INSERT INTO zone_fine VALUES(?,?,?,?)", [(z, b, *r) for (z, b), r in tables.get("zone_fine", {}).items()])
    con.executemany("INSERT INTO machine_fine VALUES(?,?,?,?)", [(m, b, *r) for (m, b), r in tables.get("machine_fine", {}).items()])
    con.executemany("INSERT INTO meta VALUES(?,?)", [(k, str(v)) for k, v in {**meta, "schema": SCHEMA_VERSION}.items()])
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
        if meta.get("schema") != SCHEMA_VERSION:
            raise RuntimeError(f"{sh} was written by an older version; delete the 'shards' folder in {out_dir} and run again")
        cam, src = meta["camera"], meta["source"]
        con.execute("BEGIN")
        con.execute("INSERT INTO events SELECT ?, ts, type, subject, value, clip, thumb, ? FROM s.events", (cam, src))
        if con.execute("SELECT COUNT(*) FROM s.sqlite_master WHERE type='table' AND name='fine'").fetchone()[0]:  # shards of older runs have none
            con.execute("""INSERT INTO fine SELECT ?, bin, covered_s, people_s, people_max FROM s.fine WHERE true
                           ON CONFLICT(camera, bin) DO UPDATE SET covered_s = covered_s + excluded.covered_s, people_s = people_s + excluded.people_s,
                             people_max = max(people_max, excluded.people_max)""", (cam,))
            con.execute("""INSERT INTO zone_fine SELECT ?, zone, bin, covered_s, occupied_s FROM s.zone_fine WHERE true
                           ON CONFLICT(camera, zone, bin) DO UPDATE SET covered_s = covered_s + excluded.covered_s, occupied_s = occupied_s + excluded.occupied_s""", (cam,))
            con.execute("""INSERT INTO machine_fine SELECT ?, machine, bin, work_s, idle_s FROM s.machine_fine WHERE true
                           ON CONFLICT(camera, machine, bin) DO UPDATE SET work_s = work_s + excluded.work_s, idle_s = idle_s + excluded.idle_s""", (cam,))
        con.execute("""INSERT INTO minutes SELECT ?, minute, covered_s, people_s, people_max, det_frames FROM s.minutes WHERE true
                       ON CONFLICT(camera, minute) DO UPDATE SET covered_s = covered_s + excluded.covered_s,
                         people_s = people_s + excluded.people_s, people_max = max(people_max, excluded.people_max),
                         det_frames = det_frames + excluded.det_frames""", (cam,))
        con.execute("""INSERT INTO zone_minutes SELECT ?, zone, minute, covered_s, occupied_s, people_s FROM s.zone_minutes WHERE true
                       ON CONFLICT(camera, zone, minute) DO UPDATE SET covered_s = covered_s + excluded.covered_s,
                         occupied_s = occupied_s + excluded.occupied_s, people_s = people_s + excluded.people_s""", (cam,))
        con.execute("""INSERT INTO machine_minutes SELECT ?, machine, minute, work_s, idle_s, alarm_s, wait_s, work_att_s,
                         idle_att_s, cycles FROM s.machine_minutes WHERE true
                       ON CONFLICT(camera, machine, minute) DO UPDATE SET work_s = work_s + excluded.work_s,
                         idle_s = idle_s + excluded.idle_s, alarm_s = alarm_s + excluded.alarm_s, wait_s = wait_s + excluded.wait_s,
                         work_att_s = work_att_s + excluded.work_att_s, idle_att_s = idle_att_s + excluded.idle_att_s,
                         cycles = cycles + excluded.cycles""", (cam,))
        con.execute("INSERT OR REPLACE INTO files VALUES(?,?,?,?,?,?,?)",
                    (src, cam, float(meta["start_ts"]), float(meta["video_s"]), float(meta["processed_s"]),
                     int(meta["det_frames"]), float(meta["elapsed_s"])))
        con.execute("COMMIT")
        con.execute("DETACH s")
    con.close()
    return db
