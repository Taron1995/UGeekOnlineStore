"""Обработка одной камеры: RF-DETR -> ByteTrack -> зоны/линии -> таймеры -> события."""
from __future__ import annotations

import itertools
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from .config import AppCfg, CameraCfg
from .motion import MotionMeter
from .posture import PoseAnalyzer, is_seated, pose_motion
from .sink import EventRec, EventSink
from .timers import TimerBank

log = logging.getLogger(__name__)


def _sel(dets: sv.Detections, class_id: int) -> sv.Detections:
    if len(dets) == 0 or dets.class_id is None:
        return sv.Detections.empty()
    return dets[dets.class_id == class_id]


class CameraWorker:
    def __init__(self, cam: CameraCfg, cfg: AppCfg, detector, sink: EventSink, pose: PoseAnalyzer | None):
        self.cam, self.cfg, self.sink, self.detector, self.pose = cam, cfg, sink, detector, pose
        self.th, self.sched = cfg.thresholds, cfg.schedule
        fps = max(int(self.th.process_fps), 1)
        self.person_tracker = sv.ByteTrack(frame_rate=fps)
        self.piece_tracker = sv.ByteTrack(frame_rate=fps)
        self.zones = {w.id: sv.PolygonZone(np.array(w.station_zone), triggering_anchors=[sv.Position.BOTTOM_CENTER])
                      for w in cam.workplaces}
        self.gather = [sv.PolygonZone(np.array(z), triggering_anchors=[sv.Position.BOTTOM_CENTER])
                       for z in cam.gathering_zones]
        self.lines = {w.id: sv.LineZone(start=sv.Point(*w.piece_line[0]), end=sv.Point(*w.piece_line[1]))
                      for w in cam.workplaces if w.piece_line}
        self.motion: dict[str, MotionMeter] = {}
        self.cls = cfg.detector.classes

        t = self.th
        self.t_present = TimerBank(t.present_on_s, t.present_off_s)
        self.t_absent = TimerBank(t.absence_after_s, t.absence_clear_s)
        self.t_run = TimerBank(t.machine_on_s, t.machine_off_s)
        self.t_idle = TimerBank(t.idle_after_s, 2.0)
        self.t_phone = TimerBank(t.phone_after_s, t.phone_clear_s)
        self.t_meet = TimerBank(t.meeting_after_s, t.meeting_clear_s)
        self.t_sit = TimerBank(t.sit_still_after_s, t.sit_still_clear_s)

        self.owner: dict[int, str] = {}                      # track_id -> workplace_id
        self.zone_time: dict[tuple[int, str], float] = defaultdict(float)
        self.pose_state: dict[int, dict] = {}
        self.last_ts: float | None = None
        self.video_start: float | None = None
        self.is_file = Path(cam.source).is_file()

    # ---------- чтение кадров ----------
    def frames(self):
        src, th = self.cam.source, self.th
        while True:
            cap = cv2.VideoCapture(src)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            stride = max(int(round(fps / th.process_fps)), 1)
            if self.is_file:
                self.video_start = (datetime.fromisoformat(self.cam.start_time).timestamp()
                                    if self.cam.start_time else time.time())
            i, last = 0, 0.0
            while cap.grab():
                if self.is_file:
                    take = i % stride == 0
                    ts = self.video_start + i / fps
                else:
                    ts = time.time()
                    take = ts - last >= 1.0 / th.process_fps
                i += 1
                if not take:
                    continue
                last = ts
                ok, frame = cap.retrieve()
                if ok:
                    yield frame, ts
            cap.release()
            if self.is_file:
                return
            log.warning("%s: поток оборвался, переподключение через 5 с", self.cam.id)
            time.sleep(5)

    # ---------- события ----------
    def _emit(self, typ: str, start: float, end: float, wp: str | None = None,
              tid: int | None = None, meta: dict | None = None):
        self.sink.emit(EventRec(
            type=typ, start=start, end=end, camera_id=self.cam.id,
            workplace_id=wp if wp is not None else self.owner.get(tid),
            track_id=tid, meta=meta or {},
            video_src=self.cam.source if self.is_file else None,
            video_offset_s=(start - self.video_start) if self.is_file else None))

    def _emit_closed(self, typ: str, closed, per_track: bool):
        for key, s, e in closed:
            if typ == "MEETING":
                a, b = key
                for me, other in ((a, b), (b, a)):
                    self._emit(typ, s, e, tid=me, meta={"partner_track": other,
                                                         "partner_workplace": self.owner.get(other)})
            elif per_track:
                self._emit(typ, s, e, tid=key)
            else:
                self._emit(typ, s, e, wp=key)

    # ---------- основной шаг ----------
    def process(self, frame: np.ndarray, ts: float):
        th = self.th
        dt = (ts - self.last_ts) if self.last_ts else 1.0 / th.process_fps
        self.last_ts = ts
        shift = self.sched.shift_at(self.sched.to_dt(ts))

        x0, y0 = 0, 0
        sub = frame
        if self.cam.roi:
            x0, y0, x1, y1 = self.cam.roi
            sub = frame[y0:y1, x0:x1]
        dets = self.detector.detect(sub)
        if len(dets) and (x0 or y0):
            dets.xyxy = dets.xyxy + np.array([x0, y0, x0, y0], dtype=np.float32)

        persons = self.person_tracker.update_with_detections(_sel(dets, self.cls["person"]))
        if len(persons) and persons.tracker_id is not None:
            persons = persons[persons.tracker_id >= 0]
        phones = _sel(dets, self.cls["phone"])
        pieces = self.piece_tracker.update_with_detections(_sel(dets, self.cls["workpiece"]))
        tids = persons.tracker_id if len(persons) else np.array([], dtype=int)

        # --- присутствие у станка и привязка треков к рабочему месту
        present: set[str] = set()
        in_station = np.zeros(len(persons), dtype=bool)
        zone_of: dict[int, str] = {}
        for wp_id, zone in self.zones.items():
            mask = zone.trigger(persons) if len(persons) else np.array([], dtype=bool)
            if mask.any():
                present.add(wp_id)
            in_station |= mask
            for tid in tids[mask]:
                zone_of[int(tid)] = wp_id
                self.zone_time[(int(tid), wp_id)] += dt
                if self.zone_time[(int(tid), wp_id)] >= th.owner_bind_s:
                    self.owner[int(tid)] = wp_id
        self._emit_closed("AT_STATION", self.t_present.update(ts, present), False)
        on_shift_wps = {w.id for w in self.cam.workplaces} if shift else set()
        self._emit_closed("ABSENT", self.t_absent.update(ts, on_shift_wps - present), False)

        # --- станок: движение в ROI без учёта людей
        running: set[str] = set()
        for w in self.cam.workplaces:
            if not w.machine_roi:
                continue
            mm = self.motion.setdefault(w.id, MotionMeter(w.machine_roi, frame.shape))
            frac = mm.measure(frame, persons.xyxy if len(persons) else np.empty((0, 4)))
            if frac is not None and frac >= th.machine_motion_thr:
                running.add(w.id)
        self._emit_closed("MACHINE_RUN", self.t_run.update(ts, running), False)
        idle = {w.id for w in self.cam.workplaces
                if w.machine_roi and shift and not self.t_run.is_on(w.id)}
        self._emit_closed("MACHINE_IDLE", self.t_idle.update(ts, idle), False)

        # --- заготовки: пересечение линии выхода
        if len(pieces) and pieces.tracker_id is not None:
            for wp_id, line in self.lines.items():
                crossed_in, crossed_out = line.trigger(pieces)
                for _ in range(int(crossed_in.sum())):
                    self._emit("PIECE", ts, ts, wp=wp_id)

        # --- телефон у человека
        phone_tids: set[int] = set()
        for x1, y1, x2, y2 in (phones.xyxy if len(phones) else []):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            for (px1, py1, px2, py2), tid in zip(persons.xyxy, tids):
                mx, my = 0.1 * (px2 - px1), 0.1 * (py2 - py1)
                if px1 - mx <= cx <= px2 + mx and py1 - my <= cy <= py2 + my:
                    phone_tids.add(int(tid))
                    break
        self._emit_closed("PHONE", self.t_phone.update(ts, phone_tids), True)

        # --- собрания: ≥2 человек рядом без производственной причины
        self._emit_closed("MEETING", self.t_meet.update(ts, self._meeting_pairs(persons, tids, zone_of)), False)

        # --- сидит без движения (MediaPipe Pose, раз в pose_interval_s на человека)
        if self.pose is not None:
            self._update_pose(frame, persons, tids, ts)
        sit = {t for t, st in self.pose_state.items() if st.get("seated") and st.get("still")}
        self._emit_closed("SITTING_STILL", self.t_sit.update(ts, sit), True)

        gone = set(self.pose_state) - {int(t) for t in tids}
        for t in gone:
            if ts - self.pose_state[t]["ts"] > 30:
                del self.pose_state[t]

    def _meeting_pairs(self, persons, tids, zone_of) -> set[tuple[int, int]]:
        th, out = self.th, set()
        if len(persons) < 2:
            return out
        xyxy = persons.xyxy
        centers = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2], 1)
        heights = xyxy[:, 3] - xyxy[:, 1]
        in_gather = np.zeros(len(persons), dtype=bool)
        for g in self.gather:
            in_gather |= g.trigger(persons)
        for i, j in itertools.combinations(range(len(persons)), 2):
            ti, tj = int(tids[i]), int(tids[j])
            if in_gather[i] and in_gather[j]:
                continue
            wi, wj = zone_of.get(ti), zone_of.get(tj)
            if wi is not None and wi == wj:
                continue  # двое у одного станка — это работа
            dist = float(np.linalg.norm(centers[i] - centers[j]))
            if dist < th.meeting_dist_heights * float(heights[[i, j]].mean()):
                out.add((min(ti, tj), max(ti, tj)))
        return out

    def _update_pose(self, frame, persons, tids, ts):
        th = self.th
        H, W = frame.shape[:2]
        for box, tid in zip(persons.xyxy.astype(int), tids):
            tid = int(tid)
            st = self.pose_state.setdefault(tid, {"ts": 0.0})
            if ts - st["ts"] < th.pose_interval_s:
                continue
            x1, y1, x2, y2 = max(box[0], 0), max(box[1], 0), min(box[2], W), min(box[3], H)
            res = self.pose.analyze(frame[y1:y2, x1:x2])
            prev_ts, st["ts"] = st["ts"], ts
            if res is None:
                st["seated"] = None
                continue
            pts, vis = res
            st["seated"] = is_seated(pts, vis, th.seated_ratio)
            mot = pose_motion(st.get("pts"), pts, st.get("vis"), vis)
            st["still"] = mot is not None and mot < th.still_motion_thr
            st["pts"], st["vis"] = pts, vis

    def flush(self):
        ts = self.last_ts or time.time()
        for typ, bank, per_track in (("AT_STATION", self.t_present, False), ("ABSENT", self.t_absent, False),
                                     ("MACHINE_RUN", self.t_run, False), ("MACHINE_IDLE", self.t_idle, False),
                                     ("PHONE", self.t_phone, True), ("MEETING", self.t_meet, False),
                                     ("SITTING_STILL", self.t_sit, True)):
            self._emit_closed(typ, bank.flush(ts), per_track)

    def run(self):
        log.info("камера %s: старт (%s)", self.cam.id, self.cam.source)
        try:
            for frame, ts in self.frames():
                self.process(frame, ts)
        finally:
            self.flush()
            log.info("камера %s: остановлена", self.cam.id)
