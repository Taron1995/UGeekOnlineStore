from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from .schedule import Schedule, Shift, parse_hhmm


@dataclass
class Thresholds:
    process_fps: float = 5.0
    det_conf: float = 0.4
    # присутствие / отсутствие у станка
    present_on_s: float = 2.0
    present_off_s: float = 5.0
    absence_after_s: float = 30.0
    absence_clear_s: float = 5.0
    # станок: доля изменившихся пикселей в ROI
    machine_motion_thr: float = 0.02
    machine_on_s: float = 3.0
    machine_off_s: float = 10.0
    idle_after_s: float = 10.0
    # телефон
    phone_after_s: float = 3.0
    phone_clear_s: float = 5.0
    # собрание: расстояние между людьми в «ростах»
    meeting_after_s: float = 60.0
    meeting_clear_s: float = 10.0
    meeting_dist_heights: float = 1.5
    # сидит без движения
    pose_interval_s: float = 1.0
    seated_ratio: float = 0.5
    still_motion_thr: float = 0.02
    sit_still_after_s: float = 120.0
    sit_still_clear_s: float = 5.0
    # привязка трека к рабочему месту
    owner_bind_s: float = 10.0


@dataclass
class DetectorCfg:
    backend: str = "rfdetr"          # rfdetr | onnx
    size: str = "small"              # small | medium
    weights: str | None = None       # свои веса после дообучения
    onnx_path: str | None = None
    providers: list[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    # id классов в ВАШЕЙ модели (для COCO: person=1, cell phone=77; workpiece — после дообучения)
    classes: dict[str, int] = field(default_factory=lambda: {"person": 1, "phone": 77, "workpiece": 100})


@dataclass
class WorkplaceCfg:
    id: str
    station_zone: list[list[int]]
    machine_roi: list[list[int]] | None = None
    piece_line: list[list[int]] | None = None   # [[x1,y1],[x2,y2]]


@dataclass
class CameraCfg:
    id: str
    source: str
    workplaces: list[WorkplaceCfg]
    roi: list[int] | None = None                 # [x1,y1,x2,y2] — кроп для детектора
    gathering_zones: list[list[list[int]]] = field(default_factory=list)  # где собираться допустимо
    start_time: str | None = None                # ISO-время начала для видеофайла


@dataclass
class AppCfg:
    schedule: Schedule
    db_url: str
    clips_dir: str
    detector: DetectorCfg
    thresholds: Thresholds
    cameras: list[CameraCfg]


def _build(cls, raw: dict):
    names = {f.name for f in fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"{cls.__name__}: неизвестные ключи {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path) -> AppCfg:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    breaks = [(parse_hhmm(b["start"]), parse_hhmm(b["end"])) for b in raw["breaks"]]
    shifts = [Shift(s["name"], parse_hhmm(s["start"]), parse_hhmm(s["end"])) for s in raw["shifts"]]
    cams = []
    for c in raw["cameras"]:
        wps = [_build(WorkplaceCfg, w) for w in c.pop("workplaces")]
        cams.append(CameraCfg(workplaces=wps, **c))
    return AppCfg(
        schedule=Schedule(raw["timezone"], breaks, shifts),
        db_url=os.environ.get("DATABASE_URL", raw["db_url"]),
        clips_dir=raw.get("clips_dir", "clips"),
        detector=_build(DetectorCfg, raw.get("detector", {})),
        thresholds=_build(Thresholds, raw.get("thresholds", {})),
        cameras=cams,
    )
