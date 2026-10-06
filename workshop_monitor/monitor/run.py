"""CLI:  python -m monitor.run init-db | worker [--camera cam1] | api"""
from __future__ import annotations

import argparse
import logging
import threading

from .config import load_config
from .db import Repo, Workplace, init_db, make_engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init-db", "worker", "api"])
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--camera", help="id камеры; по умолчанию — все")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.cmd == "api":
        import uvicorn
        uvicorn.run("monitor.api:app", host="0.0.0.0", port=8000)
        return

    cfg = load_config(args.config)
    engine = make_engine(cfg.db_url)
    init_db(engine)
    repo = Repo(engine)
    with repo.session() as s:
        for cam in cfg.cameras:
            for w in cam.workplaces:
                s.merge(Workplace(id=w.id, name=w.id, camera_id=cam.id))
        s.commit()
    if args.cmd == "init-db":
        return

    from .detector import build_detector
    from .pipeline import CameraWorker
    from .posture import PoseAnalyzer
    from .sink import EventSink

    sink = EventSink(repo, cfg.schedule, cfg.clips_dir)
    cams = [c for c in cfg.cameras if not args.camera or c.id == args.camera]
    threads = []
    for cam in cams:
        w = CameraWorker(cam, cfg, build_detector(cfg.detector, cfg.thresholds.det_conf), sink, PoseAnalyzer())
        t = threading.Thread(target=w.run, name=cam.id, daemon=False)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
