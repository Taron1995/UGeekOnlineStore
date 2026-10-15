import json

import cv2
import numpy as np
import pytest

from tests.test_features import FPS, SIZE, LAMP_ROI, GREEN, lamp, make_video, src
from videoanalytics.tools import extract_frames, probe_activity, snapshot, split_coco


def textured(path, seconds):
    """Sharp, time-varying frames (noise + brightness steps) so the blur / duplicate filters let them through."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    rng = np.random.default_rng(0)
    for i in range(int(seconds * FPS)):
        f = np.clip(rng.integers(60, 120, (SIZE[1], SIZE[0], 3)) + 12 * (int(i / FPS) % 8), 0, 255).astype(np.uint8)
        w.write(f)
    w.release()


def source(path, name, seconds):
    s = src(path, seconds)
    return {**s, "rel": name, "camera": "cam3"}


def test_extract_frames_spreads_over_all_files(tmp_path):
    for n in ("a", "b"):
        textured(tmp_path / f"{n}.mp4", 24)
    srcs = [source(tmp_path / "a.mp4", "a", 24), {**source(tmp_path / "b.mp4", "b", 24), "start_ts": src(tmp_path / "x", 1)["start_ts"] + 7200}]
    n = extract_frames(srcs, str(tmp_path / "raw"), count=6)
    files = sorted(p.name for p in (tmp_path / "raw").glob("*.jpg"))
    assert n == len(files) >= 4
    assert any("_08" in f for f in files) and any("_10" in f for f in files)  # both recordings contribute (08:xx and 10:xx)


def test_extract_frames_skips_flat_frames(tmp_path):
    make_video(tmp_path / "flat.mp4", 10, lambda f, t: None)  # no texture: blurry by definition
    assert extract_frames([source(tmp_path / "flat.mp4", "flat", 10)], str(tmp_path / "raw"), count=5) == 0


def coco(tmp_path, n=30):
    (tmp_path / "img").mkdir()
    images, anns = [], []
    for i in range(n):
        cv2.imwrite(str(tmp_path / "img" / f"f{i:03d}.jpg"), np.zeros((40, 60, 3), np.uint8))
        images.append(dict(id=i + 1, file_name=f"sub/f{i:03d}.jpg", width=60, height=40))  # CVAT-style relative name
        anns.append(dict(id=i + 1, image_id=i + 1, category_id=1, bbox=[1, 2, 10, 10], area=100, iscrowd=0))
    p = tmp_path / "instances.json"
    p.write_text(json.dumps(dict(images=images, annotations=anns, categories=[dict(id=1, name="workpiece")])))
    return p


def test_split_coco_makes_rfdetr_layout(tmp_path):
    counts = split_coco(str(tmp_path / "img"), str(coco(tmp_path)), str(tmp_path / "ds"), valid=0.2, test=0.1)
    assert counts == {"test": 3, "valid": 6, "train": 21}
    for name, n in counts.items():
        data = json.loads((tmp_path / "ds" / name / "_annotations.coco.json").read_text())
        assert len(data["images"]) == n == len(data["annotations"]) == len(list((tmp_path / "ds" / name).glob("*.jpg")))
        assert data["categories"][0]["name"] == "workpiece" and "/" not in data["images"][0]["file_name"]


def test_split_coco_names_missing_images(tmp_path):
    p = coco(tmp_path)
    next((tmp_path / "img").glob("f000.jpg")).unlink()
    with pytest.raises(FileNotFoundError, match="f000.jpg"):
        split_coco(str(tmp_path / "img"), str(p), str(tmp_path / "ds"))


def test_snapshot_reads_the_lamp_colour(tmp_path):
    make_video(tmp_path / "l.mp4", 3, lambda f, t: lamp(f, GREEN))
    out = tmp_path / "snap.jpg"
    snapshot(str(tmp_path / "l.mp4"), out, dict(machines=[dict(name="press", source="lamp", roi=LAMP_ROI)], zones=[dict(name="z", polygon=[[0, 0], [1, 0], [1, 1]])]), 1.0)
    img = cv2.imread(str(out))
    assert img.shape == (SIZE[1], SIZE[0], 3)
    assert (img[72:108, 144:176].astype(int) != 90).any()  # the lamp ROI outline / label were drawn on the frame


def test_probe_prints_activity_per_person(tmp_path, capsys):
    make_video(tmp_path / "p.mp4", 14, lambda f, t: None)
    box = (40, 100, 70, 180, 0.9)
    probe_activity(src(tmp_path / "p.mp4", 14), {}, lambda f: [box], "motion", None, 0.0, 12.0)
    out = capsys.readouterr().out
    assert "method=motion" in out and "100%" in out  # a person on a still picture is always below the threshold
