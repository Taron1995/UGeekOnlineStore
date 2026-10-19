"""Marking zones by hand: the shapes <-> config logic and saving (no window), then the window itself (needs a display)."""
import sqlite3

import pytest
import yaml

from tests.test_features import make_video
from videoanalytics import zoneeditor as ze
from videoanalytics.analyze import analyze
from videoanalytics.config import ConfigError, check_config, read_config
from videoanalytics.zoneeditor import Shape, apply_shapes, new_shape, next_name, open_session, problems, shapes_of

SQ = lambda x, y, s=0.2: [[round(a, 4), round(b, 4)] for a, b in [[x, y], [x + s, y], [x + s, y + s], [x, y + s]]]  # as the editor stores them: 4 decimals
LEFT_BOX, RIGHT_BOX = (40, 60, 70, 140), (240, 60, 270, 140)  # feet at x = 55 and 255 of 320
two_people = lambda frame: [(*LEFT_BOX, 0.9), (*RIGHT_BOX, 0.9)]


def video(path, seconds=20):
    path.parent.mkdir(parents=True, exist_ok=True)
    make_video(path, seconds, lambda f, t: None)
    return path


# ---- shapes <-> config ----------------------------------------------------------------------------------------------------
CAM = dict(match="cam1", schedule=dict(work=[["08:00", "17:00"]]), gathering=dict(min_duration_s=120),
           zones=[dict(name="press", kind="workstation", polygon=SQ(0.1, 0.5), hold_s=9.0, schedule=dict(work=[["09:00", "12:00"]])),
                  dict(name="lounge", kind="rest", polygon=SQ(0.6, 0.6))],
           machines=[dict(name="m1", source="motion", roi=SQ(0.1, 0.2), zone="press", motion_thr=0.05, window_s=20, cycle=dict(min_work_s=2.0)),
                     dict(name="l1", source="lamp", roi=SQ(0.8, 0.1, 0.05), work_colors=["green", "yellow"])],
           areas=[dict(name="floor", polygon=SQ(0, 0, 1), mode="watch")])


def test_shapes_round_trip_keeps_every_setting():
    shapes = shapes_of(CAM)
    assert [(s.kind, s.name) for s in shapes] == [("watch", "floor"), ("station", "press"), ("rest", "lounge"), ("motion", "m1"), ("lamp", "l1")]
    back = apply_shapes(CAM, shapes)
    assert back["zones"][0] == dict(CAM["zones"][0], polygon=CAM["zones"][0]["polygon"]) and back["match"] == "cam1"  # hold_s and the zone's own schedule survive
    assert back["machines"][0]["cycle"] == {"min_work_s": 2.0} and back["machines"][0]["motion_thr"] == 0.05 and back["machines"][1]["work_colors"] == ["green", "yellow"]
    assert back["schedule"] == CAM["schedule"] and back["gathering"] == CAM["gathering"]  # other keys untouched
    assert back["machines"][1].get("zone") is None  # l1 was not linked and stays so (a loaded machine keeps its link choice)
    check_config(dict(cameras=dict(cam1=back)))


def test_a_moved_shape_keeps_its_settings_a_new_one_gets_defaults():
    shapes = shapes_of(CAM)
    shapes[1].points = SQ(0.3, 0.3)
    shapes.append(new_shape("motion", "motion1", SQ(0.4, 0.1)))
    shapes.append(new_shape("lamp", "lamp1", SQ(0.9, 0.9, 0.05)))
    back = apply_shapes(CAM, shapes)
    assert back["zones"][0]["polygon"] == SQ(0.3, 0.3) and back["zones"][0]["hold_s"] == 9.0
    motion, lamp = back["machines"][2], back["machines"][3]
    assert motion["motion_thr"] == 0.02 and motion["window_s"] == 10 and lamp["work_colors"] == ["green"]  # what the scene learning gives new machines too


def test_coordinates_are_clamped_and_rounded():
    back = apply_shapes({}, [new_shape("station", "a", [[-0.2, 0.123456789], [1.4, 0.0], [0.5, 1.0]])])
    assert back["zones"][0]["polygon"] == [[0.0, 0.1235], [1.0, 0.0], [0.5, 1.0]]


def test_machines_find_their_station_unless_told_otherwise():
    station = new_shape("station", "spot1", SQ(0.1, 0.5))
    near, far, none, chosen = (new_shape("motion", "near", SQ(0.1, 0.3)), new_shape("motion", "far", SQ(0.8, 0.0)),
                               new_shape("motion", "none", SQ(0.1, 0.3)), new_shape("lamp", "chosen", SQ(0.8, 0.0, 0.05)))
    none.zone, chosen.zone = "", "spot1"
    got = {m["name"]: m.get("zone") for m in apply_shapes({}, [station, near, far, none, chosen])["machines"]}
    assert got == {"near": "spot1", "far": None, "none": None, "chosen": "spot1"}  # nearest within reach / too far / "(none)" / picked by hand


def test_a_machine_of_a_deleted_station_looks_for_the_nearest_again_and_the_setup_stays_valid():
    shapes = shapes_of(CAM)
    shapes = [s for s in shapes if s.name != "press"]  # m1 was linked to it
    back = apply_shapes(CAM, shapes)
    assert "zone" not in back["machines"][0]  # no station left to belong to
    check_config(dict(cameras=dict(cam1=back)))  # the learned-config trap: a link to a zone that no longer exists would be refused
    bench = new_shape("station", "bench", SQ(0.15, 0.45))
    again = apply_shapes(CAM, shapes + [bench])
    assert again["machines"][0]["zone"] == "bench"  # another station is near: the machine belongs to it now


def test_names_follow_the_kind_and_skip_taken_ones():
    taken = [new_shape("station", "spot1", SQ(0, 0)), new_shape("station", "spot3", SQ(0.3, 0))]
    assert next_name("station", taken) == "spot2" and next_name("watch", taken) == "area1" and next_name("lamp", taken) == "lamp1"
    assert next_name("station", taken + [new_shape("station", "spot2", SQ(0.6, 0))]) == "spot4"


def test_problems_are_said_in_plain_words():
    ok = [new_shape("station", "a", SQ(0.1, 0.1)), new_shape("watch", "a", SQ(0, 0, 1)), new_shape("motion", "a", SQ(0.5, 0.5))]  # one name in three kinds of thing: fine
    assert problems(ok) == []
    bad = [new_shape("station", "a", SQ(0.1, 0.1)), new_shape("rest", "a", SQ(0.5, 0.5)), new_shape("station", " ", SQ(0.7, 0.7)),
           Shape("station", "b", [[0, 0], [1, 1]]), new_shape("lamp", "tiny", SQ(0.5, 0.5, 0.01))]
    text = " ".join(problems(bad))
    for fragment in ("'a' is used twice among the zones", "has no name", "'b' needs at least 3 corners", "'tiny' is too small"):
        assert fragment in text


# ---- inputs and saving -------------------------------------------------------------------------------------------------------
def two_cameras(tmp_path):
    video(tmp_path / "cam1_20261001_080000.mp4")
    video(tmp_path / "cam2_20261001_080000.mp4")
    return tmp_path


def test_a_new_setup_has_every_camera_and_nothing_drawn(tmp_path):
    session = open_session([str(two_cameras(tmp_path))])
    assert session.created and session.cameras == ["cam1", "cam2"] and session.out == tmp_path.resolve() / "_analysis"
    assert session.setup_path == session.out / "cameras.auto.yaml" and not session.setup_path.exists()
    assert all(c["zones"] == [] and c["machines"] == [] and "match" in c for c in session.cfg["cameras"].values())
    assert session.shapes("cam1") == []


def test_saving_writes_the_setup_pictures_and_a_backup(tmp_path):
    session = open_session([str(two_cameras(tmp_path))])
    frame = (session.files["cam1"][0]["path"], 5.0)
    shapes = [new_shape("watch", "floor", SQ(0.05, 0.05, 0.9)), new_shape("station", "spot1", SQ(0.1, 0.5)), new_shape("motion", "motion1", SQ(0.1, 0.2))]
    path = session.save({"cam1": shapes}, {"cam1": frame, "cam2": (session.files["cam2"][0]["path"], 5.0)})  # cam2 was looked at, not changed
    cfg = read_config(path)  # valid for the analysis
    assert sorted(cfg["cameras"]) == ["cam1", "cam2"] and cfg["cameras"]["cam2"]["zones"] == []  # the camera that was not touched is there, unchanged
    cam1 = cfg["cameras"]["cam1"]
    assert [a["name"] for a in cam1["areas"]] == ["floor"] and cam1["zones"][0]["name"] == "spot1" and cam1["machines"][0]["zone"] == "spot1"
    assert path.read_text(encoding="utf-8").startswith("# Your setup") and "polygon: [[" in path.read_text(encoding="utf-8")  # polygons on one line
    assert (session.out / "setup" / "cam1.jpg").exists() and not (session.out / "setup" / "cam2.jpg").exists()
    assert read_config(path) and yaml.safe_load((session.out / "setup" / "setup.json").read_text()) == {"cam1": {"image": "setup/cam1.jpg"}}
    assert not session.created and not path.with_name(path.name + ".bak").exists()  # first write: nothing to back up
    session.save({"cam1": shapes[:2]})
    assert yaml.safe_load(path.with_name(path.name + ".bak").read_text())["cameras"]["cam1"]["machines"]  # the previous version is kept


def test_an_existing_setup_is_loaded_and_its_other_keys_survive(tmp_path):
    two_cameras(tmp_path)
    out = tmp_path / "_analysis"
    out.mkdir()
    (out / "cameras.auto.yaml").write_text(yaml.safe_dump(dict(options=dict(fps=5), schedule=dict(work=[["08:00", "17:00"]]),
                                                               cameras=dict(cam1=dict(CAM, match="cam1"), cam2=dict(match="cam2")))))
    session = open_session([str(tmp_path)])
    assert not session.created and [s.name for s in session.shapes("cam1")] == ["floor", "press", "lounge", "m1", "l1"]
    path = session.save({"cam2": [new_shape("station", "spot1", SQ(0.2, 0.2))]})
    cfg = read_config(path)
    assert cfg["options"] == {"fps": 5} and cfg["schedule"] == dict(work=[["08:00", "17:00"]]) and cfg["cameras"]["cam1"]["zones"][0]["hold_s"] == 9.0
    assert cfg["cameras"]["cam2"]["zones"][0]["name"] == "spot1"


def test_a_setup_the_analysis_would_refuse_is_not_written(tmp_path):
    session = open_session([str(two_cameras(tmp_path))])
    with pytest.raises(ConfigError, match="'a' is used twice"):
        session.save({"cam1": [new_shape("station", "a", SQ(0.1, 0.1)), new_shape("rest", "a", SQ(0.5, 0.5))]})
    assert not session.setup_path.exists()


def test_a_refused_save_leaves_the_files_alone(tmp_path):
    session = open_session([str(two_cameras(tmp_path))])
    path = session.save({"cam1": [new_shape("station", "spot1", SQ(0.1, 0.1))]})
    before = path.read_text(encoding="utf-8")
    broken = new_shape("station", "spot2", SQ(0.5, 0.5))
    broken.extra["colour"] = "red"  # a key the analysis does not know: its own check refuses the file
    with pytest.raises(ConfigError, match="unknown key 'colour'"):
        session.save({"cam1": [new_shape("station", "spot1", SQ(0.1, 0.1)), broken]})
    assert path.read_text(encoding="utf-8") == before and not path.with_name(path.name + ".bak").exists()  # not even a backup was made


def test_the_cameras_of_the_editor_are_those_of_the_analysis(tmp_path):
    for name in ("video1", "video2"):
        video(tmp_path / f"{name}.mp4")
    assert open_session([str(tmp_path)]).cameras == ["video1", "video2"]
    assert open_session([str(tmp_path)], one_camera="Lathe 3").cameras == ["Lathe_3"]  # same switches as analyze
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "notes.txt").write_text("x")
    with pytest.raises(SystemExit, match="no video files"):
        open_session([str(tmp_path / "empty")])


def test_what_was_drawn_is_what_the_analysis_uses(tmp_path):
    """Editor -> analyze, end to end: a tracking area on the left half means only the left person is counted."""
    v = video(tmp_path / "cam1_20261001_080000.mp4", 20)
    session = open_session([str(v)])
    left_only = [new_shape("watch", "floor", [[0, 0], [0.5, 0], [0.5, 1], [0, 1]])]
    session.save({"cam1": left_only}, {"cam1": (str(v), 5.0)})
    analyze([str(v)], detect=two_people, open_report=False, workers=1)
    con = sqlite3.connect(session.out / "analytics.db")
    people_s, covered_s = con.execute("SELECT SUM(people_s), SUM(covered_s) FROM minutes").fetchone()
    assert people_s / covered_s == pytest.approx(1.0, abs=0.1)  # one person in view on average, not two
    report = (session.out / "report.html").read_text(encoding="utf-8")
    assert "tracking area <b>floor</b>" in report and "drawn by hand" in report and 'src="setup/cam1.jpg"' in report


def test_the_window_offers_six_kinds_of_shape():
    assert list(ze.KINDS) == ["watch", "ignore", "station", "rest", "motion", "lamp"] and all(len(v) == 3 for v in ze.KINDS.values())


def test_the_zones_command_passes_everything_on(tmp_path, monkeypatch):
    from videoanalytics.cli import main
    seen = []
    monkeypatch.setattr(ze, "main", lambda paths, **kw: seen.append((paths, kw)))
    main(["zones", "a.mp4", "b", "--out", "res", "--config", "c.yaml", "--camera", "cam2", "--time-regex", "x", "--start", "2026-10-01 08:30", "--camera-name", "Lathe 3"])
    (paths, kw), = seen
    assert paths == ["a.mp4", "b"] and kw["out"] == "res" and kw["config"] == "c.yaml" and kw["camera"] == "cam2" and kw["time_regex"] == "x"
    assert kw["one_camera"] == "Lathe 3" and kw["start"] == pytest.approx(1790843400.0)  # 2026-10-01 08:30 as wall-clock seconds
    seen.clear()
    main(["zones", "a.mp4", "--one-camera"])
    assert seen[0][1]["one_camera"] == "" and seen[0][1]["start"] is None and seen[0][1]["camera"] is None
