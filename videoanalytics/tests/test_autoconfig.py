import itertools

import numpy as np

from tests.test_features import SIZE, make_video, src
from videoanalytics.autoconfig import build_camera_config, learn_scene, sample_windows
from videoanalytics.config import check_config, normalize_camera
from videoanalytics.zones import Zone

W, H = SIZE


def inside(zone_poly, x, y):
    return Zone("z", zone_poly).contains(x, y, W, H)


# ---------- stations: where people stay ----------
STILL_A, STILL_B = (40, 90, 70, 170), (230, 90, 260, 170)  # feet at (55, 170) and (245, 170)


def stations_detect():
    n = itertools.count()

    def detect(frame):
        i = next(n)
        walker = ((i * 37) % 280, 90, (i * 37) % 280 + 30, 170)  # crosses the picture quickly, never stands still
        return [(*STILL_A, 0.9), (*STILL_B, 0.9), (*walker, 0.9)]
    return detect


def test_stations_are_where_people_stand_still_not_where_they_walk(tmp_path):
    v = tmp_path / "s.mp4"
    make_video(v, 80, lambda f, t: None)
    cfg, info, _ = learn_scene([src(v, 80)], stations_detect(), minutes=2, fps=2, lamps=True)
    zones = cfg["zones"]
    assert len(zones) == 2 and info["frames"] > 100
    assert any(inside(z["polygon"], 55, 170) for z in zones) and any(inside(z["polygon"], 245, 170) for z in zones)
    assert not any(inside(z["polygon"], 150, 170) for z in zones)  # the middle of the picture is only walked through
    assert all(z["kind"] == "workstation" for z in zones) and [z["name"] for z in zones] == ["spot1", "spot2"]


def test_a_short_stop_does_not_make_a_station(tmp_path):
    """Standing 40 s counts only 25 s (the first dwell_s = 15 s are the arrival), below the 30 s a station needs."""
    v = tmp_path / "s.mp4"
    make_video(v, 60, lambda f, t: None)
    n = itertools.count()
    cfg, _, _ = learn_scene([src(v, 60)], lambda f: [(*STILL_A, 0.9)] if next(n) < 80 else [], minutes=2, fps=2, lamps=True, dwell_s=15)
    assert cfg["zones"] == []
    n = itertools.count()
    cfg, _, _ = learn_scene([src(v, 60)], lambda f: [(*STILL_A, 0.9)] if next(n) < 110 else [], minutes=2, fps=2, lamps=True, dwell_s=15)
    assert len(cfg["zones"]) == 1  # 55 s standing -> 40 s counted -> a station


# ---------- machines: persistent motion, minus people and clock overlays ----------
MACHINE = (200, 40, 260, 100)


def draw_scene(f, t):
    x = 200 + int(t * 2 * 20) % 40  # a bar sweeping inside the machine region, visible at every sample
    f[50:90, x:x + 15] = 255
    f[2:14, 5:80] = 255 if int(t) % 2 == 0 else 90  # the CCTV clock overlay changes every second
    px = int(t * 30) % 280  # a blue "person" walking through (exactly what the stub detector reports)
    f[100:170, px:px + 30] = (255, 0, 0)


def blue_person(frame):
    m = (frame[:, :, 0] > 200) & (frame[:, :, 1] < 50) & (frame[:, :, 2] < 50)
    ys, xs = np.where(m)
    return [(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()), 0.9)] if len(xs) else []


def test_machine_found_people_and_clock_ignored(tmp_path):
    v = tmp_path / "m.mp4"
    make_video(v, 40, draw_scene)
    cfg, info, _ = learn_scene([src(v, 40)], blue_person, minutes=2, fps=2, lamps=True)
    (m,) = cfg["machines"]
    xs = [p[0] * W for p in m["roi"]]
    ys = [p[1] * H for p in m["roi"]]
    assert m["source"] == "motion" and m["name"] == "motion1"
    assert min(xs) <= MACHINE[0] + 12 and max(xs) >= MACHINE[0] + 40 and min(ys) <= 55 and max(ys) >= 85  # covers the sweeping bar
    assert min(ys) > 20  # not the clock at the top of the picture
    assert info["motion"][0]["freq"] > 0.5


# ---------- lamps: a switching stack light, not an always-on LED or the REC dot ----------
def draw_lamps(f, t):
    phase = int(t // 3) % 3  # green, yellow, red, ...
    for i, (colour, lit) in enumerate([((0, 0, 255), phase == 2), ((0, 255, 255), phase == 1), ((0, 255, 0), phase == 0)]):
        y = 20 + 14 * i
        f[y:y + 8, 150:160] = colour if lit else (60, 60, 60)
    f[150:154, 30:34] = (0, 255, 0)  # always-on power LED
    f[120:128, 250:256] = (0, 0, 255)  # a static two-colour sign (red + yellow, both always lit)
    f[130:138, 250:256] = (0, 255, 255)
    f[4:10, 10:16] = (0, 0, 255) if int(t) % 2 == 0 else 90  # blinking REC dot in the overlay band


def test_stack_light_found_always_on_led_and_rec_dot_ignored(tmp_path):
    v = tmp_path / "l.mp4"
    make_video(v, 40, draw_lamps)
    cfg, info, _ = learn_scene([src(v, 40)], lambda f: [], minutes=2, fps=2, lamps=True)
    lamps = [m for m in cfg["machines"] if m["source"] == "lamp"]
    assert len(lamps) == 1
    xs, ys = [p[0] * W for p in lamps[0]["roi"]], [p[1] * H for p in lamps[0]["roi"]]
    assert min(xs) <= 150 and max(xs) >= 160 and min(ys) <= 22 and max(ys) >= 55
    assert set(info["lamps"][0]["colours"]) == {"red", "yellow", "green"} and lamps[0]["work_colors"] == ["green"]


def test_lamps_are_not_looked_for_unless_asked(tmp_path):
    v = tmp_path / "l.mp4"
    make_video(v, 40, draw_lamps)  # a real stack light is in the picture, but nobody said the machines have lamps
    cfg, info, _ = learn_scene([src(v, 40)], lambda f: [], minutes=2, fps=2)
    assert info["lamps"] == [] and info["lamps_looked_for"] is False and [m for m in cfg["machines"] if m["source"] == "lamp"] == []
    _, asked, _ = learn_scene([src(v, 40)], lambda f: [], minutes=2, fps=2, lamps=True)
    assert len(asked["lamps"]) == 1 and asked["lamps_looked_for"] is True


# ---------- config + helpers ----------
def test_machines_are_linked_to_the_nearest_station_and_config_is_valid():
    spots = [dict(polygon=[[0.0, 0.5], [0.3, 0.5], [0.3, 1.0], [0.0, 1.0]]), dict(polygon=[[0.7, 0.5], [1.0, 0.5], [1.0, 1.0], [0.7, 1.0]])]
    cfg = build_camera_config(spots, [dict(polygon=[[0.6, 0.2], [0.8, 0.2], [0.8, 0.5], [0.6, 0.5]])], [dict(polygon=[[0.5, 0.0], [0.55, 0.0], [0.55, 0.1], [0.5, 0.1]])])
    by = {m["name"]: m for m in cfg["machines"]}
    assert by["motion1"]["zone"] == "spot2" and "zone" not in by["lamp1"]  # the lamp is too far from both stations
    check_config({"cameras": {"cam": cfg}})
    assert normalize_camera(cfg)["gathering"] == {"min_people": 2, "min_duration_s": 180}
    import yaml
    assert yaml.safe_load(yaml.safe_dump({"cameras": {"cam": cfg}}))["cameras"]["cam"] == cfg  # plain Python types only


def test_sample_windows_spread_over_footage_and_respect_hours():
    s1, s2 = src("/x/a.mp4", 3600), {**src("/x/b.mp4", 3600), "start_ts": src("/x", 1)["start_ts"] + 3600}
    wins = sample_windows([s1, s2], total_s=600, k=5)
    assert len(wins) == 5 and all(abs(l - 120) < 1e-6 for _, _, l in wins)
    assert {w[0]["path"] for w in wins} == {"/x/a.mp4", "/x/b.mp4"}
    assert sample_windows([src("/x/c.mp4", 100)], total_s=600, k=5) == [(src("/x/c.mp4", 100), 0.0, 100.0)]  # short footage: all of it
    assert sample_windows([s1], total_s=600, k=5, hours=[[10, 12]]) == []  # the recording starts at 08:00 and lasts one hour


# ---------- people in front of things must not change what is learned ----------
def test_static_coloured_object_hidden_by_a_passing_person_is_not_a_lamp(tmp_path):
    """Real footage had orange traffic cones reported as blinking lamps: a person walking past hid them part of the time,
    and the hidden frames were counted as 'off'. Hidden frames must be unknown, not off."""
    v = tmp_path / "c.mp4"
    make_video(v, 40, lambda f, t: f.__setitem__((slice(120, 140), slice(100, 112)), (0, 200, 255)))
    n = itertools.count()
    cover = lambda f: [(90.0, 100.0, 125.0, 170.0, 0.9)] if 20 <= next(n) < 60 else []  # a person stands in front for half the time
    cfg, info, _ = learn_scene([src(v, 40)], cover, minutes=2, fps=2, lamps=True)
    assert [m for m in cfg["machines"] if m["source"] == "lamp"] == [] and info["lamps"] == []


def test_machine_mostly_hidden_by_its_operator_is_still_found(tmp_path):
    v = tmp_path / "o.mp4"
    make_video(v, 80, lambda f, t: f.__setitem__((slice(50, 90), slice(200 + int(t * 40) % 40, 215 + int(t * 40) % 40)), 255))
    n = itertools.count()
    operator = lambda f: [(190.0, 40.0, 275.0, 110.0, 0.9)] if next(n) % 10 else []  # in front of the machine 90 % of the frames
    cfg, info, _ = learn_scene([src(v, 80)], operator, minutes=2, fps=2, lamps=True)
    (m,) = [m for m in cfg["machines"] if m["source"] == "motion"]
    assert info["motion"][0]["freq"] > 0.6  # frequency over the frames in which the machine was visible (unnormalised it would be ~0.08)
    xs = [p[0] * W for p in m["roi"]]
    assert min(xs) <= 205 and max(xs) >= 240
