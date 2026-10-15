# videoanalytics - offline people, machine and workpiece analytics for multi-day CCTV footage

Հայերեն՝ [README.hy.md](README.hy.md)

Runs on recordings you already have (days of footage, many cameras), fully offline, camera-only (no machine sensors).

| # | Indicator | How it is measured | Config key | Status |
|---|---|---|---|---|
| 1 | **Time at the machine** | person detection + a zone around the station, occupancy per minute (a short hold time bridges detection gaps) | `zones` | verified (synthetic + real video) |
| 2 | **Absence outside breaks** | the same zone + a work/break `schedule`; only minutes where someone is expected count | `schedule` | verified (synthetic, unit-tested) |
| 3 | **Gatherings of 2+ workers** | people closer than `dist_k` x body height, outside the work zones, longer than `min_duration_s`; frame + clip as evidence | `gathering` | verified on real footage; thresholds not calibrated |
| 4 | **Machine work / idle** | signal-lamp colour (green/yellow/red, HSV) **or** motion in a small zone of the working part, people masked out | `machines` | lamp + motion verified on synthetic video, **not on real machines** |
| 5 | **Workpiece count** | RF-DETR trained on your part + tracker + `supervision.LineZone` at the machine exit; **alternative**: machine cycles (motion -> stop = one machining) | `workpieces`, `machines[].cycle` | counting verified with COCO weights on real video and a toy training; **your parts need your annotated data** |
| 6 | **Seated without movement** | per-person body activity over time: frame difference (default) or MediaPipe Pose (landmarks + posture) | `inactivity` | motion verified on real video; pose measured, see the warning below |
| 7 | People counting | tracker + counting lines (in / out), unique appearances | `lines` | verified (real video, stable across 2-10 fps) |

Everything is offline; only permissive licenses (see *Licenses*). No face recognition.

## Install

```
pip install -e .                    # core: people, zones, schedule, lamp, motion, gatherings, cycles, inactivity (motion)
pip install -e ".[pose]"            # + MediaPipe Pose for inactivity.method: pose
pip install -e ".[workpieces]"      # + RF-DETR and supervision (installs PyTorch) for workpiece counting
pip install -e ".[train]"           # + what is needed to fine-tune RF-DETR on your parts
```
Python 3.10+. MediaPipe ships wheels for Windows (x64/ARM), Linux and macOS-ARM. On Linux it also needs `sudo apt install libegl1 libgles2`.

**PyCharm:** open the `videoanalytics` folder (the one with `pyproject.toml`) as the project, create a virtualenv, run the `pip install` line above in
the PyCharm terminal, then open `run_pipeline.py`, edit the constants at the top (`VIDEOS`, `CONFIG`, ...), choose `STEP`, press Run.
From a terminal everything is `python -m videoanalytics <command> --help`.

## Workflow

```
python -m videoanalytics models --pose                              # downloads models/yolox_tiny.onnx (+ MediaPipe pose model)
cp cameras.example.yaml cameras.yaml                                # camera names must appear in the file paths
python -m videoanalytics scan --config cameras.yaml --videos D:/footage --model models/yolox_tiny.onnx   # files, timestamps, run-time estimate
python -m videoanalytics snap --config cameras.yaml --videos D:/footage --out out                         # out/snap/<camera>.jpg: grid, zones, lamp colour
python -m videoanalytics run  --config cameras.yaml --videos D:/footage --model models/yolox_tiny.onnx --out out --limit-files 4   # trial
python -m videoanalytics run  --config cameras.yaml --videos D:/footage --model models/yolox_tiny.onnx --out out                    # everything
python -m videoanalytics report --config cameras.yaml --out out     # rebuild the report (e.g. after changing the schedule)
```
`out/report.html` is the result. `run` can be stopped (Ctrl+C) and started again: finished files are skipped. The schedule only affects the
report, so you can change it and run `report` again without reprocessing any video.

## Footage layout

Any folder tree, searched recursively. Two things are recovered from each **relative path**:

* **camera** - the camera name from the config appears as a whole word (`cam1` matches `cam1_...mp4` and `/cam1/`, not `cam10` or `webcam1`).
  For other naming add `match: "regex"` to the camera.
* **start time** - recognised: `20261001_080000`, `2026-10-01_08-00-00`, `2026-10-01_08.00.00`, `20261001080000`, and date folder + time file
  name (`2026-10-01/cam1/08-00-00.mp4`). Anything else: `--time-regex` (6 groups: Y m d H M S). If no time is found the file modification time is
  used and `scan` warns. Times are treated as wall-clock (no time zones).

`scan` prints files per camera, hours of video, date range, the features configured per camera and every problem (unmatched, no timestamp,
unreadable files). **Run it first.**

## Configuration

`cameras.example.yaml` shows all features with comments. Coordinates are normalised (x, y in 0..1); `snap` draws a 10 % grid. **Every key is
validated on start** (unknown keys, polygons given in pixels, a machine linked to a zone that does not exist, ...) and reported together.

* `zones`: `kind: workstation` (default; expected to be staffed, excluded from gatherings) or any other kind such as `rest`. `hold_s` (default 5)
  keeps a zone "occupied" this long after the last detection.
* `schedule` (global, per camera or per zone): `work` and `breaks` as `["HH:MM", "HH:MM"]` pairs (overnight ranges work), `days` (Monday = 1).
  Absence = minutes inside work hours, outside breaks, in which a workstation zone was (almost) empty. Without a schedule you only get
  "time at the station". Rest zones never get an absence.
* `machines` (several per camera): `source: lamp` - `roi` is a tight polygon around the lamp stack, `work_colors` (default `[green]`), tune
  `sat_min` / `val_min` / `min_frac` with `snap`, which prints the colour the reader sees and its per-colour scores on the snapshot; red time is
  reported as alarm, yellow as wait. `source: motion` - `roi` covers the moving part (spindle, table), `motion_thr`, `window_s`, `on_frac`.
  `zone:` links the operator zone: the report then shows work **without an operator**. `cycle:` counts cycles (motion machines use their own
  short window, set a camera-level `fps: 5` so a few-second cycle is resolved).
* `gathering`: `min_people` (2), `dist_k` (1.0 x mean body height), `min_duration_s` (180), `gap_s` (10, detection gaps bridged).
* `inactivity`: `method: motion | pose`, `zones` (watch only people in these zones), `min_duration_s`, `motion_thr`, `require_seated` (pose only).
* `workpieces`: `weights`, `classes`, `threshold`, `line`, `count: in | out | both`, `model: nano | small | medium | large`.
* camera-level `fps` overrides `options.fps` for that camera (cycles and workpieces need 4-6).

## Workpiece counting: from your footage to counts

RF-DETR has to be taught what *your* part looks like; nothing here can replace annotated examples of it.

```
python -m videoanalytics frames --config cameras.yaml --videos D:/footage --camera cam3 --count 400 --out dataset/raw   # frames spread over all days
#   annotate dataset/raw in CVAT / Label Studio / Roboflow (one class, e.g. "workpiece"), export COCO
python -m videoanalytics split-coco --images dataset/raw --annotations instances.json --out dataset
python -m videoanalytics train --dataset dataset --model nano --epochs 50 --out output       # needs ".[train]"; use a GPU
#   -> tests checkpoint_best_total.pth and last_ema.pth on your validation images (recall / precision) and prints the one to use:
#      workpieces: {weights: ..., classes: [workpiece], line: ...}
```
The counting line sits where parts leave the machine zone. `in` is the crossing from the right to the left side of `p1 -> p2` (same rule
as the people lines). `supervision.LineZone` is used for the counting; `supervision.ByteTrack` is **not**: it is deprecated and removed in
supervision 0.31, so the repo's own tracker supplies the track ids. Use the Apache-2.0 sizes (nano/small/medium/large); RF-DETR XL / 2XL are
under the PML license and are not exposed.

If you cannot annotate yet, count **machine cycles** (`machines[].cycle`): one motion -> stop = one machining. It needs no training, but only
equals the part count if one cycle makes one part.

## Calibration (do this before trusting numbers)

* **Lamp**: `snap` shows the detected colour and scores on the frame. Overexposed lamps look white on camera; lower `sat_min`, tighten the ROI.
* **Machine motion**: `motion_thr` / `window_s` per machine on an hour whose true state you know.
* **Inactivity**: `python -m videoanalytics probe --config cameras.yaml --videos D:/footage --camera cam2 --model models/yolox_tiny.onnx --start 600 --seconds 120`
  prints the measured activity per person. Run it where someone sits still and where they work, then set `motion_thr` between the two.
  On real pedestrians the frame-difference activity was 0.4-0.7 while walking; on a still person with video-compression noise added it was exactly 0.
  The default threshold 0.01 is a starting point, not a measurement of a seated worker.
* **Pose warning**: on 60-83 px tall people (typical for wide CCTV) MediaPipe found a pose in 5 of 7 crops, and the *same still person* with only
  compression noise added had a pose-activity median of 0.02 and a 95th percentile of 0.25 of body height - pose cannot tell "still" from "moving"
  at that size. That is why `motion` is the default. Use `pose` only for close-up cameras (large people); I could not measure that case.
* **Gatherings**: two people walking side by side also qualify if they stay close for `min_duration_s`; raise it (the default is 3 minutes).

## Output

| File | Content |
|---|---|
| `report.html` | cameras overview; stations (at the station / expected / absent); machines (working %, red lamp, cycles, work without operator); per camera hour x day heatmaps (people, station occupancy, machine working %, cycles); longest idle periods, absences, gatherings, people without movement - with frame and clip |
| `hourly.csv` | camera, date, hour, analysed minutes, avg/peak people, appearances, line in/out, gatherings, inactive people, workpieces |
| `machines_hourly.csv` | camera, machine, date, hour, work / idle / alarm / wait minutes, cycles, work attended by the operator |
| `stations_daily.csv`, `absences.csv` | per zone and day: at the station, expected, present, absent, absence %; every absence run (>= 5 min) |
| `events.csv`, `idle_periods.csv`, `analytics.db` | every event with absolute time; idle stretches; SQLite (`events`, `minutes`, `zone_minutes`, `machine_minutes`, `files`) |
| `clips/<camera>/` | `.jpg` frame + `.mp4` (OpenCV `mp4v`: plays in VLC / Windows Media Player, not inline in browsers) |

## Speed - plan this before starting

Measured on a 4-vCPU cloud sandbox: YOLOX-tiny 416 takes **65 ms per model inference on 1 thread (82 ms per frame including resize + NMS), 29 ms on
4 threads**, so N single-thread worker processes (what `run` does, `--workers`, default = CPU count) beat one multi-threaded process. A 640x360
test file cost 0.24 CPU-seconds per video-second at 3 fps with the detector on every sampled frame. Real cost depends on resolution and codec
(decoding 1080p H.264/H.265 can dominate), so **`scan --model ...` benchmarks one of your files and prints an upper-bound wall-clock estimate**.

Other measured costs: RF-DETR nano (384 px) ~180-200 ms per frame on that CPU (use it only on the cameras that count parts, with `--workers 1-2`,
PyTorch uses all cores and ~1 GB RAM per process); MediaPipe pose 37-43 ms per person crop.

Levers, biggest first: `--hours` (only working hours), `--fps`, the motion gate (on by default; compare with `--no-gate` on one hour before
trusting it), `--hw-decode` (D3D11VA/VAAPI via OpenCV; not tested here), more workers or a GPU box. `models --yolox-s` gives a more accurate person
detector for small people (284 ms vs 82 ms per frame, 3.5x).

## What was verified, and what was not

Verified (synthetic video with known answers, plus the OpenCV sample `vtest.avi`, 768x576 pedestrians):

* Person detector decode/NMS; tracker 28 / 26 / 28 / 26 IDs on an 80 s clip at 10 / 5 / 3.3 / 2 fps (plain IoU matching gave 56 and 163); line counts stable (14 in / 18 out at 10 fps).
* RF-DETR nano + the tracker + `supervision.LineZone` on the same clip: **14 in / 18 out**, identical to YOLOX + the repo's own line.
* Train -> load -> count with RF-DETR on a toy synthetic dataset (24 images, nano, CPU, about a minute per epoch): the `split-coco` output is accepted,
  `train` writes the checkpoints, `from_checkpoint` restores the class names. After 10 epochs `last_ema.pth` finds the part with confidence 0.85-0.88 and
  boxes within a pixel of the truth, and the counting line gives exactly one `in` for a part sliding across it. **But** `checkpoint_best_total.pth` (the
  library's "best by validation mAP") was the near-untrained epoch-0 weights - mAP saturates at 1.0 on such a trivial dataset - and found nothing at
  confidence 0.5 (recall 0 %). That is why `train` now tests both candidates on your validation images and recommends one.
* Lamp colours (green/yellow/red, blinking), motion machine, cycles, gatherings, occupancy, schedule/absence, idle runs, report numbers: unit tests with
  mutation checks (the tests fail when the logic is broken on purpose).
* Gatherings on real footage: groups outside the work zone are found and the evidence frame shows them.

**Not verified:** anything on your cameras (night/IR, occlusion, crowds, lamp overexposure); machine WORK/IDLE against real machines (it is a motion /
lamp proxy, not a machine signal); the pose backend on large people; real workpiece accuracy; `--hw-decode`; Windows (nothing here ran on Windows).
Measure on a hand-counted sample hour before quoting numbers.

Known limits: tracks restart per file, so a person standing across a file boundary counts twice (appearances, gatherings, inactivity may split);
the first `window_s` seconds of each file are not counted as machine work/idle; absence needs the zone polygon to cover where the person really stands
(foot point).

## Licenses (commercial use)

| Component | License |
|---|---|
| YOLOX code + COCO-trained `yolox_tiny.onnx` (Megvii) | Apache-2.0 |
| ONNX Runtime | MIT |
| OpenCV (`opencv-python-headless`) | Apache-2.0; its wheels bundle FFmpeg (LGPL-2.1) - check this against your contract's wording |
| MediaPipe + pose model | Apache-2.0 |
| RF-DETR package and N / S / M / L weights | Apache-2.0 (XL / 2XL: PML 1.0, **not used**) |
| supervision | MIT |
| PyTorch (RF-DETR dependency), NumPy, PyYAML | BSD, BSD, MIT |

Ultralytics YOLOv5/v8/11 are deliberately **not** used (AGPL-3.0). Check the training-data terms of the pretrained weights before shipping; a model
fine-tuned on your own footage (workpieces) avoids that for the part detector.

## Tests

`pytest tests` (tracker, zones/lines, detector post-processing, path/time parsing, active hours, config validation, lamp, machine/cycles, occupancy,
gatherings, inactivity, schedule/absence, report, tools, workpieces). Tests that need MediaPipe / supervision skip when the package is missing.
