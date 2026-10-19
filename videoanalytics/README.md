# videoanalytics - offline people, machine and workpiece analytics for multi-day CCTV footage

Հայերեն՝ [README.hy.md](README.hy.md)

Runs on recordings you already have (days of footage, many cameras), fully offline, camera-only (no machine sensors).

| # | Indicator | How it is measured | Config key | Status |
|---|---|---|---|---|
| 1 | **Time at the machine** | person detection + a zone around the station, occupancy per minute (a short hold time bridges detection gaps) | `zones` | verified (synthetic + real video) |
| 2 | **Absence outside breaks** | the same zone + a work/break `schedule`; only minutes where someone is expected count | `schedule` | verified (synthetic, unit-tested) |
| 3 | **Gatherings of 2+ workers** | people closer than `dist_k` x body height, outside the work zones, longer than `min_duration_s`; frame + clip as evidence | `gathering` | verified on real footage; thresholds not calibrated |
| 4 | **Machine work / idle** | **no lamp needed:** a person has stayed at the machine's *processing section* (drawn by hand) while something there moves; **or** motion of the machine's own moving part, people masked out; **or**, only for machines that have one, the colour of a signal lamp (green/yellow/red, HSV) | `machines` | all three verified on synthetic video (the processing section also run on real pedestrian footage), **not on real machines** |
| 5 | **Workpiece count** | RF-DETR trained on your part + tracker + `supervision.LineZone` at the machine exit; **alternative**: machine cycles (motion -> stop = one machining) | `workpieces`, `machines[].cycle` | counting verified with COCO weights on real video and a toy training; **your parts need your annotated data** |
| 6 | **Seated without movement** | per-person body activity over time: frame difference (default) or MediaPipe Pose (landmarks + posture) | `inactivity` | motion verified on real video; pose measured, see the warning below |
| 7 | People counting | tracker + counting lines (in / out), unique appearances | `lines` | verified (real video, stable across 2-10 fps) |

Everything is offline; only permissive licenses (see *Licenses*). No face recognition.

**Shortest way:** `python -m videoanalytics analyze <video-or-folder>` (or drag the video onto `analyze_video.bat`, or `python -m videoanalytics gui`).
The program learns where people work and which machines there are, analyses everything and opens the report - see *Quick start*.
First time with your own files, or something looks wrong: `python -m videoanalytics doctor <video-or-folder>` (a 10-60 second check, see *Quick start*).

## Install

```
pip install -e .                    # core: people, zones, schedule, lamp, motion, gatherings, cycles, inactivity (motion)
pip install -e ".[pose]"            # + MediaPipe Pose for inactivity.method: pose
pip install -e ".[workpieces]"      # + RF-DETR and supervision (installs PyTorch) for workpiece counting
pip install -e ".[train]"           # + what is needed to fine-tune RF-DETR on your parts
```
Python 3.10+. MediaPipe ships wheels for Windows (x64/ARM), Linux and macOS-ARM. On Linux it also needs `sudo apt install libegl1 libgles2`.
The window (`gui`) uses tkinter, which the normal Python installers for Windows and macOS include (Linux: `sudo apt install python3-tk`).

**PyCharm:** open the `videoanalytics` folder (the one with `pyproject.toml`) as the project, create a virtualenv, run the `pip install` line above in
the PyCharm terminal, then open `run_pipeline.py`, set `VIDEOS` to your video or folder (`STEP` is already `"analyze"`; also `"doctor"`, `"zones"`, `"gui"`) and press Run.
Or open `run_gui.pyw` and press Run for the window. From a terminal everything is `python -m videoanalytics <command> --help`.

## Quick start - give it a video, get the report

```
python -m videoanalytics analyze D:\footage\cam1_20261001_080000.mp4   # one recording ...
python -m videoanalytics analyze D:\footage                              # ... or a whole folder: all cameras, all days (searched recursively)
python -m videoanalytics gui                                             # the same in a window: choose, press Start
```
Windows: drag the video or folder onto `analyze_video.bat` (double-click it without a file to get the window). Linux / macOS: `./analyze_video.sh <path>`.

No camera config is needed. The program does this by itself:

1. **Cameras and times** come from the file names and folders (`cam1_20261001_080000.mp4` -> camera `cam1`, 2026-10-01 08:00:00;
   `2026-10-01/cam2/08-00-00.mp4` -> `cam2`; files that carry no name get the folder name).
2. **Model**: on the first run the person detector (`yolox_tiny.onnx`, 20 MB, Apache-2.0) is downloaded into `models/` (or `~/.videoanalytics/models`, or
   `$VIDAN_MODELS`). On a computer without internet, put that file there once; the error message gives the address.
3. **Scene learning**, about 6 minutes of footage per camera, sampled across all of its recordings (`--learn-minutes`; `--hours 8-20` keeps it to working hours):
   * a **station** (work zone) = a place where a person stays put for at least 15 s, 30 s or more in total;
   * a **machine** = a spot that keeps moving while the people are masked out (clock overlays and picture borders are ignored). **Signal lamps are not looked
     for** unless you say that your machines have them (`--lamps`, or the checkbox in the window): then a small light that switches colour or blinks becomes a
     **signal lamp** (green = working, everything else = not working);
   * a machine that **has no lamp and is worked on by people** (a bench, a press fed by hand, a lathe with an operator) cannot be found by itself: draw its
     **processing section** in the zone editor (next section). The machine counts as working while a person has stayed at that section for at least 8 s and something in it moves;
   * a camera where nothing is found just gets people counting.
4. **Analysis** with all CPU cores. Recordings longer than 30 minutes are cut into chunks of about 20 minutes, so one big file also keeps every core busy (`--chunk-min`).
   Before the long part it lists the cameras it found (check the names!) and, for more than 2 hours of video, prints an upper-bound time estimate with the ways to shorten it.
   You can stop it (Ctrl+C / the Stop button) and start it again: finished work is not repeated.
5. **Report**: `report.html` (+ CSV files) in the results folder - `<video>_analysis` next to a single video, `<folder>/_analysis` for a folder, or `--out` -
   opens in the browser. A recording of a few minutes gets timelines at 5-second resolution; days of footage get hour x day heatmaps.

**Check the "Learned setup" block at the top of the report first.** It draws what was found on a frame of every camera. It is a best guess made from the
footage, not the truth: a person who simply stands somewhere (a guard at the gate) becomes a "station", a waving banner can look like a "machine", a
machine that never ran in the sampled minutes is not found. To correct it, **draw the zones yourself** (next section), or open `<results>/cameras.auto.yaml`,
delete or move a wrong zone or machine, add a work `schedule` (absences need one) and run the same command again. The file is used as it is and never
overwritten without a copy (`--rediscover` learns again and keeps the previous file as `cameras.auto.yaml.bak`); only the cameras whose setup changed are
computed again.

### Marking the zones by hand: where to follow, stations, machines

```
python -m videoanalytics zones D:\footage        # the same paths as for analyze; in the window: "Mark zones by hand…"; in PyCharm: STEP = "zones"
```
A window shows a frame of a camera (choose the camera, the recording and the moment at the top right). Pick what to draw, click the corners of the shape, finish it with a
double-click or Enter (right-click / Backspace removes the last corner, Esc cancels), and **Save**:

| Draw | Meaning | Config |
|---|---|---|
| Tracking area (blue) | people are followed **only** where this area is - the rest of the picture (a street behind the window, a neighbouring hall) is not analysed | `areas`, `mode: watch` |
| Ignored area (red) | nobody standing here is analysed (a TV, a poster with people, a mirror) | `areas`, `mode: ignore` |
| Station (green) | a person works here: time at the station, absences | `zones`, `kind: workstation` |
| Rest or common zone | a rest corner: time spent there, never an "absence" | `zones`, `kind: rest` |
| Machine: processing section (dark orange) | the part of the machine where people work on it - **no lamp needed.** Working = a person has stayed here for at least 8 s **and** something in the section moves; a person who only walks by, or stands still, is not work. Draw it tight around the working area | `machines`, `source: activity` |
| Machine: moving part (orange) | work / idle from motion in this spot while nobody is in front of it (people masked out): a machine that runs by itself | `machines`, `source: motion` |
| Machine: signal lamp (magenta) | work / idle from the lamp colour (draw it tight around the lamp) - only for machines that have one | `machines`, `source: lamp` |

A moving part or a lamp belongs to the nearest station (so the report can show work without an operator); change it in "Machine belongs to". A processing section has no such link: it only counts work while people are there. The shapes are saved into
`<results>/cameras.auto.yaml` - the setup `analyze` uses - and a picture of the camera with the shapes drawn in goes to `setup/`; the file that was there is kept as `.bak`.
Start with an analysis run, then correct what the program learned: the window opens with those shapes, which can be deleted and redrawn. Without an earlier run the
window starts empty and creates the setup for all cameras (cameras you do not draw on get people counting only, as in the learned setup of a camera where nothing was found).
After saving, run the analysis again: **only the cameras whose zones changed are computed again.** The foot point of a person decides whether they are inside an area or a zone.

Other options: `--config cameras.yaml` (your own file, see below), `--no-auto` (people counting only), `--lamps` (your machines have signal lamps: look for them while learning),
`--workers N`, `--fps`, `--hours`, `--time-regex` (unusual file names), `--no-open`; `python -m videoanalytics analyze --help`.

What no automatic step can know: your **work schedule** (absences), the real machine state (people at the section, motion and lamp are proxies), and the **number of parts**
(the part detector has to be trained on your annotated frames, see *Workpiece counting*).

### Recordings straight from a camera, and when something looks wrong

```
python -m videoanalytics doctor D:\footage      # 10-60 seconds, even for days of footage; the window has "Check the videos first"
```
`doctor` reads every recording, looks closely at the first / middle / last one of each camera (codec, size, frame rate, how fast it decodes and seeks,
whether the picture has colour), measures the speed of this computer and explains in plain words what it found:

* **no date/time in the file names** (typical for downloads): the file modification time would be used, which is when you saved the file. Say when it was
  recorded: `--start "2026-10-01 08:00"`; files without a time then follow each other in name order (part2 before part10).
* **several parts of ONE camera** (`video1.mp4`, `video2.mp4`, ...): the cameras are told apart by the file names, so these would be several cameras.
  `--one-camera` (or `--camera-name Lathe3`) makes them one. `doctor` says so when the names differ only by a number or when the pictures look alike.
  Names that contain a time (`cam1_20261001_080000.mp4`) and per-camera folders (`cam1/video.mp4`, `cam2/video.mp4`) are told apart reliably.
* **recorder formats** (`.dav`, raw `.h264`): they are tried, and if they cannot be read the message says how to convert them
  (`ffmpeg -i input.dav -c copy output.mp4`, or export MP4 from the recorder's own software). A file that does not say how long it is is read once from
  start to end; a path with Armenian / Russian letters that Windows cannot open should be copied to an ASCII-only folder such as `C:\videos`.
* a picture without colour (night / infrared), slow decoding, small pictures, gaps between the files, a view shared by several "cameras".

The text is saved as `doctor.txt`; `analyze` writes the same text plus what it learned and measured as `diagnostics.txt` in the results folder, even if a
later step fails. **If something looks wrong, send that file** (or press "Copy the log" in the window): it holds relative file names, numbers and
versions, no pictures and no folder names of your computer.

## Full control: your own cameras.yaml

For exact zones, several lamps, a schedule, workpiece counting or pose: write the config yourself (or start from `cameras.auto.yaml`, it has the same format).

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
  name (`2026-10-01/cam1/08-00-00.mp4`). Anything else: `--time-regex` (6 groups: Y m d H M S), or `--start "2026-10-01 08:00"` when the files carry
  no time at all. If no time is found the file modification time is used and `scan` / `doctor` warn. Times are treated as wall-clock (no time zones).

`scan` prints files per camera, hours of video, date range, the features configured per camera and every problem (unmatched, no timestamp,
unreadable files). **Run it first.**

## Configuration

`cameras.example.yaml` shows all features with comments. Coordinates are normalised (x, y in 0..1); `snap` draws a 10 % grid. **Every key is
validated on start** (unknown keys, polygons given in pixels, a machine linked to a zone that does not exist, ...) and reported together.

* `areas`: `{name, polygon, mode}`, `mode: watch` (default) or `ignore`. If there is a `watch` area, only people whose feet are inside one are analysed (counted, tracked, in zones,
  gatherings, inactivity); people whose feet are inside an `ignore` area never are. Machine ROIs do not depend on areas. Normally drawn with `zones`, see above.
* `zones`: `kind: workstation` (default; expected to be staffed, excluded from gatherings) or any other kind such as `rest`. `hold_s` (default 5)
  keeps a zone "occupied" this long after the last detection.
* `schedule` (global, per camera or per zone): `work` and `breaks` as `["HH:MM", "HH:MM"]` pairs (overnight ranges work), `days` (Monday = 1).
  Absence = minutes inside work hours, outside breaks, in which a workstation zone was (almost) empty. Without a schedule you only get
  "time at the station". Rest zones never get an absence.
* `machines` (several per camera): `source: lamp` - `roi` is a tight polygon around the lamp stack, `work_colors` (default `[green]`), tune
  `sat_min` / `val_min` / `min_frac` with `snap`, which prints the colour the reader sees and its per-colour scores on the snapshot; red time is
  reported as alarm, yellow as wait. `source: motion` - `roi` covers the moving part (spindle, table), `motion_thr`, `window_s`, `on_frac`.
  `source: activity` (a machine without a lamp that people work on) - `roi` is the processing section: the machine works while a person has stayed in it for `min_present_s`
  (8 s; at least `min_overlap` = 0.15 of their box or of the section lies in it) **and** more than `motion_thr` of the section's picture changes (`window_s`, `on_frac` as for motion).
  People are not masked out here (their work is the movement), so draw the section small: someone slowly strolling through a large one counts too.
  `zone:` (moving parts and lamps) links the operator zone: the report then shows work **without an operator**. `cycle:` counts cycles (motion machines use their own
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
* **Processing section**: on an hour whose true state you know compare the machine's "Work" with reality, then tune `min_present_s` (so that a passer-by does not count),
  `motion_thr` (how much must move) and the size of the section.
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
| `report.html` | cameras overview; the learned setup drawn on a frame of every camera; stations (at the station / expected / absent); machines (working %, red lamp, cycles, work without operator); per camera a timeline (recordings up to 12 h: people, station occupancy, machine working share, 5 s to 5 min per point) and hour x day heatmaps (recordings over 3 h: people, station occupancy, machine working %, cycles); longest idle periods, absences, gatherings, people without movement - with frame and clip |
| `cameras.auto.yaml`, `setup/` | the setup used (learned and / or drawn by hand; editable, used again on the next run; the previous one is kept as `.bak`) and its pictures with the zones drawn in |
| `timeline.csv`, `stations_timeline.csv`, `machines_timeline.csv` | the 5-second values behind the timelines (only for recordings up to 12 h) |
| `hourly.csv` | camera, date, hour, analysed minutes, avg/peak people, appearances, line in/out, gatherings, inactive people, workpieces |
| `machines_hourly.csv` | camera, machine, date, hour, work / idle / alarm / wait minutes, cycles, work attended by the operator |
| `stations_daily.csv`, `absences.csv` | per zone and day: at the station, expected, present, absent, absence %; every absence run (>= 5 min) |
| `events.csv`, `idle_periods.csv`, `analytics.db` | every event with absolute time; idle stretches; SQLite (`events`, `minutes`, `zone_minutes`, `machine_minutes`, `fine`, `zone_fine`, `machine_fine`, `files`) |
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
* Zone editor (logic tests with mutation checks; the window driven with synthetic mouse and keyboard events on a virtual display): shapes <-> config keep every other setting, machines
  find their station again when theirs is deleted, a setup the analysis would refuse is not written. On the real pedestrian clip, a tracking area over the left half changed the
  average number of people in view from 5.78 to 2.03 and only that camera was computed again.
* Machine work from people at a processing section (`activity`; unit tests with mutation checks): a person who stays and works makes the machine work; a passer-by, a person
  standing still, a machine that runs with nobody there and a person far from the section do not; a missed detection does not restart the clock; what is drawn in the editor is what
  `analyze` uses. On the real pedestrian clip (`vtest.avi`) a section in the middle of the picture was "working" for 15 of 69 s - one person stayed in it for about 16 s - and an
  empty patch of grass for 0 s; the same footage frozen with 7 people standing in view gave 0 s (nothing moves). That checks the plumbing with real detections, not the accuracy on a machine.
* Camera codecs and containers, made with ffmpeg from the real pedestrian clip and run through `doctor` and `analyze` without errors: H.264 in MP4 (also 25 fps with a
  250-frame key interval, and 1080p), in MKV, in MPEG-TS and fragmented MP4; H.265 / HEVC; a raw `.h264` stream (no length, no frame rate: flagged, read once).
  OpenCV's FFmpeg messages about not-yet-decodable frames are switched off.
* One-step `analyze` (unit tests with mutation checks, plus real runs): a video in -> setup learned -> report out. On synthetic scenes the stations (people standing
  still), a stack light and a moving machine are found while decoys are ignored (an always-on LED, a blinking REC dot, a static two-colour sign, traffic cones
  that a passing person hides for a moment, a clock overlay, a short stop of less than 40 s), also when the operator hides most of the machine. On the real
  clips: walking pedestrians (`vtest.avi`) -> no station, as nobody stays; the same footage frozen with 7 people standing -> 4 stations; a 3-camera folder
  took 38 s end to end, including the first-run model download. An edited setup is recomputed, an unchanged one reused; chunked and unchunked totals are equal;
  the window was run for real under a virtual display (start, progress, report); Stop and the failure cases are tested with a stand-in command.

**Not verified:** anything on your cameras (night/IR, occlusion, crowds, lamp overexposure); machine WORK/IDLE against real machines (people at the processing
section, motion and lamp are proxies, not a machine signal; whether "a person stays 8 s and something moves" is the right rule for your machines is unknown until you compare it with an hour you checked by hand); the pose backend on large people; real workpiece accuracy; `--hw-decode`; Windows (nothing here ran on Windows,
including `analyze_video.bat`, the window and the zone editor under Windows; the editor was not tried on a high-resolution or macOS screen). **The automatic setup has not been tried on footage from a real workshop** - how many
false stations / machines it finds and how many real ones it misses there is unknown until you look at the "Learned setup" block of your first run.
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

`pytest tests` (tracker, zones/lines, detector post-processing, path/time parsing, active hours, config validation, lamp, machine/cycles, work by people at a processing section, occupancy,
gatherings, inactivity, schedule/absence, report and timelines, tools, workpieces, scene learning, the one-step run, camera-style inputs, doctor / diagnostics, the window). Tests that need
MediaPipe / supervision skip when the package is missing; the window tests need a display (Linux without one: `xvfb-run python -m pytest tests/test_gui.py`)
and skip without it.
