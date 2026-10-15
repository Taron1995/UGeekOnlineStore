# videoanalytics - offline people & machine-activity analytics for multi-day CCTV footage

Runs on the footage you already have (days of recordings from many cameras), fully offline:

| Step | What | Component |
|---|---|---|
| sample | 2-3 frames/s from each file, optional hours-of-day filter | OpenCV |
| gate | skips the detector while a scene is still (re-checks every 3 s) | frame difference |
| detect | persons | YOLOX-tiny, ONNX Runtime (CPU) |
| track | IDs, with velocity prediction so 2-3 fps still works | own code (`tracker.py`) |
| events | person appeared, zone enter/exit + dwell time, counting-line in/out | polygons/lines from the config |
| machine | WORK / IDLE from motion inside the machine zone, people masked out | frame difference + hysteresis |
| output | per-file SQLite shards -> `analytics.db`, `report.html`, CSVs, frame + clip per machine state change | |

No face recognition, no cloud, no AGPL/GPL code (see *Licenses*).

## Quick start (Հայերեն համառոտ)

```
pip install -e .                                   # կամ pip install -r requirements.txt
python -c "import urllib.request as u; u.urlretrieve('https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx','yolox_tiny.onnx')"
cp cameras.example.yaml cameras.yaml               # camera names must appear in the file paths

python -m videoanalytics scan --config cameras.yaml --videos D:/footage --model yolox_tiny.onnx   # files, timestamps, run-time estimate
python -m videoanalytics snap --config cameras.yaml --videos D:/footage --out out                 # out/snap/<camera>.jpg with grid -> draw zones
python -m videoanalytics run  --config cameras.yaml --videos D:/footage --model yolox_tiny.onnx --out out --limit-files 4   # trial
python -m videoanalytics run  --config cameras.yaml --videos D:/footage --model yolox_tiny.onnx --out out                    # everything
```
`out/report.html` is the result. `run` can be stopped (Ctrl+C) and started again: finished files are skipped.

## Footage layout

Any folder tree, searched recursively. Two things are recovered from each **relative path**:

* **camera** - the camera name from the config appears as a whole word (`cam1` matches `cam1_...mp4` and `/cam1/`, not `cam10`
  or `webcam1`). For other naming add `match: "regex"` to the camera.
* **start time** - recognised: `20261001_080000`, `2026-10-01_08-00-00`, `2026-10-01_08.00.00`, `20261001080000`, and
  date folder + time file name (`2026-10-01/cam1/08-00-00.mp4`). Anything else: `--time-regex` (6 groups: Y m d H M S).
  If no time is found the file modification time is used and `scan` warns. Times are treated as wall-clock (no time zones).

`scan` prints files per camera, hours of video, date range and every problem (unmatched, no timestamp, unreadable/corrupt files).
**Run it first** and check that the date range matches what you expect.

## Config (`cameras.yaml`)

See `cameras.example.yaml`. All coordinates are normalized (x, y in 0..1); `snap` draws a 10 % grid and the configured zones on one
frame per camera so you can read coordinates off and verify them.

* `options`: `fps` (analysed frames/s, default 3), `conf`, `active_hours` (e.g. `[[8, 20]]`, `[[22, 6]]` wraps midnight),
  `gate_motion`, `keepalive_s`, `clips`, `max_clips_per_file`. Most can be overridden on the command line (`--fps`, `--hours 8-20`, `--no-gate`).
* per camera: `zones` (workstations/areas), `lines` (counting; `in` = crossing from the right to the left side of p1->p2),
  `machine` (polygon + `motion_thr`, `window_s`).

## Output

| File | Content |
|---|---|
| `report.html` | overview per camera, hour x day heatmaps (people, machine working %), longest idle periods with frame + clip |
| `hourly.csv` | camera, date, hour, analysed minutes, avg/peak people, appearances, line in/out, machine work/idle minutes |
| `events.csv` | every event with absolute time, `idle_periods.csv`, `analytics.db` (SQLite: `events`, `minutes`, `files`) |
| `clips/<camera>/` | `.jpg` frame + `.mp4` (OpenCV `mp4v`: plays in VLC / Windows Media Player, not inline in browsers) |

## Speed - plan this before starting

Measured on a 4-vCPU cloud sandbox, YOLOX-tiny 416: **65 ms per model inference on 1 thread (82 ms per frame including resize + NMS), 29 ms on 4 threads** - so N single-thread worker
processes (what `run` does, `--workers`, default = CPU count) beat one multi-threaded process. A 640x360 test file cost 0.24 CPU-seconds per
video-second at 3 fps with the detector on every sampled frame. Real cost depends on resolution and codec (decoding 1080p H.264/H.265 can
dominate), so **`scan --model yolox_tiny.onnx` benchmarks one of your files and prints an upper-bound wall-clock estimate** for all footage.

Levers, biggest first: `--hours` (analyse only working hours), `--fps` (2 instead of 3), the motion gate (on by default; compare with
`--no-gate` on one hour before trusting it - it can miss a small, fast person in a very quiet scene), `--hw-decode` (tries D3D11VA/VAAPI via OpenCV; not tested here, no GPU in the sandbox), more workers or a GPU box.
`yolox_s.onnx` (640 px; same download URL, different file name) is the accuracy upgrade: on the same test frame it found the same 7 people with higher
scores (weakest 0.50 vs 0.39) but took **284 ms vs 82 ms per frame (3.5x)**. The detector reads the input size from the model file.

## What was verified, and what was not

* Detector decode/NMS on a real pedestrian video (OpenCV sample `vtest.avi`, 768x576): boxes sit on all visible people in the frames inspected.
* Tracker on the same 80 s clip: plain IoU matching gave 56 IDs at 10 fps and 163 at 3.3 fps (fragmentation). With velocity prediction +
  distance pass: **28 / 26 / 28 / 26 IDs at 10 / 5 / 3.3 / 2 fps**, average people in view 5.8 / 5.8 / 5.8 / 5.7. There is no ground truth for the
  true number of people, so only the *stability across frame rates* is established. Line counts were also stable (14 in / 18 out at 10 fps, 14 / 17 at 3.3 fps).
* Machine WORK/IDLE is **a motion proxy, not a machine signal**: it reads "motion in this zone", with people masked out. It is only meaningful
  where the machine visibly moves; tune `motion_thr`/`window_s` per machine on a known hour. Not validated against real machines.
* Not validated on your cameras at all (night/IR, occlusion, crowding). Measure on a hand-counted sample hour before quoting numbers.
* Appearances count confirmed tracks; tracks restart per file, so a person standing across a file boundary counts twice.
* The first `window_s` seconds of each file are not counted as machine work/idle (state needs a full window).

## Licenses (commercial use)

| Component | License |
|---|---|
| YOLOX code + COCO-trained `yolox_tiny.onnx` (Megvii) | Apache-2.0 |
| ONNX Runtime | MIT |
| OpenCV (`opencv-python-headless`) | Apache-2.0; its wheels bundle FFmpeg (LGPL-2.1) - check this against your contract's wording |
| NumPy, PyYAML | BSD, MIT |

Ultralytics YOLOv5/v8/11 are deliberately **not** used (AGPL-3.0). Check the model weights' training-data terms before shipping; to avoid
them, fine-tune on your own footage (not part of this MVP).

## Tests

`pytest tests` (tracker, zones/lines, detector post-processing, path/time parsing, active hours, machine state,
end-to-end pipeline with a stub detector on synthetic video, shard merge, report).
