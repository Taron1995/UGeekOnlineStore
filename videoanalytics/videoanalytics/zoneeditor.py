"""Marking the zones by hand: draw tracking areas, stations and machines (the section where people work on one, its moving part, a signal lamp)
on a frame of a camera, and save them as the setup that `analyze` uses. The window (tkinter) is at the bottom; everything above it - shapes <->
config, checks, saving - has no window and is tested alone."""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .analyze import default_out, resolve_inputs
from .autoconfig import build_camera_config, nearest_station
from .config import DEFAULTS, ConfigError, normalize_camera, read_config
from .setupfile import camera_picture, read_info, setup_file, write_info, write_setup
from .sources import discover, discover_auto, parse_start_arg
from .tools import read_frame

try:
    import tkinter as tk
    from tkinter import messagebox, simpledialog, ttk
except ImportError:  # python without Tk (Linux: sudo apt install python3-tk)
    tk = None

NEEDS_TK = ("The zone editor needs tkinter (part of the normal Python installer; on Linux: sudo apt install python3-tk).\n"
            "Without it, write the zones into cameras.auto.yaml by hand (see README.md).")
MIN_AREA = 0.0005  # a shape smaller than 0.05 % of the picture is a misclick

# kind -> (what it is called in the window, colour, prefix of the automatic names)
KINDS = {
    "watch": ("Tracking area: follow people only here", "#2f6fed", "area"),
    "ignore": ("Ignored area: nobody here is analysed", "#e5484d", "ignore"),
    "station": ("Station: a person works here", "#2ea043", "spot"),
    "rest": ("Rest or common zone", "#12b3a8", "rest"),
    "activity": ("Machine: processing section (people work here)", "#e8590c", "work"),
    "motion": ("Machine: moving part (runs by itself)", "#f08c00", "motion"),
    "lamp": ("Machine: signal lamp", "#d63fd0", "lamp"),
}
MACHINE_KINDS = ("activity", "motion", "lamp")  # shapes that are machines (their names live apart from the zones')
LINKED = ("motion", "lamp")  # machines that can belong to a station; in a processing section the people are there by definition
DEFAULT_EXTRA = {"activity": {"motion_thr": 0.02, "window_s": 10, "min_present_s": 8},
                 "motion": {"motion_thr": 0.02, "window_s": 10}, "lamp": {"work_colors": ["green"]}}  # what the scene learning gives new machines too


def kind_word(kind: str) -> str:
    """What a shape is called in the list and in messages: 'station', 'processing section', 'signal lamp', ..."""
    head, _, rest = KINDS[kind][0].partition(": ")
    return (rest.split(" (")[0] if kind in MACHINE_KINDS else head).lower()


# ---- shapes <-> config --------------------------------------------------------------------------------------------------
@dataclass
class Shape:
    kind: str  # a key of KINDS
    name: str
    points: list  # [[x, y], ...] in 0..1 (x left -> right, y top -> bottom)
    zone: str | None = None  # machines: the station they belong to; None = the nearest one (automatic), "" = none
    extra: dict = field(default_factory=dict)  # the other keys of the entry it came from (hold_s, motion_thr, schedule, ...): kept when saved


def new_shape(kind: str, name: str, points) -> Shape:
    return Shape(kind, name, [[float(x), float(y)] for x, y in points], None if kind in LINKED else "", dict(DEFAULT_EXTRA.get(kind, {})))


def shapes_of(cam_cfg: dict) -> list[Shape]:
    """The areas, zones and machines of a camera's config as shapes."""
    cam = normalize_camera(cam_cfg)
    pts = lambda poly: [[float(x), float(y)] for x, y in poly]
    out = []
    for a in cam.get("areas") or []:
        out.append(Shape("ignore" if a.get("mode", "watch") == "ignore" else "watch", a["name"], pts(a["polygon"]),
                         extra={k: v for k, v in a.items() if k not in ("name", "polygon", "mode")}))
    for z in cam.get("zones") or []:
        out.append(Shape("station" if z.get("kind", "workstation") == "workstation" else "rest", z["name"], pts(z["polygon"]),
                         extra={k: v for k, v in z.items() if k not in ("name", "polygon")}))
    for m in cam["machines"]:
        out.append(Shape(m["source"] if m["source"] in MACHINE_KINDS else "motion", m["name"], pts(m["roi"]), m.get("zone", ""),
                         {k: v for k, v in m.items() if k not in ("name", "roi", "zone", "source")}))
    return out


def polygon_area(points) -> float:
    a = np.array(points, float)
    x, y = a[:, 0], a[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def next_name(kind: str, shapes: list[Shape]) -> str:
    prefix, taken = KINDS[kind][2], {s.name for s in shapes}
    return next(f"{prefix}{i}" for i in range(1, len(taken) + 2) if f"{prefix}{i}" not in taken)


def problems(shapes: list[Shape]) -> list[str]:
    """Everything that would make a saved setup wrong or refused, in plain words."""
    out, seen = [], {}
    for s in shapes:
        space = "area" if s.kind in ("watch", "ignore") else "machine" if s.kind in MACHINE_KINDS else "zone"  # names only have to differ within a kind of thing
        if not s.name.strip():
            out.append(f"a {kind_word(s.kind)} has no name")
        if (space, s.name) in seen:
            out.append(f"the name '{s.name}' is used twice among the {space}s")
        seen[(space, s.name)] = True
        if len(s.points) < 3:
            out.append(f"'{s.name}' needs at least 3 corners")
        elif polygon_area(s.points) < MIN_AREA:
            out.append(f"'{s.name}' is too small to be meant (a misclick?)")
    return out


def apply_shapes(cam_cfg: dict, shapes: list[Shape], link_dist: float = 0.3) -> dict:
    """The camera's config with its areas, zones and machines replaced by the shapes. Every other key (match, schedule, gathering, inactivity,
    workpieces, ...) stays, and so do the settings of an entry that was only moved or resized. Machines get the nearest station unless a
    link was chosen."""
    old = dict(cam_cfg)
    r4 = lambda poly: [[round(min(1.0, max(0.0, float(x))), 4), round(min(1.0, max(0.0, float(y))), 4)] for x, y in poly]
    cam = {k: v for k, v in old.items() if k == "match"}
    zones = []
    for s in shapes:
        if s.kind in ("station", "rest"):
            custom = s.extra.get("kind")
            kind = "workstation" if s.kind == "station" else (custom if custom and custom != "workstation" else "rest")
            zones.append({"name": s.name, "kind": kind, "polygon": r4(s.points), **{k: v for k, v in s.extra.items() if k != "kind"}})
    stations = [(z["name"], z["polygon"]) for z in zones if z["kind"] == "workstation"]
    machines = []
    for s in shapes:
        if s.kind in MACHINE_KINDS:
            poly = r4(s.points)
            link = None
            if s.kind in LINKED:
                link = s.zone if s.zone in {n for n, _ in stations} else (nearest_station(poly, stations, link_dist) if s.zone != "" else None)
            machines.append({"name": s.name, "source": s.kind, "roi": poly, **({"zone": link} if link else {}), **s.extra})
    cam["zones"] = zones
    areas = [{"name": s.name, "mode": s.kind, "polygon": r4(s.points), **s.extra} for s in shapes if s.kind in ("watch", "ignore")]
    if areas:
        cam["areas"] = areas
    cam["machines"] = machines
    cam.update({k: v for k, v in old.items() if k not in ("match", "zones", "areas", "machines", "machine")})
    return cam


# ---- the session: inputs, setup file, saving -------------------------------------------------------------------------------
class Session:
    """The cameras found in the given videos and the setup to edit: the existing one (cameras.auto.yaml in the results folder, or --config),
    else a new one with every camera present and nothing drawn yet."""

    def __init__(self, root: Path, out: Path, setup_path: Path, cfg: dict, files: dict, created: bool):
        self.root, self.out, self.setup_path, self.cfg, self.files, self.created = root, out, setup_path, cfg, files, created

    @property
    def cameras(self) -> list[str]:
        return sorted(self.files)

    def shapes(self, camera: str) -> list[Shape]:
        return shapes_of(self.cfg["cameras"].get(camera) or {})

    def save(self, edited: dict[str, list[Shape]], frames: dict[str, tuple[str, float]] | None = None) -> Path:
        """Writes the setup with the edited cameras' shapes (the others stay as they are) and a picture of each edited camera. The file that was
        there is kept as <name>.bak. Raises ConfigError (nothing written) if the result is not a valid setup."""
        bad = [f"{cam}: {p}" for cam, shapes in edited.items() for p in problems(shapes)]
        if bad:
            raise ConfigError("the shapes cannot be saved:\n  " + "\n  ".join(bad))
        cameras = dict(self.cfg["cameras"])
        for cam, shapes in edited.items():
            cameras[cam] = apply_shapes(cameras.get(cam) or {"match": ".*"}, shapes)
        cfg = {**self.cfg, "cameras": cameras}
        path = write_setup(self.setup_path, cfg)  # checks first: a setup the analysis would refuse is not written, nothing is touched
        info = read_info(self.out)
        for cam, (video, at_s) in (frames or {}).items():
            if cam in edited and (picture := camera_picture(self.out, cam, video, at_s, normalize_camera(cameras[cam]))):
                info[cam] = {**info.get(cam, {}), "image": picture}
        write_info(self.out, info)
        self.cfg, self.created = cfg, False
        return path


def open_session(paths, *, out=None, config=None, time_regex=None, start=None, one_camera=None) -> Session:
    """The same cameras and the same setup file that `analyze` would use for these paths."""
    root, files = resolve_inputs(paths)
    out = Path(out) if out else default_out(root, paths)
    setup_path = Path(config) if config else setup_file(out)
    start_ts = parse_start_arg(start) if isinstance(start, str) else start
    if setup_path.exists():
        cfg = read_config(setup_path)
        srcs, _ = discover(str(root), {k: normalize_camera(v) for k, v in cfg["cameras"].items()}, time_regex, only=files, exclude=[out], start=start_ts)
        created = False
    else:
        srcs, _, match = discover_auto(str(root), files, time_regex, exclude=[out], start=start_ts, one_camera=one_camera)
        cams = {cam: {"match": m["match"], **build_camera_config([], [], [])} for cam, m in match.items()}
        cfg = {"options": {"fps": DEFAULTS["fps"], "conf": DEFAULTS["conf"], "clips": DEFAULTS["clips"]}, "cameras": cams}
        created = True
    by: dict = {}
    for s in sorted(srcs, key=lambda s: s["start_ts"]):
        by.setdefault(s["camera"], []).append(s)
    if not by:
        raise SystemExit("no readable video files")
    return Session(root, out, setup_path, cfg, by, created)


# ---- the window -----------------------------------------------------------------------------------------------------------
class ZoneEditor:
    """A frame of the camera on a canvas: click the corners of a shape, finish it, repeat; pick the kind of shape on the right."""

    def __init__(self, parent, session: Session, *, camera: str | None = None, on_saved=None):
        if tk is None:
            raise RuntimeError(NEEDS_TK)
        self.top, self.session, self.on_saved = parent, session, on_saved
        sw, sh = parent.winfo_screenwidth(), parent.winfo_screenheight()
        self.CANVAS = (max(560, min(960, sw - 450)), max(320, min(560, sh - 230)))  # the picture as large as the screen allows, the panel on the right stays visible
        self.list_rows = 6 if sh >= 760 else 4  # the panel sets the height of the window: on a small screen the list gets shorter, so Save stays above the taskbar
        self.shapes: dict[str, list[Shape]] = {}  # per camera, loaded when the camera is first shown
        self.dirty: set[str] = set()  # cameras changed since the last save
        self.points: list[list[float]] = []  # the shape being drawn
        self.cursor = None
        self.camera = camera if camera in session.files else session.cameras[0]
        self.frame_at: dict[str, tuple[str, float]] = {}  # the frame shown per camera: the picture saved with the setup
        self.photo = None
        self.geom = (0, 0, 1.0, 1, 1)  # x offset, y offset, scale, image width, height
        parent.title(f"Mark the zones - {session.out}")
        self.kind_var, self.name_var, self.link_var, self.time_var = tk.StringVar(value="station"), tk.StringVar(), tk.StringVar(), tk.DoubleVar(value=0.0)
        self.camera_var, self.file_var = tk.StringVar(value=self.camera), tk.StringVar()
        self.status_var, self.coord_var = tk.StringVar(), tk.StringVar()
        self._layout()
        self._show_camera(self.camera, first=True)

    # ---- layout ----
    def _layout(self):
        f = ttk.Frame(self.top, padding=8)
        f.pack(fill="both", expand=True)
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(f, width=self.CANVAS[0], height=self.CANVAS[1], background="#202020", highlightthickness=0, cursor="crosshair", takefocus=True)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        side = ttk.Frame(f, padding=(10, 0, 0, 0))
        side.grid(row=0, column=1, sticky="ns")
        self.camera_box = self._row(side, "Camera", ttk.Combobox(side, textvariable=self.camera_var, state="readonly", values=self.session.cameras, width=28), 0)
        self.camera_box.bind("<<ComboboxSelected>>", lambda e: self._show_camera(self.camera_var.get()))
        self.file_box = self._row(side, "Recording", ttk.Combobox(side, textvariable=self.file_var, state="readonly", width=28), 1)
        self.file_box.bind("<<ComboboxSelected>>", lambda e: self._load_frame(at=0.0))
        self.scale = ttk.Scale(side, from_=0, to=1, variable=self.time_var, command=lambda v: self._time_label())
        self._row(side, "Moment", self.scale, 2)
        self.scale.bind("<ButtonRelease-1>", lambda e: self._load_frame())
        self.time_label = ttk.Label(side, text="0:00")
        self.time_label.grid(row=3, column=0, sticky="w")
        ttk.Button(side, text="Another frame", command=self.random_frame).grid(row=3, column=1, sticky="e")
        kinds = ttk.LabelFrame(side, text="Draw", padding=6)
        kinds.grid(row=4, column=0, columnspan=2, sticky="ew", pady=8)
        for i, (key, (label, colour, _)) in enumerate(KINDS.items()):
            tk.Radiobutton(kinds, text=label, value=key, variable=self.kind_var, command=self._kind_changed, anchor="w", fg=colour, selectcolor="#ffffff",
                           activeforeground=colour).grid(row=i, column=0, sticky="ew")
        ttk.Label(side, text="Name of the next shape").grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Entry(side, textvariable=self.name_var).grid(row=6, column=0, columnspan=2, sticky="ew")
        btns = ttk.Frame(side)
        btns.grid(row=7, column=0, columnspan=2, sticky="ew", pady=4)
        for i, (text, cmd) in enumerate((("Finish shape", self.finish), ("Remove last corner", self.undo_point), ("Cancel", self.cancel))):
            ttk.Button(btns, text=text, command=cmd).grid(row=0, column=i, padx=1)
        ttk.Label(side, text="Shapes of this camera").grid(row=8, column=0, columnspan=2, sticky="w", pady=(6, 0))
        listf = ttk.Frame(side)
        listf.grid(row=9, column=0, columnspan=2, sticky="ew")
        listf.columnconfigure(0, weight=1)
        self.listbox = tk.Listbox(listf, height=self.list_rows, activestyle="none", exportselection=False)
        self.list_scroll = ttk.Scrollbar(listf, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=self.list_scroll.set)
        self.listbox.grid(row=0, column=0, sticky="ew")
        self.list_scroll.grid(row=0, column=1, sticky="ns")
        self.listbox.bind("<<ListboxSelect>>", lambda e: self._selected())
        edit = ttk.Frame(side)
        edit.grid(row=10, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(edit, text="Delete", command=self.delete_selected).grid(row=0, column=0, padx=1)
        ttk.Button(edit, text="Rename…", command=self.rename_selected).grid(row=0, column=1, padx=1)
        self.link_box = ttk.Combobox(side, textvariable=self.link_var, state="disabled", width=28)
        self._row(side, "Machine belongs to", self.link_box, 11)
        self.link_box.bind("<<ComboboxSelected>>", lambda e: self._link_changed())
        ttk.Label(f, textvariable=self.status_var, wraplength=1100, justify="left").grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        foot = ttk.Frame(f)
        foot.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(foot, foreground="gray", text="Click the corners of a shape · double-click or Enter finishes it · right-click or Backspace removes the last corner · Esc cancels").pack(side="left")
        ttk.Button(foot, text="Close", command=self.close).pack(side="right")
        ttk.Button(foot, text="Save", command=self.save).pack(side="right", padx=6)
        ttk.Label(foot, textvariable=self.coord_var, width=20, foreground="gray").pack(side="right", padx=10)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Double-Button-1>", self._on_double)
        self.canvas.bind("<Button-3>", lambda e: self.undo_point())
        self.canvas.bind("<Button-2>", lambda e: self.undo_point())
        self.canvas.bind("<Motion>", self._on_motion)
        self.top.bind("<Return>", lambda e: self.finish() if self.points else None)
        self.top.bind("<Escape>", lambda e: self.cancel())
        self.top.bind("<BackSpace>", lambda e: self.undo_point() if self.points and not isinstance(self.top.focus_get(), ttk.Entry) else None)
        self.top.bind("<Delete>", lambda e: None if self.points or isinstance(self.top.focus_get(), ttk.Entry) else self.delete_selected())
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        self._kind_changed()

    @staticmethod
    def _row(parent, label, widget, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(2, 0))
        widget.grid(row=row, column=1, sticky="ew", pady=(2, 0))
        return widget

    # ---- the camera and its frame ----
    def _show_camera(self, camera: str, first: bool = False):
        if self.points:
            self.cancel()
        self.camera = camera
        self.camera_var.set(camera)
        self.shapes.setdefault(camera, self.session.shapes(camera))
        files = self.session.files[camera]
        step = max(1, -(-len(files) // 400))  # days of footage are hundreds of files: the list shows every n-th, spread over all of them
        self.file_box.configure(values=[s["rel"] for s in files[::step]])
        mid = files[len(files) // 2]
        self.file_var.set(mid["rel"])
        self._load_frame(at=min(30.0, mid["duration"] / 2), first=first)
        self._fill_list()
        self._kind_changed()

    def _source(self) -> dict:
        return next(s for s in self.session.files[self.camera] if s["rel"] == self.file_var.get())

    def _load_frame(self, at: float | None = None, first: bool = False):
        src = self._source()
        dur = max(1.0, float(src["duration"]))
        self.scale.configure(to=dur)
        if at is not None:
            self.time_var.set(at)
        at = float(self.time_var.get())
        try:
            frame = read_frame(src["path"], at)
        except OSError as e:
            self.status_var.set(f"This frame cannot be read ({e}); try another moment or recording.")
            return
        self.frame_at[self.camera] = (src["path"], at)
        h, w = frame.shape[:2]
        cw, ch = self.CANVAS
        k = min(cw / w, ch / h)
        iw, ih = max(1, int(w * k)), max(1, int(h * k))
        rgb = cv2.cvtColor(cv2.resize(frame, (iw, ih), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self.photo = tk.PhotoImage(data=b"P6\n%d %d\n255\n" % (iw, ih) + rgb.tobytes())  # a PPM: understood by every Tk, no extra package
        self.geom = ((cw - iw) // 2, (ch - ih) // 2, k, iw, ih)
        self._time_label()
        self._redraw()
        if not first:
            self.status_var.set(f"{self.camera}: {src['rel']} at {self._clock(at)}")

    @staticmethod
    def _clock(sec: float) -> str:
        sec = int(sec)
        return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60}:{sec % 60:02d}"

    def _time_label(self):
        self.time_label.configure(text=f"{self._clock(float(self.time_var.get()))} of {self._clock(float(self._source()['duration']))}")

    def random_frame(self):
        self._load_frame(at=random.uniform(0, max(1.0, float(self._source()["duration"]) - 1)))

    # ---- coordinates ----
    def to_norm(self, cx: float, cy: float) -> list[float]:
        ox, oy, k, iw, ih = self.geom
        return [min(1.0, max(0.0, (cx - ox) / iw)), min(1.0, max(0.0, (cy - oy) / ih))]

    def to_canvas(self, p) -> tuple[float, float]:
        ox, oy, k, iw, ih = self.geom
        return ox + p[0] * iw, oy + p[1] * ih

    # ---- drawing a shape ----
    def add_point(self, x: float, y: float):
        """x, y in 0..1."""
        self.points.append([x, y])
        self._redraw()
        self.status_var.set(f"{len(self.points)} corner(s). Double-click or press Enter to finish the shape.")

    def undo_point(self):
        if self.points:
            self.points.pop()
            self._redraw()

    def cancel(self):
        self.points = []
        self.cursor = None
        self._redraw()

    def finish(self):
        kind = self.kind_var.get()
        pts = list(self.points)
        while len(pts) > 1 and self._near(pts[-1], pts[-2]):  # the second click of a double-click lands on the first
            pts.pop()
        if len(pts) < 3 or polygon_area(pts) < MIN_AREA:
            self.status_var.set("A shape needs at least 3 corners that enclose some area. Keep clicking, or press Esc to start again.")
            return None
        name = self.name_var.get().strip() or next_name(kind, self.shapes[self.camera])
        shape = new_shape(kind, name, pts)
        clash = [p for p in problems(self.shapes[self.camera] + [shape]) if "used twice" in p]
        if clash:
            self.status_var.set(f"The name '{name}' is already taken: type another one in 'Name of the next shape'.")
            return None
        self.shapes[self.camera].append(shape)
        self.dirty.add(self.camera)
        self.points, self.cursor = [], None
        self._fill_list(select=len(self.shapes[self.camera]) - 1)
        self._kind_changed()
        self._redraw()
        self.status_var.set(f"Added {kind_word(kind)} '{name}'. Draw the next one, or Save.")
        return shape

    def _near(self, a, b) -> bool:
        (ax, ay), (bx, by) = self.to_canvas(a), self.to_canvas(b)
        return abs(ax - bx) < 5 and abs(ay - by) < 5

    def _on_click(self, event):
        self.canvas.focus_set()
        self.add_point(*self.to_norm(event.x, event.y))

    def _on_double(self, event):
        """The last corner is where the double-click was (Tk delivers the second click only as the double-click, not as a click)."""
        self.add_point(*self.to_norm(event.x, event.y))
        self.finish()

    def _on_motion(self, event):
        self.cursor = self.to_norm(event.x, event.y)
        self.coord_var.set(f"x {self.cursor[0]:.3f}   y {self.cursor[1]:.3f}")
        if self.points:
            self._redraw()

    def _kind_changed(self):
        self.name_var.set(next_name(self.kind_var.get(), self.shapes.get(self.camera, [])))

    # ---- the list of shapes ----
    def _fill_list(self, select: int | None = None):
        self.listbox.delete(0, "end")
        for s in self.shapes[self.camera]:
            self.listbox.insert("end", f"{s.name}   ({kind_word(s.kind)})")
        if select is not None:
            self.listbox.selection_set(select)
        self._selected()

    def selected_index(self) -> int | None:
        sel = self.listbox.curselection()
        return sel[0] if sel else None

    def _selected(self):
        i = self.selected_index()
        machine = i is not None and self.shapes[self.camera][i].kind in LINKED
        stations = [s.name for s in self.shapes[self.camera] if s.kind == "station"]
        self.link_box.configure(state="readonly" if machine else "disabled", values=["(nearest station)", "(none)", *stations])
        if machine:
            z = self.shapes[self.camera][i].zone
            self.link_var.set("(nearest station)" if z is None else "(none)" if z == "" else z)
        else:
            self.link_var.set("")
        self._redraw()

    def _link_changed(self):
        i = self.selected_index()
        if i is not None:
            v = self.link_var.get()
            self.shapes[self.camera][i].zone = None if v == "(nearest station)" else "" if v == "(none)" else v
            self.dirty.add(self.camera)

    def delete_selected(self):
        i = self.selected_index()
        if i is None:
            return
        gone = self.shapes[self.camera].pop(i)
        for s in self.shapes[self.camera]:  # machines of a deleted station look for the nearest one again
            if s.zone == gone.name:
                s.zone = None
        self.dirty.add(self.camera)
        self._fill_list()
        self._kind_changed()
        self.status_var.set(f"Deleted '{gone.name}'.")

    def rename_selected(self):
        i = self.selected_index()
        if i is None:
            return
        shape = self.shapes[self.camera][i]
        new = simpledialog.askstring("Rename", "New name:", initialvalue=shape.name, parent=self.top)
        if not new or not new.strip() or new.strip() == shape.name:
            return
        new = new.strip()
        trial = [Shape(s.kind, new if s is shape else s.name, s.points) for s in self.shapes[self.camera]]
        if any("used twice" in p for p in problems(trial)):
            self.status_var.set(f"The name '{new}' is already taken.")
            return
        for s in self.shapes[self.camera]:
            if s.zone == shape.name:
                s.zone = new
        shape.name = new
        self.dirty.add(self.camera)
        self._fill_list(select=i)

    # ---- painting ----
    def _redraw(self):
        c = self.canvas
        c.delete("all")
        c.create_rectangle(0, 0, *self.CANVAS, fill="#202020", outline="")
        if self.photo is not None:
            c.create_image(self.geom[0], self.geom[1], image=self.photo, anchor="nw")
        chosen = self.selected_index()
        for i, s in enumerate(self.shapes.get(self.camera, [])):
            colour = KINDS[s.kind][1]
            pts = [v for p in s.points for v in self.to_canvas(p)]
            c.create_polygon(*pts, outline=colour, fill="", width=4 if i == chosen else 2)
            x, y = self.to_canvas(min(s.points, key=lambda p: (p[1], p[0])))
            tag = c.create_text(x + 3, y - 9, text=s.name, anchor="w", fill="#ffffff", font=("TkDefaultFont", 9, "bold"))
            bbox = c.bbox(tag)
            if bbox:
                c.create_rectangle(bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1, fill=colour, outline="")
                c.tag_raise(tag)
            if i == chosen:
                for p in s.points:
                    px, py = self.to_canvas(p)
                    c.create_rectangle(px - 4, py - 4, px + 4, py + 4, fill="#ffffff", outline=colour)
        if self.points:
            colour = KINDS[self.kind_var.get()][1]
            cp = [v for p in self.points for v in self.to_canvas(p)]
            if len(self.points) > 1:
                c.create_line(*cp, fill=colour, width=2)
            if self.cursor:
                c.create_line(*cp[-2:], *self.to_canvas(self.cursor), fill=colour, width=1, dash=(4, 3))
            for p in self.points:
                px, py = self.to_canvas(p)
                c.create_oval(px - 4, py - 4, px + 4, py + 4, fill=colour, outline="#ffffff")

    # ---- saving ----
    def save(self) -> Path | None:
        if self.points:
            self.finish()
        try:
            path = self.session.save({cam: self.shapes[cam] for cam in self.dirty}, self.frame_at)
        except ConfigError as e:
            messagebox.showerror("Mark the zones", str(e), parent=self.top)
            return None
        except OSError as e:
            messagebox.showerror("Mark the zones", f"The setup could not be written: {e}", parent=self.top)
            return None
        self.dirty.clear()
        self.status_var.set(f"Saved {path}. Run the analysis again (same command, or Start in the window): the cameras whose zones changed are computed again.")
        if self.on_saved:
            self.on_saved(path)
        return path

    def close(self):
        if self.dirty and not messagebox.askyesno("Mark the zones", "There are changes that were not saved. Close without saving?", parent=self.top):
            return
        self.top.destroy()


def edit_zones(parent, paths, **kw) -> ZoneEditor:
    """Opens the editor in a window of its own on top of `parent` (the main window)."""
    camera, on_saved = kw.pop("camera", None), kw.pop("on_saved", None)
    session = open_session(paths, **kw)  # before the window opens: a problem with the inputs is reported, not shown as an empty window
    return ZoneEditor(tk.Toplevel(parent), session, camera=camera, on_saved=on_saved)


def main(paths, **kw):
    if tk is None:
        sys.exit(NEEDS_TK)
    camera = kw.pop("camera", None)
    session = open_session(paths, **kw)
    root = tk.Tk()
    ZoneEditor(root, session, camera=camera, on_saved=lambda p: print(
        f"Setup saved: {p}\nRun the analysis again on the same videos (python -m videoanalytics analyze ...): the cameras whose zones changed are computed again.", flush=True))
    root.mainloop()
