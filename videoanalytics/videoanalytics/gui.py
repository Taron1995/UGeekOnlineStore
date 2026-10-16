"""A small window for the one-step analysis: choose recordings (or a folder), press Start, read the report.

The window only starts `python -m videoanalytics analyze ... --no-open` as a separate process and shows its output, so the analysis
never depends on the window (and the other way round). The work is resumable: a stopped run continues where it stopped."""
from __future__ import annotations

import os
import queue
import re
import signal
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

from .sources import VIDEO_EXT, parse_start_arg

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # python without Tk (Linux: sudo apt install python3-tk)
    tk = None

ROOT = Path(__file__).resolve().parent.parent  # the project folder: `python -m videoanalytics` works from here without installing
NO_WINDOW, NEW_GROUP = 0x08000000, 0x00000200  # Windows process flags: no console window for the child, own process group
PROGRESS = re.compile(r"^\[(\d+)/(\d+)\]")
REPORT = re.compile(r"^REPORT:\s*(.+?)\s*$")
SAVED = re.compile(r"^Saved as\s+(.+?)\s*$")  # the last line of `doctor`
NEEDS_TK = ("The window needs tkinter (part of the normal Python installer; on Linux: sudo apt install python3-tk).\n"
            "Without it use the command line:  python -m videoanalytics analyze <video-or-folder>")


def python_exe() -> str:
    """The interpreter for the child process: python.exe, also when this window was started with pythonw.exe (no console)."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))
    return str(exe)


def build_command(paths, *, out=None, workers=None, relearn=False, python=None, doctor=False, one_camera=False, start=None) -> list[str]:
    """doctor=True: the quick check of the recordings instead of the analysis."""
    cmd = [python or python_exe(), "-m", "videoanalytics", "doctor" if doctor else "analyze", *[os.path.abspath(p) for p in paths]]
    if not doctor:
        cmd.append("--no-open")
    if out:
        cmd += ["--out", os.path.abspath(out)]
    if workers:
        cmd += ["--workers", str(int(workers))]
    if relearn and not doctor:
        cmd.append("--rediscover")
    if one_camera:
        cmd.append("--one-camera")
    if start:
        cmd += ["--start", start]
    return cmd


def parse_progress(line: str):
    """'[3/5] cam1_...: 90s video ...' -> (3, 5)"""
    m = PROGRESS.match(line)
    return (int(m[1]), int(m[2])) if m else None


def parse_report(line: str):
    """'REPORT: /path/to/report.html' -> the path"""
    m = REPORT.match(line)
    return m[1] if m else None


def parse_saved(line: str):
    """'Saved as /path/to/doctor.txt' -> the path"""
    m = SAVED.match(line)
    return m[1] if m else None


def kill_tree(proc: subprocess.Popen):
    """Stops the analysis and its worker processes."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, creationflags=NO_WINDOW, check=False)
        else:
            os.killpg(proc.pid, signal.SIGTERM)  # the child was started in its own session
    except OSError:
        proc.kill()


class Job:
    """One analysis in a separate process. Its output lines (and finally the exit code, an int) arrive on `lines`."""

    def __init__(self, cmd: list[str], cwd: Path = ROOT):
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}  # lines arrive at once; paths in any alphabet survive
        extra = dict(creationflags=NO_WINDOW | NEW_GROUP) if os.name == "nt" else dict(start_new_session=True)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd, env=env, **extra)
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for raw in iter(self.proc.stdout.readline, b""):
            self.lines.put(raw.decode("utf-8", "replace").rstrip("\r\n"))
        self.lines.put(self.proc.wait())

    def stop(self):
        kill_tree(self.proc)


def open_path(path: str):
    if os.name == "nt":
        os.startfile(path)  # noqa: S606
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])


def open_report(path: str):
    webbrowser.open(Path(path).resolve().as_uri())


class App:
    def __init__(self, root, paths=(), *, build=build_command, show_report=open_report, show_folder=open_path):
        if tk is None:
            raise RuntimeError(NEEDS_TK)
        self.root, self.build, self.show_report, self.show_folder = root, build, show_report, show_folder
        self.paths: list[str] = []
        self.job: Job | None = None
        self.report: str | None = None  # the finished report
        self.saved: str | None = None  # doctor.txt after a check
        self.mode = "analyze"
        self.stopping = False
        root.title("Video analytics")
        root.minsize(760, 640)
        self.out_var, self.workers_var, self.start_var = tk.StringVar(), tk.StringVar(value="0"), tk.StringVar()
        self.relearn_var, self.open_var, self.one_var = tk.BooleanVar(value=False), tk.BooleanVar(value=True), tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Choose the recordings, then press Start.")
        self._layout()
        self.add_paths(paths)

    # ---- layout ----------------------------------------------------------------------------------------------------
    def _layout(self):
        f = ttk.Frame(self.root, padding=14)
        f.pack(fill="both", expand=True)
        f.columnconfigure(0, weight=1)
        ttk.Label(f, text="Video analytics", font=("TkDefaultFont", 16, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(f, wraplength=720, justify="left", text="Choose the recordings and press Start. The program finds out by itself where the people "
                  "work and which machines there are, analyses everything and shows the report.").grid(row=1, column=0, sticky="w", pady=(2, 10))

        box = ttk.LabelFrame(f, text="1. What to analyse (video files and / or folders)", padding=8)
        box.grid(row=2, column=0, sticky="ew")
        box.columnconfigure(0, weight=1)
        self.listbox = tk.Listbox(box, height=4, selectmode="extended", activestyle="none")
        self.listbox.grid(row=0, column=0, rowspan=4, sticky="ew")
        for i, (text, cmd) in enumerate((("Add video files…", self.pick_files), ("Add a folder…", self.pick_folder),
                                         ("Remove selected", self.remove_selected), ("Clear", self.clear))):
            ttk.Button(box, text=text, command=cmd).grid(row=i, column=1, sticky="ew", padx=(8, 0), pady=1)

        opt = ttk.LabelFrame(f, text="2. Options (all optional)", padding=8)
        opt.grid(row=3, column=0, sticky="ew", pady=8)
        opt.columnconfigure(1, weight=1)
        ttk.Label(opt, text="Results folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(opt, textvariable=self.out_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(opt, text="Browse…", command=self.pick_out).grid(row=0, column=2)
        ttk.Label(opt, text="(empty: next to the video)", foreground="gray").grid(row=1, column=1, sticky="w", padx=6)
        ttk.Checkbutton(opt, text="Learn the setup again (ignore the setup of an earlier run)", variable=self.relearn_var).grid(row=2, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(opt, text="All these files are parts of ONE camera (their names do not tell the cameras apart)", variable=self.one_var).grid(
            row=3, column=0, columnspan=3, sticky="w")
        ttk.Label(opt, text="Recorded at (files with no date in their name):").grid(row=4, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(opt, textvariable=self.start_var, width=20).grid(row=4, column=1, sticky="w", padx=6, pady=(4, 0))
        ttk.Label(opt, text="e.g. 2026-10-01 08:30", foreground="gray").grid(row=4, column=2, sticky="e", pady=(4, 0))
        ttk.Label(opt, text="Parallel processes (0 = automatic):").grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Spinbox(opt, from_=0, to=64, width=5, textvariable=self.workers_var).grid(row=5, column=2, sticky="e", pady=(4, 0))
        ttk.Checkbutton(opt, text="Open the report when it is finished", variable=self.open_var).grid(row=6, column=0, columnspan=3, sticky="w")

        run = ttk.Frame(f)
        run.grid(row=4, column=0, sticky="ew")
        run.columnconfigure(3, weight=1)
        self.start_btn = ttk.Button(run, text="Start", command=self.start)
        self.start_btn.grid(row=0, column=0)
        self.check_btn = ttk.Button(run, text="Check the videos first", command=self.check)
        self.check_btn.grid(row=0, column=1, padx=6)
        self.stop_btn = ttk.Button(run, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.grid(row=0, column=2)
        self.bar = ttk.Progressbar(run, maximum=100)
        self.bar.grid(row=0, column=3, sticky="ew", padx=6)
        self.status = ttk.Label(f, textvariable=self.status_var, wraplength=720, justify="left")
        self.status.grid(row=5, column=0, sticky="nw", pady=(6, 0))
        f.rowconfigure(5, minsize=44)  # room for two lines, so the log below does not jump

        logf = ttk.Frame(f)
        logf.grid(row=6, column=0, sticky="nsew", pady=6)
        f.rowconfigure(6, weight=3)
        logf.columnconfigure(0, weight=1)
        logf.rowconfigure(0, weight=1)
        self.log = tk.Text(logf, height=12, wrap="word", state="disabled", font="TkFixedFont")
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        end = ttk.Frame(f)
        end.grid(row=7, column=0, sticky="w")
        self.report_btn = ttk.Button(end, text="Open the report", command=self.open_report, state="disabled")
        self.report_btn.grid(row=0, column=0)
        self.folder_btn = ttk.Button(end, text="Open the results folder", command=self.open_folder, state="disabled")
        self.folder_btn.grid(row=0, column=1, padx=6)
        ttk.Button(end, text="Copy the log", command=self.copy_log).grid(row=0, column=2)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # ---- choosing the input ----------------------------------------------------------------------------------------
    def add_paths(self, paths):
        for p in paths:
            p = os.path.abspath(p)
            if p not in self.paths:
                self.paths.append(p)
                self.listbox.insert("end", p)

    def pick_files(self):
        kinds = " ".join(f"*{e}" for e in sorted(VIDEO_EXT))
        self.add_paths(filedialog.askopenfilenames(title="Choose video files", filetypes=[("Video files", kinds), ("All files", "*.*")]))

    def pick_folder(self):
        folder = filedialog.askdirectory(title="Choose a folder with recordings", mustexist=True)
        if folder:
            self.add_paths([folder])

    def pick_out(self):
        folder = filedialog.askdirectory(title="Choose the results folder")
        if folder:
            self.out_var.set(folder)

    def remove_selected(self):
        for i in reversed(self.listbox.curselection()):
            self.listbox.delete(i)
            del self.paths[i]

    def clear(self):
        self.listbox.delete(0, "end")
        self.paths.clear()

    # ---- running ---------------------------------------------------------------------------------------------------
    def start(self):
        self._launch(doctor=False)

    def check(self):
        """The quick check of the recordings (what the program sees in them, how fast this computer is): a text to copy and send."""
        self._launch(doctor=True)

    def _launch(self, doctor: bool):
        if self.job is not None:
            return
        if not self.paths:
            messagebox.showinfo("Video analytics", "Add at least one video file or folder first.")
            return
        start = self.start_var.get().strip()
        if start:
            try:
                parse_start_arg(start)
            except ValueError as e:
                messagebox.showerror("Video analytics", str(e))
                return
        try:
            workers = max(0, int(self.workers_var.get() or 0))
        except ValueError:
            workers = 0
        cmd = self.build(self.paths, out=self.out_var.get().strip() or None, workers=workers or None, relearn=self.relearn_var.get(),
                         doctor=doctor, one_camera=self.one_var.get(), start=start or None)
        self.report, self.saved, self.stopping, self.mode = None, None, False, "check" if doctor else "analyze"
        self._set_log("")
        try:
            self.job = Job(cmd)
        except OSError as e:
            self._status(f"Could not start the {'check' if doctor else 'analysis'}: {e}", bad=True)
            return
        self.bar.configure(mode="indeterminate")  # no way to tell how long it takes yet
        self.bar.start(15)
        for btn in (self.start_btn, self.check_btn, self.report_btn, self.folder_btn):
            btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._status("Checking the recordings …" if doctor else "Working … the first minutes go to learning the scene.")
        self.root.after(100, self._pump)

    def stop(self):
        if self.job is not None:
            self.stopping = True
            self._status("Stopping …")
            self.job.stop()

    def _pump(self):
        job = self.job
        if job is None:
            return
        try:
            while True:
                item = job.lines.get_nowait()
                if isinstance(item, int):
                    self._finished(item)
                    return
                self._line(item)
        except queue.Empty:
            pass
        self.root.after(100, self._pump)

    def _line(self, text: str):
        self._append(text)
        done = parse_progress(text)
        if done:
            self._determinate()
            self.bar["value"] = 100.0 * done[0] / max(done[1], 1)
        rep, saved = parse_report(text), parse_saved(text)
        if rep:
            self.report = rep
        if saved:
            self.saved = saved
        if text.strip() and not rep:
            self._status(text.strip()[:140])

    def _finished(self, code: int):
        self.job = None
        self._determinate()
        self.start_btn.configure(state="normal")
        self.check_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        ready = self.report is not None and Path(self.report).is_file()
        if self.stopping:
            self._status("Stopped. Press Start to go on where it stopped: finished files are not repeated.")
        elif self.mode == "check" and code == 0:
            self.bar["value"] = 100
            self._status("The check is done. Press 'Copy the log' and paste it into the chat if something looks wrong, or press Start.")
            if self.saved and Path(self.saved).parent.is_dir():
                self.folder_btn.configure(state="normal")
            return
        elif code == 0 and ready:
            self.bar["value"] = 100
            self._status(f"Done. The report is ready: {self.report}")
        else:
            what = "check" if self.mode == "check" else "analysis"
            self._status(f"The {what} did not finish (exit code {code}). The last lines of the log below say why.", bad=True)
        if ready:
            self.report_btn.configure(state="normal")
            self.folder_btn.configure(state="normal")
            if code == 0 and not self.stopping and self.open_var.get():
                self.open_report()

    def open_report(self):
        if self.report:
            self.show_report(self.report)

    def open_folder(self):
        target = self.report or self.saved
        if target:
            self.show_folder(str(Path(target).parent))

    def copy_log(self):
        """The whole log onto the clipboard, to paste into a message."""
        self.root.clipboard_clear()
        self.root.clipboard_append(self.log.get("1.0", "end-1c"))
        self.root.update()
        self._status("The log is copied: paste it into your message.")

    def close(self):
        if self.job is not None:
            if not messagebox.askyesno("Video analytics", "The analysis is still running. Stop it and close?"):
                return
            self.job.stop()
        self.root.destroy()

    # ---- log / status ----------------------------------------------------------------------------------------------
    def _determinate(self):
        if str(self.bar.cget("mode")) == "indeterminate":
            self.bar.stop()
            self.bar.configure(mode="determinate")
            self.bar["value"] = 0

    def _status(self, text: str, bad: bool = False):
        self.status_var.set(text)
        self.status.configure(foreground="#b3261e" if bad else "")

    def _set_log(self, text: str):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.insert("end", text)
        self.log.configure(state="disabled")

    def _append(self, text: str):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 3000:  # a multi-day run prints a lot: keep the window light
            self.log.delete("1.0", "1000.0")
        self.log.see("end")
        self.log.configure(state="disabled")


def main(paths=()):
    if tk is None:
        sys.exit(NEEDS_TK)
    try:  # crisp text on high-resolution Windows screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError, ImportError):
        pass
    root = tk.Tk()
    App(root, paths)
    root.mainloop()
