import os
import sys
import time

import pytest

from videoanalytics import gui
from videoanalytics.gui import Job, build_command, kill_tree, parse_progress, parse_report, parse_saved


def test_command_for_the_analysis_process(tmp_path):
    cmd = build_command([tmp_path / "a b.mp4", "relative"], out=tmp_path / "res", workers=3, relearn=True, python="py")
    assert cmd[:4] == ["py", "-m", "videoanalytics", "analyze"] and "--no-open" in cmd  # the window opens the report itself
    assert cmd[4] == str(tmp_path / "a b.mp4") and cmd[5] == os.path.abspath("relative")  # absolute: the child runs in the project folder
    assert cmd[cmd.index("--out") + 1] == str(tmp_path / "res") and cmd[cmd.index("--workers") + 1] == "3" and "--rediscover" in cmd
    plain = build_command(["x.mp4"], python="py")
    assert "--out" not in plain and "--workers" not in plain and "--rediscover" not in plain


def test_command_for_the_check_and_for_camera_options(tmp_path):
    check = build_command([tmp_path / "a.mp4"], out=tmp_path / "res", workers=2, relearn=True, python="py", doctor=True, one_camera=True,
                          start="2026-10-01 08:30")
    assert check[:5] == ["py", "-m", "videoanalytics", "doctor", str(tmp_path / "a.mp4")]
    assert "--no-open" not in check and "--rediscover" not in check  # neither applies to the check
    assert check[check.index("--out") + 1] == str(tmp_path / "res") and "--one-camera" in check and check[check.index("--start") + 1] == "2026-10-01 08:30"
    run = build_command(["x.mp4"], python="py", one_camera=True, start="2026-10-01 08:30")
    assert run[3] == "analyze" and "--one-camera" in run and "--start" in run and "--no-open" in run


def test_output_lines_are_understood():
    assert parse_progress("[3/5] cam1_20261001.avi: 80s video, 265/265 frames detected, 20s") == (3, 5)
    assert parse_progress("5 job(s) to process (0 already done)") is None and parse_progress("  [1/2] indented") is None
    assert parse_report("REPORT: /tmp/a b/report.html") == "/tmp/a b/report.html"
    assert parse_report("Report: x") is None and parse_report("note REPORT: x") is None
    assert parse_saved("Saved as /tmp/a b/doctor.txt") == "/tmp/a b/doctor.txt" and parse_saved("Saved in x") is None


@pytest.mark.skipif(os.name == "nt", reason="process groups are checked on POSIX")
def test_kill_tree_stops_the_children_too(tmp_path):
    marker = tmp_path / "child_pid"
    script = ("import os, subprocess, sys, time\n"
              f"c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\nopen(r'{marker}', 'w').write(str(c.pid))\ntime.sleep(60)")
    job = Job([sys.executable, "-c", script], cwd=tmp_path)
    for _ in range(100):
        if marker.exists() and marker.read_text():
            break
        time.sleep(0.05)
    child = int(marker.read_text())
    kill_tree(job.proc)
    assert job.proc.wait(5) is not None
    for _ in range(100):
        try:
            os.kill(child, 0)
        except OSError:
            break
        time.sleep(0.05)
    else:
        os.kill(child, 9)
        pytest.fail("the worker process of the analysis kept running after Stop")


# ---- the window itself (needs a display: xvfb-run python -m pytest ...) -------------------------------------------------------
@pytest.fixture
def root():
    tk = pytest.importorskip("tkinter")  # only the window tests need Tk (and a display); the helpers above always run
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    yield r
    r.destroy()


def fake(script):
    """Replaces the analysis command by a small script, so the window can be tested without a model or a video."""
    return lambda paths, **kw: [sys.executable, "-c", script]


def wait(root, cond, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        root.update()
        if cond():
            return True
        time.sleep(0.02)
    return False


def window(root, script, **kw):
    seen = dict(report=[], folder=[])
    app = gui.App(root, show_report=seen["report"].append, show_folder=seen["folder"].append, build=fake(script), **kw)
    app.add_paths(["clip.mp4"])
    return app, seen


def test_a_finished_run_enables_the_report_and_opens_it(root, tmp_path):
    report = tmp_path / "report.html"
    report.write_text("<html></html>")
    app, seen = window(root, f"print('Learning the scene'); print('[1/2] a.mp4: 10s video'); print('[2/2] b.mp4: 10s video'); print('REPORT: {report}')")
    app.start()
    assert str(app.start_btn.cget("state")) == "disabled" and str(app.stop_btn.cget("state")) == "normal"
    assert wait(root, lambda: app.job is None)
    assert app.bar["value"] == 100 and str(app.bar.cget("mode")) == "determinate" and app.status_var.get().startswith("Done") and str(report) in app.status_var.get()
    assert "Learning the scene" in app.log.get("1.0", "end") and "[2/2] b.mp4" in app.log.get("1.0", "end")
    assert str(app.report_btn.cget("state")) == "normal" and str(app.folder_btn.cget("state")) == "normal" and str(app.start_btn.cget("state")) == "normal"
    assert seen["report"] == [str(report)]  # opened because "open when finished" is on
    app.open_folder()
    assert seen["folder"] == [str(tmp_path)]


def test_report_is_not_opened_when_the_box_is_unticked(root, tmp_path):
    report = tmp_path / "report.html"
    report.write_text("x")
    app, seen = window(root, f"print('REPORT: {report}')")
    app.open_var.set(False)
    app.start()
    assert wait(root, lambda: app.job is None)
    assert seen["report"] == [] and str(app.report_btn.cget("state")) == "normal"


def test_a_failed_run_says_so_and_offers_no_report(root):
    app, seen = window(root, "import sys; print('Traceback: boom'); sys.exit(3)")
    app.start()
    assert wait(root, lambda: app.job is None)
    assert "did not finish" in app.status_var.get() and "exit code 3" in app.status_var.get()
    assert str(app.report_btn.cget("state")) == "disabled" and seen["report"] == [] and "boom" in app.log.get("1.0", "end")
    assert str(app.start_btn.cget("state")) == "normal"  # can try again


def test_a_report_that_does_not_exist_is_not_offered(root, tmp_path):
    app, seen = window(root, f"print('REPORT: {tmp_path / 'missing.html'}')")
    app.start()
    assert wait(root, lambda: app.job is None)
    assert "did not finish" in app.status_var.get() and str(app.report_btn.cget("state")) == "disabled" and seen["report"] == []


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX shell script")
def test_stop_ends_the_run_and_a_new_start_is_possible(root):
    app, _ = window(root, "import time; print('working', flush=True); time.sleep(60)")
    app.start()
    assert wait(root, lambda: "working" in app.log.get("1.0", "end"))
    assert str(app.bar.cget("mode")) == "indeterminate"  # no file finished yet: nothing to show a percentage of
    app.stop()
    assert wait(root, lambda: app.job is None, 10)
    assert app.status_var.get().startswith("Stopped") and str(app.start_btn.cget("state")) == "normal"


def test_lines_arrive_while_the_analysis_is_still_running(root, monkeypatch):
    monkeypatch.delenv("PYTHONUNBUFFERED", raising=False)  # the window must not depend on the caller's environment
    app, _ = window(root, "import time; print('[1/9] early line'); time.sleep(30)")  # no flush: the child must be started unbuffered
    app.start()
    try:
        assert wait(root, lambda: "early line" in app.log.get("1.0", "end"), 8)
        assert str(app.bar.cget("mode")) == "determinate" and 10 < app.bar["value"] < 12  # 1 of 9 files
    finally:
        app.stop()
        wait(root, lambda: app.job is None, 10)


def test_list_of_inputs(root, tmp_path):
    app = gui.App(root, [str(tmp_path / "a.mp4"), str(tmp_path / "a.mp4"), str(tmp_path)], build=fake("pass"))
    assert app.paths == [str(tmp_path / "a.mp4"), str(tmp_path)]  # no duplicates, absolute
    app.listbox.selection_set(0)
    app.remove_selected()
    assert app.paths == [str(tmp_path)] and app.listbox.size() == 1
    app.clear()
    assert app.paths == [] and app.listbox.size() == 0


def test_start_without_input_does_not_run(root, monkeypatch):
    asked = []
    monkeypatch.setattr(gui.messagebox, "showinfo", lambda *a, **k: asked.append(a))
    app = gui.App(root, build=fake("print('should not run')"))
    app.start()
    assert asked and app.job is None


def recording(script, seen):
    def build(paths, **kw):
        seen.append(kw)
        return [sys.executable, "-c", script]
    return build


def test_check_runs_the_doctor_and_needs_no_report(root, tmp_path):
    saved = tmp_path / "res" / "doctor.txt"
    saved.parent.mkdir()
    saved.write_text("x")
    seen = []
    app = gui.App(root, build=recording(f"print('== Computer'); print('Saved as {saved}')", seen), show_report=lambda p: seen.append(("report", p)))
    app.add_paths(["clip.mp4"])
    app.check()
    assert wait(root, lambda: app.job is None)
    assert seen[0]["doctor"] is True and seen[-1] == seen[0]  # no report was opened
    assert app.status_var.get().startswith("The check is done") and "Copy the log" in app.status_var.get()
    assert str(app.report_btn.cget("state")) == "disabled" and str(app.folder_btn.cget("state")) == "normal"  # doctor.txt's folder can be opened
    assert str(app.start_btn.cget("state")) == "normal" and str(app.check_btn.cget("state")) == "normal"
    assert "== Computer" in app.log.get("1.0", "end")


def test_a_failed_check_says_so(root):
    app, _ = window(root, "import sys; print('boom'); sys.exit(2)")
    app.check()
    assert wait(root, lambda: app.job is None)
    assert "check did not finish" in app.status_var.get() and "exit code 2" in app.status_var.get()


def test_camera_options_reach_the_command(root):
    seen = []
    app = gui.App(root, build=recording("pass", seen))
    app.add_paths(["clip.mp4"])
    app.one_var.set(True)
    app.start_var.set(" 2026-10-01 08:30 ")
    app.workers_var.set("3")
    app.start()
    assert wait(root, lambda: app.job is None)
    assert seen == [dict(out=None, workers=3, relearn=False, doctor=False, one_camera=True, start="2026-10-01 08:30")]


def test_a_start_time_that_cannot_be_read_is_refused_before_anything_runs(root, monkeypatch):
    shown = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *a, **k: shown.append(a))
    seen = []
    app = gui.App(root, build=recording("print('should not run')", seen))
    app.add_paths(["clip.mp4"])
    app.start_var.set("tomorrow")
    app.start()
    assert shown and "start time" in shown[0][1] and app.job is None and seen == []


def test_the_log_can_be_copied(root):
    app, _ = window(root, "print('line one'); print('line two')")
    app.start()
    assert wait(root, lambda: app.job is None)
    app.copy_log()
    assert root.clipboard_get() == "line one\nline two\n"


def test_mark_zones_opens_the_editor_with_the_chosen_options(root):
    calls = []
    app = gui.App(root, zones=lambda parent, paths, **kw: calls.append((parent, paths, kw)))
    app.add_paths(["clip.mp4"])
    app.out_var.set("/tmp/results")
    app.one_var.set(True)
    app.start_var.set(" 2026-10-01 08:30 ")
    app.mark_zones()
    (parent, paths, kw), = calls
    assert parent is root and paths == [os.path.abspath("clip.mp4")]
    assert kw["out"] == "/tmp/results" and kw["start"] == "2026-10-01 08:30" and kw["one_camera"] == "" and callable(kw["on_saved"])
    kw["on_saved"]("/tmp/results/cameras.auto.yaml")
    assert "zones are saved" in app.status_var.get() and "Zones saved: /tmp/results/cameras.auto.yaml" in app.log.get("1.0", "end")


def test_mark_zones_needs_input_and_reports_problems(root, monkeypatch):
    shown = []
    monkeypatch.setattr(gui.messagebox, "showinfo", lambda *a, **k: shown.append(("info", a[1])))
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *a, **k: shown.append(("error", a[1])))

    def refuses(parent, paths, **kw):
        raise SystemExit("no video files in /x")

    app = gui.App(root, zones=refuses)
    app.mark_zones()  # nothing chosen
    app.add_paths(["clip.mp4"])
    app.mark_zones()  # the editor cannot open: the reason is shown, not lost
    app.start_var.set("tomorrow")
    app.mark_zones()  # an unreadable start time is refused before the editor opens
    assert shown[0][0] == "info" and shown[1] == ("error", "no video files in /x") and shown[2][0] == "error" and "start time" in shown[2][1]


def test_the_zone_button_waits_while_something_runs(root):
    app, _ = window(root, "import time; time.sleep(30)")
    app.start()
    assert str(app.zones_btn.cget("state")) == "disabled"
    app.stop()
    wait(root, lambda: app.job is None, 10)
    assert str(app.zones_btn.cget("state")) == "normal"
