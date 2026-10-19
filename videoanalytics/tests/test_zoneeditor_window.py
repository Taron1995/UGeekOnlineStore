"""The zone editor window: drawing with the mouse and keyboard, the list, saving (needs a display: xvfb-run python -m pytest ...)."""
import pytest

from tests.test_features import make_video
from videoanalytics import zoneeditor as ze
from videoanalytics.config import read_config
from videoanalytics.zoneeditor import ZoneEditor, new_shape, open_session

SQ = lambda x, y, s=0.2: [[x, y], [x + s, y], [x + s, y + s], [x, y + s]]
flat = lambda points: [v for p in points for v in p]


@pytest.fixture
def root():
    tk = pytest.importorskip("tkinter")  # the logic tests above need neither Tk nor a display
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    yield r
    try:
        r.destroy()
    except tk.TclError:  # a test that closed the window already
        pass


@pytest.fixture
def editor(root, tmp_path):
    for cam in ("cam1", "cam2"):
        make_video(tmp_path / f"{cam}_20261001_080000.mp4", 20, lambda f, t: None)
    saved = []
    ed = ZoneEditor(root, open_session([str(tmp_path)]), on_saved=saved.append)
    ed.saved = saved
    root.update()
    return ed


def click(ed, nx, ny, button="<Button-1>"):
    x, y = ed.to_canvas([nx, ny])
    ed.canvas.event_generate(button, x=int(round(x)), y=int(round(y)))
    ed.top.update()


def draw(ed, corners):
    for nx, ny in corners:
        click(ed, nx, ny)
    return ed.finish()


def test_the_frame_is_shown_and_coordinates_map_both_ways(editor):
    assert editor.photo is not None and editor.camera == "cam1" and editor.file_var.get().startswith("cam1_")
    for p in ([0.0, 0.0], [1.0, 1.0], [0.25, 0.8]):
        assert editor.to_norm(*editor.to_canvas(p)) == pytest.approx(p, abs=1e-9)
    assert editor.to_norm(-50, 9999) == [0.0, 1.0]  # a click outside the picture lands on its edge
    editor.random_frame()
    assert 0 <= editor.time_var.get() < 20 and "of 0:20" in editor.time_label.cget("text")


def test_clicks_make_a_station_with_the_next_free_name(editor):
    shape = draw(editor, [(0.1, 0.5), (0.3, 0.5), (0.3, 0.8), (0.1, 0.8)])
    assert (shape.kind, shape.name) == ("station", "spot1") and flat(shape.points) == pytest.approx(flat([[0.1, 0.5], [0.3, 0.5], [0.3, 0.8], [0.1, 0.8]]), abs=2e-3)
    assert editor.listbox.get(0).startswith("spot1") and editor.name_var.get() == "spot2" and editor.dirty == {"cam1"} and editor.points == []


def test_double_click_adds_the_last_corner_and_finishes(editor):
    for p in ((0.1, 0.5), (0.3, 0.5), (0.3, 0.8)):
        click(editor, *p)
    x, y = (int(v) for v in editor.to_canvas([0.1, 0.8]))
    for t in (1000, 1060):  # two quick presses at one spot: the first is a click, the second arrives as the double-click
        editor.canvas.event_generate("<ButtonPress-1>", x=x, y=y, time=t)
        editor.canvas.event_generate("<ButtonRelease-1>", x=x, y=y, time=t + 20)
        editor.top.update()
    (shape,) = editor.shapes["cam1"]
    assert len(shape.points) == 4 and shape.points[-1] == pytest.approx([0.1, 0.8], abs=2e-3)  # the double-click's two points became one corner


def test_enter_finishes_backspace_and_right_click_remove_a_corner_escape_cancels(editor):
    editor.top.focus_force()
    editor.top.update()
    for p in ((0.1, 0.5), (0.3, 0.5), (0.3, 0.8), (0.9, 0.9)):
        click(editor, *p)
    click(editor, 0, 0, "<Button-3>")  # right-click: the last corner goes
    assert len(editor.points) == 3
    editor.top.event_generate("<BackSpace>")
    editor.top.update()
    assert len(editor.points) == 2
    editor.top.event_generate("<Escape>")
    editor.top.update()
    assert editor.points == [] and editor.shapes["cam1"] == []
    for p in ((0.1, 0.5), (0.3, 0.5), (0.3, 0.8)):
        click(editor, *p)
    editor.top.event_generate("<Return>")
    editor.top.update()
    assert [s.name for s in editor.shapes["cam1"]] == ["spot1"]


def test_too_few_corners_is_explained_not_saved(editor):
    for p in ((0.1, 0.5), (0.3, 0.5)):
        click(editor, *p)
    assert editor.finish() is None and "at least 3 corners" in editor.status_var.get() and editor.shapes["cam1"] == []
    assert len(editor.points) == 2  # still drawing: the user can go on


def test_each_kind_has_its_own_names_and_a_name_can_be_chosen(editor):
    editor.kind_var.set("lamp")
    editor._kind_changed()
    assert editor.name_var.get() == "lamp1"
    draw(editor, [(0.8, 0.1), (0.85, 0.1), (0.85, 0.2), (0.8, 0.2)])
    editor.kind_var.set("watch")
    editor._kind_changed()
    editor.name_var.set("floor")
    draw(editor, [(0.02, 0.02), (0.98, 0.02), (0.98, 0.98), (0.02, 0.98)])
    assert [(s.kind, s.name) for s in editor.shapes["cam1"]] == [("lamp", "lamp1"), ("watch", "floor")]
    editor.kind_var.set("watch")
    editor.name_var.set("floor")  # taken
    assert draw(editor, [(0.1, 0.1), (0.2, 0.1), (0.2, 0.2), (0.1, 0.2)]) is None and "already taken" in editor.status_var.get()
    assert len(editor.shapes["cam1"]) == 2


@pytest.mark.parametrize("screen_h, rows", [(1080, 6), (768, 6), (720, 4)])
def test_the_window_fits_the_screen_so_that_save_stays_in_sight(root, tmp_path, monkeypatch, screen_h, rows):
    """The panel on the right sets the height of the window: on a 1366x768 laptop Save sat behind the taskbar."""
    make_video(tmp_path / "cam1_20261001_080000.mp4", 10, lambda f, t: None)
    monkeypatch.setattr(root, "winfo_screenheight", lambda: screen_h)
    monkeypatch.setattr(root, "winfo_screenwidth", lambda: 1366)
    ed = ZoneEditor(root, open_session([str(tmp_path)]))
    root.update()
    assert ed.list_rows == rows and int(ed.listbox.cget("height")) == rows
    assert ed.top.winfo_reqheight() <= screen_h - 72 and ed.top.winfo_reqwidth() <= 1366  # title bar + taskbar below, the screen's width across


def test_a_long_list_of_shapes_scrolls(editor):
    for i in range(9):
        draw(editor, SQ(0.05 + 0.1 * i, 0.5, 0.05))
    editor.top.update()
    assert editor.listbox.size() == 9 and editor.listbox.yview()[1] < 1.0  # more shapes than rows: the rest is below
    assert editor.list_scroll.get() == pytest.approx(editor.listbox.yview())  # the scroll bar shows where the visible part is
    editor.listbox.yview_moveto(1.0)
    editor.top.update()
    assert editor.listbox.yview()[1] == 1.0 and editor.list_scroll.get()[1] == 1.0


def test_a_processing_section_is_listed_by_what_it_is_and_belongs_to_no_station(editor):
    draw(editor, [(0.1, 0.5), (0.3, 0.5), (0.3, 0.8), (0.1, 0.8)])  # spot1
    editor.kind_var.set("activity")
    editor._kind_changed()
    assert editor.name_var.get() == "work1"
    shape = draw(editor, [(0.1, 0.2), (0.3, 0.2), (0.3, 0.4), (0.1, 0.4)])  # right above the station, where a moving part would be linked to it
    assert (shape.kind, shape.name, shape.zone) == ("activity", "work1", "") and "Added processing section 'work1'" in editor.status_var.get()
    assert [editor.listbox.get(i) for i in range(2)] == ["spot1   (station)", "work1   (processing section)"]
    editor.listbox.selection_clear(0, "end")
    editor.listbox.selection_set(1)
    editor.listbox.event_generate("<<ListboxSelect>>")
    assert str(editor.link_box.cget("state")) == "disabled"  # nothing to choose: the people who work there are the station's
    (machine,) = read_config(editor.save())["cameras"]["cam1"]["machines"]
    assert machine["source"] == "activity" and "zone" not in machine and machine["min_present_s"] == 8


def test_select_link_and_delete(editor):
    draw(editor, [(0.1, 0.5), (0.3, 0.5), (0.3, 0.8), (0.1, 0.8)])  # spot1
    editor.kind_var.set("motion")
    editor._kind_changed()
    draw(editor, [(0.1, 0.2), (0.3, 0.2), (0.3, 0.4), (0.1, 0.4)])  # motion1, near it
    editor.listbox.selection_clear(0, "end")
    editor.listbox.selection_set(1)
    editor.listbox.event_generate("<<ListboxSelect>>")
    assert str(editor.link_box.cget("state")) == "readonly" and list(editor.link_box.cget("values")) == ["(nearest station)", "(none)", "spot1"]
    editor.link_var.set("(none)")
    editor._link_changed()
    assert editor.shapes["cam1"][1].zone == ""
    editor.link_var.set("spot1")
    editor._link_changed()
    assert editor.shapes["cam1"][1].zone == "spot1"
    editor.listbox.selection_clear(0, "end")
    editor.listbox.selection_set(0)
    editor.listbox.event_generate("<<ListboxSelect>>")
    assert str(editor.link_box.cget("state")) == "disabled"  # a station has nothing to belong to
    editor.delete_selected()
    assert [s.name for s in editor.shapes["cam1"]] == ["motion1"] and editor.shapes["cam1"][0].zone is None  # looks for the nearest station again


def test_rename_keeps_the_links(editor, monkeypatch):
    draw(editor, [(0.1, 0.5), (0.3, 0.5), (0.3, 0.8), (0.1, 0.8)])  # spot1
    draw(editor, [(0.5, 0.5), (0.7, 0.5), (0.7, 0.8), (0.5, 0.8)])  # spot2
    editor.kind_var.set("motion")
    editor._kind_changed()
    draw(editor, [(0.1, 0.2), (0.3, 0.2), (0.3, 0.4), (0.1, 0.4)])
    editor.shapes["cam1"][2].zone = "spot1"
    editor.listbox.selection_set(0)
    asked = iter(["press", "spot2", "motion1"])  # a free name; one taken by the other station; one that only another kind of thing has
    monkeypatch.setattr(ze.simpledialog, "askstring", lambda *a, **k: next(asked))
    editor.rename_selected()
    assert [s.name for s in editor.shapes["cam1"]] == ["press", "spot2", "motion1"] and editor.shapes["cam1"][2].zone == "press"
    editor.rename_selected()
    assert editor.shapes["cam1"][0].name == "press" and "already taken" in editor.status_var.get()
    editor.rename_selected()
    assert editor.shapes["cam1"][0].name == "motion1"  # a station and a machine may share a name


def test_two_cameras_keep_their_own_shapes_and_both_are_saved(editor, tmp_path):
    draw(editor, [(0.1, 0.5), (0.3, 0.5), (0.3, 0.8), (0.1, 0.8)])
    editor.camera_var.set("cam2")
    editor.camera_box.event_generate("<<ComboboxSelected>>")
    editor.top.update()
    assert editor.camera == "cam2" and editor.shapes["cam2"] == [] and editor.listbox.size() == 0 and editor.file_var.get().startswith("cam2_")
    editor.kind_var.set("watch")
    editor._kind_changed()
    draw(editor, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    path = editor.save()
    cfg = read_config(path)
    assert [z["name"] for z in cfg["cameras"]["cam1"]["zones"]] == ["spot1"] and [a["name"] for a in cfg["cameras"]["cam2"]["areas"]] == ["area1"]
    assert editor.saved == [path] and not editor.dirty and "Saved" in editor.status_var.get()
    assert (editor.session.out / "setup" / "cam1.jpg").exists() and (editor.session.out / "setup" / "cam2.jpg").exists()


def test_save_finishes_a_shape_that_is_still_being_drawn(editor):
    for p in ((0.1, 0.5), (0.3, 0.5), (0.3, 0.8)):
        click(editor, *p)
    path = editor.save()
    assert [z["name"] for z in read_config(path)["cameras"]["cam1"]["zones"]] == ["spot1"]


def test_a_save_that_cannot_be_written_is_explained(editor, monkeypatch):
    shown = []
    monkeypatch.setattr(ze.messagebox, "showerror", lambda *a, **k: shown.append(a[1]))
    shape = draw(editor, [(0.1, 0.5), (0.3, 0.5), (0.3, 0.8), (0.1, 0.8)])
    shape.extra["colour"] = "red"
    assert editor.save() is None and "unknown key 'colour'" in shown[0] and editor.dirty == {"cam1"}  # still unsaved, still open


def test_closing_with_unsaved_shapes_asks_first(editor, monkeypatch):
    draw(editor, [(0.1, 0.5), (0.3, 0.5), (0.3, 0.8), (0.1, 0.8)])
    answers = iter([False, True])
    monkeypatch.setattr(ze.messagebox, "askyesno", lambda *a, **k: next(answers))
    editor.close()
    assert editor.top.winfo_exists()  # "no": stays open
    editor.close()
    with pytest.raises(Exception):
        editor.top.winfo_exists()  # "yes": gone


def test_the_editor_opens_on_top_of_the_main_window(root, tmp_path):
    make_video(tmp_path / "cam1_20261001_080000.mp4", 10, lambda f, t: None)
    ed = ze.edit_zones(root, [str(tmp_path)], camera="cam1")
    assert isinstance(ed, ZoneEditor) and ed.top is not root and ed.camera == "cam1"
    ed.top.destroy()
    assert new_shape("station", "a", SQ(0, 0)).name == "a"


def test_the_cursor_position_is_shown_without_hiding_the_messages(editor):
    draw(editor, [(0.1, 0.5), (0.3, 0.5), (0.3, 0.8), (0.1, 0.8)])
    message = editor.status_var.get()
    x, y = editor.to_canvas([0.5, 0.5])
    editor.canvas.event_generate("<Motion>", x=int(x), y=int(y))
    editor.top.update()
    shown = editor.coord_var.get().split()  # "x 0.500   y 0.499": pixels are whole numbers
    assert editor.status_var.get() == message and float(shown[1]) == pytest.approx(0.5, abs=2e-3) and float(shown[3]) == pytest.approx(0.5, abs=2e-3)


def test_days_of_footage_are_listed_spread_over_all_files(root, tmp_path):
    make_video(tmp_path / "cam1_20261001_080000.mp4", 10, lambda f, t: None)
    session = open_session([str(tmp_path)])
    base = session.files["cam1"][0]
    session.files["cam1"] = [dict(base, rel=f"cam1/part{i:04d}.mp4", start_ts=base["start_ts"] + 3600 * i) for i in range(1000)]  # 1000 recordings, one real file behind all
    ed = ZoneEditor(root, session)
    values = list(ed.file_box.cget("values"))
    assert 300 <= len(values) <= 400 and values[0] == "cam1/part0000.mp4" and values[-1] > "cam1/part0900.mp4"  # from the first day to the last
    assert ed.file_var.get() == "cam1/part0500.mp4" and ed.photo is not None  # the one shown is the middle one, listed or not
