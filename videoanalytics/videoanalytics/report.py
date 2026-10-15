"""Reports from analytics.db: overview tables, per-camera hour x day heatmaps, incidents, CSV exports."""
from __future__ import annotations

import csv
import html
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .config import normalize_camera
from .metrics import absence_stats, idle_runs
from .schedule import Schedule

# Sequential blue ramp, steps 100 -> 700 (dataviz reference palette). Light mode: more = darker.
# Dark mode flips the anchor so that "more" is lighter and zero recedes into the dark surface.
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
N = len(RAMP)
INK_DARK, INK_LIGHT = "#0b0b0b", "#f0efec"
MIN_COVER_S = 60  # a cell needs at least a minute of analysed video, otherwise it is shown as "no data"


def _lum(hex_):
    c = [int(hex_[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _contrast(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _best_text(bg):
    return max((INK_DARK, INK_LIGHT), key=lambda ink: _contrast(ink, bg))


def _ramp_rules(flip: bool, scope: str = "") -> str:
    return "".join(f"{scope}.h{n}{{background:{RAMP[N - 1 - n] if flip else RAMP[n]};"
                   f"color:{_best_text(RAMP[N - 1 - n] if flip else RAMP[n])}}}" for n in range(N))


CSS = f"""
:root{{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10)}}
{_ramp_rules(False)}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10)}}
{_ramp_rules(True, ':root:not([data-theme="light"]) ')}}}
:root[data-theme="dark"]{{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10)}}
{_ramp_rules(True, ':root[data-theme="dark"] ')}
section>h2:first-child{{margin-top:4px}}
body{{margin:0;padding:24px 16px;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:0 auto}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 8px}} h3{{font-size:14px;margin:18px 0 6px;color:var(--ink2);font-weight:600}}
p.sub,p.note,caption{{color:var(--ink2);margin:0 0 10px}} p.note{{font-size:12px;color:var(--muted)}}
section{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 16px 16px;margin:14px 0}}
table{{border-collapse:separate;border-spacing:0;width:100%}} th,td{{padding:6px 10px;text-align:right;border-bottom:1px solid var(--grid)}}
th{{color:var(--ink2);font-weight:600}} .l{{text-align:left}}
td.num{{font-variant-numeric:tabular-nums}}
.scroll{{overflow-x:auto}}
table.heat{{width:auto;border-spacing:2px}} table.heat th,table.heat td{{border:0;padding:0;text-align:center}}
table.heat th{{font-size:11px;font-weight:500;color:var(--muted);min-width:34px;padding-bottom:2px}}
table.heat th.d{{text-align:left;padding-right:10px;white-space:nowrap;color:var(--ink2);font-size:12px}}
table.heat td{{height:28px;min-width:34px;border-radius:3px;font-size:11px;font-variant-numeric:tabular-nums}}
table.heat td.none{{background:transparent;color:var(--muted);box-shadow:inset 0 0 0 1px var(--grid)}}
table.heat td:focus-visible{{outline:2px solid var(--ink);outline-offset:1px}}
.legend{{display:flex;flex-wrap:wrap;align-items:center;gap:4px 8px;margin:8px 0 0;font-size:12px;color:var(--ink2)}}
.legend i{{display:inline-block;width:22px;height:10px}} .legend .bar{{display:flex}}
img.th{{width:160px;border-radius:4px;display:block}} a{{color:inherit}}
.tag{{font-weight:400;color:var(--muted)}}
"""


def _legend(lo: str, hi: str, unit: str) -> str:
    cells = "".join(f'<i class="h{n}"></i>' for n in range(N))
    return (f'<div class="legend"><span>{html.escape(lo)}</span><span class="bar">{cells}</span>'
            f'<span>{html.escape(hi)}</span><span>{html.escape(unit)}</span>'
            '<span>· outlined cell = no (or &lt; 1 min of) analysed video</span></div>')


def _heat(days, cells, fmt, vmax, title_fmt) -> str:
    """days: sorted date strings; cells: {(date, hour): (value|None, detail)}."""
    head = "".join(f"<th scope=col>{h:02d}</th>" for h in range(24))
    rows = []
    for d in days:
        tds = []
        for h in range(24):
            v, detail = cells.get((d, h), (None, ""))
            if v is None:
                tds.append('<td class="none" title="no data">–</td>')
            else:
                n = 0 if vmax <= 0 else min(N - 1, int(v / vmax * N))
                tds.append(f'<td tabindex="0" class="h{n}" title="{html.escape(title_fmt(d, h, v, detail))}">{fmt(v)}</td>')
        rows.append(f"<tr><th class=d scope=row>{d}</th>{''.join(tds)}</tr>")
    return f'<div class="scroll"><table class="heat"><tr><th></th>{head}</tr>{"".join(rows)}</table></div>'


def _t(ts, f="%Y-%m-%d %H:%M:%S"):
    return datetime.fromtimestamp(ts, timezone.utc).strftime(f)


def _hk(h):
    return _t(h * 3600, "%Y-%m-%d"), int(_t(h * 3600, "%H"))


def _num(pattern, text, default=0):
    m = re.search(pattern, text or "")
    return int(m.group(1)) if m else default


def _link(clip, thumb):
    if not clip:
        return ""
    return (f'<a href="{html.escape(clip)}"><img class="th" loading="lazy" src="{html.escape(thumb)}" alt="evidence frame"></a>'
            if thumb else f'<a href="{html.escape(clip)}">clip</a>')


def _table(headers, rows, left=(0,)):
    """Columns listed in `left` hold text (left aligned); all others are numbers (right aligned, tabular figures)."""
    head = "".join(f"<th{' class=l' if i in left else ''}>{html.escape(h)}" for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f"<td class={'l' if i in left else 'num'}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><tr>{head}</tr>{body}</table></div>'


def _schedule_for(cfg, cam_cfg, zone_cfg):
    for s in (zone_cfg.get("schedule"), cam_cfg.get("schedule"), (cfg or {}).get("schedule")):
        if s:
            return Schedule(s)
    return None


def write_reports(out_dir: Path, idle_min_s: float = 300.0, cfg: dict | None = None) -> Path:
    con = sqlite3.connect(out_dir / "analytics.db")
    ccfg = {k: normalize_camera(v) for k, v in ((cfg or {}).get("cameras") or {}).items()}
    cams = [r[0] for r in con.execute("SELECT camera FROM minutes GROUP BY camera ORDER BY camera")]

    # ---- people, per camera and hour
    hourly = defaultdict(lambda: dict(cov=0.0, ps=0.0, pmax=0, app=0, lin=0, lout=0, gath=0, inact=0, wp=0))
    for cam, h, cov, ps, pmax in con.execute(
            "SELECT camera, minute/60, SUM(covered_s), SUM(people_s), MAX(people_max) FROM minutes GROUP BY camera, minute/60"):
        hourly[(cam, h)].update(cov=cov, ps=ps, pmax=pmax)
    kind_key = {"person_appeared": "app", "gathering": "gath", "inactive": "inact", "workpiece": "wp"}
    for cam, ts, typ, val in con.execute(
            "SELECT camera, ts, type, value FROM events WHERE type IN ('person_appeared','line_cross','gathering','inactive','workpiece')"):
        r = hourly[(cam, int(ts // 3600))]
        if typ in kind_key:
            r[kind_key[typ]] += 1
        else:
            r["lin" if val.endswith(":in") else "lout"] += 1
    with open(out_dir / "hourly.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["camera", "date", "hour", "analysed_min", "avg_people", "peak_people", "appearances", "line_in", "line_out",
                    "gatherings", "inactive_people", "workpieces"])
        for (cam, h), r in sorted(hourly.items()):
            d, hr = _hk(h)
            w.writerow([cam, d, f"{hr:02d}", round(r["cov"] / 60, 1), round(r["ps"] / r["cov"], 3) if r["cov"] else "", r["pmax"],
                        r["app"], r["lin"], r["lout"], r["gath"], r["inact"], r["wp"]])

    # ---- machines, per camera / machine / hour
    mh = {}
    for cam, mach, h, *v in con.execute(
            "SELECT camera, machine, minute/60, SUM(work_s), SUM(idle_s), SUM(alarm_s), SUM(wait_s), SUM(work_att_s), "
            "SUM(idle_att_s), SUM(cycles) FROM machine_minutes GROUP BY camera, machine, minute/60"):
        mh[(cam, mach, h)] = v
    with open(out_dir / "machines_hourly.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["camera", "machine", "date", "hour", "work_min", "idle_min", "alarm_min", "wait_min", "cycles", "work_attended_min"])
        for (cam, mach, h), v in sorted(mh.items()):
            d, hr = _hk(h)
            w.writerow([cam, mach, d, f"{hr:02d}", round(v[0] / 60, 1), round(v[1] / 60, 1), round(v[2] / 60, 1), round(v[3] / 60, 1),
                        v[6], round(v[4] / 60, 1)])

    # ---- stations (zones): presence, absence outside breaks
    zh = {}
    for cam, zone, h, cov, occ, ps in con.execute(
            "SELECT camera, zone, minute/60, SUM(covered_s), SUM(occupied_s), SUM(people_s) FROM zone_minutes GROUP BY camera, zone, minute/60"):
        zh[(cam, zone, h)] = (cov, occ, ps)
    zones = sorted({(c, z) for c, z, _ in zh})
    daily_rows, absence_rows, station_csv = [], [], []
    zone_info = {}
    for cam, zone in zones:
        zcfg = next((z for z in ccfg.get(cam, {}).get("zones", []) if z.get("name") == zone), {})
        # only work stations are expected to be staffed; a rest or common zone has no "absence"
        sched = _schedule_for(cfg, ccfg.get(cam, {}), zcfg) if zcfg.get("kind", "workstation") == "workstation" else None
        rows = con.execute("SELECT minute, covered_s, occupied_s FROM zone_minutes WHERE camera=? AND zone=? ORDER BY minute", (cam, zone)).fetchall()
        at = defaultdict(float)
        for m, cov, occ in rows:
            at[_t(m * 60, "%Y-%m-%d")] += occ
        daily, runs = absence_stats(rows, sched) if sched else ({}, [])
        zone_info[(cam, zone)] = dict(kind=zcfg.get("kind", "workstation"), sched=sched is not None)
        for d in sorted(at):
            e = daily.get(d)
            station_csv.append([cam, zone, d, round(at[d] / 60, 1)] + ([round(e["expected_s"] / 60, 1), round(e["present_s"] / 60, 1), round(e["absent_s"] / 60, 1),
                                round(100 * e["absent_s"] / e["expected_s"], 1) if e["expected_s"] else ""] if e else ["", "", "", ""]))
        for start, n, absent in runs:
            absence_rows.append((cam, zone, start * 60.0, n, absent))
    with open(out_dir / "stations_daily.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["camera", "zone", "date", "at_station_min", "expected_min", "present_min", "absent_min", "absence_pct"])
        w.writerows(station_csv)
    absence_rows.sort(key=lambda r: -r[3])
    with open(out_dir / "absences.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["camera", "zone", "start", "duration_min", "empty_min"])
        w.writerows([(c, z, _t(t0), n, round(a / 60, 1)) for c, z, t0, n, a in absence_rows])

    # ---- idle stretches per machine (minute timeline: spans file boundaries)
    idle_ev = defaultdict(list)
    for cam, mach, ts, clip, thumb in con.execute(
            "SELECT camera, subject, ts, clip, thumb FROM events WHERE type='machine_state' AND value='idle' AND clip!='' ORDER BY ts"):
        idle_ev[(cam, mach)].append((ts, clip, thumb))
    idle_list = []
    for cam, mach in sorted({(c, m) for c, m, _ in mh}):
        rows = con.execute("SELECT minute, work_s, idle_s FROM machine_minutes WHERE camera=? AND machine=? ORDER BY minute", (cam, mach)).fetchall()
        for start, idle in idle_runs(rows, idle_min_s):
            near = [e for e in idle_ev[(cam, mach)] if start * 60 - 120 <= e[0] <= start * 60 + 120]
            idle_list.append((cam, mach, start * 60.0, idle, near[0][1] if near else "", near[0][2] if near else ""))
    idle_list.sort(key=lambda p: -p[3])
    with open(out_dir / "idle_periods.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["camera", "machine", "start", "duration_min", "clip"])
        w.writerows([(c, m, _t(t0), round(d / 60, 1), cl) for c, m, t0, d, cl, _ in idle_list])

    ev = con.execute("SELECT camera, ts, type, subject, value, clip, source FROM events ORDER BY camera, ts").fetchall()
    with open(out_dir / "events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["camera", "time", "type", "subject", "value", "clip", "source"])
        w.writerows([(c, _t(ts), ty, s, v, cl, src) for c, ts, ty, s, v, cl, src in ev])

    # ---- HTML: overview tables
    cam_rows = []
    for cam in cams:
        hs = {h: r for (c, h), r in hourly.items() if c == cam}
        cov, ps = sum(r["cov"] for r in hs.values()), sum(r["ps"] for r in hs.values())
        cam_rows.append([html.escape(cam), f"{cov / 3600:.1f}", f"{ps / cov if cov else 0:.2f}", max((r["pmax"] for r in hs.values()), default=0),
                         sum(r["app"] for r in hs.values()), f"{sum(r['lin'] for r in hs.values())} / {sum(r['lout'] for r in hs.values())}",
                         sum(r["gath"] for r in hs.values()), sum(r["inact"] for r in hs.values()), sum(r["wp"] for r in hs.values())])
    mach_rows, st_rows = [], []
    for cam, mach in sorted({(c, m) for c, m, _ in mh}):
        v = [sum(x[i] for (c, m, _), x in mh.items() if (c, m) == (cam, mach)) for i in range(7)]
        mcfg = next((m for m in ccfg.get(cam, {}).get("machines", []) if m["name"] == mach), {})
        linked = bool(mcfg.get("zone")) if mcfg else (v[4] + v[5] > 0)
        src_kind = mcfg.get("source", "")
        mach_rows.append([html.escape(cam), html.escape(mach) + (f' <span class="tag">{src_kind}</span>' if src_kind else ""),
                          f"{100 * v[0] / (v[0] + v[1]):.0f}%" if v[0] + v[1] else "–", f"{v[0] / 3600:.1f}", f"{v[1] / 3600:.1f}",
                          f"{v[2] / 60:.0f}", f"{v[6]:.0f}", f"{(v[0] - v[4]) / 60:.0f}" if linked else "–"])
    for (cam, zone), info in sorted(zone_info.items()):
        rows = [r for r in station_csv if r[0] == cam and r[1] == zone]
        at = sum(r[3] for r in rows)
        exp = sum(r[4] for r in rows if r[4] != "")
        ab = sum(r[6] for r in rows if r[6] != "")
        st_rows.append([html.escape(cam), html.escape(zone) + f' <span class="tag">{info["kind"]}</span>', f"{at / 60:.1f}",
                        f"{exp / 60:.1f}" if info["sched"] else "–", f"{ab / 60:.1f}" if info["sched"] else "–",
                        f"{100 * ab / exp:.0f}%" if info["sched"] and exp else "–"])

    # ---- HTML: per-camera heatmaps
    sections = []
    for cam in cams:
        hs = {h: r for (c, h), r in hourly.items() if c == cam}
        hrs = sorted(set(hs) | {h for (c, _, h) in mh if c == cam} | {h for (c, _, h) in zh if c == cam})
        days = [_t(d * 86400, "%Y-%m-%d") for d in range(int(hrs[0] * 3600 // 86400), int(hrs[-1] * 3600 // 86400) + 1)]
        ppl = {}
        for h, r in hs.items():
            if r["cov"] >= MIN_COVER_S:
                ppl[_hk(h)] = (r["ps"] / r["cov"], f"peak {r['pmax']}, {r['cov'] / 60:.0f} min analysed, {r['app']} appearances")
        pmax = max((v for v, _ in ppl.values()), default=0)
        body = ("<h3>Average people in view, by hour</h3>"
                + _heat(days, ppl, lambda v: f"{v:.1f}" if v >= 0.05 else "0", pmax, lambda d, h, v, det: f"{d} {h:02d}:00 · avg {v:.2f} people · {det}")
                + _legend("0", f"{pmax:.1f}", "people (hourly mean)"))
        for (c, zone), info in sorted(zone_info.items()):
            if c != cam:
                continue
            cells = {_hk(h): (100 * occ / cov, f"{occ / 60:.0f} of {cov / 60:.0f} min occupied") for (c2, z2, h), (cov, occ, _) in zh.items()
                     if (c2, z2) == (cam, zone) and cov >= MIN_COVER_S}
            body += (f"<h3>Station <b>{html.escape(zone)}</b> <span class='tag'>({info['kind']})</span>: occupied by someone, by hour</h3>"
                     + _heat(days, cells, lambda v: f"{v:.0f}", 100, lambda d, h, v, det: f"{d} {h:02d}:00 · {v:.0f}% occupied · {det}")
                     + _legend("0", "100", "% of analysed time with a person in the zone"))
        for (c, mach) in sorted({(c, m) for c, m, _ in mh if c == cam}):
            cells, ccells = {}, {}
            for (c2, m2, h), v in mh.items():
                if (c2, m2) != (cam, mach):
                    continue
                if v[0] + v[1] >= MIN_COVER_S:
                    cells[_hk(h)] = (100 * v[0] / (v[0] + v[1]), f"{v[0] / 60:.0f} min work / {v[1] / 60:.0f} min idle" + (f" / {v[2] / 60:.0f} min red" if v[2] else ""))
                    ccells[_hk(h)] = (v[6], f"{v[6]} cycles")
            body += (f"<h3>Machine <b>{html.escape(mach)}</b>: working share, by hour</h3>"
                     + _heat(days, cells, lambda v: f"{v:.0f}", 100, lambda d, h, v, det: f"{d} {h:02d}:00 · {v:.0f}% working · {det}")
                     + _legend("0", "100", "% of analysed time working"))
            cmax = max((v for v, _ in ccells.values()), default=0)
            if cmax > 0:
                body += (f"<h3>Machine <b>{html.escape(mach)}</b>: completed cycles, by hour</h3>"
                         + _heat(days, ccells, lambda v: f"{v:.0f}", cmax, lambda d, h, v, det: f"{d} {h:02d}:00 · {det}")
                         + _legend("0", f"{cmax:.0f}", "cycles per hour"))
        sections.append(f"<section><h2>{html.escape(cam)}</h2>{body}</section>")

    # ---- HTML: incident lists
    def incidents(title, intro, rows, headers, left):
        if not rows:
            return ""
        return f"<section><h2>{title}</h2><p class=sub>{intro}</p>{_table(headers, rows, left)}</section>"

    idle_sec = incidents("Longest machine idle periods",
                         f"Stretches of at least {idle_min_s / 60:.0f} min with (almost) no work, minute resolution. A frame and clip are attached when one was "
                         "saved near the start; clips open in a video player such as VLC or Windows Media Player.",
                         [[html.escape(c), html.escape(m), _t(t0), f"{d / 60:.0f} min", _link(cl, th)] for c, m, t0, d, cl, th in idle_list[:40]],
                         ["Camera", "Machine", "Start", "Idle", "Frame"], (0, 1, 2, 4))
    abs_sec = incidents("Longest absences from a station",
                        "Consecutive minutes inside work hours (outside breaks) in which the station zone was empty. Presence has a hold time, "
                        "so short detection gaps do not count as absence.",
                        [[html.escape(c), html.escape(z), _t(t0), f"{n} min"] for c, z, t0, n, _ in absence_rows[:40]],
                        ["Camera", "Zone", "Start", "Empty for"], (0, 1, 2))
    gath = con.execute("SELECT camera, ts, value, clip, thumb FROM events WHERE type='gathering' ORDER BY ts").fetchall()
    gath.sort(key=lambda r: -_num(r"duration=(\d+)s", r[2]))
    gath_sec = incidents("Gatherings", "Two or more people standing close together outside the work zones for longer than the configured time.",
                         [[html.escape(c), _t(ts), f"{_num(r'duration=(\d+)s', v) / 60:.1f} min", _num(r"people=(\d+)", v), _link(cl, th)]
                          for c, ts, v, cl, th in gath[:40]], ["Camera", "Start", "Duration", "Max people", "Frame"], (0, 1, 4))
    ina = con.execute("SELECT camera, ts, subject, value, clip, thumb FROM events WHERE type='inactive' ORDER BY ts").fetchall()
    ina.sort(key=lambda r: -_num(r"duration=(\d+)s", r[3]))
    ina_sec = incidents("People without movement", "A person whose body hardly moved for longer than the configured time. Method and threshold are set per camera and "
                        "need calibration (see README).",
                        [[html.escape(c), _t(ts), f"{_num(r'duration=(\d+)s', v) / 60:.1f} min", html.escape(v.split("posture=")[-1].split()[0] if "posture=" in v else ""),
                          html.escape(v.split("zone=")[-1] if "zone=" in v else ""), _link(cl, th)] for c, ts, s, v, cl, th in ina[:40]],
                        ["Camera", "Start", "Duration", "Posture", "Zone", "Frame"], (0, 1, 3, 4, 5))

    fcount, fvid, fcpu = con.execute("SELECT COUNT(*), COALESCE(SUM(processed_s),0), COALESCE(SUM(elapsed_s),0) FROM files").fetchone()
    parts = [f"<section><h2>Cameras</h2>{_table(['Camera', 'Analysed (h)', 'Avg people in view', 'Peak people', 'Appearances', 'Line in / out', 'Gatherings', 'People without movement', 'Workpieces counted'], cam_rows)}"
             "<p class='note'>Appearances = confirmed person tracks (one person can produce several if tracking breaks). Tables: hourly.csv, events.csv.</p></section>"]
    if st_rows:
        parts.append(f"<section><h2>Stations</h2>{_table(['Camera', 'Zone', 'At the station (h)', 'Expected (h)', 'Absent (h)', 'Absence'], st_rows, (0, 1))}"
                     "<p class='note'>Expected / absent need a <code>schedule</code> (work hours, breaks, weekdays) in the config; absence counts only minutes "
                     "the schedule expects someone there. Tables: stations_daily.csv, absences.csv.</p></section>")
    if mach_rows:
        parts.append(f"<section><h2>Machines</h2>{_table(['Camera', 'Machine', 'Working', 'Work (h)', 'Idle (h)', 'Red lamp (min)', 'Cycles', 'Work without operator (min)'], mach_rows, (0, 1))}"
                     "<p class='note'>Working = signal lamp colour (lamp source) or motion in the machine zone (motion source): a proxy, not a machine signal. "
                     "Work without operator needs the machine linked to a station zone. Table: machines_hourly.csv.</p></section>")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Video analytics report</title><style>{CSS}</style></head>
<body><main><h1>Video analytics report</h1>
<p class="sub">{len(cams)} camera(s) · {fcount} file(s) · {fvid / 3600:.1f} h of video analysed in {fcpu / 3600:.1f} CPU-hours.
Times are the wall-clock times found in the file names.</p>
{''.join(parts)}{''.join(sections)}{idle_sec}{abs_sec}{gath_sec}{ina_sec}</main></body></html>"""
    (out_dir / "report.html").write_text(page, encoding="utf-8")
    con.close()
    return out_dir / "report.html"
