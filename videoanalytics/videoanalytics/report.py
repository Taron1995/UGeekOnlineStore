"""Reports from analytics.db: overview, per-camera hour x day heatmaps, long idle periods, CSV exports."""
from __future__ import annotations

import csv
import html
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

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
th{{color:var(--ink2);font-weight:600}} th:first-child,td:first-child{{text-align:left}}
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


def write_reports(out_dir: Path, idle_min_s: float = 300.0) -> Path:
    con = sqlite3.connect(out_dir / "analytics.db")
    cams = [r[0] for r in con.execute("SELECT DISTINCT camera FROM minutes ORDER BY 1")]

    hourly = defaultdict(lambda: dict(cov=0.0, ps=0.0, pmax=0, work=0.0, idle=0.0, app=0, lin=0, lout=0))
    for cam, h, cov, ps, pmax, work, idle in con.execute(
            "SELECT camera, minute/60, SUM(covered_s), SUM(people_s), MAX(people_max), SUM(work_s), SUM(idle_s) "
            "FROM minutes GROUP BY camera, minute/60"):
        hourly[(cam, h)].update(cov=cov, ps=ps, pmax=pmax, work=work, idle=idle)
    for cam, ts, typ, val in con.execute(
            "SELECT camera, ts, type, value FROM events WHERE type IN ('person_appeared','line_cross')"):
        r = hourly[(cam, int(ts // 3600))]
        if typ == "person_appeared":
            r["app"] += 1
        elif val.endswith(":in"):
            r["lin"] += 1
        else:
            r["lout"] += 1

    with open(out_dir / "hourly.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["camera", "date", "hour", "analysed_min", "avg_people", "peak_people", "appearances",
                    "line_in", "line_out", "machine_work_min", "machine_idle_min"])
        for (cam, h), r in sorted(hourly.items()):
            w.writerow([cam, _t(h * 3600, "%Y-%m-%d"), _t(h * 3600, "%H"), round(r["cov"] / 60, 1),
                        round(r["ps"] / r["cov"], 3) if r["cov"] else "", r["pmax"], r["app"], r["lin"], r["lout"],
                        round(r["work"] / 60, 1), round(r["idle"] / 60, 1)])

    ev = con.execute("SELECT camera, ts, type, subject, value, clip, source FROM events ORDER BY camera, ts").fetchall()
    with open(out_dir / "events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["camera", "time", "type", "subject", "value", "clip", "source"])
        w.writerows([(c, _t(ts), ty, s, v, cl, src) for c, ts, ty, s, v, cl, src in ev])

    # Idle periods come from the per-minute timeline (not from event pairs), so they also span file boundaries.
    idle_ev = defaultdict(list)
    for cam, ts, clip, thumb in con.execute(
            "SELECT camera, ts, clip, thumb FROM events WHERE type='machine_state' AND value='idle' AND clip!='' ORDER BY ts"):
        idle_ev[cam].append((ts, clip, thumb))
    idle_periods, run = [], None
    for cam, minute, work, idle in con.execute(
            "SELECT camera, minute, work_s, idle_s FROM minutes WHERE work_s + idle_s >= 30 ORDER BY camera, minute"):
        mostly_idle = idle >= 0.9 * (work + idle)
        if run and (cam, minute - 1) == (run[0], run[2]) and mostly_idle:
            run[2], run[3] = minute, run[3] + idle
            continue
        if run:
            idle_periods.append(run)
        run = [cam, minute, minute, idle] if mostly_idle else None
    if run:
        idle_periods.append(run)
    found = []
    for cam, m0, m1, dur in idle_periods:
        t0 = m0 * 60.0
        if dur < idle_min_s:
            continue
        near = [e for e in idle_ev[cam] if t0 - 120 <= e[0] <= t0 + 120]
        found.append((cam, t0, dur, near[0][1] if near else "", near[0][2] if near else ""))
    idle_periods = found
    idle_periods.sort(key=lambda p: -p[2])
    with open(out_dir / "idle_periods.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["camera", "start", "duration_min", "clip"])
        w.writerows([(c, _t(t0), round(d / 60, 1), cl) for c, t0, d, cl, _ in idle_periods])

    overview, sections = [], []
    for cam in cams:
        hs = {h: r for (c, h), r in hourly.items() if c == cam}
        cov = sum(r["cov"] for r in hs.values())
        ps = sum(r["ps"] for r in hs.values())
        work, idle = sum(r["work"] for r in hs.values()), sum(r["idle"] for r in hs.values())
        has_machine = work + idle > 0
        overview.append(
            f"<tr><td>{html.escape(cam)}</td><td class=num>{cov / 3600:.1f}</td>"
            f"<td class=num>{ps / cov if cov else 0:.2f}</td><td class=num>{max(r['pmax'] for r in hs.values())}</td>"
            f"<td class=num>{sum(r['app'] for r in hs.values())}</td>"
            f"<td class=num>{sum(r['lin'] for r in hs.values())} / {sum(r['lout'] for r in hs.values())}</td>"
            f"<td class=num>{f'{100 * work / (work + idle):.0f}%' if has_machine else '–'}</td></tr>")
        hrs = sorted(hs)
        d0, d1 = int(hrs[0] * 3600 // 86400), int(hrs[-1] * 3600 // 86400)
        days = [_t(d * 86400, "%Y-%m-%d") for d in range(d0, d1 + 1)]
        ppl, mach = {}, {}
        for h, r in hs.items():
            key = (_t(h * 3600, "%Y-%m-%d"), int(_t(h * 3600, "%H")))
            if r["cov"] >= MIN_COVER_S:
                ppl[key] = (r["ps"] / r["cov"], f"peak {r['pmax']}, {r['cov'] / 60:.0f} min analysed, {r['app']} appearances")
            if r["work"] + r["idle"] >= MIN_COVER_S:
                mach[key] = (100 * r["work"] / (r["work"] + r["idle"]), f"{r['work'] / 60:.0f} min work / {r['idle'] / 60:.0f} min idle")
        pmax = max((v for v, _ in ppl.values()), default=0)
        body = (f"<h3>Average people in view, by hour</h3>"
                + _heat(days, ppl, lambda v: f"{v:.1f}" if v >= 0.05 else "0", pmax,
                        lambda d, h, v, det: f"{d} {h:02d}:00 · avg {v:.2f} people · {det}")
                + _legend("0", f"{pmax:.1f}", "people (hourly mean)"))
        if has_machine:
            body += ("<h3>Machine working share, by hour</h3>"
                     + _heat(days, mach, lambda v: f"{v:.0f}", 100,
                             lambda d, h, v, det: f"{d} {h:02d}:00 · {v:.0f}% working · {det}")
                     + _legend("0", "100", "% of analysed time with motion in the machine zone"))
        sections.append(f"<section><h2>{html.escape(cam)}</h2>{body}</section>")

    idle_rows = "".join(
        f"<tr><td>{html.escape(c)}</td><td>{_t(t0)}</td><td class=num>{d / 60:.0f} min</td><td>"
        + (f'<a href="{html.escape(cl)}"><img class="th" loading="lazy" src="{html.escape(th)}" alt="frame at idle start"></a>'
           if cl else "") + "</td></tr>" for c, t0, d, cl, th in idle_periods[:40])
    idle_sec = (f"<section><h2>Longest machine idle periods</h2><p class=sub>Stretches of at least {idle_min_s / 60:.0f} min "
                f"without motion in the machine zone (minute resolution). A frame and clip are attached when one was saved near "
                f"the start; clips open in a video player such as VLC or Windows Media Player.</p>"
                f"<div class=scroll><table><tr><th>Camera<th>Start<th>Duration<th>Frame</tr>{idle_rows}</table></div>"
                f"<p class=note>{len(idle_periods)} period(s) in total, full list in idle_periods.csv.</p></section>"
                if idle_periods else "")

    fcount, fvid, fcpu = con.execute("SELECT COUNT(*), COALESCE(SUM(processed_s),0), COALESCE(SUM(elapsed_s),0) FROM files").fetchone()
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Video analytics report</title><style>{CSS}</style></head>
<body><main><h1>Video analytics report</h1>
<p class="sub">{len(cams)} camera(s) · {fcount} file(s) · {fvid / 3600:.1f} h of video analysed in {fcpu / 3600:.1f} CPU-hours.
Times are the wall-clock times found in the file names.</p>
<section><h2>Overview</h2><div class="scroll"><table><tr><th>Camera<th>Analysed (h)<th>Avg people in view<th>Peak people<th>Appearances<th>Line in / out<th>Machine working</tr>
{''.join(overview)}</table></div>
<p class="note">Appearances = confirmed person tracks (one person can produce several if tracking breaks). Machine working = share of
time with motion in the configured machine zone, a proxy and not a machine signal. Tables: hourly.csv, events.csv, idle_periods.csv.</p></section>
{''.join(sections)}{idle_sec}</main></body></html>"""
    (out_dir / "report.html").write_text(page, encoding="utf-8")
    con.close()
    return out_dir / "report.html"
