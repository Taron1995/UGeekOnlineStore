"""Reports from analytics.db: overview tables, per-camera hour x day heatmaps, incidents, CSV exports."""
from __future__ import annotations

import csv
import html
import json
import math
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .config import normalize_camera
from .metrics import absence_stats, idle_runs
from .schedule import Schedule
from .storage import FINE_S

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
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834}}
{_ramp_rules(False)}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926}}
{_ramp_rules(True, ':root:not([data-theme="light"]) ')}}}
:root[data-theme="dark"]{{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926}}
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
.tlwrap{{position:relative;overflow-x:auto}} svg.tl{{display:block;width:100%;min-width:640px;height:auto}}
.tl-grid{{stroke:var(--grid);stroke-width:1}} .tl-text{{fill:var(--muted);font-size:11px;font-family:inherit}}
.tl-line{{fill:none;stroke:var(--s1);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}} .tl-area{{fill:var(--s1);opacity:.10}}
.tl-dot{{fill:var(--s1)}} .tl-bar{{fill:var(--s2)}} .tl-cross{{stroke:var(--muted);stroke-width:1;visibility:hidden}}
.tl-tip{{position:absolute;pointer-events:none;visibility:hidden;background:var(--surface);color:var(--ink);border:1px solid var(--border);
border-radius:6px;padding:4px 8px;font-size:12px;white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,.18)}} .tl-tip b{{font-size:14px}}
.setupgrid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}} .setupgrid figure{{margin:0}}
.setupgrid img{{width:100%;border-radius:6px;display:block;border:1px solid var(--border)}}
.setupgrid figcaption{{font-size:13px;color:var(--ink2);padding-top:6px}} .setupgrid ul{{margin:4px 0 0 18px;padding:0}}
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


TL_W, TL_L, TL_R, TL_T, TL_B = 1000, 46, 10, 12, 24
TL_H, TL_BAR_H = 150, 112
TL_STEPS = (5, 10, 15, 30, 60, 120, 300, 600)  # seconds per drawn point: the finest that keeps a chart at TL_POINTS points or fewer
TL_POINTS = 150
TL_MAX_S = 12 * 3600  # recordings spanning more than this get the hour x day heatmaps only
HEAT_MIN_S = 3 * 3600  # recordings spanning less need no heatmap: the timeline already shows everything
TICKS = (5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400)  # spacing of the time labels, seconds
TL_SCRIPT = """<script>(function(){document.querySelectorAll('svg.tl').forEach(function(svg){
var v=JSON.parse(svg.getAttribute('data-v')),d=JSON.parse(svg.getAttribute('data-d')||'[]'),t0=+svg.getAttribute('data-t0'),
step=+svg.getAttribute('data-step'),x0=+svg.getAttribute('data-x0'),w=+svg.getAttribute('data-w'),unit=svg.getAttribute('data-unit')||'',
n=v.length,wrap=svg.parentNode,cross=svg.querySelector('.tl-cross'),tip=document.createElement('div');
tip.className='tl-tip';wrap.appendChild(tip);
function p(k){return ('0'+k).slice(-2);}
function clock(s,full){var t=new Date(s*1000);return p(t.getUTCHours())+':'+p(t.getUTCMinutes())+(full?':'+p(t.getUTCSeconds()):'');}
function hide(){cross.style.visibility='hidden';tip.style.visibility='hidden';}
svg.addEventListener('pointermove',function(e){var r=svg.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*1000,i=Math.floor((x-x0)/w*n);
if(i<0||i>=n){hide();return;}var cx=x0+(i+0.5)*w/n,a=t0+i*step,full=step<60;
cross.setAttribute('x1',cx);cross.setAttribute('x2',cx);cross.style.visibility='visible';
tip.textContent='';var b=document.createElement('b');b.textContent=v[i]===null?'no data':v[i]+' '+unit;tip.appendChild(b);
tip.appendChild(document.createTextNode('  '+clock(a,full)+'\\u2013'+clock(a+step,full)+(d[i]?'  \\u00b7  '+d[i]:'')));
tip.style.visibility='visible';var px=cx/1000*r.width,tw=tip.offsetWidth,left=px+10;if(left+tw>r.width){left=px-tw-10;}
tip.style.left=Math.max(0,left)+'px';tip.style.top='2px';});
svg.addEventListener('pointerleave',hide);});})();</script>"""


def _dur(seconds: float) -> str:
    """12 s, 7.5 min, 3.2 h: the unit follows the size (a 90 s clip must not read '0.0 h')."""
    s = float(seconds)
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        m = s / 60
        return f"{m:.0f} min" if m >= 10 else f"{m:.1f} min"
    return f"{s / 3600:.1f} h"


def _clock(ts: float, seconds: bool = False) -> str:
    return _t(ts, "%H:%M:%S" if seconds else "%H:%M")


class _Grid:
    """The points drawn in the charts of one camera: buckets of `step` seconds starting at `t0`. One grid per camera, so its charts line up."""

    def __init__(self, base: int, starts):
        lo, hi = min(starts), max(starts) + base
        self.span = hi - lo
        self.step = next((s for s in TL_STEPS if s >= base and self.span / s <= TL_POINTS), TL_STEPS[-1])
        self.t0 = lo // self.step * self.step  # buckets begin on round clock times
        self.n = max(1, math.ceil((hi - self.t0) / self.step))
        self.min_cov = min(0.3 * self.step, 2.0)  # analysed seconds a bucket needs before its value is shown

    def at(self, start) -> int:
        return int((start - self.t0) // self.step)


def _tick(v: float) -> str:
    return f"{v:.0f}" if abs(v - round(v)) < 1e-9 or v >= 10 else f"{v:.1f}"


def _tl_svg(values, grid: _Grid, vmax: float, kind: str, unit: str, label: str, details=None, height: int = TL_H) -> str:
    """values: one number (or None = no data) per grid bucket. kind: 'line' (2 px, 10 % area wash) or 'bars' (rounded tops, 2 px gaps).
    Hover (crosshair + readout, with `details` text per bucket) comes from TL_SCRIPT; the numbers are also in the CSV files."""
    n = grid.n
    pw, ph = TL_W - TL_L - TL_R, height - TL_T - TL_B
    base = TL_T + ph
    sx = lambda i: TL_L + (i + 0.5) * pw / n
    sy = lambda v: TL_T + ph * (1 - min(max(v, 0), vmax) / vmax)
    g = []
    for frac in (0, 0.5, 1):
        y = TL_T + ph * (1 - frac)
        g.append(f'<line class="tl-grid" x1="{TL_L}" x2="{TL_W - TL_R}" y1="{y:.1f}" y2="{y:.1f}"/>'
                 f'<text class="tl-text" x="{TL_L - 6}" y="{y + 4:.1f}" text-anchor="end">{_tick(vmax * frac)}</text>')
    total = n * grid.step
    tick = next((s for s in TICKS if total / s <= 8), TICKS[-1])
    t = math.ceil(grid.t0 / tick) * tick
    while t <= grid.t0 + total:  # labels sit on round clock times
        x = TL_L + (t - grid.t0) * pw / total
        if TL_L - 12 <= x <= TL_W - 30:
            g.append(f'<line class="tl-grid" x1="{x:.1f}" x2="{x:.1f}" y1="{base}" y2="{base + 4}"/>'
                     f'<text class="tl-text" x="{x:.1f}" y="{height - 6}" text-anchor="middle">{_clock(t, tick < 60)}</text>')
        t += tick
    marks = []
    if kind == "line":
        runs, run = [], []
        for i, v in enumerate(list(values) + [None]):
            if v is None:
                if run:
                    runs.append(run)
                    run = []
            else:
                run.append((sx(i), sy(v)))
        for r in runs:
            if len(r) == 1:
                marks.append(f'<circle class="tl-dot" cx="{r[0][0]:.1f}" cy="{r[0][1]:.1f}" r="3"/>')
                continue
            pts = " L".join(f"{x:.1f},{y:.1f}" for x, y in r)
            marks.append(f'<path class="tl-area" d="M{r[0][0]:.1f},{base:.1f} L{pts} L{r[-1][0]:.1f},{base:.1f}Z"/><path class="tl-line" d="M{pts}"/>')
        known = [(v, i) for i, v in enumerate(values) if v is not None]
        if known:  # one direct label: the extreme
            v, i = max(known)
            marks.append(f'<text class="tl-text" x="{min(max(sx(i), TL_L + 24), TL_W - 30):.1f}" y="{max(sy(v) - 6, 9):.1f}" text-anchor="middle">peak {_tick(v)}</text>')
    else:
        bw = max(1.0, min(24.0, pw / n - 2.0))
        for i, v in enumerate(values):
            h = 0 if v is None else ph * min(v, vmax) / vmax
            if h < 0.5:
                continue
            x, y, r = sx(i) - bw / 2, base - h, min(4.0, bw / 2, h) if bw >= 4 else 0.0
            marks.append(f'<path class="tl-bar" d="M{x:.1f},{base:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} H{x + bw - r:.1f} '
                         f'Q{x + bw:.1f},{y:.1f} {x + bw:.1f},{y + r:.1f} V{base:.1f}Z"/>')
    js = lambda obj: html.escape(json.dumps(obj, ensure_ascii=False), quote=True)
    return (f'<div class="tlwrap"><svg class="tl" viewBox="0 0 {TL_W} {height}" role="img" aria-label="{html.escape(label, quote=True)}" '
            f'data-v="{js([None if v is None else round(v, 2) for v in values])}" data-d="{js(list(details or []))}" '
            f'data-t0="{grid.t0}" data-step="{grid.step}" data-x0="{TL_L}" data-w="{pw}" data-unit="{html.escape(unit, quote=True)}">'
            f'{"".join(g)}{"".join(marks)}<line class="tl-cross" y1="{TL_T}" y2="{base}" x1="0" x2="0"/></svg></div>')


def _series(con, cam: str):
    """Rows (start_second, ...) of the people / zone / machine series at the finest resolution stored: 5 s, or per minute for shards
    written before the 5 s tables existed. Returns (base seconds, people rows, {zone: rows}, {machine: rows})."""
    ppl = con.execute("SELECT bin * ?, covered_s, people_s, people_max FROM fine WHERE camera=? ORDER BY bin", (FINE_S, cam)).fetchall()
    if ppl:
        base, key, zt, mt = FINE_S, "bin", "zone_fine", "machine_fine"
    else:
        ppl = con.execute("SELECT minute * 60, covered_s, people_s, people_max FROM minutes WHERE camera=? ORDER BY minute", (cam,)).fetchall()
        base, key, zt, mt = 60, "minute", "zone_minutes", "machine_minutes"
    zones, machines = defaultdict(list), defaultdict(list)
    for z, s, cov, occ in con.execute(f"SELECT zone, {key} * ?, covered_s, occupied_s FROM {zt} WHERE camera=? ORDER BY zone, {key}", (base, cam)):
        zones[z].append((s, cov, occ))
    for m, s, work, idle in con.execute(f"SELECT machine, {key} * ?, work_s, idle_s FROM {mt} WHERE camera=? ORDER BY machine, {key}", (base, cam)):
        machines[m].append((s, work, idle))
    return base, ppl, dict(zones), dict(machines)


def _timelines(cam: str, grid: _Grid, ppl, zones, machines, zone_kind: dict, csv_rows: dict) -> str:
    """People line, one occupancy bar chart per station, one working-share bar chart per machine; fills the csv rows."""
    n = grid.n
    cov, ps, pk = [0.0] * n, [0.0] * n, [0] * n
    for s, c, p, m in ppl:
        i = grid.at(s)
        cov[i] += c
        ps[i] += p
        pk[i] = max(pk[i], m)
    vals = [ps[i] / cov[i] if cov[i] >= grid.min_cov else None for i in range(n)]
    det = [f"peak {pk[i]}" if vals[i] is not None else "" for i in range(n)]
    vmax = max(2, 2 * math.ceil(max((v for v in vals if v is not None), default=0) / 2))  # even, so the middle grid line is a whole number
    csv_rows["people"] += [[cam, _t(s), round(c, 1), round(p / c, 3) if c else "", m] for s, c, p, m in ppl]
    out = ("<h3>People in view over time</h3>" + _tl_svg(vals, grid, vmax, "line", "people", f"People in view over time, {cam}", det)
           + f"<p class=note>Each point is the average over {_dur(grid.step)} of video; hover for the peak. Raw values: timeline.csv.</p>")
    for zone, rows in zones.items():
        cv, oc = [0.0] * n, [0.0] * n
        for s, c, o in rows:
            cv[grid.at(s)] += c
            oc[grid.at(s)] += o
        v = [100 * oc[i] / cv[i] if cv[i] >= grid.min_cov else None for i in range(n)]
        d = [f"{oc[i]:.0f} of {cv[i]:.0f} s occupied" if v[i] is not None else "" for i in range(n)]
        out += (f"<h3>Station <b>{html.escape(zone)}</b> <span class='tag'>({zone_kind.get(zone, 'workstation')})</span>: someone there</h3>"
                + _tl_svg(v, grid, 100, "bars", "% occupied", f"Station {zone}: share of time with a person there, {cam}", d, TL_BAR_H))
        csv_rows["zones"] += [[cam, zone, _t(s), round(c, 1), round(o, 1)] for s, c, o in rows]
    for mach, rows in machines.items():
        wk, idl = [0.0] * n, [0.0] * n
        for s, w, i_ in rows:
            wk[grid.at(s)] += w
            idl[grid.at(s)] += i_
        v = [100 * wk[i] / (wk[i] + idl[i]) if wk[i] + idl[i] >= grid.min_cov else None for i in range(n)]
        d = [f"{wk[i]:.0f} s working, {idl[i]:.0f} s idle" if v[i] is not None else "" for i in range(n)]
        out += (f"<h3>Machine <b>{html.escape(mach)}</b>: working share</h3>"
                + _tl_svg(v, grid, 100, "bars", "% working", f"Machine {mach}: share of time working, {cam}", d, TL_BAR_H))
        csv_rows["machines"] += [[cam, mach, _t(s), round(w, 1), round(i_, 1)] for s, w, i_ in rows]
    return out


def _setup_section(setup: dict) -> str:
    """What the automatic setup found, with the overlay image per camera, so it can be checked at a glance."""
    if not setup:
        return ""
    figs = []
    for cam, info in sorted(setup.items()):
        items = [f"station <b>spot{i}</b>: people stood here for {_dur(s['total_s'])} in total (sampled footage)" for i, s in enumerate(info.get("spots", []), 1)]
        items += [f"signal lamp <b>lamp{i}</b>: colours seen {html.escape(', '.join(m['colours']))}" for i, m in enumerate(info.get("lamps", []), 1)]
        items += [f"moving machine zone <b>motion{i}</b>: moving in {100 * m['freq']:.0f}% of the frames" for i, m in enumerate(info.get("motion", []), 1)]
        ul = "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>" if items else "<p class=note>Nothing found: people counting only.</p>"
        img = f'<img loading="lazy" src="{html.escape(info["image"], quote=True)}" alt="learned setup of {html.escape(cam, quote=True)}">' if info.get("image") else ""
        figs.append(f"<figure>{img}<figcaption><b>{html.escape(cam)}</b> · learned from {info.get('sampled_s', 0)} s of footage{ul}</figcaption></figure>")
    return ("<section><h2>Learned setup <span class='tag'>automatic - please check</span></h2>"
            "<p class=sub>Green = stations (where people stayed), orange = machines (a moving part, or a lamp). This is a best guess made from the footage itself: "
            "if something is wrong or missing, edit <code>cameras.auto.yaml</code> in the results folder (delete or move a zone, add a "
            "<code>schedule</code>) and run the same command again.</p>"
            f"<div class='setupgrid'>{''.join(figs)}</div></section>")


def write_reports(out_dir: Path, idle_min_s: float = 300.0, cfg: dict | None = None) -> Path:
    con = sqlite3.connect(out_dir / "analytics.db")
    setup_file = out_dir / "setup" / "setup.json"
    setup = json.loads(setup_file.read_text(encoding="utf-8")) if setup_file.exists() else {}
    tl = dict(people=[], zones=[], machines=[])  # csv rows behind the timeline charts
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
        cam_rows.append([html.escape(cam), _dur(cov), f"{ps / cov if cov else 0:.2f}", max((r["pmax"] for r in hs.values()), default=0),
                         sum(r["app"] for r in hs.values()), f"{sum(r['lin'] for r in hs.values())} / {sum(r['lout'] for r in hs.values())}",
                         sum(r["gath"] for r in hs.values()), sum(r["inact"] for r in hs.values()), sum(r["wp"] for r in hs.values())])
    mach_rows, st_rows = [], []
    for cam, mach in sorted({(c, m) for c, m, _ in mh}):
        v = [sum(x[i] for (c, m, _), x in mh.items() if (c, m) == (cam, mach)) for i in range(7)]
        mcfg = next((m for m in ccfg.get(cam, {}).get("machines", []) if m["name"] == mach), {})
        linked = bool(mcfg.get("zone")) if mcfg else (v[4] + v[5] > 0)
        src_kind = mcfg.get("source", "")
        mach_rows.append([html.escape(cam), html.escape(mach) + (f' <span class="tag">{src_kind}</span>' if src_kind else ""),
                          f"{100 * v[0] / (v[0] + v[1]):.0f}%" if v[0] + v[1] else "–", _dur(v[0]), _dur(v[1]),
                          _dur(v[2]), f"{v[6]:.0f}", _dur(v[0] - v[4]) if linked else "–"])
    for (cam, zone), info in sorted(zone_info.items()):
        rows = [r for r in station_csv if r[0] == cam and r[1] == zone]
        at = sum(r[3] for r in rows)
        exp = sum(r[4] for r in rows if r[4] != "")
        ab = sum(r[6] for r in rows if r[6] != "")
        st_rows.append([html.escape(cam), html.escape(zone) + f' <span class="tag">{info["kind"]}</span>', _dur(at * 60),
                        _dur(exp * 60) if info["sched"] else "–", _dur(ab * 60) if info["sched"] else "–",
                        f"{100 * ab / exp:.0f}%" if info["sched"] and exp else "–"])

    # ---- HTML: one section per camera. Short recordings get timelines (5 s resolution), long ones hour x day heatmaps, in between both.
    sections = []
    for cam in cams:
        hs = {h: r for (c, h), r in hourly.items() if c == cam}
        base, prow, zrows, mrows = _series(con, cam)
        grid = _Grid(base, [r[0] for r in prow]) if prow else None
        show_tl = grid is not None and grid.n >= 2 and grid.span <= TL_MAX_S
        show_heat = not show_tl or grid.span > HEAT_MIN_S
        body = ""
        if show_tl:
            body += _timelines(cam, grid, prow, zrows, mrows, {z: i["kind"] for (c, z), i in zone_info.items() if c == cam}, tl)
        if show_heat:
            hrs = sorted(set(hs) | {h for (c, _, h) in mh if c == cam} | {h for (c, _, h) in zh if c == cam})
            days = [_t(d * 86400, "%Y-%m-%d") for d in range(int(hrs[0] * 3600 // 86400), int(hrs[-1] * 3600 // 86400) + 1)]
            ppl = {}
            for h, r in hs.items():
                if r["cov"] >= MIN_COVER_S:
                    ppl[_hk(h)] = (r["ps"] / r["cov"], f"peak {r['pmax']}, {r['cov'] / 60:.0f} min analysed, {r['app']} appearances")
            pmax = max((v for v, _ in ppl.values()), default=0)
            body += ("<h3>Average people in view, by hour</h3>"
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
    parts = [f"<section><h2>Cameras</h2>{_table(['Camera', 'Analysed', 'Avg people in view', 'Peak people', 'Appearances', 'Line in / out', 'Gatherings', 'People without movement', 'Workpieces counted'], cam_rows)}"
             "<p class='note'>Appearances = confirmed person tracks (one person can produce several if tracking breaks). Tables: hourly.csv, events.csv.</p></section>"]
    if st_rows:
        parts.append(f"<section><h2>Stations</h2>{_table(['Camera', 'Zone', 'At the station', 'Expected', 'Absent', 'Absence'], st_rows, (0, 1))}"
                     "<p class='note'>Expected / absent need a <code>schedule</code> (work hours, breaks, weekdays) in the config; absence counts only minutes "
                     "the schedule expects someone there. Tables: stations_daily.csv, absences.csv.</p></section>")
    if mach_rows:
        parts.append(f"<section><h2>Machines</h2>{_table(['Camera', 'Machine', 'Working', 'Work', 'Idle', 'Red lamp', 'Cycles', 'Work without operator'], mach_rows, (0, 1))}"
                     "<p class='note'>Working = signal lamp colour (lamp source) or motion in the machine zone (motion source): a proxy, not a machine signal. "
                     "Work without operator needs the machine linked to a station zone. Table: machines_hourly.csv.</p></section>")
    for name, header, rows in (("timeline.csv", ["camera", "time", "analysed_s", "avg_people", "peak_people"], tl["people"]),
                               ("stations_timeline.csv", ["camera", "zone", "time", "analysed_s", "occupied_s"], tl["zones"]),
                               ("machines_timeline.csv", ["camera", "machine", "time", "work_s", "idle_s"], tl["machines"])):
        (out_dir / name).unlink(missing_ok=True)  # a file of an earlier run must not be mistaken for this one
        if rows:
            with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows)
    (out_dir / "machines_minutely.csv").unlink(missing_ok=True)  # written by earlier versions
    parts.insert(1, _setup_section(setup))  # right after the cameras overview: this is what the reader should verify first
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Video analytics report</title><style>{CSS}</style></head>
<body><main><h1>Video analytics report</h1>
<p class="sub">{len(cams)} camera(s) · {fcount} file(s) · {_dur(fvid)} of video analysed in {_dur(fcpu)} of processing time.
Times are the wall-clock times found in the file names.</p>
{''.join(parts)}{''.join(sections)}{idle_sec}{abs_sec}{gath_sec}{ina_sec}</main>{TL_SCRIPT}</body></html>"""
    (out_dir / "report.html").write_text(page, encoding="utf-8")
    con.close()
    return out_dir / "report.html"
