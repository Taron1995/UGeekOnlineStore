'use strict';
/* Երևան Նավիգատոր — mobile web navigator for Yerevan.
 * Map: OpenStreetMap · Search: Nominatim · Routing: OSRM (routing.openstreetmap.de)
 */

// ---------- config ----------
const YEREVAN_CENTER = [40.1792, 44.4991];
const YEREVAN_BOUNDS = L.latLngBounds([40.03, 44.33], [40.30, 44.70]);
const ROUTERS = {
  car:  'https://routing.openstreetmap.de/routed-car/route/v1/driving/',
  foot: 'https://routing.openstreetmap.de/routed-foot/route/v1/driving/',
  bike: 'https://routing.openstreetmap.de/routed-bike/route/v1/driving/',
};
const NOMINATIM = 'https://nominatim.openstreetmap.org/search';
const MAX_ACCEPTED_ACCURACY = 80;   // m — worse fixes are ignored once we have a good one
const SNAP_DISTANCE = 30;           // m — snap the marker onto the route when this close
const ARRIVE_DISTANCE = 25;         // m
const REROUTE_MIN_INTERVAL = 10000; // ms

// ---------- state ----------
const S = {
  mode: 'car', pos: null, rawPos: null, accuracy: null, heading: null, compass: null, speed: 0,
  dest: null, destName: '', route: null, navigating: false, follow: true, voice: false,
  offRouteCount: 0, lastReroute: 0, lastSpokenStep: -1, lastSpokenNear: -1, wakeLock: null,
};

// ---------- map ----------
const map = L.map('map', {
  zoomControl: false, maxBounds: YEREVAN_BOUNDS.pad(0.5), minZoom: 11, maxZoom: 19,
}).setView(YEREVAN_CENTER, 13);
const onlineTiles = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19, attribution: '© OpenStreetMap',
});

// Offline mode: map, search and routing work from data stored on the phone.
// If the offline data is missing, the app falls back to online services.
async function initOffline() {
  const bar = $('loadBar'), box = $('loading');
  try {
    box.hidden = false;
    await Offline.load('data/', f => { bar.style.width = Math.round(f * 100) + '%'; $('loadPct').textContent = Math.round(f * 100) + '%'; });
    const offlineLayer = Offline.mapLayer().addTo(map);
    L.control.layers({ 'Քարտեզ (offline)': offlineLayer, 'OSM (ինտերնետով)': onlineTiles }, null, { position: 'bottomleft' }).addTo(map);
    const d = new Date(Offline.meta.built);
    $('netBadge').textContent = '📴 Offline քարտեզ ' + d.toLocaleDateString('hy-AM');
  } catch (e) {
    console.warn('offline data not available', e);
    onlineTiles.addTo(map);
    $('netBadge').textContent = '🌐 Online ռեժիմ';
  } finally { box.hidden = true; }
}

const meIcon = L.divIcon({ className: 'me-icon', iconSize: [40, 40], iconAnchor: [20, 20],
  html: '<div class="me-rot" style="position:relative;width:40px;height:40px"><div class="me-cone"></div><div class="me-dot"></div></div>' });
const meMarker = L.marker(YEREVAN_CENTER, { icon: meIcon, interactive: false, zIndexOffset: 1000 });
const accCircle = L.circle(YEREVAN_CENTER, { radius: 1, color: '#1e88e5', weight: 1, fillOpacity: 0.12, interactive: false });
const destMarker = L.marker(YEREVAN_CENTER);
const routeLine = L.polyline([], { color: '#1565c0', weight: 7, opacity: 0.85 });
const passedLine = L.polyline([], { color: '#90a4ae', weight: 7, opacity: 0.8 });

const $ = id => document.getElementById(id);
const status = t => { $('status').textContent = t; };

// ---------- geometry ----------
const R = 6371000, rad = d => d * Math.PI / 180;
function dist(a, b) { // haversine, a/b = [lat, lng]
  const dLat = rad(b[0] - a[0]), dLng = rad(b[1] - a[1]);
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(rad(a[0])) * Math.cos(rad(b[0])) * Math.sin(dLng / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}
function bearing(a, b) {
  const y = Math.sin(rad(b[1] - a[1])) * Math.cos(rad(b[0]));
  const x = Math.cos(rad(a[0])) * Math.sin(rad(b[0])) - Math.sin(rad(a[0])) * Math.cos(rad(b[0])) * Math.cos(rad(b[1] - a[1]));
  return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
}
// Projects p onto segment a-b (local flat approximation — fine at city scale)
function projectOnSegment(p, a, b) {
  const k = Math.cos(rad(p[0]));
  const ax = a[1] * k, ay = a[0], bx = b[1] * k, by = b[0], px = p[1] * k, py = p[0];
  const dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy;
  let t = len2 ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0;
  t = Math.max(0, Math.min(1, t));
  const q = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  return { point: q, t, d: dist(p, q) };
}
// Nearest point on route; searches near the last known index first for speed and to avoid jumping to parallel streets
function locateOnRoute(p) {
  const r = S.route, c = r.coords;
  let best = null;
  const from = Math.max(0, (r.lastIdx || 0) - 5), to = Math.min(c.length - 1, (r.lastIdx || 0) + 60);
  const scan = (i0, i1) => {
    for (let i = i0; i < i1; i++) {
      const pr = projectOnSegment(p, c[i], c[i + 1]);
      if (!best || pr.d < best.d) best = { ...pr, idx: i };
    }
  };
  scan(from, to);
  if (!best || best.d > SNAP_DISTANCE) scan(0, c.length - 1);
  r.lastIdx = best.idx;
  best.along = r.cum[best.idx] + dist(c[best.idx], best.point);
  return best;
}

// ---------- GPS ----------
function onPosition(p) {
  const { latitude, longitude, accuracy, heading, speed } = p.coords;
  // Ignore very inaccurate fixes (cell/wifi) once we have a good GPS fix less than 15s old
  if (S.accuracy !== null && accuracy > MAX_ACCEPTED_ACCURACY && S.accuracy <= MAX_ACCEPTED_ACCURACY &&
      Date.now() - S.lastFixTime < 15000) return;

  const raw = [latitude, longitude];
  // light smoothing: weight new fix by relative accuracy (simple 1D Kalman per axis)
  // (only for small jitter — real movement larger than the noise is taken as is, so the marker never lags behind)
  if (S.rawPos && S.accuracy && dist(S.rawPos, raw) < 2 * Math.max(accuracy, S.accuracy)) {
    const dt = (p.timestamp - S.lastFixTs) / 1000;
    const varPrev = S.accuracy ** 2 + (dt * Math.max(S.speed || 0, 3)) ** 2; // grow uncertainty with time/speed
    const g = varPrev / (varPrev + accuracy ** 2);
    S.rawPos = [S.rawPos[0] + g * (raw[0] - S.rawPos[0]), S.rawPos[1] + g * (raw[1] - S.rawPos[1])];
  } else S.rawPos = raw;
  S.accuracy = accuracy; S.lastFixTs = p.timestamp; S.lastFixTime = Date.now();
  S.speed = speed || 0;
  if (heading !== null && !isNaN(heading) && S.speed > 1.5) S.heading = heading;

  let shown = S.rawPos;
  if (S.route) {
    const loc = locateOnRoute(S.rawPos);
    const tol = Math.max(SNAP_DISTANCE, accuracy);
    if (loc.d <= tol) {
      shown = loc.point; S.offRouteCount = 0;
      if (S.speed > 1.5 || S.heading === null) S.heading = bearing(S.route.coords[loc.idx], S.route.coords[loc.idx + 1]);
      if (S.navigating) updateGuidance(loc);
    } else if (S.navigating && ++S.offRouteCount >= 3 && Date.now() - S.lastReroute > REROUTE_MIN_INTERVAL) {
      status('Շեղվեցիք երթուղուց — վերահաշվարկ…'); speak('Վերահաշվարկում եմ երթուղին');
      buildRoute(true);
    }
  }
  S.pos = shown;

  meMarker.setLatLng(shown); accCircle.setLatLng(S.rawPos).setRadius(accuracy);
  if (!map.hasLayer(meMarker)) { meMarker.addTo(map); accCircle.addTo(map); map.setView(shown, 16); }
  rotateMarker();
  if (S.follow) map.panTo(shown, { animate: true, duration: 0.5 });
  if (!S.navigating) status(`GPS ճշգրտություն՝ ±${Math.round(accuracy)} մ`);
}
function onGeoError(e) {
  if (e.code !== 1 && S.lastFixTime && Date.now() - S.lastFixTime < 15000) return; // a short gap — keep last fix
  const msg = { 1: 'Թույլ տվեք տեղորոշումը բրաուզերի կարգավորումներում', 2: 'GPS ազդանշան չկա', 3: 'GPS-ը չպատասխանեց' }[e.code] || e.message;
  status('⚠ ' + msg);
}
function startGps() {
  if (!('geolocation' in navigator)) return status('Այս սարքը չի աջակցում տեղորոշում');
  if (!isSecureContext) status('⚠ GPS-ը աշխատում է միայն HTTPS-ով');
  navigator.geolocation.watchPosition(onPosition, onGeoError, { enableHighAccuracy: true, maximumAge: 0, timeout: 20000 });
}

// Compass (used when standing still / walking slowly)
function onOrientation(e) {
  let h = null;
  if (typeof e.webkitCompassHeading === 'number') h = e.webkitCompassHeading;           // iOS
  else if (e.absolute && e.alpha !== null) h = (360 - e.alpha) % 360;                   // Android
  if (h === null) return;
  h = (h + (screen.orientation ? screen.orientation.angle : 0)) % 360;
  S.compass = h;
  if (S.speed <= 1.5) rotateMarker();
}
function startCompass() {
  const add = () => {
    window.addEventListener('deviceorientationabsolute', onOrientation, true);
    window.addEventListener('deviceorientation', onOrientation, true);
  };
  if (window.DeviceOrientationEvent && typeof DeviceOrientationEvent.requestPermission === 'function') {
    DeviceOrientationEvent.requestPermission().then(r => r === 'granted' && add()).catch(() => {});
  } else add();
}
function rotateMarker() {
  const h = S.speed > 1.5 && S.heading !== null ? S.heading : (S.compass ?? S.heading);
  const el = meMarker.getElement && meMarker.getElement();
  const rot = el && el.querySelector('.me-rot');
  if (rot) { rot.style.transform = `rotate(${h ?? 0}deg)`; rot.querySelector('.me-cone').style.display = h === null ? 'none' : ''; }
}

// ---------- search ----------
let searchTimer, searchAbort;
$('search').addEventListener('input', e => {
  clearTimeout(searchTimer);
  const q = e.target.value.trim();
  if (q.length < 3) { $('results').innerHTML = ''; return; }
  if (Offline.ready) { searchTimer = setTimeout(() => showResults(Offline.search(q, S.pos)), 150); return; }
  searchTimer = setTimeout(() => search(q), 600); // Nominatim policy: max 1 req/s
});
function showResults(list) {
  const ul = $('results'); ul.innerHTML = '';
  if (!list.length) { ul.innerHTML = '<li>Ոչինչ չգտնվեց</li>'; return; }
  for (const r of list) {
    const li = document.createElement('li');
    li.innerHTML = `${esc(r.name)}<small>${esc(r.sub)}</small>`;
    li.onclick = () => { ul.innerHTML = ''; $('search').value = r.name; $('search').blur(); setDest([r.lat, r.lon], r.name); };
    ul.appendChild(li);
  }
}
async function search(q) {
  searchAbort?.abort(); searchAbort = new AbortController();
  const b = YEREVAN_BOUNDS;
  const url = `${NOMINATIM}?format=jsonv2&limit=8&countrycodes=am&bounded=1&accept-language=hy,ru,en` +
    `&viewbox=${b.getWest()},${b.getNorth()},${b.getEast()},${b.getSouth()}&q=${encodeURIComponent(q)}`;
  try {
    const data = await (await fetch(url, { signal: searchAbort.signal })).json();
    showResults(data.map(r => {
      const [name, ...rest] = r.display_name.split(', ');
      return { name, sub: rest.slice(0, 3).join(', '), lat: +r.lat, lon: +r.lon };
    }));
  } catch (e) { if (e.name !== 'AbortError') status('Որոնումը ձախողվեց — ստուգեք ինտերնետը'); }
}
const esc = s => s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
$('clearBtn').onclick = () => { $('search').value = ''; $('results').innerHTML = ''; stopNav(true); };

// long-press / tap on map sets destination
const pointName = ll => (Offline.ready && Offline.nearestName(ll[0], ll[1])) || 'Նշված կետ';
map.on('contextmenu', e => setDest([e.latlng.lat, e.latlng.lng], pointName([e.latlng.lat, e.latlng.lng])));
map.on('click', e => { if (!S.route && !S.navigating) setDest([e.latlng.lat, e.latlng.lng], pointName([e.latlng.lat, e.latlng.lng])); });
map.on('dragstart', () => setFollow(false));

function setDest(ll, name) {
  S.dest = ll; S.destName = name;
  destMarker.setLatLng(ll).addTo(map);
  buildRoute(false);
}

// ---------- routing ----------
async function buildRoute(isReroute) {
  if (!S.dest) return;
  if (!S.pos) { status('Սպասում ենք ձեր դիրքին՝ երթուղի կառուցելու համար…'); setTimeout(() => buildRoute(isReroute), 2000); return; }
  S.lastReroute = Date.now(); S.offRouteCount = 0;
  const from = S.rawPos || S.pos;
  if (Offline.ready) {
    const t0 = performance.now();
    const r = Offline.route(from, S.dest, S.mode);
    if (!r) { status('Երթուղի չգտնվեց'); return; }
    setRoute(r);
    console.log(`offline route ${Math.round(r.distance)} m in ${Math.round(performance.now() - t0)} ms`);
    if (!isReroute) map.fitBounds(routeLine.getBounds(), { padding: [60, 60] });
    return;
  }
  let url = `${ROUTERS[S.mode]}${from[1]},${from[0]};${S.dest[1]},${S.dest[0]}?overview=full&geometries=geojson&steps=true`;
  // when driving, prefer a route continuing in the current direction of travel
  if (S.mode === 'car' && S.heading !== null && S.speed > 2) url += `&bearings=${Math.round(S.heading)},45;`;
  try {
    const data = await (await fetch(url)).json();
    if (data.code !== 'Ok' || !data.routes.length) throw new Error(data.message || data.code);
    setRoute(data.routes[0]);
    if (!isReroute) map.fitBounds(routeLine.getBounds(), { padding: [60, 60] });
  } catch (e) {
    status('Երթուղի չգտնվեց: ' + e.message);
  }
}
function setRoute(r) {
  const coords = r.geometry.coordinates.map(([lng, lat]) => [lat, lng]);
  const cum = [0];
  for (let i = 1; i < coords.length; i++) cum.push(cum[i - 1] + dist(coords[i - 1], coords[i]));
  const steps = [];
  for (const leg of r.legs) for (const s of leg.steps) {
    const loc = [s.maneuver.location[1], s.maneuver.location[0]];
    let bi = 0, bd = Infinity; // vertex closest to maneuver point
    for (let i = 0; i < coords.length; i++) { const d = dist(loc, coords[i]); if (d < bd) { bd = d; bi = i; } }
    steps.push({ ...s, at: cum[bi] });
  }
  S.route = { coords, cum, steps, total: cum[cum.length - 1], duration: r.duration, lastIdx: 0 };
  S.lastSpokenStep = -1; S.lastSpokenNear = -1;
  routeLine.setLatLngs(coords).addTo(map); passedLine.setLatLngs([]).addTo(map);
  passedLine.bringToFront();
  $('summary').innerHTML = `<b>${fmtDist(S.route.total)}</b> · ${fmtTime(r.duration)} — ${esc(S.destName)}`;
  $('panel').hidden = false; document.body.classList.add('hasPanel');
  if (S.navigating) updateGuidance(locateOnRoute(S.rawPos));
}

// ---------- guidance ----------
function updateGuidance(loc) {
  const r = S.route, along = loc.along, remaining = r.total - along;
  passedLine.setLatLngs([...r.coords.slice(0, loc.idx + 1), loc.point]);
  if (remaining < ARRIVE_DISTANCE || dist(S.rawPos, S.dest) < ARRIVE_DISTANCE) {
    showInstr('🏁', '', 'Դուք հասաք նպատակակետին'); speak('Դուք հասաք նպատակակետին');
    setTimeout(() => stopNav(true), 4000); S.navigating = false; return;
  }
  let i = r.steps.findIndex(s => s.at > along + 5);
  if (i < 0) i = r.steps.length - 1;
  const step = r.steps[i], d = step.at - along;
  const text = instruction(step);
  showInstr(icon(step), fmtDist(d), text);
  const eta = r.duration * (remaining / r.total);
  status(`Մնաց ${fmtDist(remaining)} · ${fmtTime(eta)} · ±${Math.round(S.accuracy)} մ`);
  $('summary').innerHTML = `<b>${fmtDist(remaining)}</b> · ${fmtTime(eta)} — ${esc(S.destName)}`;

  const near = S.mode === 'car' ? 60 : 25;
  if (i !== S.lastSpokenStep && d > near) { S.lastSpokenStep = i; speak(`${fmtDist(d)} հետո ${text}`); }
  if (i !== S.lastSpokenNear && d <= near) { S.lastSpokenNear = i; S.lastSpokenStep = i; speak(text); }
}
function showInstr(ic, d, t) {
  $('instruction').hidden = false; $('instrIcon').textContent = ic; $('instrDist').textContent = d; $('instrText').textContent = t;
}
const MOD = {
  'straight': 'ուղիղ', 'slight right': 'թեթևակի աջ', 'right': 'աջ', 'sharp right': 'կտրուկ աջ',
  'uturn': 'հետադարձ', 'sharp left': 'կտրուկ ձախ', 'left': 'ձախ', 'slight left': 'թեթևակի ձախ',
};
function instruction(s) {
  const m = s.maneuver, street = s.name ? ` դեպի ${s.name}` : '', mod = MOD[m.modifier] || '';
  switch (m.type) {
    case 'depart': return `Շարժվեք${street}`;
    case 'arrive': return 'Նպատակակետ';
    case 'roundabout': case 'rotary': case 'roundabout turn':
      return `Շրջանաձև հրապարակում ${m.exit ? m.exit + '-րդ ելքով' : ''} դուրս եկեք${street}`;
    case 'exit roundabout': case 'exit rotary': return `Դուրս եկեք շրջանից${street}`;
    case 'merge': return `Միացեք երթևեկությանը${street}`;
    case 'on ramp': return `Մտեք արագընթաց ճանապարհ${street}`;
    case 'off ramp': return `Դուրս եկեք ճանապարհից${street}`;
    case 'fork': return `Ճյուղավորման մոտ պահեք ${mod}${street}`;
    case 'end of road': return `Ճանապարհի վերջում թեքվեք ${mod}${street}`;
    case 'continue': case 'new name': return m.modifier && m.modifier !== 'straight' ? `Պահեք ${mod}${street}` : `Շարունակեք ուղիղ${street}`;
    default: return m.modifier === 'uturn' ? `Կատարեք հետադարձ${street}` :
      m.modifier === 'straight' ? `Շարունակեք ուղիղ${street}` : `Թեքվեք ${mod}${street}`;
  }
}
function icon(s) {
  const t = s.maneuver.type, m = s.maneuver.modifier || '';
  if (t === 'arrive') return '🏁';
  if (t.includes('roundabout') || t.includes('rotary')) return '⟳';
  if (m === 'uturn') return '↶';
  if (m.includes('left')) return m.includes('slight') ? '↖' : '↰';
  if (m.includes('right')) return m.includes('slight') ? '↗' : '↱';
  return '⬆';
}
const fmtDist = m => m >= 1000 ? (m / 1000).toFixed(m >= 10000 ? 0 : 1) + ' կմ' : Math.max(10, Math.round(m / 10) * 10) + ' մ';
const fmtTime = s => { const min = Math.round(s / 60); return min < 60 ? `${Math.max(1, min)} րոպե` : `${Math.floor(min / 60)} ժ ${min % 60} ր`; };

function speak(t) {
  if (!S.voice || !('speechSynthesis' in window)) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(t); u.lang = 'hy-AM';
  const v = speechSynthesis.getVoices().find(v => v.lang.startsWith('hy'));
  if (v) u.voice = v;
  speechSynthesis.speak(u);
}

// ---------- navigation control ----------
async function startNav() {
  if (!S.route) return;
  S.navigating = true; document.body.classList.add('navigating');
  setFollow(true); map.setZoom(17);
  startCompass();
  await lockScreen();
  if (S.rawPos) updateGuidance(locateOnRoute(S.rawPos));
}
function stopNav(clearAll) {
  S.navigating = false; document.body.classList.remove('navigating');
  $('instruction').hidden = true;
  S.wakeLock?.release?.(); S.wakeLock = null;
  if (clearAll) {
    S.route = null; S.dest = null;
    routeLine.remove(); passedLine.remove(); destMarker.remove();
    $('panel').hidden = true; document.body.classList.remove('hasPanel');
  }
}
function setFollow(on) { S.follow = on; $('locateBtn').classList.toggle('on', on); }

$('startBtn').onclick = startNav;
$('stopBtn').onclick = () => stopNav(true);
$('locateBtn').onclick = () => { setFollow(true); startCompass(); if (S.pos) map.setView(S.pos, Math.max(map.getZoom(), 16)); };
$('voiceBtn').onclick = () => {
  S.voice = !S.voice; $('voiceBtn').classList.toggle('on', S.voice); $('voiceBtn').textContent = S.voice ? '🔊' : '🔇';
  if (S.voice) speak('Ձայնային ուղեցույցը միացված է');
};
$('voiceBtn').textContent = '🔇';
for (const b of document.querySelectorAll('#modes button')) b.onclick = () => {
  document.querySelectorAll('#modes button').forEach(x => x.classList.toggle('active', x === b));
  S.mode = b.dataset.mode; if (S.dest) buildRoute(false);
};
document.addEventListener('visibilitychange', async () => {
  if (document.visibilityState === 'visible' && S.navigating && !S.wakeLock) {
    await lockScreen();
  }
});
async function lockScreen() {
  try {
    S.wakeLock = await navigator.wakeLock?.request('screen');
    S.wakeLock?.addEventListener('release', () => { S.wakeLock = null; });
  } catch {}
}

setFollow(true);
initOffline();
startGps();
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
