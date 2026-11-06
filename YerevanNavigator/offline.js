'use strict';
/* Offline engine of Yerevan Navigator: everything runs on the phone, no internet needed.
 *  - Offline.load()        downloads (first time) / reads from cache the data files
 *  - Offline.mapLayer()    Leaflet layer that draws the map from vector data
 *  - Offline.route()       A* routing on the road graph (car / foot / bike) with turn instructions
 *  - Offline.search()      street / address / place search
 *  - Offline.nearestName() street name near a point
 * Data files are produced by tools/build-offline-data.mjs from OpenStreetMap.
 */
const Offline = (() => {
  const D = {}; // decoded data
  const toRad = Math.PI / 180;
  const R = 6371000;
  function hav(lat1, lon1, lat2, lon2) {
    const dLat = (lat2 - lat1) * toRad, dLon = (lon2 - lon1) * toRad;
    const h = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * toRad) * Math.cos(lat2 * toRad) * Math.sin(dLon / 2) ** 2;
    return 2 * R * Math.asin(Math.min(1, Math.sqrt(h)));
  }
  function brg(lat1, lon1, lat2, lon2) {
    const y = Math.sin((lon2 - lon1) * toRad) * Math.cos(lat2 * toRad);
    const x = Math.cos(lat1 * toRad) * Math.sin(lat2 * toRad) - Math.sin(lat1 * toRad) * Math.cos(lat2 * toRad) * Math.cos((lon2 - lon1) * toRad);
    return (Math.atan2(y, x) / toRad + 360) % 360;
  }
  // Web-Mercator, normalized to 0..1
  const mx = lon => (lon + 180) / 360;
  const my = lat => { const s = Math.sin(lat * toRad); return 0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI); };

  // ------------------------------------------------------------------ loading
  async function fetchJson(url, onBytes) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    if (!res.body || !onBytes) return res.json();
    const reader = res.body.getReader(), chunks = []; let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); got += value.length; onBytes(value.length);
    }
    const buf = new Uint8Array(got); let o = 0;
    for (const c of chunks) { buf.set(c, o); o += c.length; }
    return JSON.parse(new TextDecoder().decode(buf));
  }

  async function load(base, onProgress) {
    const meta = await fetchJson(base + 'meta.json');
    const total = meta.sizes.roads + meta.sizes.areas + meta.sizes.places;
    let got = 0; const tick = n => { got += n; onProgress && onProgress(Math.min(1, got / total)); };
    const [roads, areas, places] = await Promise.all(
      ['roads.json', 'areas.json', 'places.json'].map(f => fetchJson(base + f, tick)));
    D.meta = meta;
    decodeRoads(roads); decodeAreas(areas); decodePlaces(places);
    buildSegmentIndex();
    return meta;
  }

  function decodeFlat(flat, bbox, scale) { // delta ints -> {lat:Float64Array, lon:Float64Array}
    const n = flat.length / 2, lat = new Float64Array(n), lon = new Float64Array(n);
    let x = 0, y = 0;
    for (let i = 0; i < n; i++) { x += flat[2 * i]; y += flat[2 * i + 1]; lon[i] = bbox.w + x / scale; lat[i] = bbox.s + y / scale; }
    return { lat, lon };
  }

  // ------------------------------------------------------------------ roads / graph
  function decodeRoads(r) {
    const { lat, lon } = decodeFlat(r.nodes, r.bbox, r.scale);
    D.lat = lat; D.lon = lon; D.names = r.names; D.hw = r.hw; D.bbox = r.bbox;
    const W = r.ways.length;
    D.wHw = new Uint8Array(W); D.wName = new Int32Array(W); D.wFlags = new Uint16Array(W); D.wSpeed = new Uint8Array(W);
    D.wRefs = new Array(W);
    let segs = 0;
    for (let i = 0; i < W; i++) {
      const w = r.ways[i];
      D.wHw[i] = w[0]; D.wName[i] = w[1]; D.wFlags[i] = w[2]; D.wSpeed[i] = w[3];
      D.wRefs[i] = Int32Array.from(w.slice(4)); segs += w.length - 5;
    }
    // CSR adjacency, two directed edges per segment
    const N = lat.length, deg = new Int32Array(N + 1);
    for (const refs of D.wRefs) for (let k = 0; k + 1 < refs.length; k++) { deg[refs[k]]++; deg[refs[k + 1]]++; }
    const start = new Int32Array(N + 1);
    for (let i = 0; i < N; i++) start[i + 1] = start[i] + deg[i];
    const fill = start.slice(0, N);
    const E = start[N];
    D.eTo = new Int32Array(E); D.eWay = new Int32Array(E); D.eLen = new Float32Array(E); D.eFwd = new Uint8Array(E);
    for (let w = 0; w < W; w++) {
      const refs = D.wRefs[w];
      for (let k = 0; k + 1 < refs.length; k++) {
        const a = refs[k], b = refs[k + 1], len = hav(lat[a], lon[a], lat[b], lon[b]);
        let e = fill[a]++; D.eTo[e] = b; D.eWay[e] = w; D.eLen[e] = len; D.eFwd[e] = 1;
        e = fill[b]++; D.eTo[e] = a; D.eWay[e] = w; D.eLen[e] = len; D.eFwd[e] = 0;
      }
    }
    D.eStart = start; D.N = N;
    D.maxCarSpeed = D.wSpeed.reduce((m, v) => Math.max(m, v), 30);
    D.hwId = name => D.hw.indexOf(name);
  }

  const MODE_BIT = { car: 1, foot: 2, bike: 4 };
  function canUse(w, fwd, mode) {
    const f = D.wFlags[w];
    if (!(f & MODE_BIT[mode])) return false;
    if (mode === 'foot') return true;
    if (mode === 'bike' && !(f & 32)) return true;
    if (f & 8) return fwd === 1;
    if (f & 16) return fwd === 0;
    return true;
  }
  function speedOf(w, mode) { // m/s
    const hw = D.hw[D.wHw[w]], f = D.wFlags[w];
    if (mode === 'car') return D.wSpeed[w] / 3.6;
    if (mode === 'foot') return (hw === 'steps' ? 3 : 5) / 3.6;
    if (f & 512) return 5 / 3.6;                       // bike must be pushed
    if (hw === 'cycleway') return 18 / 3.6;
    if (hw === 'path' || hw === 'track') return 11 / 3.6;
    if (/^(primary|trunk|secondary)/.test(hw)) return 15 / 3.6; // slower = prefer calmer streets
    return 16 / 3.6;
  }
  const maxSpeed = mode => mode === 'car' ? D.maxCarSpeed / 3.6 : mode === 'foot' ? 5 / 3.6 : 18 / 3.6;

  // spatial grid of segments for snapping
  const CELL = 0.002; // deg ≈ 170–220 m
  function cellKey(lat, lon) { return Math.floor((lat - D.bbox.s) / CELL) * 4096 + Math.floor((lon - D.bbox.w) / CELL); }
  function buildSegmentIndex() {
    const grid = new Map();
    D.wRefs.forEach((refs, w) => {
      for (let k = 0; k + 1 < refs.length; k++) {
        const a = refs[k], b = refs[k + 1];
        const r0 = Math.floor((Math.min(D.lat[a], D.lat[b]) - D.bbox.s) / CELL), r1 = Math.floor((Math.max(D.lat[a], D.lat[b]) - D.bbox.s) / CELL);
        const c0 = Math.floor((Math.min(D.lon[a], D.lon[b]) - D.bbox.w) / CELL), c1 = Math.floor((Math.max(D.lon[a], D.lon[b]) - D.bbox.w) / CELL);
        for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) {
          const key = r * 4096 + c; let arr = grid.get(key);
          if (!arr) grid.set(key, arr = []);
          arr.push(w, k);
        }
      }
    });
    D.segGrid = grid;
  }
  function project(lat, lon, a, b) {
    const k = Math.cos(lat * toRad);
    const ax = D.lon[a] * k, ay = D.lat[a], bx = D.lon[b] * k, by = D.lat[b], px = lon * k;
    const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
    let t = l2 ? ((px - ax) * dx + (lat - ay) * dy) / l2 : 0; t = Math.max(0, Math.min(1, t));
    const plat = D.lat[a] + (D.lat[b] - D.lat[a]) * t, plon = D.lon[a] + (D.lon[b] - D.lon[a]) * t;
    return { t, lat: plat, lon: plon, d: hav(lat, lon, plat, plon) };
  }
  // nearest usable segment; filter(w) decides which ways count
  function nearestSeg(lat, lon, filter) {
    const r0 = Math.floor((lat - D.bbox.s) / CELL), c0 = Math.floor((lon - D.bbox.w) / CELL);
    let best = null;
    for (let ring = 0; ring < 12; ring++) {
      for (let r = r0 - ring; r <= r0 + ring; r++) for (let c = c0 - ring; c <= c0 + ring; c++) {
        if (Math.max(Math.abs(r - r0), Math.abs(c - c0)) !== ring) continue;
        const arr = D.segGrid.get(r * 4096 + c); if (!arr) continue;
        for (let i = 0; i < arr.length; i += 2) {
          const w = arr[i], k = arr[i + 1];
          if (!filter(w)) continue;
          const refs = D.wRefs[w], p = project(lat, lon, refs[k], refs[k + 1]);
          if (!best || p.d < best.d) best = { ...p, w, k, a: refs[k], b: refs[k + 1] };
        }
      }
      if (best && best.d < ring * CELL * 80000) break; // ring is already farther than best
    }
    return best;
  }

  // binary heap keyed by f
  class Heap {
    constructor() { this.k = []; this.v = []; }
    get size() { return this.k.length; }
    push(key, val) {
      const k = this.k, v = this.v; let i = k.length; k.push(key); v.push(val);
      while (i > 0) { const p = (i - 1) >> 1; if (k[p] <= key) break; k[i] = k[p]; v[i] = v[p]; i = p; }
      k[i] = key; v[i] = val;
    }
    pop() {
      const k = this.k, v = this.v, top = v[0], lk = k.pop(), lv = v.pop(), n = k.length;
      if (n) {
        let i = 0;
        for (;;) {
          let c = 2 * i + 1; if (c >= n) break;
          if (c + 1 < n && k[c + 1] < k[c]) c++;
          if (k[c] >= lk) break;
          k[i] = k[c]; v[i] = v[c]; i = c;
        }
        k[i] = lk; v[i] = lv;
      }
      return top;
    }
    peekKey() { return this.k[0]; }
  }

  let G, PE, ST, stamp = 0;
  function route(from, to, mode) {
    if (!D.N) return null;
    const filt = w => !!(D.wFlags[w] & MODE_BIT[mode]);
    const S = nearestSeg(from[0], from[1], filt), T = nearestSeg(to[0], to[1], filt);
    if (!S || !T) return null;
    if (!G || G.length < D.N) { G = new Float64Array(D.N); PE = new Int32Array(D.N); ST = new Int32Array(D.N); }
    stamp++;
    const vmax = maxSpeed(mode), tlat = T.lat, tlon = T.lon;
    const h = n => hav(D.lat[n], D.lon[n], tlat, tlon) / vmax;
    const segLen = (seg) => hav(D.lat[seg.a], D.lon[seg.a], D.lat[seg.b], D.lon[seg.b]);
    const sLen = segLen(S), tLen = segLen(T), sv = speedOf(S.w, mode), tv = speedOf(T.w, mode);
    const heap = new Heap();
    const relax = (n, g, pe) => {
      if (ST[n] === stamp && G[n] <= g) return;
      ST[n] = stamp; G[n] = g; PE[n] = pe; heap.push(g + h(n), n);
    };
    // sources: walk from the projected start point to either end of its segment
    if (canUse(S.w, 0, mode)) relax(S.a, S.t * sLen / sv, -1);
    if (canUse(S.w, 1, mode)) relax(S.b, (1 - S.t) * sLen / sv, -2);
    // direct (start and end on the same segment)
    let best = Infinity, bestEnd = -1;
    if (S.w === T.w && S.k === T.k) {
      const fwd = T.t >= S.t ? 1 : 0;
      if (canUse(S.w, fwd, mode)) { best = Math.abs(T.t - S.t) * sLen / sv; bestEnd = -3; }
    }
    const tA = canUse(T.w, 1, mode) ? T.t * tLen / tv : Infinity;       // reach T.a then go forward
    const tB = canUse(T.w, 0, mode) ? (1 - T.t) * tLen / tv : Infinity; // reach T.b then go backward
    let iter = 0;
    while (heap.size) {
      if (heap.peekKey() >= best) break;
      const u = heap.pop(), g = G[u];
      if (++iter > 3e6) break;
      if (u === T.a && g + tA < best) { best = g + tA; bestEnd = u; }
      if (u === T.b && g + tB < best) { best = g + tB; bestEnd = u; }
      for (let e = D.eStart[u], end = D.eStart[u + 1]; e < end; e++) {
        const w = D.eWay[e];
        if (!canUse(w, D.eFwd[e], mode)) continue;
        const v = D.eTo[e], ng = g + D.eLen[e] / speedOf(w, mode);
        if (ST[v] === stamp && G[v] <= ng) continue;
        ST[v] = stamp; G[v] = ng; PE[v] = e; heap.push(ng + h(v), v);
      }
    }
    if (best === Infinity) return null;

    // ---- reconstruct: list of points + the way used to arrive at each point
    const pts = [[S.lat, S.lon]], ways = [S.w], nodes = [-1];
    if (bestEnd !== -3) {
      const chain = []; let n = bestEnd;
      while (true) {
        chain.push(n);
        const pe = PE[n];
        if (pe < 0) break;
        // find source of edge pe: it is the node whose adjacency contains pe
        n = edgeSource(pe);
      }
      chain.reverse();
      for (let i = 0; i < chain.length; i++) {
        const n2 = chain[i], pe = PE[n2];
        pts.push([D.lat[n2], D.lon[n2]]); nodes.push(n2);
        ways.push(pe >= 0 ? D.eWay[pe] : S.w);
      }
      ways.push(T.w);
    } else ways.push(S.w);
    pts.push([T.lat, T.lon]); nodes.push(-1);
    return buildResult(pts, ways, nodes, mode, best);
  }
  function edgeSource(e) { // binary search in eStart
    let lo = 0, hi = D.N - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (D.eStart[mid] <= e) lo = mid; else hi = mid - 1; }
    return lo;
  }

  // ---- turn-by-turn instructions
  // ways[i] = way used to travel from pts[i-1] to pts[i]
  function buildResult(pts, ways, nodes, mode, duration) {
    const cum = [0];
    for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + hav(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1]));
    const nameOf = w => D.names[D.wName[w]] || '';
    const isRb = w => !!(D.wFlags[w] & 64);
    const bearingAround = (i, dir) => { // bearing over ~20 m before (dir=-1) / after (dir=1) point i
      let j = i;
      while (j + dir >= 0 && j + dir < pts.length) { j += dir; if (Math.abs(cum[j] - cum[i]) > 20) break; }
      if (j === i) return null;
      return dir < 0 ? brg(pts[j][0], pts[j][1], pts[i][0], pts[i][1]) : brg(pts[i][0], pts[i][1], pts[j][0], pts[j][1]);
    };
    const options = (n, mode) => { // number of usable directions at a node
      if (n < 0) return 0; let c = 0;
      for (let e = D.eStart[n]; e < D.eStart[n + 1]; e++) if (canUse(D.eWay[e], D.eFwd[e], mode)) c++;
      return c;
    };
    const modifier = d => {
      const a = Math.abs(d);
      if (a < 20) return 'straight';
      const side = d > 0 ? 'right' : 'left';
      if (a < 45) return 'slight ' + side;
      if (a < 125) return side;
      if (a < 165) return 'sharp ' + side;
      return 'uturn';
    };
    const loc = i => [pts[i][1], pts[i][0]];
    const steps = [{ name: nameOf(ways[1]), maneuver: { type: 'depart', location: loc(0) }, idx: 0 }];
    let i = 1;
    while (i < pts.length - 1) {
      const win = ways[i], wout = ways[i + 1];
      if (win === wout) { i++; continue; }
      // roundabout: count exits until we leave it
      if (isRb(wout) && !isRb(win)) {
        let j = i + 1, exits = 0;
        while (j < pts.length - 1 && isRb(ways[j + 1])) { if (options(nodes[j], mode) > 2) exits++; j++; }
        exits++;
        const outName = nameOf(ways[Math.min(j + 1, ways.length - 1)]);
        steps.push({ name: outName, maneuver: { type: 'roundabout', exit: exits, location: loc(i) }, idx: i });
        i = j + 1; continue;
      }
      const bin = bearingAround(i, -1), bout = bearingAround(i, 1);
      const d = bin === null || bout === null ? 0 : ((bout - bin + 540) % 360) - 180;
      const mod = modifier(d), nIn = nameOf(win), nOut = nameOf(wout);
      const choice = options(nodes[i], mode) > 2;
      if (mod !== 'straight' && (choice || nIn !== nOut)) {
        steps.push({ name: nOut, maneuver: { type: 'turn', modifier: mod, location: loc(i) }, idx: i });
      } else if (nOut && nIn !== nOut && choice) {
        steps.push({ name: nOut, maneuver: { type: 'new name', modifier: 'straight', location: loc(i) }, idx: i });
      }
      i++;
    }
    // merge steps closer than 15 m (OSM micro-segments at complex junctions)
    const merged = [];
    for (const s of steps) {
      const prev = merged[merged.length - 1];
      if (prev && prev.maneuver.type !== 'depart' && cum[s.idx] - cum[prev.idx] < 15 && s.maneuver.type !== 'roundabout') { prev.name = s.name; continue; }
      merged.push(s);
    }
    merged.push({ name: '', maneuver: { type: 'arrive', location: loc(pts.length - 1) }, idx: pts.length - 1 });
    return {
      distance: cum[cum.length - 1], duration,
      geometry: { coordinates: pts.map(p => [p[1], p[0]]) },
      legs: [{ steps: merged }],
    };
  }

  function nearestName(lat, lon) {
    const s = nearestSeg(lat, lon, w => !!D.names[D.wName[w]]);
    return s && s.d < 150 ? D.names[D.wName[s.w]] : '';
  }

  // ------------------------------------------------------------------ search
  const HY_LAT = { 'ա': 'a', 'բ': 'b', 'գ': 'g', 'դ': 'd', 'ե': 'e', 'զ': 'z', 'է': 'e', 'ը': 'y', 'թ': 't', 'ժ': 'zh', 'ի': 'i', 'լ': 'l', 'խ': 'kh',
    'ծ': 'ts', 'կ': 'k', 'հ': 'h', 'ձ': 'dz', 'ղ': 'gh', 'ճ': 'ch', 'մ': 'm', 'յ': 'y', 'ն': 'n', 'շ': 'sh', 'ո': 'o', 'չ': 'ch', 'պ': 'p',
    'ջ': 'j', 'ռ': 'r', 'ս': 's', 'վ': 'v', 'տ': 't', 'ր': 'r', 'ց': 'ts', 'ու': 'u', 'փ': 'p', 'ք': 'k', 'և': 'ev', 'օ': 'o', 'ֆ': 'f' };
  const RU_LAT = { 'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e', 'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l',
    'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh',
    'щ': 'sh', 'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya' };
  const norm = s => s.toLowerCase().replace(/և/g, 'եվ').replace(/ё/g, 'е').replace(/[«»"'`’.,()\-–—/№#]/g, ' ').replace(/\s+/g, ' ').trim();
  const translit = s => s.replace(/ու/g, 'u').replace(/./g, c => HY_LAT[c] ?? RU_LAT[c] ?? c).replace(/kh/g, 'x');
  const simplifyLat = s => s.replace(/kh/g, 'x').replace(/yan\b/g, 'yan').replace(/[yj]/g, 'i').replace(/ts/g, 'c').replace(/w/g, 'v');

  const KIND_HY = {
    street: 'Փողոց', address: 'Հասցե', 'amenity=cafe': 'Սրճարան', 'amenity=restaurant': 'Ռեստորան', 'amenity=fast_food': 'Արագ սնունդ',
    'amenity=bar': 'Բար', 'amenity=pub': 'Փաբ', 'amenity=pharmacy': 'Դեղատուն', 'amenity=hospital': 'Հիվանդանոց', 'amenity=clinic': 'Կլինիկա',
    'amenity=doctors': 'Բժիշկ', 'amenity=dentist': 'Ատամնաբույժ', 'amenity=bank': 'Բանկ', 'amenity=atm': 'Բանկոմատ', 'amenity=fuel': 'Լցակայան',
    'amenity=school': 'Դպրոց', 'amenity=kindergarten': 'Մանկապարտեզ', 'amenity=university': 'Համալսարան', 'amenity=college': 'Քոլեջ',
    'amenity=parking': 'Կայանատեղի', 'amenity=place_of_worship': 'Եկեղեցի', 'amenity=theatre': 'Թատրոն', 'amenity=cinema': 'Կինոթատրոն',
    'amenity=post_office': 'Փոստ', 'amenity=police': 'Ոստիկանություն', 'amenity=marketplace': 'Շուկա', 'amenity=library': 'Գրադարան',
    'amenity=bus_station': 'Ավտոկայան', 'amenity=car_wash': 'Ավտոլվացում', 'shop=supermarket': 'Սուպերմարկետ', 'shop=mall': 'Առևտրի կենտրոն',
    'shop=convenience': 'Մթերային խանութ', 'shop=bakery': 'Հացատուն', 'shop=clothes': 'Հագուստ', 'tourism=hotel': 'Հյուրանոց',
    'tourism=hostel': 'Հոսթել', 'tourism=museum': 'Թանգարան', 'tourism=attraction': 'Տեսարժան վայր', 'tourism=artwork': 'Արձան',
    'tourism=viewpoint': 'Դիտակետ', 'leisure=park': 'Այգի', 'leisure=stadium': 'Մարզադաշտ', 'leisure=fitness_centre': 'Ֆիթնես',
    'historic=memorial': 'Հուշարձան', 'historic=monument': 'Հուշարձան', 'railway=station': 'Կայարան', 'railway=subway_entrance': 'Մետրոյի մուտք',
    'public_transport=station': 'Կայան', 'public_transport=stop_position': 'Կանգառ', 'public_transport=platform': 'Կանգառ',
  };
  const kindLabel = k => KIND_HY[k] || (k.startsWith('shop=') ? 'Խանութ' : k.startsWith('place=') ? 'Թաղամաս / վայր' :
    k.startsWith('office=') ? 'Գրասենյակ' : k.startsWith('tourism=') ? 'Զբոսաշրջություն' : k.startsWith('railway=') ? 'Կայարան' : '');
  const KIND_BOOST = k => k === 'street' ? 3 : k.startsWith('place=') || k === 'railway=station' || k.startsWith('tourism=') ? 2.5 : k === 'address' ? 1 : 2;

  function decodePlaces(p) {
    D.places = p.places.map(([name, alt, x, y, kind, addr]) => {
      const all = norm([name, alt].join(' ').replace(/\|/g, ' '));
      return { name, kind: p.kinds[kind], addr: addr || '', lat: p.bbox.s + y / p.scale, lon: p.bbox.w + x / p.scale,
        text: all + ' ' + simplifyLat(translit(all)), nname: norm(name) };
    });
  }
  function search(q, near) {
    if (!D.places) return [];
    const nq = norm(q); if (!nq) return [];
    const tokens = nq.split(' ');
    const latTokens = tokens.map(t => simplifyLat(translit(t)));
    const out = [];
    for (const p of D.places) {
      let ok = true;
      for (let i = 0; i < tokens.length; i++) if (!p.text.includes(tokens[i]) && !p.text.includes(latTokens[i])) { ok = false; break; }
      if (!ok) continue;
      let score = KIND_BOOST(p.kind);
      if (p.nname === nq) score += 10; else if (p.nname.startsWith(nq)) score += 6;
      else if ((' ' + p.text).includes(' ' + tokens[0]) || (' ' + p.text).includes(' ' + latTokens[0])) score += 3;
      if (p.kind === 'address' && /\d/.test(nq)) score += 4;
      score -= p.name.length / 60;
      if (near) score -= hav(near[0], near[1], p.lat, p.lon) / 8000;
      out.push([score, p]);
    }
    out.sort((a, b) => b[0] - a[0]);
    return out.slice(0, 10).map(([, p]) => ({ name: p.name, lat: p.lat, lon: p.lon,
      sub: [kindLabel(p.kind), p.addr].filter(Boolean).join(' · ') }));
  }

  // ------------------------------------------------------------------ map drawing
  const CELLZ = 13, CN = 1 << CELLZ; // spatial index cell = one z13 tile
  function indexFeature(index, f) {
    const c0 = Math.floor(f.x0 * CN), c1 = Math.floor(f.x1 * CN), r0 = Math.floor(f.y0 * CN), r1 = Math.floor(f.y1 * CN);
    for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) {
      const k = r * CN + c; let a = index.get(k); if (!a) index.set(k, a = []); a.push(f);
    }
  }
  function makeFeature(lat, lon, extra) {
    const n = lat.length, xy = new Float64Array(2 * n);
    let x0 = 1, x1 = 0, y0 = 1, y1 = 0;
    for (let i = 0; i < n; i++) {
      const x = mx(lon[i]), y = my(lat[i]); xy[2 * i] = x; xy[2 * i + 1] = y;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    return Object.assign({ xy, x0, x1, y0, y1, mark: 0 }, extra);
  }
  function decodeAreas(a) {
    D.areaIndex = new Map();
    for (const [layer, items] of Object.entries(a.layers)) for (const flat of items) {
      const { lat, lon } = decodeFlat(flat, a.bbox, a.scale);
      indexFeature(D.areaIndex, makeFeature(lat, lon, { layer }));
    }
    // roads for drawing
    D.roadIndex = new Map();
    for (let w = 0; w < D.wRefs.length; w++) {
      const refs = D.wRefs[w], lat = new Float64Array(refs.length), lon = new Float64Array(refs.length);
      for (let i = 0; i < refs.length; i++) { lat[i] = D.lat[refs[i]]; lon[i] = D.lon[refs[i]]; }
      const hw = D.hw[D.wHw[w]];
      indexFeature(D.roadIndex, makeFeature(lat, lon, { cls: roadClass(hw), name: D.names[D.wName[w]], hw,
        tunnel: !!(D.wFlags[w] & 256), bridge: !!(D.wFlags[w] & 128) }));
    }
  }
  function roadClass(hw) {
    if (hw.startsWith('motorway')) return 0; if (hw.startsWith('trunk')) return 1; if (hw.startsWith('primary')) return 2;
    if (hw.startsWith('secondary')) return 3; if (hw.startsWith('tertiary')) return 4;
    if (hw === 'residential' || hw === 'unclassified' || hw === 'living_street' || hw === 'road') return 5;
    if (hw === 'service') return 6; if (hw === 'pedestrian') return 7; return 8; // footway, path, steps, cycleway, track
  }
  //                motorway   trunk      primary    secondary  tertiary   resid.     service    pedestr.   path
  const ROAD_FILL = ['#e990a0', '#f9b29c', '#fcd6a4', '#f7fabf', '#ffffff', '#ffffff', '#ffffff', '#ededf3', '#f07f6a'];
  const ROAD_CASE = ['#c24e6b', '#c84e2f', '#a06b00', '#707d05', '#b0a99f', '#c2b9ae', '#c2b9ae', '#b5b5c8', null];
  const ROAD_W16 =  [12, 11, 10, 9, 8, 6.5, 4, 5, 1.6]; // px at zoom 16
  const ROAD_MINZ = [0, 0, 0, 11, 12, 13, 14, 14, 15];
  const AREA_STYLE = {
    forest: ['#add19e'], grass: ['#cdebb0'], park: ['#c8facc'], cemetery: ['#aacbaf'], pitch: ['#aae0cb', '#88c0a7'],
    water: ['#aad3df'], building: ['#d9d0c9', '#c4b6ab'],
  };
  const AREA_ORDER = ['forest', 'grass', 'cemetery', 'park', 'pitch', 'water'];

  function query(index, x0, y0, x1, y1, mark) {
    const out = [], c0 = Math.floor(x0 * CN), c1 = Math.floor(x1 * CN), r0 = Math.floor(y0 * CN), r1 = Math.floor(y1 * CN);
    for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) {
      const a = index.get(r * CN + c); if (!a) continue;
      for (const f of a) if (f.mark !== mark && f.x1 >= x0 && f.x0 <= x1 && f.y1 >= y0 && f.y0 <= y1) { f.mark = mark; out.push(f); }
    }
    return out;
  }

  let drawMark = 0;
  function drawTile(ctx, tx, ty, z, size, ratio) {
    const scale = (1 << z) * 256, s = ratio;
    const pad = 24 / scale;
    const x0 = tx / (1 << z) - pad, y0 = ty / (1 << z) - pad, x1 = (tx + 1) / (1 << z) + pad, y1 = (ty + 1) / (1 << z) + pad;
    const ox = tx * 256, oy = ty * 256;
    const path = (f, close) => {
      const xy = f.xy; ctx.moveTo((xy[0] * scale - ox) * s, (xy[1] * scale - oy) * s);
      for (let i = 2; i < xy.length; i += 2) ctx.lineTo((xy[i] * scale - ox) * s, (xy[i + 1] * scale - oy) * s);
      if (close) ctx.closePath();
    };
    ctx.fillStyle = '#f2efe9'; ctx.fillRect(0, 0, size, size);
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';

    const areas = query(D.areaIndex, x0, y0, x1, y1, ++drawMark);
    for (const layer of AREA_ORDER) {
      const st = AREA_STYLE[layer]; ctx.beginPath(); let any = false;
      for (const f of areas) if (f.layer === layer) { path(f, true); any = true; }
      if (any) { ctx.fillStyle = st[0]; ctx.fill(); if (st[1]) { ctx.strokeStyle = st[1]; ctx.lineWidth = s; ctx.stroke(); } }
    }
    for (const [layer, color, w] of [['river', '#aad3df', 6], ['stream', '#aad3df', 2]]) {
      ctx.beginPath(); let any = false;
      for (const f of areas) if (f.layer === layer) { path(f); any = true; }
      if (any) { ctx.strokeStyle = color; ctx.lineWidth = Math.max(1, w * Math.pow(2, (z - 16) * 0.6)) * s; ctx.stroke(); }
    }
    if (z >= 15) {
      ctx.beginPath(); let any = false;
      for (const f of areas) if (f.layer === 'building') { path(f, true); any = true; }
      if (any) { ctx.fillStyle = '#d9d0c9'; ctx.fill(); if (z >= 16) { ctx.strokeStyle = '#c4b6ab'; ctx.lineWidth = 0.7 * s; ctx.stroke(); } }
    }
    for (const [layer, color, dash] of [['rail', '#707070', [6, 6]], ['tram', '#9a9a9a', null]]) {
      ctx.beginPath(); let any = false;
      for (const f of areas) if (f.layer === layer) { path(f); any = true; }
      if (any && z >= 12) { ctx.strokeStyle = color; ctx.lineWidth = 1.5 * s; ctx.setLineDash(dash ? dash.map(v => v * s) : []); ctx.stroke(); ctx.setLineDash([]); }
    }

    const roads = query(D.roadIndex, x0, y0, x1, y1, ++drawMark).filter(f => z >= ROAD_MINZ[f.cls]);
    const zf = Math.max(0.25, Math.pow(2, (z - 16) * 0.75));
    const width = c => Math.max(c >= 8 ? 1 : 1.2, ROAD_W16[c] * zf);
    // tunnels faded, casings first, then fills from minor to major
    for (let c = 8; c >= 0; c--) {
      if (!ROAD_CASE[c]) continue;
      ctx.beginPath(); let any = false;
      for (const f of roads) if (f.cls === c) { path(f); any = true; }
      if (any && z >= 13) { ctx.strokeStyle = ROAD_CASE[c]; ctx.lineWidth = (width(c) + 2) * s; ctx.stroke(); }
    }
    for (let c = 8; c >= 0; c--) {
      ctx.beginPath(); let any = false;
      for (const f of roads) if (f.cls === c) { path(f); any = true; }
      if (!any) continue;
      ctx.strokeStyle = ROAD_FILL[c]; ctx.lineWidth = width(c) * s;
      if (c === 8) ctx.setLineDash([3 * s, 3 * s]);
      ctx.stroke(); ctx.setLineDash([]);
    }

    // street names
    if (z >= 15) {
      const boxes = [];
      ctx.font = `${(z >= 17 ? 12 : 11) * s}px -apple-system, Roboto, Arial, sans-serif`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      const sorted = roads.filter(f => f.name && (z >= 16 || f.cls <= 4)).sort((a, b) => a.cls - b.cls);
      for (const f of sorted) {
        const xy = f.xy; let best = null, bl = 0;
        for (let i = 0; i + 3 < xy.length; i += 2) {
          const ax = (xy[i] * scale - ox) * s, ay = (xy[i + 1] * scale - oy) * s, bx = (xy[i + 2] * scale - ox) * s, by = (xy[i + 3] * scale - oy) * s;
          const c = clip(ax, ay, bx, by, 0, size); if (!c) continue;
          const l = Math.hypot(c[2] - c[0], c[3] - c[1]);
          if (l > bl) { bl = l; best = c; }
        }
        if (!best) continue;
        const tw = ctx.measureText(f.name).width;
        if (bl < tw + 8 * s) continue;
        const cx = (best[0] + best[2]) / 2, cy = (best[1] + best[3]) / 2, half = tw / 2 + 4 * s;
        const m = 2 * s;
        if (cx - half < m || cx + half > size - m || cy - half < m || cy + half > size - m) continue; // keep label inside tile
        if (boxes.some(b => Math.abs(b[0] - cx) < (b[2] + half) && Math.abs(b[1] - cy) < 14 * s)) continue;
        boxes.push([cx, cy, half]);
        let ang = Math.atan2(best[3] - best[1], best[2] - best[0]);
        if (ang > Math.PI / 2) ang -= Math.PI; if (ang < -Math.PI / 2) ang += Math.PI;
        ctx.save(); ctx.translate(cx, cy); ctx.rotate(ang);
        ctx.lineWidth = 3 * s; ctx.strokeStyle = '#ffffffdd'; ctx.strokeText(f.name, 0, 0);
        ctx.fillStyle = '#333'; ctx.fillText(f.name, 0, 0);
        ctx.restore();
      }
    }
  }

  // Liang–Barsky: part of segment inside [lo,hi]² or null
  function clip(x0, y0, x1, y1, lo, hi) {
    let t0 = 0, t1 = 1; const dx = x1 - x0, dy = y1 - y0;
    for (const [p, q] of [[-dx, x0 - lo], [dx, hi - x0], [-dy, y0 - lo], [dy, hi - y0]]) {
      if (p === 0) { if (q < 0) return null; continue; }
      const r = q / p;
      if (p < 0) { if (r > t1) return null; if (r > t0) t0 = r; } else { if (r < t0) return null; if (r < t1) t1 = r; }
    }
    return [x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy];
  }

  function mapLayer() {
    const Layer = L.GridLayer.extend({
      createTile(coords) {
        const ratio = Math.min(2, window.devicePixelRatio || 1), size = 256 * ratio;
        const c = document.createElement('canvas'); c.width = c.height = size;
        drawTile(c.getContext('2d'), coords.x, coords.y, coords.z, size, ratio);
        return c;
      },
    });
    return new Layer({ maxZoom: 19, keepBuffer: 3, attribution: '© OpenStreetMap contributors' });
  }

  return { load, route, search, mapLayer, nearestName, get ready() { return !!D.N; }, get meta() { return D.meta; } };
})();
