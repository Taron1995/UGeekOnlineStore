#!/usr/bin/env node
/*
 * Builds the offline data of Yerevan Navigator from OpenStreetMap.
 *
 *   node tools/build-offline-data.mjs                 # downloads from Overpass API (needs internet)
 *   node tools/build-offline-data.mjs --input x.json  # uses an already downloaded Overpass JSON
 *
 * Output (YerevanNavigator/data/):
 *   roads.json   – road network: drawn on the map and used for offline routing
 *   areas.json   – buildings, water, parks, railways (map drawing)
 *   places.json  – streets, addresses and points of interest (offline search)
 *   meta.json    – build date and sizes
 *
 * Coordinates are stored as integers in micro-degrees relative to the south-west corner
 * of the bbox and delta-encoded to keep files small.
 * Map data © OpenStreetMap contributors, ODbL.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.join(HERE, '..', 'data');
const BBOX = { s: 40.05, w: 44.35, n: 40.26, e: 44.64 }; // Yerevan with a margin
const SCALE = 1e6;
const ENDPOINTS = [
  'https://overpass-api.de/api/interpreter',
  'https://overpass.kumi.systems/api/interpreter',
  'https://maps.mail.ru/osm/tools/overpass/api/interpreter',
];

const QUERY = `[out:json][timeout:900][maxsize:2000000000][bbox:${BBOX.s},${BBOX.w},${BBOX.n},${BBOX.e}];
(
  way[highway];
  way[building];
  way[natural~"^(water|wood|scrub|grassland)$"];
  way[waterway~"^(river|canal|stream|riverbank)$"];
  way[water];
  way[leisure~"^(park|garden|pitch|stadium|playground)$"];
  way[landuse~"^(grass|forest|cemetery|meadow|recreation_ground|village_green|orchard|vineyard|reservoir|basin)$"];
  way[railway~"^(rail|tram|light_rail)$"];
  rel[type=multipolygon][natural];
  rel[type=multipolygon][leisure];
  rel[type=multipolygon][landuse];
  rel[type=multipolygon][building];
  rel[type=multipolygon][water];
  node[name][~"^(amenity|shop|tourism|leisure|historic|office|railway|public_transport|place|healthcare|craft|sport|man_made)$"~"."];
  way[name][~"^(amenity|shop|tourism|leisure|historic|office|healthcare)$"~"."];
  node["addr:housenumber"];
  way["addr:housenumber"];
);
out geom qt;`;

// ---------------------------------------------------------------- download
async function download() {
  let lastErr;
  for (let attempt = 0; attempt < 3; attempt++) {
    for (const url of ENDPOINTS) {
      try {
        console.log(`Downloading from ${url} …`);
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded', 'User-Agent': 'YerevanNavigator-offline-build/1.0' },
          body: 'data=' + encodeURIComponent(QUERY),
          signal: AbortSignal.timeout(20 * 60 * 1000),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
        const json = await res.json();
        if (!json.elements?.length) throw new Error('empty answer: ' + JSON.stringify(json.remark || ''));
        if (json.remark && /error|timed out/i.test(json.remark)) throw new Error(json.remark);
        return json;
      } catch (e) {
        lastErr = e; console.warn('  failed:', e.message);
      }
    }
    await new Promise(r => setTimeout(r, 30000 * (attempt + 1)));
  }
  throw lastErr;
}

// ---------------------------------------------------------------- helpers
const qx = lon => Math.round((lon - BBOX.w) * SCALE);
const qy = lat => Math.round((lat - BBOX.s) * SCALE);
function deltaEncode(points) { // points: [[x,y]...] ints -> flat delta array
  const out = []; let px = 0, py = 0;
  for (const [x, y] of points) { out.push(x - px, y - py); px = x; py = y; }
  return out;
}
function simplify(pts, tol) { // Douglas–Peucker on int points
  if (pts.length <= 3) return pts;
  const keep = new Uint8Array(pts.length); keep[0] = keep[pts.length - 1] = 1;
  const stack = [[0, pts.length - 1]], t2 = tol * tol;
  while (stack.length) {
    const [a, b] = stack.pop();
    const [ax, ay] = pts[a], [bx, by] = pts[b];
    const dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy;
    let md = -1, mi = -1;
    for (let i = a + 1; i < b; i++) {
      const [px, py] = pts[i];
      let t = len2 ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0; t = Math.max(0, Math.min(1, t));
      const ex = ax + t * dx - px, ey = ay + t * dy - py, d = ex * ex + ey * ey;
      if (d > md) { md = d; mi = i; }
    }
    if (md > t2) { keep[mi] = 1; stack.push([a, mi], [mi, b]); }
  }
  return pts.filter((_, i) => keep[i]);
}
const geomPts = g => g.filter(Boolean).map(p => [qx(p.lon), qy(p.lat)]);
function centroid(g) {
  let x = 0, y = 0, n = 0;
  for (const p of g) if (p) { x += p.lon; y += p.lat; n++; }
  return n ? [x / n, y / n] : null;
}
const yes = v => v === 'yes' || v === 'designated' || v === 'permissive' || v === 'destination';
const no = v => v === 'no' || v === 'private';

// ---------------------------------------------------------------- roads
const CAR = new Set(['motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link', 'secondary', 'secondary_link',
  'tertiary', 'tertiary_link', 'unclassified', 'residential', 'living_street', 'service', 'road']);
const FOOT_ONLY = new Set(['footway', 'pedestrian', 'path', 'steps', 'cycleway', 'track', 'bridleway', 'corridor']);
const SPEED = { motorway: 90, motorway_link: 50, trunk: 70, trunk_link: 40, primary: 55, primary_link: 35, secondary: 50,
  secondary_link: 30, tertiary: 40, tertiary_link: 25, unclassified: 30, residential: 25, living_street: 10, service: 15, road: 25, track: 15 };

function buildRoads(elements) {
  const hwList = [], hwIdx = new Map(), names = [''], nameIdx = new Map([['', 0]]);
  const nodeIndex = new Map(), nodePts = [], ways = [], streetInfo = new Map();
  const idx = (map, list, v) => { if (!map.has(v)) { map.set(v, list.length); list.push(v); } return map.get(v); };

  for (const el of elements) {
    if (el.type !== 'way' || !el.tags?.highway || !el.geometry || !el.nodes) continue;
    const t = el.tags, hw = t.highway;
    if (!CAR.has(hw) && !FOOT_ONLY.has(hw)) continue;
    if (t.area === 'yes' && hw !== 'pedestrian') continue;
    const access = t.access;
    const car = CAR.has(hw) && !no(t.motor_vehicle) && !no(t.motorcar) &&
      !(no(access) && !yes(t.motor_vehicle) && !yes(t.motorcar));
    const foot = !no(t.foot) && !hw.startsWith('motorway') && !(no(access) && !yes(t.foot));
    const bike = !no(t.bicycle) && !hw.startsWith('motorway') && hw !== 'steps' && !(no(access) && !yes(t.bicycle));
    if (!car && !foot && !bike) continue;

    const roundabout = t.junction === 'roundabout' || t.junction === 'circular';
    let oneway = 0; // 1 = forward, 2 = reverse
    if (t.oneway === 'yes' || t.oneway === '1' || t.oneway === 'true' || (roundabout && t.oneway !== 'no') ||
        (hw === 'motorway' && t.oneway !== 'no')) oneway = 1;
    if (t.oneway === '-1' || t.oneway === 'reverse') oneway = 2;
    const bikeOneway = oneway && t['oneway:bicycle'] !== 'no' && !/opposite/.test(t.cycleway || '');

    let flags = 0;
    if (car) flags |= 1; if (foot) flags |= 2; if (bike) flags |= 4;
    if (oneway === 1) flags |= 8; if (oneway === 2) flags |= 16; if (bikeOneway) flags |= 32;
    if (roundabout) flags |= 64; if (t.bridge && t.bridge !== 'no') flags |= 128; if (t.tunnel && t.tunnel !== 'no') flags |= 256;
    const dismount = !(yes(t.bicycle)) && (hw === 'footway' || hw === 'pedestrian' || hw === 'corridor');
    if (dismount) flags |= 512;

    let speed = SPEED[hw] || 25;
    const ms = parseInt(t.maxspeed, 10);
    if (ms > 0 && ms < 150) speed = Math.round(Math.min(speed * 1.4, ms * 0.85));

    const name = t.name || t['name:hy'] || t.ref || '';
    const refs = [];
    for (let i = 0; i < el.nodes.length; i++) {
      const g = el.geometry[i]; if (!g) continue;
      let ni = nodeIndex.get(el.nodes[i]);
      if (ni === undefined) { ni = nodePts.length; nodeIndex.set(el.nodes[i], ni); nodePts.push([qx(g.lon), qy(g.lat)]); }
      refs.push(ni);
    }
    if (refs.length < 2) continue;
    ways.push([idx(hwIdx, hwList, hw), idx(nameIdx, names, name), flags, speed, ...refs]);

    if (t.name) { // remember streets for search
      const s = streetInfo.get(t.name) || { names: new Set([t.name]), best: null, bestLen: 0, car: false };
      for (const k of ['name:hy', 'name:ru', 'name:en', 'old_name', 'alt_name', 'int_name']) if (t[k]) s.names.add(t[k]);
      if (refs.length > s.bestLen) { s.bestLen = refs.length; s.best = nodePts[refs[refs.length >> 1]]; }
      s.car ||= car;
      streetInfo.set(t.name, s);
    }
  }
  return {
    file: { bbox: BBOX, scale: SCALE, hw: hwList, names, nodes: deltaEncode(nodePts), ways },
    streetInfo,
  };
}

// ---------------------------------------------------------------- areas
function areaLayer(t) {
  if (t.building && t.building !== 'no') return 'building';
  if (t.natural === 'water' || t.water || t.waterway === 'riverbank' || t.landuse === 'reservoir' || t.landuse === 'basin') return 'water';
  if (t.waterway && /^(river|canal|stream)$/.test(t.waterway)) return t.waterway === 'stream' ? 'stream' : 'river';
  if (t.railway && /^(rail|light_rail)$/.test(t.railway) && !t.tunnel) return 'rail';
  if (t.railway === 'tram') return 'tram';
  if (t.leisure === 'park' || t.leisure === 'garden' || t.landuse === 'village_green' || t.landuse === 'recreation_ground') return 'park';
  if (t.leisure === 'pitch' || t.leisure === 'stadium' || t.leisure === 'playground') return 'pitch';
  if (t.natural === 'wood' || t.landuse === 'forest' || t.landuse === 'orchard' || t.landuse === 'vineyard') return 'forest';
  if (t.landuse === 'grass' || t.landuse === 'meadow' || t.natural === 'scrub' || t.natural === 'grassland') return 'grass';
  if (t.landuse === 'cemetery') return 'cemetery';
  return null;
}
const AREA_TOL = { building: 4, water: 15, park: 15, pitch: 6, forest: 20, grass: 15, cemetery: 15, river: 10, stream: 10, rail: 8, tram: 8 };

function buildAreas(elements) {
  const layers = {};
  const add = (layer, g) => {
    let pts = simplify(geomPts(g), AREA_TOL[layer] || 8);
    if (pts.length < 2) return;
    (layers[layer] ||= []).push(deltaEncode(pts));
  };
  for (const el of elements) {
    if (!el.tags) continue;
    if (el.type === 'way' && el.geometry) {
      const l = areaLayer(el.tags);
      if (l && !(el.tags.highway && l !== 'building')) add(l, el.geometry);
    } else if (el.type === 'relation' && el.members) {
      const l = areaLayer(el.tags);
      if (!l) continue;
      for (const m of el.members) if (m.type === 'way' && m.role !== 'inner' && m.geometry) add(l, m.geometry);
    }
  }
  return { bbox: BBOX, scale: SCALE, layers };
}

// ---------------------------------------------------------------- places (search)
const POI_KEYS = ['amenity', 'shop', 'tourism', 'leisure', 'historic', 'office', 'railway', 'public_transport', 'place', 'healthcare', 'craft', 'sport', 'man_made'];
function buildPlaces(elements, streetInfo) {
  const kinds = ['street', 'address'], kindIdx = new Map([['street', 0], ['address', 1]]);
  const places = [], seen = new Set();
  // streets
  for (const [name, s] of streetInfo) {
    const alt = [...s.names].filter(n => n !== name).join('|');
    places.push([name, alt, s.best[0], s.best[1], 0]);
  }
  for (const el of elements) {
    const t = el.tags; if (!t) continue;
    const pt = el.type === 'node' ? [el.lon, el.lat] : el.geometry ? centroid(el.geometry) : null;
    if (!pt) continue;
    const x = qx(pt[0]), y = qy(pt[1]);
    if (x < 0 || y < 0 || pt[0] > BBOX.e || pt[1] > BBOX.n) continue;
    const key = POI_KEYS.find(k => t[k]);
    if (t.name && key && !t.highway) {
      const kind = `${key}=${t[key]}`;
      if (!kindIdx.has(kind)) { kindIdx.set(kind, kinds.length); kinds.push(kind); }
      const alt = ['name:hy', 'name:ru', 'name:en', 'alt_name', 'old_name', 'brand'].map(k => t[k]).filter(v => v && v !== t.name);
      const id = `${t.name}|${x >> 8}|${y >> 8}`;
      if (!seen.has(id)) { seen.add(id); places.push([t.name, [...new Set(alt)].join('|'), x, y, kindIdx.get(kind), t['addr:street'] ? `${t['addr:street']} ${t['addr:housenumber'] || ''}`.trim() : '']); }
    }
    if (t['addr:housenumber'] && t['addr:street']) {
      const label = `${t['addr:street']} ${t['addr:housenumber']}`;
      const id = `a|${label}`;
      if (seen.has(id)) continue; seen.add(id);
      const s = streetInfo.get(t['addr:street']);
      const alt = s ? [...s.names].filter(n => n !== t['addr:street']).map(n => `${n} ${t['addr:housenumber']}`).join('|') : '';
      places.push([label, alt, x, y, 1]);
    }
  }
  return { bbox: BBOX, scale: SCALE, kinds, places };
}

// ---------------------------------------------------------------- main
async function main() {
  const inArg = process.argv.indexOf('--input');
  let osm;
  if (inArg > 0) osm = JSON.parse(fs.readFileSync(process.argv[inArg + 1], 'utf8'));
  else {
    osm = await download();
    if (process.argv.includes('--save-raw')) fs.writeFileSync(path.join(HERE, 'overpass-raw.json'), JSON.stringify(osm));
  }
  const elements = osm.elements;
  console.log(`OSM elements: ${elements.length}`);

  const { file: roads, streetInfo } = buildRoads(elements);
  const areas = buildAreas(elements);
  const places = buildPlaces(elements, streetInfo);

  fs.mkdirSync(OUT, { recursive: true });
  const write = (name, obj) => {
    const s = JSON.stringify(obj); fs.writeFileSync(path.join(OUT, name), s);
    console.log(`  ${name}: ${(s.length / 1e6).toFixed(2)} MB`); return s.length;
  };
  const meta = {
    built: new Date().toISOString(), source: 'OpenStreetMap contributors (ODbL)', bbox: BBOX,
    sizes: { roads: write('roads.json', roads), areas: write('areas.json', areas), places: write('places.json', places) },
    counts: { roadNodes: roads.nodes.length / 2, roads: roads.ways.length, places: places.places.length,
      buildings: areas.layers.building?.length || 0 },
  };
  write('meta.json', meta);
  console.log(meta.counts);
}
main().catch(e => { console.error(e); process.exit(1); });
