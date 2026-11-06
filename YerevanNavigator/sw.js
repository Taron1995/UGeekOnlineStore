// Offline support: the app, the library and the Yerevan map data are stored on the phone.
const SHELL = 'shell-v2', DATA = 'data-v1', TILES = 'tiles-v1', MAX_TILES = 2000;
const ASSETS = ['./', 'index.html', 'style.css', 'app.js', 'offline.js', 'manifest.json', 'icon.svg',
  'vendor/leaflet/leaflet.css', 'vendor/leaflet/leaflet.js', 'vendor/leaflet/images/marker-icon.png',
  'vendor/leaflet/images/marker-icon-2x.png', 'vendor/leaflet/images/marker-shadow.png',
  'vendor/leaflet/images/layers.png', 'vendor/leaflet/images/layers-2x.png'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(SHELL).then(c => c.addAll(ASSETS)));
  self.skipWaiting();
});
self.addEventListener('activate', e => e.waitUntil((async () => {
  for (const k of await caches.keys()) if (![SHELL, DATA, TILES].includes(k)) await caches.delete(k);
  await self.clients.claim();
})()));

self.addEventListener('fetch', e => {
  const req = e.request, url = new URL(req.url);
  if (req.method !== 'GET') return;
  if (url.origin === location.origin && url.pathname.includes('/data/')) {
    // map data: from the phone first (instant, offline); refreshed in the background when online
    e.respondWith(caches.open(DATA).then(async c => {
      const hit = await c.match(req, { ignoreSearch: true });
      const net = fetch(req).then(res => { if (res.ok) c.put(req, res.clone()); return res; });
      if (hit) { e.waitUntil(net.catch(() => {})); return hit; }
      return net;
    }));
  } else if (url.origin === location.origin) {
    // app files: network first so updates arrive, cache when offline
    e.respondWith(fetch(req).then(res => {
      if (res.ok) { const copy = res.clone(); caches.open(SHELL).then(c => c.put(req, copy)); }
      return res;
    }).catch(() => caches.match(req, { ignoreSearch: true })));
  } else if (url.hostname.endsWith('tile.openstreetmap.org')) {
    // optional online layer: keep only what the user has viewed
    e.respondWith(caches.open(TILES).then(async c => {
      const hit = await c.match(req); if (hit) return hit;
      const res = await fetch(req);
      if (res.ok) { c.put(req, res.clone()); c.keys().then(k => k.length > MAX_TILES && c.delete(k[0])); }
      return res;
    }));
  }
});
