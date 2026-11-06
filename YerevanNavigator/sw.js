// Caches the app shell and map tiles that were already viewed (works with weak signal).
const SHELL = 'shell-v1', TILES = 'tiles-v1', MAX_TILES = 3000;
const ASSETS = ['./', 'index.html', 'style.css', 'app.js', 'manifest.json', 'icon.svg',
  'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css', 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'];

self.addEventListener('install', e => { e.waitUntil(caches.open(SHELL).then(c => c.addAll(ASSETS))); self.skipWaiting(); });
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (url.hostname.endsWith('tile.openstreetmap.org')) {
    e.respondWith(caches.open(TILES).then(async c => {
      const hit = await c.match(e.request);
      if (hit) return hit;
      const res = await fetch(e.request);
      if (res.ok) { c.put(e.request, res.clone()); trim(c); }
      return res;
    }));
  } else if (e.request.method === 'GET' && (url.origin === location.origin || url.hostname === 'unpkg.com')) {
    // network first, fall back to cache
    e.respondWith(fetch(e.request).then(res => {
      const copy = res.clone(); caches.open(SHELL).then(c => c.put(e.request, copy)); return res;
    }).catch(() => caches.match(e.request)));
  }
});

async function trim(cache) {
  const keys = await cache.keys();
  for (let i = 0; i < keys.length - MAX_TILES; i++) await cache.delete(keys[i]);
}
