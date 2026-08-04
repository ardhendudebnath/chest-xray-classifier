/* Service worker: caches the app shell so the page opens without the network.
 *
 * Predictions are never cached. /predict and /explain are cross-origin POSTs to
 * the model server, and a cached X-ray result shown against a different image
 * would be worse than no result at all — so anything that is not one of the
 * shell files listed here goes straight to the network, every time.
 *
 * Only registers over HTTPS or on localhost. On a plain-HTTP LAN address it
 * never runs, which costs nothing: the page still works, it just is not
 * available offline.
 */

const CACHE = 'cxr-shell-v1';

const SHELL = [
  '.',
  'index.html',
  'styles.css',
  'app.js',
  'manifest.webmanifest',
  'icons/icon-192.png',
  'icons/icon-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE)
      .then((cache) => cache.addAll(SHELL))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((key) => key !== CACHE).map((key) => caches.delete(key))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const request = event.request;

  // Never come between the app and the model server.
  if (request.method !== 'GET') return;
  if (new URL(request.url).origin !== self.location.origin) return;

  event.respondWith(
    caches.match(request).then((hit) => hit || fetch(request))
  );
});
