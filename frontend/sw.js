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

/* Bump this on every shell change. The activate handler deletes every cache
 * whose name is not this one, so the rename is what evicts the old copy --
 * leaving it fixed means the old files are never purged. */
const CACHE = 'cxr-shell-v2';

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

  /* Network first, cache only as the offline fallback.
   *
   * This was cache-first, which is the usual choice for an app shell and is
   * wrong here. Cache-first serves whatever it stored on the first visit and
   * never checks again, so a rebuilt page never reaches anyone who had already
   * opened it. That is not merely stale: app.js carries the class list, and an
   * older copy talking to a newer API renders one bar per class it knows about
   * and silently drops the rest — probabilities that no longer sum to 100% and
   * a missing class, with nothing on screen saying so.
   *
   * Offline still works: the cache answers whenever the network does not. The
   * cost is a network round trip on each shell request while online, which for
   * a handful of small files is not worth a wrong label. */
  event.respondWith(
    fetch(request)
      .then((response) => {
        if (response.ok) {
          const copy = response.clone();
          caches.open(CACHE).then((cache) => cache.put(request, copy));
        }
        return response;
      })
      .catch(() => caches.match(request))
  );
});
