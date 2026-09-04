/* Service worker: makes the page usable offline after a first visit.

   Application code goes to the network first, cache as fallback — a new
   deploy is picked up straight away. Serving it cache-first would freeze
   the app on its first version until the cache name changed.

   The model and the ONNX Runtime engine go to the cache first: large
   immutable binaries, not worth re-downloading. After one successful
   segmentation while online, aeroplane mode works. */

const CACHE = "iseg-viewer-v7";

// application code: always revalidated
const APP = [
  "./",
  "./index.html",
  "./app.css",
  "./app.js",
  "./manifest.json",
];

// binaries: cached once and for all
const BINARIES = [
  // only the default variant and the first sample ship with the install;
  // the others are cached when they are first selected
  "./model-separable.onnx",
  "./sample-1.bin",
  "./icon-192.png",
  "./icon-512.png",
];

const isAppCode = (url) =>
  url.origin === self.location.origin &&
  /(\/|\.html|\.css|\.js|\.json)$/.test(url.pathname) &&
  !url.pathname.endsWith("/sw.js");

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE)
      // addAll fails as a whole if a single entry is missing
      .then((c) => Promise.allSettled([...APP, ...BINARIES].map((u) => c.add(u))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((names) => Promise.all(names.filter((n) => n !== CACHE).map((n) => caches.delete(n))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;

  const url = new URL(e.request.url);
  if (url.pathname.endsWith("/sw.js")) return;   // never served from cache

  e.respondWith(isAppCode(url) ? networkFirst(e.request) : cacheFirst(e.request));
});

async function networkFirst(request) {
  try {
    const response = await fetch(request);
    if (response && response.ok) {
      const copy = response.clone();
      caches.open(CACHE).then((c) => c.put(request, copy));
    }
    return response;
  } catch (e) {
    const cached = await caches.match(request);
    if (cached) return cached;
    throw e;
  }
}

async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;

  const response = await fetch(request);
  // Opaque CDN responses are stored too: they cannot be inspected, but
  // they can be replayed as-is.
  if (response && (response.ok || response.type === "opaque")) {
    const copy = response.clone();
    caches.open(CACHE).then((c) => c.put(request, copy));
  }
  return response;
}
