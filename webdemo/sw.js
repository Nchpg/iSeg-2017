/* Service worker: makes the page usable offline after a first visit.

   Two strategies, depending on what the file is:

   - Application code (HTML, CSS, JS) goes to the NETWORK first, with
     the cache as a fallback. A freshly deployed version is therefore
     picked up straight away, and the page still works offline. Serving
     these files cache-first would freeze the app on its very first
     version until the cache name changes: the classic PWA trap, and one
     the client side cannot recover from on its own.

   - The model and the ONNX Runtime engine go to the CACHE first, with
     the network as a fallback. These are large, immutable binaries;
     re-downloading them on every visit would be pure waste.

   After one successful segmentation while online, everything needed is
   cached and aeroplane mode works. */

const CACHE = "iseg-viewer-v4";

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
  "./model.onnx",
  "./sample.bin.gz",
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
