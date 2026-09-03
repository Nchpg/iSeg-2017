/* Service worker : rend la page utilisable hors ligne apres une premiere
   visite.

   Strategie « cache d'abord, reseau en secours », avec mise en cache de
   toute reponse valide. Les fichiers du site sont pre-caches a
   l'installation ; le moteur ONNX Runtime, charge depuis un CDN, est mis
   en cache au premier passage. Apres une inference reussie en ligne,
   plus rien n'est necessaire.

   Le modele (model.onnx, 0,43 Mo) fait partie des fichiers pre-caches :
   apres l'installation il est disponible hors ligne. */

const CACHE = "iseg-viewer-v2";

const LOCAL = [
  "./",
  "./index.html",
  "./app.css",
  "./app.js",
  "./model.onnx",
  "./manifest.json",
  "./icon-192.png",
  "./icon-512.png",
];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE)
      // addAll echoue en bloc si une entree manque : on tolere les absences
      .then((c) => Promise.allSettled(LOCAL.map((u) => c.add(u))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((noms) => Promise.all(noms.filter((n) => n !== CACHE).map((n) => caches.delete(n))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;

  e.respondWith(
    caches.match(e.request).then((enCache) => {
      if (enCache) return enCache;
      return fetch(e.request).then((reponse) => {
        // On archive aussi les reponses opaques du CDN : elles sont
        // inutilisables a inspecter mais rejouables telles quelles.
        if (reponse && (reponse.ok || reponse.type === "opaque")) {
          const copie = reponse.clone();
          caches.open(CACHE).then((c) => c.put(e.request, copie));
        }
        return reponse;
      });
    })
  );
});
