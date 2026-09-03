/* Service worker : rend la page utilisable hors ligne apres une premiere
   visite.

   Deux strategies, selon la nature du fichier :

   - Le code de l'application (HTML, CSS, JS) part du RESEAU, avec le
     cache en secours. Une nouvelle version deployee est donc prise en
     compte immediatement, et la page reste utilisable hors ligne. Un
     « cache d'abord » sur ces fichiers figerait l'application dans sa
     version initiale jusqu'a un changement de nom de cache : c'est le
     piege classique des PWA, et il n'est pas rattrapable cote client.

   - Le modele et le moteur ONNX Runtime partent du CACHE, avec le
     reseau en secours. Ce sont des binaires lourds et immuables : les
     retelecharger a chaque visite serait du gaspillage.

   Apres une segmentation reussie en ligne, tout le necessaire est en
   cache et le mode avion fonctionne. */

const CACHE = "iseg-viewer-v3";

// code de l'application : toujours revalide
const APPLI = [
  "./",
  "./index.html",
  "./app.css",
  "./app.js",
  "./manifest.json",
];

// binaires : mis en cache une fois pour toutes
const BINAIRES = [
  "./model.onnx",
  "./icon-192.png",
  "./icon-512.png",
];

const estCodeAppli = (url) =>
  url.origin === self.location.origin &&
  /(\/|\.html|\.css|\.js|\.json)$/.test(url.pathname) &&
  !url.pathname.endsWith("/sw.js");

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE)
      // addAll echoue en bloc si une seule entree manque
      .then((c) => Promise.allSettled([...APPLI, ...BINAIRES].map((u) => c.add(u))))
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

  const url = new URL(e.request.url);
  if (url.pathname.endsWith("/sw.js")) return;   // jamais servi depuis le cache

  e.respondWith(estCodeAppli(url) ? reseauDabord(e.request) : cacheDabord(e.request));
});

async function reseauDabord(requete) {
  try {
    const reponse = await fetch(requete);
    if (reponse && reponse.ok) {
      const copie = reponse.clone();
      caches.open(CACHE).then((c) => c.put(requete, copie));
    }
    return reponse;
  } catch (e) {
    const enCache = await caches.match(requete);
    if (enCache) return enCache;
    throw e;
  }
}

async function cacheDabord(requete) {
  const enCache = await caches.match(requete);
  if (enCache) return enCache;

  const reponse = await fetch(requete);
  // On archive aussi les reponses opaques du CDN : inutilisables a
  // inspecter, mais rejouables telles quelles.
  if (reponse && (reponse.ok || reponse.type === "opaque")) {
    const copie = reponse.clone();
    caches.open(CACHE).then((c) => c.put(requete, copie));
  }
  return reponse;
}
