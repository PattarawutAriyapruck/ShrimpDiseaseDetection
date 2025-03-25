const CACHE_NAME = "shrimp-cache-v1";
const urlsToCache = [
    "/",
    "/static/icons/android-chrome-192x192.png",
    "/static/icons/android-chrome-512x512.png"
];

self.addEventListener("install", event => {
    event.waitUntil(
        caches.open(CACHE_NAME).then(cache => {
            console.log("Caching resources");
            return cache.addAll(urlsToCache);
        })
    );
});

self.addEventListener("fetch", event => {
    console.log("Fetching:", event.request.url);
    event.respondWith(
        caches.match(event.request).then(response => {
            return response || fetch(event.request).catch(() => new Response("Offline"));
        })
    );
});

self.addEventListener("activate", event => {
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames.map(cacheName => {
                    if (cacheName !== CACHE_NAME) {
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
});
