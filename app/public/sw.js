// Service worker do app:
//  1) cache offline (app shell + modelo + .wasm), como no yoloinferApp original;
//  2) injeta os cabeçalhos COOP/COEP em todas as respostas. O GitHub Pages não deixa
//     configurar cabeçalhos, e sem eles a página não fica "cross-origin isolated", o que
//     impede o ONNX Runtime de usar WASM com várias threads (mesma técnica do coi-serviceworker).
const CACHE = 'calcadas-v2';
const SHELL = [
   './',
   './index.html',
   './manifest.json',
   './icons/icon-192.png',
   './icons/icon-512.png',
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE).then(cache => cache.addAll(SHELL))
   );
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
     )
   );
  self.clients.claim();
});

function withCoiHeaders(response) {
  if (!response || response.status === 0) return response;
  const headers = new Headers(response.headers);
  headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
  headers.set('Cross-Origin-Resource-Policy', 'same-origin');
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET') return;
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return;
  if ((url.pathname.includes('/models/') && !url.pathname.endsWith('.json')) || url.pathname.endsWith('.wasm') || url.pathname.endsWith('.mjs')) {
    event.respondWith(cacheFirst(event.request).then(withCoiHeaders));
    return;
   }
   event.respondWith(staleWhileRevalidate(event.request).then(withCoiHeaders));
});

async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (response.ok) {
    const clone = response.clone();
    caches.open(CACHE).then(cache => cache.put(request, clone));
   }
   return response;
}

async function staleWhileRevalidate(request) {
  const cached = await caches.match(request);
  const fresh = fetch(request).then(response => {
    if (response.ok) {
      const clone = response.clone();
      caches.open(CACHE).then(cache => cache.put(request, clone));
     }
     return response;
   }).catch(() => cached);
  // HTML/JS/CSS/JSON sempre da rede quando possível (evita ficar preso numa versão antiga)
  if (request.mode === 'navigate' || /\.(js|css|json)$/.test(new URL(request.url).pathname)) {
    return (await fresh) || cached || new Response('Network error', { status: 500 });
  }
  return cached || (await fresh) || new Response('Network error', { status: 500 });
}
