import { defineConfig } from 'vite';
import { existsSync, readFileSync, readdirSync, copyFileSync } from 'fs';
import { join } from 'path';
import basicSsl from '@vitejs/plugin-basic-ssl';

const ortDist = join(process.cwd(), 'node_modules', 'onnxruntime-web', 'dist');
// bundle 'onnxruntime-web/webgpu' (v1.22) usa o binário JSEP (WebGPU + WASM) + seu loader .mjs
const isWasm = f => /^ort-wasm-simd-threaded\.jsep\.(wasm|mjs)$/.test(f);

// Serve (dev) e copia (build) os .wasm do onnxruntime-web, para o app não depender
// de CDN externo durante a demonstração.
function ortWasmPlugin() {
  let outDir = 'dist';
  return {
    name: 'ort-wasm',
    configResolved(cfg) { outDir = cfg.build.outDir; },
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const match = req.url?.match(/\/(ort-wasm[^?/]*\.(?:wasm|mjs))/);
        if (match) {
          const wasmPath = join(ortDist, match[1]);
          if (existsSync(wasmPath)) {
            res.setHeader('Content-Type', wasmPath.endsWith('.mjs') ? 'text/javascript' : 'application/wasm');
            res.end(readFileSync(wasmPath));
            return;
          }
        }
        next();
      });
    },
    closeBundle() {
      readdirSync(ortDist).filter(isWasm).forEach(f => copyFileSync(join(ortDist, f), join(outDir, f)));
    },
  };
}

const isDev = process.env.NODE_ENV !== 'production';
const COI_HEADERS = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  // base relativa: funciona tanto em https://<user>.github.io/<repo>/ quanto em localhost
  base: './',
  plugins: [ortWasmPlugin(), ...(isDev ? [basicSsl()] : [])],
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  build: {
    target: 'es2020',
  },
  // cabeçalhos de isolamento (permitem WASM multithread) também no servidor local
  server: { host: true, headers: COI_HEADERS },
  preview: { headers: COI_HEADERS },
});
