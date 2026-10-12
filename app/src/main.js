import * as inf from './inference.js';

const video          = document.getElementById('video');
const staticImg      = document.getElementById('static-img');
const overlay        = document.getElementById('overlay');
const ctx            = overlay.getContext('2d');
const placeholder    = document.getElementById('placeholder');
const loader         = document.getElementById('loader');
const loaderMsg      = document.getElementById('loader-msg');
const errorScreen    = document.getElementById('error-screen');
const errorMsg       = document.getElementById('error-msg');
const fpsBadge       = document.getElementById('fps-badge');
const modelBadge     = document.getElementById('model-badge');
const btnCamera      = document.getElementById('btn-camera');
const btnUpload      = document.getElementById('btn-upload');
const btnModel       = document.getElementById('btn-model');
const btnCapture     = document.getElementById('btn-capture');
const fileInput      = document.getElementById('file-input');
const modelInput     = document.getElementById('model-input');
const modelSelect    = document.getElementById('model-select');
const confSlider     = document.getElementById('conf-slider');
const confVal        = document.getElementById('conf-val');
const iouSlider      = document.getElementById('iou-slider');
const iouVal         = document.getElementById('iou-val');
const bboxToggleRow  = document.getElementById('bbox-toggle-row');
const bboxToggle     = document.getElementById('bbox-toggle');
const viewport       = document.getElementById('viewport');
const btnTheme       = document.getElementById('btn-theme');

let session            = null;
let cameraRunning      = false;
let inferenceRunning   = false;
let rafId              = null;
let fpsSmooth          = 0;
let msSmooth           = 0;
let lastInfStart       = 0;
let displayedW         = 1;
let displayedH         = 1;
let offsetX            = 0;
let offsetY            = 0;
let currentModelName   = 'detection';

function updateDisplayMetrics() {
  const state = inf.getLastPreprocessState();
  const srcW = state.srcW;
  const srcH = state.srcH;
  const canvasW = overlay.width;
  const canvasH = overlay.height;
  const srcAspect = srcW / srcH;
  const canvasAspect = canvasW / canvasH;

  if (srcAspect > canvasAspect) {
    displayedW = canvasW;
    displayedH = canvasW / srcAspect;
  } else {
    displayedH = canvasH;
    displayedW = canvasH * srcAspect;
  }

  offsetX = (canvasW - displayedW) / 2;
  offsetY = (canvasH - displayedH) / 2;

  console.log(`updateDisplayMetrics: src=(${srcW}x${srcH}) canvas=(${canvasW}x${canvasH}) displayed=(${displayedW.toFixed(1)}x${displayedH.toFixed(1)}) offset=(${offsetX.toFixed(1)}, ${offsetY.toFixed(1)})`);
}

function toCanvasCoords(x, y) {
  const state = inf.getLastPreprocessState();
  const srcW = state.srcW;
  const srcH = state.srcH;
  const canvasW = overlay.width;
  const canvasH = overlay.height;
  const srcAspect = srcW / srcH;
  const canvasAspect = canvasW / canvasH;

  let dW, dH, oX, oY;
  if (srcAspect > canvasAspect) {
    dW = canvasW;
    dH = canvasW / srcAspect;
  } else {
    dH = canvasH;
    dW = canvasH * srcAspect;
  }
  oX = (canvasW - dW) / 2;
  oY = (canvasH - dH) / 2;

  // Letterbox state uses { scale, padX, padY }; crop state uses { sx, sy, sw, sh }
  let srcX, srcY;
  if (state.scale !== undefined) {
    srcX = (x - state.padX) / state.scale;
    srcY = (y - state.padY) / state.scale;
  } else {
    srcX = state.sx + x * (state.sw / state.inW);
    srcY = state.sy + y * (state.sh / state.inW);
  }

  return {
    x: oX + srcX * (dW / srcW),
    y: oY + srcY * (dH / srcH),
  };
}

// ── Theme ─────────────────────────────────────────────────

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  btnTheme.textContent = theme === 'dark' ? '☀' : '🌙';
  localStorage.setItem('yoloinfer-theme', theme);
}

btnTheme.addEventListener('click', () => {
  applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
});

// ─────────────────────────────────────────────────────────

inf.setConf(0.30);
inf.setIou(0.45);
updateConfDisplay();
updateIouDisplay();

btnUpload.querySelector('span').textContent = '📤 Foto';
btnModel.querySelector('span').textContent = '🤖 Modelo';

// Uma cor fixa por classe (no app original a cor variava por instância).
// Ordem = ordem das classes no data.yaml: buraco, lixo, piso_tatil, poste
const COLORS = ['#ff1744', '#ff9100', '#ffd600', '#00b0ff', '#d500f9', '#00e676'];
const CLASS_NAMES = ['buraco', 'lixo', 'piso_tatil', 'poste'];
const POSE_NAMES  = ['person'];
const COCO_NAMES  = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'];
// Modelos embarcados no app. "calcadas" é o nosso YOLOv8n treinado (metadados em
// models/calcadas.json: imgsz + nomes das classes, gerados no notebook de exportação).
// "coco" é o yolov8n.onnx original do yoloinferApp (192x192, 80 classes) para comparação.
const BUILTIN_MODELS = {
  calcadas: { path: 'models/calcadas.onnx', meta: 'models/calcadas.json' },
  coco:     { path: 'models/yolov8n.onnx',  imgsz: 192, names: null },
};
const MODEL_LABELS = { detection: 'Detecção', pose: 'Pose', segmentation: 'Segmentação' };
let modelType = 'detection';
const statsEl = document.getElementById('stats');

let showBboxes = true;
let normalizedDetCoords = true; // true = custom model (0-1 → ×INPUT_W); false = standard YOLO (pixel)
let detectionClassNames = CLASS_NAMES; // updated at load time based on detected numClasses

function hexToRgb(hex) {
  return [parseInt(hex.slice(1,3),16), parseInt(hex.slice(3,5),16), parseInt(hex.slice(5,7),16)];
}

function getClassName(classId) {
  if (modelType === 'pose') return POSE_NAMES[classId] ?? 'c' + classId;
  if (modelType === 'segmentation') return COCO_NAMES[classId] ?? 'c' + classId;
  return detectionClassNames[classId] ?? 'c' + classId;
}
const kptPairs = [
    [0,1],[0,2],[1,3],[2,4],          // face
    [5,6],                             // shoulder–shoulder
    [5,7],[6,8],[7,9],[8,10],         // arms
    [5,11],[6,12],[11,12],            // torso + hip–hip
    [11,13],[12,14],[13,15],[14,16],  // legs
];

function updateConfDisplay() {
  inf.setConf(parseFloat(confSlider.value) / 100);
  confVal.textContent = (parseFloat(confSlider.value) / 100).toFixed(2);
}

function updateIouDisplay() {
  inf.setIou(parseFloat(iouSlider.value) / 100);
  iouVal.textContent = (parseFloat(iouSlider.value) / 100).toFixed(2);
}

confSlider.addEventListener('input', updateConfDisplay);
iouSlider.addEventListener('input', updateIouDisplay);

confSlider.addEventListener('change', () => {
  if (!staticImg.hidden) runStaticImage();
});

iouSlider.addEventListener('change', () => {
  if (!staticImg.hidden) runStaticImage();
});

function resizeCanvas() {
  const rect = overlay.getBoundingClientRect();
  overlay.width    = Math.max(1, Math.round(rect.width));
  overlay.height   = Math.max(1, Math.round(rect.height));
  updateDisplayMetrics();
}

function updateModelBadge(modelType) {
  let filename;
  if (BUILTIN_MODELS[currentModelName]) {
    filename = BUILTIN_MODELS[currentModelName].path.split('/').pop();
  } else {
    filename = currentModelName + '.onnx';
  }
  const { w, h } = inf.getInputSize();
  const be = inf.getBackend();
  const beLabel = be.name === 'webgpu' ? 'WebGPU' : 'WASM ×' + be.threads;
  modelBadge.textContent = filename + '  ·  ' + w + 'x' + h + '  ·  ' + beLabel;
  modelBadge.hidden = false;
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: 'environment',
      width:  { ideal: 1280 },
      height: { ideal: 720  },
    },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise(r => { video.onloadedmetadata = r; });
  await video.play();
  // Match viewport to camera's actual aspect ratio so the feed fills the box
  if (video.videoWidth && video.videoHeight) {
    viewport.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
  }
  viewport.classList.add('camera-live');
}

function stopCamera() {
  cameraRunning  = false;
  inferenceRunning = false;
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }
  video.hidden = true;
  fpsBadge.hidden = true;
  btnCamera.textContent = '📷 Câmera';
  btnCamera.classList.remove('running');
  btnCapture.classList.remove('active');
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  viewport.style.aspectRatio = '';
  viewport.classList.remove('camera-live');
}

async function runFrame() {
  if (!cameraRunning) return;
  if (inferenceRunning) {
    rafId = requestAnimationFrame(runFrame);
    return;
  }
  inferenceRunning = true;
  const t0 = performance.now();
  try {
    const results = await doInference(video);
    drawResults(results);
    const infMs = performance.now() - t0;
    msSmooth = msSmooth ? msSmooth * 0.85 + infMs * 0.15 : infMs;
    if (lastInfStart > 0) {
      const dt = t0 - lastInfStart;
      fpsSmooth = fpsSmooth * 0.85 + (1000 / dt) * 0.15;
      fpsBadge.textContent = Math.round(fpsSmooth) + ' FPS · ' + Math.round(msSmooth) + ' ms';
    }
    lastInfStart = t0;
  } catch (e) {
    console.error(e);
  }
  inferenceRunning = false;
  if (cameraRunning) rafId = requestAnimationFrame(runFrame);
}

async function runStaticImage() {
  if (!staticImg.complete || staticImg.naturalWidth === 0) return;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  try {
    const results = await doInference(staticImg);
    drawResults(results);
  } catch (e) {
    console.error(e);
  }
}

function drawResults(results) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  updateStats(results);
  if (!results || results.length === 0) return;
  results.forEach((r) => {
    const c = COLORS[r.classId % COLORS.length];
    const drawBox_ = r.type === 'bbox' || (showBboxes && (r.type === 'keypoints' || r.type === 'mask'));
    if (drawBox_) drawBox(r, c);
    if (r.type === 'keypoints') drawKeypoints(r, c);
    if (r.type === 'mask')      drawMask(r, c);
  });
}

function drawBox(r, c) {
  const p1 = toCanvasCoords(r.x1, r.y1);
  const p2 = toCanvasCoords(r.x2, r.y2);
  const w = p2.x - p1.x;
  const h = p2.y - p1.y;

  ctx.strokeStyle = c;
  ctx.lineWidth    = 3;
  ctx.setLineDash([]);
  ctx.strokeRect(p1.x, p1.y, w, h);
  const lbl = getClassName(r.classId) + ': ' + (r.conf * 100).toFixed(0) + '%';
  ctx.font    = 'bold 12px -apple-system, sans-serif';
  const lw = ctx.measureText(lbl).width;
  const lx = p1.x;
  const ly = Math.max(5, p1.y - 22);
  ctx.fillStyle = c;
  ctx.fillRect(lx, ly, lw + 8, 17);
  ctx.fillStyle = '#000';
  ctx.fillText(lbl, lx + 4, ly + 13);
}

function drawKeypoints(r, c) {
  ctx.strokeStyle = c;
  ctx.lineWidth    = 1.5;
  ctx.setLineDash([]);
  kptPairs.forEach(([i, j]) => {
    const kp1 = r.kpts[i], kp2 = r.kpts[j];
    if (kp1.conf > 0.2 && kp2.conf > 0.2) {
      const p1 = toCanvasCoords(kp1.x, kp1.y);
      const p2 = toCanvasCoords(kp2.x, kp2.y);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }
  });
  ctx.fillStyle = c;
  r.kpts.forEach(k => {
    if (k.conf > 0.2) {
      const p = toCanvasCoords(k.x, k.y);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  });
}

const _maskCanvas = document.createElement('canvas');
_maskCanvas.width = 48;
_maskCanvas.height = 48;
const _maskCtx = _maskCanvas.getContext('2d');

function drawMask(r, c) {
  if (!r.mask) return;
  const imgData = _maskCtx.createImageData(48, 48);
  const [cr, cg, cb] = hexToRgb(c);
  for (let i = 0; i < 48 * 48; i++) {
    const base = i * 4;
    imgData.data[base]     = cr;
    imgData.data[base + 1] = cg;
    imgData.data[base + 2] = cb;
    imgData.data[base + 3] = r.mask[i] > 0.5 ? 110 : 0;
  }
  _maskCtx.putImageData(imgData, 0, 0);

  // Draw full mask without bbox clip so sigmoid contours follow object shape
  const tl = toCanvasCoords(0, 0);
  const br = toCanvasCoords(inf.getInputSize().w, inf.getInputSize().h);
  ctx.drawImage(_maskCanvas, tl.x, tl.y, br.x - tl.x, br.y - tl.y);
}

// Contador de detecções por classe no quadro atual (resumo da "auditoria").
function updateStats(results) {
  if (!statsEl) return;
  if (modelType !== 'detection' || !results) { statsEl.innerHTML = ''; return; }
  const counts = {};
  results.forEach(r => { counts[r.classId] = (counts[r.classId] || 0) + 1; });
  const ids = detectionClassNames.length <= 10
    ? detectionClassNames.map((_, i) => i)
    : Object.keys(counts).map(Number);
  statsEl.innerHTML = ids.map(i =>
    `<span class="stat${counts[i] ? '' : ' zero'}"><i style="background:${COLORS[i % COLORS.length]}"></i>${getClassName(i)} <b>${counts[i] || 0}</b></span>`
  ).join('');
}

// Descobre o tamanho de entrada de um ONNX desconhecido testando tamanhos comuns.
async function probeInputSize(preferred) {
  const sizes = [...new Set([preferred, 640, 320, 416, 256, 192, 480, 512].filter(Boolean))];
  for (const s of sizes) {
    inf.setInputSize(s, s);
    try { await inf.detectOutputConventions(); return s; } catch (e) { /* tenta o próximo */ }
  }
  throw new Error('não foi possível descobrir o tamanho de entrada do modelo');
}

// Run once after every model load to detect type, coordinate convention,
// and preprocessing mode so the very first inference frame is correct.
async function configureLoadedModel(meta = {}) {
  await probeInputSize(meta.imgsz);
  const conventions = await inf.detectOutputConventions();
  if (!conventions) return;
  const { type, normalizedCoords, numClasses } = conventions;

  modelType = type;
  normalizedDetCoords = normalizedCoords;
  if (type === 'detection') {
    if (meta.names && meta.names.length === numClasses) detectionClassNames = meta.names;
    else if (numClasses === 80) detectionClassNames = COCO_NAMES;
    else if (numClasses === CLASS_NAMES.length) detectionClassNames = CLASS_NAMES;
    else detectionClassNames = Array.from({ length: numClasses }, (_, i) => 'classe ' + i);
  }

  // Preprocessing: custom normalized detection → crop; everything else → letterbox
  inf.setPreprocessMode(type === 'detection' && normalizedCoords ? 'crop' : 'letterbox');

  bboxToggleRow.hidden = type === 'detection';
  if (type !== 'detection') {
    showBboxes = false;
    bboxToggle.checked = false;
  }
  updateModelBadge(type);
  updateStats([]);
}

async function fetchMeta(url) {
  try {
    const r = await fetch(url, { cache: 'no-cache' });
    if (r.ok) return await r.json();
  } catch (e) { /* sem metadados */ }
  return {};
}

function modelReady() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  loader.hidden = true;
  btnCamera.disabled = false;
  btnCamera.textContent = '📷 Câmera';
  btnUpload.classList.remove('disabled');
  btnModel.classList.remove('disabled');
  btnCapture.classList.remove('active');
  fpsBadge.hidden = true;
}

async function loadBuiltin(key) {
  const def = BUILTIN_MODELS[key];
  if (cameraRunning) stopCamera();
  errorScreen.hidden = true;
  loader.hidden = false;
  loaderMsg.textContent = 'Carregando modelo...';
  currentModelName = key;
  try {
    if (session) { session.endSession?.(); session = null; }
    const meta = def.meta ? await fetchMeta(def.meta) : { imgsz: def.imgsz, names: def.names };
    // ?v=<hash do modelo> evita que o service worker sirva um modelo antigo do cache
    session = await inf.loadModel(def.path + (meta.versao ? '?v=' + meta.versao : ''));
    await configureLoadedModel(meta);
    modelReady();
  } catch (e) {
    loader.hidden = true;
    errorScreen.hidden = false;
    errorMsg.textContent = 'Não foi possível carregar o modelo ' + def.path + ': ' + e.message;
    modelBadge.textContent = 'Erro';
    modelBadge.hidden = false;
  }
}

// ── Camera ────────────────────────────────────────────────

btnCamera.addEventListener('click', async () => {
  if (cameraRunning) {
    stopCamera();
    placeholder.hidden = false;
    return;
  }
  if (!session) {
    errorMsg.textContent = 'Nenhum modelo carregado. Use o botão "Modelo".';
    errorScreen.hidden = false;
    return;
  }
  staticImg.hidden = true;
  placeholder.hidden = true;
  video.hidden      = false;
  fpsBadge.hidden   = false;
  try {
    await startCamera();
  } catch (e) {
    video.hidden = true;
    placeholder.hidden = true;
    fpsBadge.hidden = true;
    errorMsg.textContent = 'A câmera exige HTTPS e permissão de acesso. Abra o app pelo link https e permita a câmera.';
    errorScreen.hidden = false;
    return;
  }
  cameraRunning   = true;
  btnCamera.textContent = '⏹ Parar';
  btnCamera.classList.add('running');
  fpsSmooth = 0;
  lastInfStart = 0;
  resizeCanvas();
  rafId = requestAnimationFrame(runFrame);
});

// ── Image Upload ──────────────────────────────────────────

fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file || !session) return;
  stopCamera();
  placeholder.hidden = true;
  const url = URL.createObjectURL(file);
  staticImg.src = url;
  await new Promise(r => { staticImg.onload = r; });
  staticImg.hidden = false;
  resizeCanvas();
  btnCapture.classList.add('active');
  await runStaticImage();
  URL.revokeObjectURL(url);
  fileInput.value = '';
});

// ── Model Upload ──────────────────────────────────────────

async function loadModelFromFile(file) {
  if (!file) return;
  if (cameraRunning) stopCamera();
  const url = URL.createObjectURL(file);
  currentModelName = file.name.replace(/\.[^/.]+$/, '');
  errorScreen.hidden = true;
  loader.hidden = false;
  loaderMsg.textContent = 'Carregando modelo...';
  try {
    if (session) { session.endSession?.(); session = null; }
    session = await inf.loadModel(url);
    await configureLoadedModel({});
    modelSelect.value = 'custom';
    modelReady();
  } catch (e) {
    loader.hidden = true;
    errorScreen.hidden = false;
    errorMsg.textContent = 'Não foi possível carregar o modelo. Use um ONNX YOLOv8 exportado pela Ultralytics. ' + e.message;
    modelBadge.textContent = 'Erro';
  }
  URL.revokeObjectURL(url);
  modelInput.value = '';
}

// Use File System Access API on desktop (no action sheet, direct file dialog).
// Fall back to <input type="file"> on iOS/unsupported browsers.
btnModel.addEventListener('click', async () => {
  if (window.showOpenFilePicker) {
    try {
      const [handle] = await window.showOpenFilePicker({
        types: [{ description: 'ONNX Model', accept: { 'application/octet-stream': ['.onnx'] } }],
        multiple: false,
      });
      await loadModelFromFile(await handle.getFile());
    } catch (e) {
      if (e.name !== 'AbortError') console.error(e);
    }
  } else {
    modelInput.click();
  }
});

modelInput.addEventListener('change', async (e) => {
  await loadModelFromFile(e.target.files[0]);
});

// ── Model Selector ────────────────────────────────────────

modelSelect.addEventListener('change', async () => {
  if (BUILTIN_MODELS[modelSelect.value]) await loadBuiltin(modelSelect.value);
});

bboxToggle.addEventListener('change', () => {
  showBboxes = bboxToggle.checked;
  if (!staticImg.hidden) runStaticImage();
});

// ── Capture Button ────────────────────────────────────────

btnCapture.addEventListener('click', () => {
  if (session) runStaticImage();
});

// ── Resize ────────────────────────────────────────────────

window.addEventListener('resize', resizeCanvas);

// ── Inference ─────────────────────────────────────────────

async function doInference(source) {
  if (!session) return [];
  const input = inf.preprocess(source);
  const feeds = { [session.inputNames[0]]: input };
  const out = await session.run(feeds);
  const output = out[session.outputNames[0]];
  const numBoxes = output.dims[2];
  const cTV = inf.getConf();
  const iTV = inf.getIou();

  const detectedType = modelType;

  if (detectedType === 'segmentation') {
    const protos = out[session.outputNames[1]];
    return inf.postprocessSegmentation(output, protos.data, numBoxes, cTV, iTV);
  }
  if (detectedType === 'pose') {
    return inf.postprocessPose(output, numBoxes, cTV, iTV);
  }
  return inf.postprocessDetection(output, numBoxes, cTV, iTV, normalizedDetCoords ? inf.getInputSize().w : 1);
}

// ── Init ──────────────────────────────────────────────────

async function init() {
  resizeCanvas();
  errorScreen.hidden = true;
  fpsBadge.hidden = true;
  modelBadge.hidden = true;
  modelSelect.value = 'calcadas';
  await loadBuiltin('calcadas');
}

init();

// PWA: registra o service worker (só no build de produção) para permitir
// "Adicionar à tela inicial" e cache do modelo.
// O SW também injeta COOP/COEP: na primeira visita a página ainda não está isolada,
// então recarregamos uma única vez quando ele assume o controle (habilita WASM multithread).
if ('serviceWorker' in navigator && import.meta.env.PROD) {
  navigator.serviceWorker.register('./sw.js').then(() => {
    if (!self.crossOriginIsolated && !sessionStorage.getItem('coi-reload')) {
      const recarrega = () => { sessionStorage.setItem('coi-reload', '1'); location.reload(); };
      if (navigator.serviceWorker.controller) recarrega();
      else navigator.serviceWorker.addEventListener('controllerchange', recarrega);
    }
  }).catch(console.error);
}
