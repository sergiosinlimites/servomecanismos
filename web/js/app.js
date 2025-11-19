import { CanvasSpec, ArmParams, Arm2R, TrefoilSpec, TrefoilGenerator, MotionSpec, TrajectoryPlanner, CM } from './sim.js';
import { SerialManager } from './serial.js';

// Estado global simple
const state = {
  canvas: new CanvasSpec(20 * CM, 20 * CM),
  armParams: new ArmParams(25.0 * CM, 18.0 * CM, [-5.0 * CM, -5.0 * CM]),
  trefoil: new TrefoilSpec(4, Math.PI / 2, 0.3, 7.5 * CM, [10.0 * CM, 10.0 * CM]),
  motion: new MotionSpec(1.0, 60, 10, true, 1.0, 1.0, 6.0, 50.0),
  planned: null, // { t, refXY, thetas, vEff, idx0, startIdxTraj, Ncycle }
  view: { minX: -10, maxX: 25, minY: -10, maxY: 25 }, // viewport mundo→canvas
  txPeriodMs: 200,
  serialEnabled: true,
  serial: new SerialManager(),
  connected: false,
  sending: false,
  startTimePerf: null,
  frameIndex: 0,
  lastTxTime: 0,
  telemetry: [], // objetos {pc_time_s, arduino_ms, q1, q2, q1_ref, q2_ref, u1, u2}
};

// Referencias DOM
const dom = {
  canvas: document.getElementById('armCanvas'),
  status: document.getElementById('statusText'),
  // trébol
  a: document.getElementById('inp_a'),
  bdeg: document.getElementById('inp_bdeg'),
  M: document.getElementById('inp_M'),
  scale: document.getElementById('inp_scale'),
  // brazo y movimiento
  d1: document.getElementById('inp_d1'),
  d2: document.getElementById('inp_d2'),
  v: document.getElementById('inp_v'),
  blend: document.getElementById('inp_blend'),
  // límites/ciclos
  wmax: document.getElementById('inp_wmax'),
  amax: document.getElementById('inp_amax'),
  cycles: document.getElementById('inp_cycles'),
  dwell: document.getElementById('inp_dwell'),
  // serial
  baud: document.getElementById('inp_baud'),
  fps: document.getElementById('inp_fps'),
  txMs: document.getElementById('inp_tx_ms'),
  chkSerial: document.getElementById('chk_serial'),
  btnConnect: document.getElementById('btnConnect'),
  btnDisconnect: document.getElementById('btnDisconnect'),
  btnStopS: document.getElementById('btnStop'),
  // PID inputs/buttons
  kp1: document.getElementById('inp_kp1'),
  kp2: document.getElementById('inp_kp2'),
  ki1: document.getElementById('inp_ki1'),
  ki2: document.getElementById('inp_ki2'),
  kd1: document.getElementById('inp_kd1'),
  kd2: document.getElementById('inp_kd2'),
  btnSendP: document.getElementById('btnSendP'),
  btnSendI: document.getElementById('btnSendI'),
  btnSendD: document.getElementById('btnSendD'),
  console: document.getElementById('serialConsole'),
  // acciones
  btnPlan: document.getElementById('btnPlan'),
  btnStart: document.getElementById('btnStart'),
  btnStop: document.getElementById('btnStop'),
  btnReset: document.getElementById('btnReset'),
  btnSaveConfig: document.getElementById('btnSaveConfig'),
  btnSaveCSV: document.getElementById('btnSaveCSV'),
  btnSaveTelemetry: document.getElementById('btnSaveTelemetry'),
  // plots
  plotTheta: document.getElementById('plot_theta'),
  plotOmega: document.getElementById('plot_omega'),
  plotAlpha: document.getElementById('plot_alpha'),
  plotJerk: document.getElementById('plot_jerk'),
  histTh1: document.getElementById('hist_th1'),
  histTh2: document.getElementById('hist_th2'),
  hist2d: document.getElementById('hist2d'),
  demOmega: document.getElementById('dem_omega'),
  demAlpha: document.getElementById('dem_alpha'),
  demEffort: document.getElementById('dem_effort'),
  stats: document.getElementById('stats'),
};

// Consola enfocada
state.serial.consoleCallback = (line) => {
  appendConsole(line);
};
state.serial.onTelemetry = (obj) => {
  state.telemetry.push(obj);
};

function appendConsole(line) {
  const pre = dom.console;
  const maxLines = 400;
  pre.textContent += (pre.textContent ? '\n' : '') + line;
  const lines = pre.textContent.split('\n');
  if (lines.length > maxLines) {
    pre.textContent = lines.slice(-maxLines).join('\n');
  }
  pre.scrollTop = pre.scrollHeight;
}

// Utilidades
function setStatus(text) {
  dom.status.textContent = text;
}

function getInputsIntoState() {
  state.trefoil.a = parseInt(dom.a.value, 10);
  state.trefoil.b = (parseFloat(dom.bdeg.value) * Math.PI) / 180.0;
  state.trefoil.M = parseFloat(dom.M.value);
  state.trefoil.scale = parseFloat(dom.scale.value) * CM;

  state.armParams.d1 = parseFloat(dom.d1.value) * CM;
  state.armParams.d2 = parseFloat(dom.d2.value) * CM;

  state.motion.speed = parseFloat(dom.v.value);
  state.motion.blendS = parseFloat(dom.blend.value);
  state.motion.wMax = parseFloat(dom.wmax.value);
  state.motion.aMax = parseFloat(dom.amax.value);
  state.motion.cycles = parseInt(dom.cycles.value, 10);
  state.motion.dwellS = parseFloat(dom.dwell.value);

  state.motion.fps = parseInt(dom.fps.value, 10);
  state.txPeriodMs = parseInt(dom.txMs.value, 10) || 200;
  state.serialEnabled = !!dom.chkSerial.checked;
}

function resetInputsToDefault() {
  dom.a.value = '4';
  dom.bdeg.value = '90';
  dom.M.value = '0.30';
  dom.scale.value = '7.5';
  dom.d1.value = '25.0';
  dom.d2.value = '18.0';
  dom.v.value = '1.0';
  dom.blend.value = '1.0';
  dom.wmax.value = '6.0';
  dom.amax.value = '50';
  dom.cycles.value = '10';
  dom.dwell.value = '1.0';
  dom.fps.value = '60';
  dom.baud.value = '115200';
  dom.chkSerial.checked = true;
}

// Planificar
function planTrajectory() {
  getInputsIntoState();
  state.armParams.checkReachRequirement();
  const arm = new Arm2R(state.armParams);
  const tref = new TrefoilGenerator(state.trefoil, state.canvas);
  const planner = new TrajectoryPlanner(arm, tref, state.motion);
  try {
    const planned = planner.build();
    state.planned = planned;
    setStatus(`OK (v efectiva=${planned.vEff.toFixed(2)} cm/s, inicio idx=${planned.idx0})`);
    fitView();
    drawInitialPose(planned);
    updateProfiles(planned);
    updateAnalysis(planned);
  } catch (e) {
    setStatus(String(e.message || e));
  }
}

// Dibujo de brazo
const ctx = dom.canvas.getContext('2d');
function resizeCanvasSquare() {
  // Asegura buffer cuadrado y escalado nítido en pantallas HiDPI
  const cssSize = Math.floor(dom.canvas.clientWidth);
  const ratio = window.devicePixelRatio || 1;
  const target = Math.max(200, Math.floor(cssSize * ratio));
  if (dom.canvas.width !== target || dom.canvas.height !== target) {
    dom.canvas.width = target;
    dom.canvas.height = target;
  }
}
window.addEventListener('resize', () => {
  resizeCanvasSquare();
  if (state.planned) {
    const i = Math.max(0, Math.min(state.frameIndex, state.planned.t.length - 1));
    const th1T = state.planned.thetas[i * 2] || Math.PI;
    const th2T = state.planned.thetas[i * 2 + 1] || 0.0;
    renderScene(th1T, th2T);
  }
});
function worldToCanvas(x, y) {
  // Vista dinámica según state.view, origen abajo-izquierda
  const w = dom.canvas.width, h = dom.canvas.height;
  const { minX, maxX, minY, maxY } = state.view;
  const sx = ((x - minX) / (maxX - minX)) * (w - 20) + 10;
  const sy = h - (((y - minY) / (maxY - minY)) * (h - 20) + 10);
  return [sx, sy];
}

// Cache de trayectoria y trazo de la punta
const trajCache = { points: [] };
const tipTrace = { xs: [], ys: [] };

function drawInitialPose(planned) {
  // Cachear puntos visibles del lienzo 20x20
  trajCache.points = [];
  for (let i = 0; i < planned.refXY.length / 2; i++) {
    const x = planned.refXY[i * 2], y = planned.refXY[i * 2 + 1];
    if (x >= 0 && x <= 20 && y >= 0 && y <= 20) trajCache.points.push([x, y]);
  }
  // Reset trazo
  tipTrace.xs = [];
  tipTrace.ys = [];
  // Render con pose inicial
  renderScene(-Math.PI / 2, 0.0);
}

function drawArmSegments(points) {
  drawArmSegmentsColor(points, '#7fd');
}
function drawArmSegmentsColor(points, color) {
  const base = points[0], elbow = points[1], tip = points[2];
  const [bx, by] = worldToCanvas(base[0], base[1]);
  const [ex, ey] = worldToCanvas(elbow[0], elbow[1]);
  const [tx, ty] = worldToCanvas(tip[0], tip[1]);
  // links
  ctx.strokeStyle = color;
  ctx.lineWidth = 3.0;
  ctx.beginPath();
  ctx.moveTo(bx, by); ctx.lineTo(ex, ey);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(ex, ey); ctx.lineTo(tx, ty);
  ctx.stroke();
  // joints
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(ex, ey, 4, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#fff';
  ctx.beginPath(); ctx.arc(tx, ty, 3, 0, Math.PI * 2); ctx.fill();
}

function renderScene(theta1, theta2) {
  ctx.clearRect(0, 0, dom.canvas.width, dom.canvas.height);
  // Marco lienzo 20x20
  ctx.strokeStyle = '#234';
  ctx.lineWidth = 2;
  const [x0, y0] = worldToCanvas(0, 0);
  const [x20, y20] = worldToCanvas(20, 20);
  ctx.strokeRect(x0, y20, x20 - x0, y0 - y20);
  // Trayectoria (en canvas)
  ctx.strokeStyle = 'rgba(120,180,255,0.5)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < trajCache.points.length; i++) {
    const [cx, cy] = worldToCanvas(trajCache.points[i][0], trajCache.points[i][1]);
    if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
  }
  ctx.stroke();
  // Trazo de punta
  if (tipTrace.xs.length > 1) {
    ctx.strokeStyle = '#6cf';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let i = 0; i < tipTrace.xs.length; i++) {
      const [tx, ty] = worldToCanvas(tipTrace.xs[i], tipTrace.ys[i]);
      if (i === 0) ctx.moveTo(tx, ty); else ctx.lineTo(tx, ty);
    }
    ctx.stroke();
  }
  // Brazo actual
  const arm = new Arm2R(state.armParams);
  const [[x1, y1], [x2, y2]] = arm.fkine(theta1, theta2);
  drawArmSegments([[state.armParams.base[0], state.armParams.base[1]], [x1, y1], [x2, y2]]);
  // Etiquetas de ángulos (θ1 y θ2). Convención visual: 0° hacia abajo
  const deg1 = (theta1 + Math.PI / 2) * 180 / Math.PI;
  const deg2 = (theta2) * 180 / Math.PI; // relativo al primer eslabón
  // Texto cerca de base y codo
  ctx.fillStyle = '#cdeaff';
  ctx.font = '12px monospace';
  const [bx, by] = worldToCanvas(state.armParams.base[0], state.armParams.base[1]);
  const [ex, ey] = worldToCanvas(x1, y1);
  ctx.fillText(`θ1=${deg1.toFixed(1)}°`, bx + 8, by - 6);
  ctx.fillText(`θ2=${deg2.toFixed(1)}°`, ex + 8, ey - 6);

  // Opcional: dibujar brazo "real" (planta) si hay telemetría reciente
  if (state.telemetry.length > 0) {
    const last = state.telemetry[state.telemetry.length - 1];
    // Convertir convención real→sim:
    // q1_arduino: 0 rad hacia abajo -> sim necesita +π/2 para ser “desde horizontal”
    // q2_arduino: (aprox absoluta) -> convertir a relativo respecto al primer eslabón
    let q1_sim = (isFinite(last.q1) ? last.q1 : 0.0) - Math.PI / 2;
    let q2_rel = (isFinite(last.q2) ? last.q2 : 0.0) - (isFinite(last.q1) ? last.q1 : 0.0);
    const [[rx1, ry1], [rx2, ry2]] = arm.fkine(q1_sim, q2_rel);
    drawArmSegmentsColor([[state.armParams.base[0], state.armParams.base[1]], [rx1, ry1], [rx2, ry2]], '#ff8b4b');
  }
}

// Auto-ajustar vista para que quepan lienzo 20x20 y alcance del brazo
function fitView() {
  const baseX = state.armParams.base[0];
  const baseY = state.armParams.base[1];
  const reach = (state.armParams.d1 + state.armParams.d2);
  const margin = 2.0;
  const minX = Math.min(0, baseX - reach) - margin;
  const maxX = Math.max(20, baseX + reach) + margin;
  const minY = Math.min(0, baseY - reach) - margin;
  const maxY = Math.max(20, baseY + reach) + margin;
  const cx = 0.5 * (minX + maxX);
  const cy = 0.5 * (minY + maxY);
  const half = Math.max(maxX - minX, maxY - minY) * 0.5;
  state.view.minX = cx - half;
  state.view.maxX = cx + half;
  state.view.minY = cy - half;
  state.view.maxY = cy + half;
}

// Pan y zoom sobre el canvas
let isPanning = false;
let panStart = null;
dom.canvas.addEventListener('mousedown', (e) => {
  isPanning = true;
  panStart = { x: e.clientX, y: e.clientY, view: { ...state.view } };
});
window.addEventListener('mouseup', () => { isPanning = false; });
window.addEventListener('mousemove', (e) => {
  if (!isPanning) return;
  const w = dom.canvas.clientWidth;
  const h = dom.canvas.clientHeight;
  const dxPx = e.clientX - panStart.x;
  const dyPx = e.clientY - panStart.y;
  const vx = (panStart.view.maxX - panStart.view.minX);
  const vy = (panStart.view.maxY - panStart.view.minY);
  const dWx = -dxPx / Math.max(1, w) * vx;
  const dWy = +dyPx / Math.max(1, h) * vy; // invertido por origen
  state.view.minX = panStart.view.minX + dWx;
  state.view.maxX = panStart.view.maxX + dWx;
  state.view.minY = panStart.view.minY + dWy;
  state.view.maxY = panStart.view.maxY + dWy;
  if (state.planned) {
    const i = Math.max(0, Math.min(state.frameIndex, state.planned.t.length - 1));
    renderScene(state.planned.thetas[i * 2] || -Math.PI / 2, state.planned.thetas[i * 2 + 1] || 0.0);
  }
});
dom.canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const scale = e.deltaY < 0 ? 0.9 : 1.1;
  const { minX, maxX, minY, maxY } = state.view;
  const cx = 0.5 * (minX + maxX);
  const cy = 0.5 * (minY + maxY);
  const halfX = (maxX - minX) * 0.5 * scale;
  const halfY = (maxY - minY) * 0.5 * scale;
  const half = Math.max(halfX, halfY); // mantener cuadrado
  state.view.minX = cx - half;
  state.view.maxX = cx + half;
  state.view.minY = cy - half;
  state.view.maxY = cy + half;
  if (state.planned) {
    const i = Math.max(0, Math.min(state.frameIndex, state.planned.t.length - 1));
    renderScene(state.planned.thetas[i * 2] || -Math.PI / 2, state.planned.thetas[i * 2 + 1] || 0.0);
  }
}, { passive: false });

// Start/Stop streaming + animación
let rafId = null;
function start() {
  if (!state.planned) { planTrajectory(); if (!state.planned) return; }
  state.motion.fps = parseInt(dom.fps.value, 10) || 60;
  getInputsIntoState(); // refresh fps/serial checkbox
  state.sending = true;
  state.startTimePerf = performance.now() / 1000.0;
  state.frameIndex = 0;
  state.lastTxTime = 0;
  state.telemetry = [];
  setStatus('Reproduciendo…');
  loop();
}

function loop() {
  if (!state.sending) return;
  const tNow = performance.now() / 1000.0;
  const tElapsed = tNow - state.startTimePerf;
  const planned = state.planned;
  // Buscar índice objetivo en la serie temporal
  let iTarget = binarySearchRight(planned.t, tElapsed);
  iTarget = Math.max(0, Math.min(iTarget, planned.t.length - 1));
  // Reproducir frames intermedios y acumular trazo de la punta dentro del lienzo
  const arm = new Arm2R(state.armParams);
  for (let k = state.frameIndex; k <= iTarget; k++) {
    const th1 = planned.thetas[k * 2];
    const th2 = planned.thetas[k * 2 + 1];
    const [[, ], [x2, y2]] = arm.fkine(th1, th2);
    if (x2 >= 0 && x2 <= 20 && y2 >= 0 && y2 <= 20) {
      tipTrace.xs.push(x2);
      tipTrace.ys.push(y2);
    }
  }
  // Envío de referencia a periodo configurado
  if (state.serialEnabled && state.connected) {
    if ((tNow - state.lastTxTime) >= (state.txPeriodMs / 1000.0)) {
      const th1 = planned.thetas[iTarget * 2];
      const th2 = planned.thetas[iTarget * 2 + 1];
      // Enviar referencias en GRADOS al Arduino
      const d = 180 / Math.PI;
      state.serial.sendR(th1 * d, th2 * d, (tNow - state.startTimePerf)).catch(() => {});
      state.lastTxTime = tNow;
    }
  }
  // Render del frame objetivo
  const th1T = planned.thetas[iTarget * 2];
  const th2T = planned.thetas[iTarget * 2 + 1];
  renderScene(th1T, th2T);
  state.frameIndex = iTarget + 1;
  // Fin
  if (state.frameIndex >= planned.t.length) {
    stop();
    return;
  }
  rafId = requestAnimationFrame(loop);
}

function stop() {
  if (!state.sending) return;
  state.sending = false;
  if (rafId) cancelAnimationFrame(rafId);
  // Stop serial
  if (state.serialEnabled && state.connected) {
    state.serial.sendS().catch(() => {});
  }
  // Actualizar overlay en perfiles (primera vuelta)
  if (state.planned) overlayTelemetryOnProfiles(state.planned);
  setStatus('Detenido');
}

function binarySearchRight(arr, x) {
  // devuelve el índice del primer elemento > x
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] <= x) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// Perfiles y análisis (Plotly)
function updateProfiles(planned) {
  const { t, thetas } = planned;
  const N = t.length;
  const th1 = new Float64Array(N), th2 = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    th1[i] = thetas[i * 2];
    th2[i] = thetas[i * 2 + 1];
  }
  const dt = 1 / Math.max(1, state.motion.fps);
  const w1 = gradient(th1, dt), w2 = gradient(th2, dt);
  const a1 = gradient(w1, dt), a2 = gradient(w2, dt);
  const j1 = gradient(a1, dt), j2 = gradient(a2, dt);

  // Convertir a grados para visualización
  const d = 180 / Math.PI;
  plotXY(dom.plotTheta, t, th1.map(v => v * d), t, th2.map(v => v * d), 'θ [°]', 'θ1', 'θ2');
  plotXY(dom.plotOmega, t, w1.map(v => v * d), t, w2.map(v => v * d), 'ω [°/s]', 'ω1', 'ω2');
  plotXY(dom.plotAlpha, t, a1.map(v => v * d), t, a2.map(v => v * d), 'α [°/s²]', 'α1', 'α2');
  plotXY(dom.plotJerk, t, j1.map(v => v * d), t, j2.map(v => v * d), 'jerk [°/s³]', 'j1', 'j2');
}

function gradient(arr, dt) {
  const N = arr.length;
  const out = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const i0 = Math.max(0, i - 1), i1 = Math.min(N - 1, i + 1);
    out[i] = (arr[i1] - arr[i0]) / (Math.max(1, (i1 - i0)) * dt);
  }
  return out;
}

function plotXY(el, x1, y1, x2, y2, yTitle, name1, name2) {
  const traces = [
    { x: Array.from(x1), y: Array.from(y1), mode: 'lines', name: name1, line: { color: '#58a6ff' } },
    { x: Array.from(x2), y: Array.from(y2), mode: 'lines', name: name2, line: { color: '#ff8b4b' } },
  ];
  Plotly.newPlot(el, traces, {
    margin: { l: 40, r: 10, t: 10, b: 30 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    xaxis: { title: 't [s]', gridcolor: '#1f2733', color: '#e6edf3' },
    yaxis: { title: yTitle, gridcolor: '#1f2733', color: '#e6edf3' },
    legend: { orientation: 'h', y: 1.12, x: 0 },
  }, { displayModeBar: false, responsive: true });
}

function updateAnalysis(planned) {
  const N = planned.t.length;
  const th1deg = new Float64Array(N), th2deg = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    th1deg[i] = planned.thetas[i * 2] * 180 / Math.PI;
    th2deg[i] = planned.thetas[i * 2 + 1] * 180 / Math.PI;
  }
  // Hist θ1, θ2
  Plotly.newPlot(dom.histTh1, [{
    x: Array.from(th1deg), type: 'histogram', nbinsx: 40, marker: { color: '#58a6ff' }
  }], plotLayout('θ₁ [°]'));
  Plotly.newPlot(dom.histTh2, [{
    x: Array.from(th2deg), type: 'histogram', nbinsx: 40, marker: { color: '#ff8b4b' }
  }], plotLayout('θ₂ [°]'));
  // Hist2D θ1 vs θ2
  Plotly.newPlot(dom.hist2d, [{
    x: Array.from(th1deg), y: Array.from(th2deg), type: 'histogram2d', colorscale: 'YlOrRd'
  }], plotLayout('θ₁ [°] vs θ₂ [°]'));

  // Demandas
  const dt = 1 / Math.max(1, state.motion.fps);
  const w1 = gradient(planned.thetas.filter((_, i) => i % 2 === 0), dt);
  const w2 = gradient(planned.thetas.filter((_, i) => i % 2 === 1), dt);
  const a1 = gradient(w1, dt);
  const a2 = gradient(w2, dt);
  const wNorm = w1.map((v, i) => Math.hypot(v * 180 / Math.PI, w2[i] * 180 / Math.PI));
  const aNorm = a1.map((v, i) => Math.hypot(v * 180 / Math.PI, a2[i] * 180 / Math.PI));
  const effort = wNorm.map((v, i) => v + 0.1 * aNorm[i]);

  scatterColored(dom.demOmega, th1deg, th2deg, wNorm, '|ω| [°/s]');
  scatterColored(dom.demAlpha, th1deg, th2deg, aNorm, '|α| [°/s²]');
  scatterColored(dom.demEffort, th1deg, th2deg, effort, 'Esfuerzo');

  // Stats
  const statsHtml = buildStatsTable(th1deg, th2deg, wNorm, aNorm, effort);
  dom.stats.innerHTML = statsHtml;
}

function plotLayout(yTitle) {
  return {
    margin: { l: 40, r: 10, t: 10, b: 30 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    xaxis: { gridcolor: '#1f2733', color: '#e6edf3' },
    yaxis: { title: yTitle, gridcolor: '#1f2733', color: '#e6edf3' },
    coloraxis: { colorbar: { outlinewidth: 0 } }
  };
}

function scatterColored(el, x, y, c, title) {
  Plotly.newPlot(el, [{
    x: Array.from(x), y: Array.from(y), mode: 'markers',
    marker: { size: 4, color: Array.from(c), colorscale: 'Viridis' }
  }], {
    ...plotLayout(title),
    yaxis: { ...plotLayout(title).yaxis, title: 'θ₂ [°]' },
    xaxis: { ...plotLayout(title).xaxis, title: 'θ₁ [°]' },
  }, { displayModeBar: false, responsive: true });
}

function buildStatsTable(th1, th2, wNorm, aNorm, effort) {
  const mean = arr => arr.reduce((s, v) => s + v, 0) / Math.max(1, arr.length);
  const std = arr => {
    const m = mean(arr);
    return Math.sqrt(mean(arr.map(v => (v - m) ** 2)));
  };
  const ptp = arr => Math.max(...arr) - Math.min(...arr);
  const mostCommon = (arr) => {
    // aproximación por histograma
    const bins = 40;
    const minV = Math.min(...arr), maxV = Math.max(...arr);
    const width = (maxV - minV) / Math.max(1, bins);
    if (width === 0) return arr[0] || 0;
    const hist = new Array(bins).fill(0);
    for (const v of arr) {
      let idx = Math.floor((v - minV) / width);
      if (idx >= bins) idx = bins - 1;
      if (idx < 0) idx = 0;
      hist[idx]++;
    }
    const k = hist.indexOf(Math.max(...hist));
    return minV + (k + 0.5) * width;
  };
  const idxMax = arr => arr.reduce((bi, v, i, a) => v > a[bi] ? i : bi, 0);

  const th1Common = mostCommon(Array.from(th1));
  const th2Common = mostCommon(Array.from(th2));
  const iV = idxMax(wNorm);
  const iA = idxMax(aNorm);
  const iE = idxMax(effort);

  return `
    <table>
      <thead><tr><th>PARÁMETRO</th><th>θ₁</th><th>θ₂</th><th>UNIDAD</th></tr></thead>
      <tbody>
        <tr><td>Más común (histograma)</td><td>${th1Common.toFixed(2)}</td><td>${th2Common.toFixed(2)}</td><td>°</td></tr>
        <tr><td>Promedio</td><td>${mean(th1).toFixed(2)}</td><td>${mean(th2).toFixed(2)}</td><td>°</td></tr>
        <tr><td>Desv. estándar</td><td>${std(th1).toFixed(2)}</td><td>${std(th2).toFixed(2)}</td><td>°</td></tr>
        <tr><td>Rango (max-min)</td><td>${ptp(th1).toFixed(2)}</td><td>${ptp(th2).toFixed(2)}</td><td>°</td></tr>
        <tr><td>Mínimo</td><td>${Math.min(...th1).toFixed(2)}</td><td>${Math.min(...th2).toFixed(2)}</td><td>°</td></tr>
        <tr><td>Máximo</td><td>${Math.max(...th1).toFixed(2)}</td><td>${Math.max(...th2).toFixed(2)}</td><td>°</td></tr>
        <tr><td>Velocidad máx |ω|</td><td colspan="2">${Math.max(...wNorm).toFixed(2)}</td><td>°/s</td></tr>
        <tr><td>Aceleración máx |α|</td><td colspan="2">${Math.max(...aNorm).toFixed(2)}</td><td>°/s²</td></tr>
        <tr><td>Mayor velocidad en</td><td>${th1[iV].toFixed(2)}</td><td>${th2[iV].toFixed(2)}</td><td>°</td></tr>
        <tr><td>Mayor aceleración en</td><td>${th1[iA].toFixed(2)}</td><td>${th2[iA].toFixed(2)}</td><td>°</td></tr>
        <tr><td>Mayor esfuerzo en</td><td>${th1[iE].toFixed(2)}</td><td>${th2[iE].toFixed(2)}</td><td>°</td></tr>
      </tbody>
    </table>
  `;
}

// Overlay con telemetría
function overlayTelemetryOnProfiles(planned) {
  if (!state.telemetry.length) return;
  const idx0 = planned.startIdxTraj;
  const idx1 = Math.min(planned.t.length, idx0 + planned.Ncycle);
  const tSeg = Array.from(planned.t.slice(idx0, idx1));
  const tRel = tSeg.map((v, i) => v - tSeg[0]);
  const tel = state.telemetry.slice(); // [{pc_time_s, q1, q2, ...}] (pc_time relativo al start)
  const tTelRel0 = tel[0].pc_time_s;
  const tTelRel = tel.map(o => o.pc_time_s - tTelRel0);
  const q1r = interp1(tTelRel, tel.map(o => o.q1), tRel);
  const q2r = interp1(tTelRel, tel.map(o => o.q2), tRel);
  const d = 180 / Math.PI;

  // Agregar como líneas discontinuas encima de las existentes
  Plotly.addTraces(dom.plotTheta, [
    { x: tRel, y: Array.from(q1r.map(v => v * d)), mode: 'lines', name: 'θ1 real', line: { dash: 'dash', color: '#58a6ff' } },
    { x: tRel, y: Array.from(q2r.map(v => v * d)), mode: 'lines', name: 'θ2 real', line: { dash: 'dash', color: '#ff8b4b' } },
  ]);
}

function interp1(xi, yi, xq) {
  const out = new Float64Array(xq.length);
  for (let k = 0; k < xq.length; k++) {
    const x = xq[k];
    if (x <= xi[0]) { out[k] = yi[0]; continue; }
    if (x >= xi[xi.length - 1]) { out[k] = yi[yi.length - 1]; continue; }
    let lo = 0, hi = xi.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (xi[mid] <= x) lo = mid; else hi = mid;
    }
    const t = (x - xi[lo]) / Math.max(1e-9, (xi[hi] - xi[lo]));
    out[k] = yi[lo] + t * (yi[hi] - yi[lo]);
  }
  return out;
}

// Guardado de archivos
function saveTextFile(filename, text) {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function onSaveConfig() {
  if (!state.planned) return;
  const p = state.planned;
  const now = new Date();
  const timestamp = now.toISOString().replace(/[-:T]/g, '').slice(0, 15);
  const lines = [];
  lines.push('='.repeat(60));
  lines.push('CONFIGURACIÓN DE TRAYECTORIA 2R - TRÉBOL');
  lines.push('='.repeat(60));
  lines.push('');
  lines.push(`Fecha y hora: ${now.toLocaleString()}`);
  lines.push('');
  lines.push('PARÁMETROS DEL TRÉBOL:');
  lines.push('-'.repeat(40));
  lines.push(`  Número de hojas (lóbulos): ${state.trefoil.a}`);
  lines.push(`  Parámetro b [grados]: ${(state.trefoil.b * 180 / Math.PI).toFixed(2)}°`);
  lines.push(`  Parámetro b [radianes]: ${state.trefoil.b.toFixed(4)} rad`);
  lines.push(`  Parámetro M (modulación): ${state.trefoil.M.toFixed(3)}`);
  lines.push(`  Escala [cm]: ${state.trefoil.scale.toFixed(2)} cm`);
  lines.push(`  Centro [cm]: (${state.trefoil.center[0].toFixed(2)}, ${state.trefoil.center[1].toFixed(2)})`);
  lines.push('');
  lines.push('PARÁMETROS DEL BRAZO:');
  lines.push('-'.repeat(40));
  lines.push(`  Longitud eslabón 1 (d1): ${state.armParams.d1.toFixed(2)} cm`);
  lines.push(`  Longitud eslabón 2 (d2): ${state.armParams.d2.toFixed(2)} cm`);
  lines.push(`  Base del brazo: (${state.armParams.base[0].toFixed(2)}, ${state.armParams.base[1].toFixed(2)}) cm`);
  lines.push(`  Alcance total (d1+d2): ${(state.armParams.d1 + state.armParams.d2).toFixed(2)} cm`);
  lines.push('');
  lines.push('PARÁMETROS DE MOVIMIENTO:');
  lines.push('-'.repeat(40));
  lines.push(`  Velocidad lineal deseada: ${state.motion.speed.toFixed(2)} cm/s`);
  lines.push(`  Velocidad lineal efectiva: ${p.vEff.toFixed(2)} cm/s`);
  lines.push(`  Frecuencia de muestreo (fps): ${state.motion.fps} Hz`);
  lines.push(`  Número de ciclos: ${state.motion.cycles}`);
  lines.push(`  Configuración de codo: ${state.motion.elbowUp ? 'up' : 'down'}`);
  lines.push(`  Tiempo de blend inicial: ${state.motion.blendS.toFixed(2)} s`);
  lines.push(`  Tiempo de espera (dwell): ${state.motion.dwellS.toFixed(2)} s`);
  lines.push('');
  lines.push('LÍMITES ARTICULARES:');
  lines.push('-'.repeat(40));
  lines.push(`  ω máxima: ${state.motion.wMax.toFixed(2)} rad/s`);
  lines.push(`  α máxima: ${state.motion.aMax.toFixed(2)} rad/s²`);
  lines.push('');
  lines.push('INFORMACIÓN DE LA TRAYECTORIA:');
  lines.push('-'.repeat(40));
  lines.push(`  Duración total: ${p.t[p.t.length - 1].toFixed(3)} s`);
  lines.push(`  Número total de puntos: ${p.t.length}`);
  lines.push(`  Índice de inicio en curva: ${p.idx0}`);
  lines.push('');
  lines.push('CONVENCIÓN DE ÁNGULOS:');
  lines.push('-'.repeat(40));
  lines.push('  θ₁: ángulo del eslabón 1 medido desde la horizontal (eje +X)');
  lines.push('  θ₂: ángulo relativo del eslabón 2 respecto al eslabón 1');
  lines.push('       (θ₂ = 0 cuando los eslabones están colineales)');
  lines.push('');
  lines.push('ARCHIVOS GENERADOS:');
  lines.push('-'.repeat(40));
  lines.push(`  Configuración: config_${timestamp}.txt`);
  lines.push(`  Datos CSV: trajectory_${timestamp}.csv`);
  lines.push('');
  lines.push('='.repeat(60));
  saveTextFile(`config_${timestamp}.txt`, lines.join('\n'));
}

function onSaveCSV() {
  if (!state.planned) return;
  const p = state.planned;
  const now = new Date();
  const timestamp = now.toISOString().replace(/[-:T]/g, '').slice(0, 15);
  const lines = [];
  lines.push('# Trayectoria 2R - Referencias angulares');
  lines.push(`# Generado: ${now.toLocaleString()}`);
  lines.push('# theta1: ángulo eslabón 1 desde horizontal [rad]');
  lines.push('# theta2: ángulo relativo eslabón 2 [rad]');
  lines.push(`# start_index=${p.startIdxTraj}`);
  lines.push(`# cycle_frames=${p.Ncycle}`);
  lines.push(`# fps=${state.motion.fps}`);
  lines.push('time_s,theta1_rad,theta2_rad');
  for (let i = 0; i < p.t.length; i++) {
    lines.push(`${p.t[i].toFixed(6)},${p.thetas[i * 2].toFixed(6)},${p.thetas[i * 2 + 1].toFixed(6)}`);
  }
  saveTextFile(`trajectory_${timestamp}.csv`, lines.join('\n'));
}

function onSaveTelemetry() {
  if (!state.telemetry.length) return;
  const now = new Date();
  const timestamp = now.toISOString().replace(/[-:T]/g, '').slice(0, 15);
  const lines = [];
  lines.push('pc_time_s,arduino_ms,q1,q2,q1_ref,q2_ref,u1,u2');
  for (const r of state.telemetry) {
    lines.push(`${r.pc_time_s.toFixed(6)},${r.arduino_ms},${r.q1.toFixed(6)},${r.q2.toFixed(6)},${r.q1_ref.toFixed(6)},${r.q2_ref.toFixed(6)},${r.u1.toFixed(6)},${r.u2.toFixed(6)}`);
  }
  saveTextFile(`telemetry_${timestamp}.csv`, lines.join('\n'));
}

// Eventos UI
dom.btnPlan.addEventListener('click', planTrajectory);
dom.btnStart.addEventListener('click', start);
dom.btnStop.addEventListener('click', stop);
dom.btnReset.addEventListener('click', () => {
  resetInputsToDefault();
  planTrajectory();
});

dom.btnConnect.addEventListener('click', async () => {
  try {
    await state.serial.connect(parseInt(dom.baud.value, 10) || 115200);
    state.connected = true;
    setStatus('Serial conectado.');
  } catch (e) {
    setStatus('Error conectando serial.');
    appendConsole(String(e.message || e));
  }
});
dom.btnDisconnect.addEventListener('click', async () => {
  await state.serial.disconnect().catch(() => {});
  state.connected = false;
  setStatus('Serial desconectado.');
});
dom.btnStopS.addEventListener('click', async () => {
  // Primero detén la simulación (para que no siga enviando R),
  // luego manda 'S' al Arduino.
  stop();
  if (state.connected) {
    await state.serial.sendS().catch(() => {});
  }
});
dom.btnSendP.addEventListener('click', async () => {
  if (!state.connected) return;
  const kp1 = parseFloat(dom.kp1.value), kp2 = parseFloat(dom.kp2.value);
  if (isFinite(kp1) && isFinite(kp2)) await state.serial.sendP(kp1, kp2).catch(() => {});
});
dom.btnSendI.addEventListener('click', async () => {
  if (!state.connected) return;
  const ki1 = parseFloat(dom.ki1.value), ki2 = parseFloat(dom.ki2.value);
  if (isFinite(ki1) && isFinite(ki2)) await state.serial.sendI(ki1, ki2).catch(() => {});
});
dom.btnSendD.addEventListener('click', async () => {
  if (!state.connected) return;
  const kd1 = parseFloat(dom.kd1.value), kd2 = parseFloat(dom.kd2.value);
  if (isFinite(kd1) && isFinite(kd2)) await state.serial.sendD(kd1, kd2).catch(() => {});
});

dom.btnSaveConfig.addEventListener('click', onSaveConfig);
dom.btnSaveCSV.addEventListener('click', onSaveCSV);
dom.btnSaveTelemetry.addEventListener('click', onSaveTelemetry);

// Arranque
resetInputsToDefault();
planTrajectory();
setStatus('Listo (usa Conectar y Start para enviar por Serial)');
resizeCanvasSquare();


