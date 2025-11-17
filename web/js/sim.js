// Simulación/planificación del 2R + generador de trébol en JS (sin dependencias externas)

export const CM = 1.0;
const EPS = 1e-9;

export function clamp(x, a, b) {
  return Math.max(a, Math.min(b, x));
}

export class CanvasSpec {
  constructor(width = 20.0 * CM, height = 20.0 * CM) {
    this.width = width;
    this.height = height;
  }
}

export class ArmParams {
  constructor(d1, d2, base = [-5.0 * CM, -5.0 * CM]) {
    this.d1 = d1;
    this.d2 = d2;
    this.base = base;
  }
  checkReachRequirement() {
    const required = 25.0 * Math.sqrt(2.0) * CM;
    if (this.d1 + this.d2 + 1e-6 < required) {
      // Solo advertencia (no lanzamos) para permitir hardware real
      console.warn(`[WARN] Alcance teórico < 25√2 cm: d1+d2=${(this.d1 + this.d2).toFixed(2)} cm.`);
    }
  }
}

export class Arm2R {
  constructor(params) { this.params = params; }
  fkine(t1, t2) {
    const [x0, y0] = this.params.base;
    const { d1, d2 } = this.params;
    const x1 = x0 + d1 * Math.cos(t1);
    const y1 = y0 + d1 * Math.sin(t1);
    const x2 = x1 + d2 * Math.cos(t1 + t2);
    const y2 = y1 + d2 * Math.sin(t1 + t2);
    return [[x1, y1], [x2, y2]];
  }
  ikine(x, y, elbowUp = true) {
    const [x0, y0] = this.params.base;
    const dx = x - x0, dy = y - y0;
    const { d1, d2 } = this.params;
    const r2 = dx * dx + dy * dy;
    const r = Math.sqrt(Math.max(0, r2));
    if (r > d1 + d2 + 1e-9 || r < Math.abs(d1 - d2) - 1e-9) return null;
    let c2 = (r2 - d1 * d1 - d2 * d2) / (2.0 * d1 * d2);
    c2 = clamp(c2, -1.0, 1.0);
    let s2 = Math.sqrt(Math.max(0.0, 1.0 - c2 * c2));
    s2 = elbowUp ? s2 : -s2;
    const t2 = Math.atan2(s2, c2);
    const k1 = d1 + d2 * c2;
    const k2 = d2 * s2;
    const t1 = Math.atan2(dy, dx) - Math.atan2(k2, k1);
    return [normalizeAngle(t1), normalizeAngle(t2)];
  }
}

export function normalizeAngle(a) {
  return ((a + Math.PI) % (2 * Math.PI)) - Math.PI;
}

export class TrefoilSpec {
  constructor(a = 4, bRad = Math.PI / 2, M = 0.3, scaleCm = 7.5 * CM, center = [10.0 * CM, 10.0 * CM]) {
    this.a = a;
    this.b = bRad;
    this.M = M;
    this.scale = scaleCm;
    this.center = center;
  }
}

export class TrefoilGenerator {
  constructor(spec, canvas) {
    this.spec = spec;
    this.canvas = canvas;
  }
  _radius(theta) {
    const s = this.spec;
    // r = scale * (1 + M*sin(a*θ + b))
    return s.scale * (1.0 + s.M * Math.sin(s.a * theta + s.b));
  }
  curveXY(N = 4000) {
    const xs = new Array(N), ys = new Array(N);
    const [xc, yc] = this.spec.center;
    for (let i = 0; i < N; i++) {
      const th = (2.0 * Math.PI * i) / N;
      const r = this._radius(th);
      xs[i] = xc + r * Math.cos(th);
      ys[i] = yc + r * Math.sin(th);
    }
    return { xs, ys };
  }
  fitInsideCanvas(margin = 0.3 * CM) {
    const [xc, yc] = this.spec.center;
    const maxRx = Math.min(xc - margin, this.canvas.width - xc - margin);
    const maxRy = Math.min(yc - margin, this.canvas.height - yc - margin);
    const maxR = Math.max(0.0, Math.min(maxRx, maxRy));
    const allowedScale = maxR / Math.max(1e-9, 1.0 + this.spec.M);
    if (this.spec.scale > allowedScale) {
      this.spec.scale = allowedScale;
    }
  }
  ensureMinDiameter(diameter, margin = 0.3 * CM) {
    const neededScale = (diameter / 2.0) / Math.max(1e-9, 1.0 + this.spec.M);
    this.spec.scale = Math.max(this.spec.scale, neededScale);
    this.fitInsideCanvas(margin);
  }
}

export class MotionSpec {
  constructor(speedCmS = 6.0, fps = 60, cycles = 10, elbowUp = true, dwellS = 1.0, blendS = 1.0, wMax = 6.0, aMax = 50.0) {
    this.speed = speedCmS;
    this.fps = fps;
    this.cycles = cycles;
    this.elbowUp = elbowUp;
    this.dwellS = dwellS;
    this.blendS = blendS;
    this.wMax = wMax;
    this.aMax = aMax;
  }
}

export class TrajectoryPlanner {
  constructor(arm, tref, motion) {
    this.arm = arm;
    this.tref = tref;
    this.motion = motion;
  }

  _arcLength(xs, ys) {
    const N = xs.length;
    const s = new Float64Array(N);
    s[0] = 0.0;
    for (let i = 1; i < N; i++) {
      const dx = xs[i] - xs[i - 1];
      const dy = ys[i] - ys[i - 1];
      s[i] = s[i - 1] + Math.hypot(dx, dy);
    }
    return s;
  }

  _checkReach(xs, ys) {
    const [x0, y0] = this.arm.params.base;
    const { d1, d2 } = this.arm.params;
    for (let i = 0; i < xs.length; i++) {
      const dx = xs[i] - x0, dy = ys[i] - y0;
      const r = Math.hypot(dx, dy);
      if (r < Math.abs(d1 - d2) - 1e-6 || r > d1 + d2 + 1e-6) return false;
    }
    return true;
  }

  _interp1(s, xs, sQuery) {
    // asume s creciente, búsqueda lineal / binaria
    const N = s.length;
    const out = new Float64Array(sQuery.length);
    for (let qi = 0; qi < sQuery.length; qi++) {
      const sq = sQuery[qi];
      if (sq <= s[0]) { out[qi] = xs[0]; continue; }
      if (sq >= s[N - 1]) { out[qi] = xs[N - 1]; continue; }
      let lo = 0, hi = N - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (s[mid] <= sq) lo = mid; else hi = mid;
      }
      const t = (sq - s[lo]) / Math.max(EPS, (s[hi] - s[lo]));
      out[qi] = xs[lo] + t * (xs[hi] - xs[lo]);
    }
    return out;
  }

  _ikPath(x, y, elbowUp) {
    const N = x.length;
    const th = new Float64Array(N * 2);
    let alt = elbowUp;
    for (let i = 0; i < N; i++) {
      const sol = this.arm.ikine(x[i], y[i], alt);
      if (!sol) {
        alt = !alt;
        const altSol = this.arm.ikine(x[i], y[i], alt);
        if (!altSol) throw new Error(`Punto inalcanzable idx=${i}`);
        th[i * 2] = altSol[0]; th[i * 2 + 1] = altSol[1];
      } else {
        th[i * 2] = sol[0]; th[i * 2 + 1] = sol[1];
      }
    }
    return th; // [t1_0, t2_0, t1_1, t2_1, ...]
  }

  _profiles(th, fps) {
    const N = th.length / 2;
    const dt = 1.0 / Math.max(1, fps);
    const w = new Float64Array(N * 2);
    const a = new Float64Array(N * 2);
    // gradiente simple (centrado interno)
    for (let i = 0; i < N; i++) {
      const i0 = Math.max(0, i - 1), i1 = Math.min(N - 1, i + 1);
      w[i * 2] = (th[i1 * 2] - th[i0 * 2]) / (Math.max(1, (i1 - i0)) * dt);
      w[i * 2 + 1] = (th[i1 * 2 + 1] - th[i0 * 2 + 1]) / (Math.max(1, (i1 - i0)) * dt);
    }
    for (let i = 0; i < N; i++) {
      const i0 = Math.max(0, i - 1), i1 = Math.min(N - 1, i + 1);
      a[i * 2] = (w[i1 * 2] - w[i0 * 2]) / (Math.max(1, (i1 - i0)) * dt);
      a[i * 2 + 1] = (w[i1 * 2 + 1] - w[i0 * 2 + 1]) / (Math.max(1, (i1 - i0)) * dt);
    }
    return { w, a };
  }

  _closestStartIndexBottom(xs, ys) {
    const [[, ], [xpark, ypark]] = this.arm.fkine(-Math.PI / 2, 0.0);
    const yc = this.tref.spec.center[1];
    let bestIdx = 0, bestD2 = Number.POSITIVE_INFINITY;
    for (let i = 0; i < xs.length; i++) {
      if (ys[i] <= yc + 1e-9) {
        const dx = xs[i] - xpark, dy = ys[i] - ypark;
        const d2 = dx * dx + dy * dy;
        if (d2 < bestD2) { bestD2 = d2; bestIdx = i; }
      }
    }
    return bestIdx;
  }

  _quinticBlend(th0, th1, N) {
    if (N <= 1) return [th1.slice()];
    const out = new Array(N);
    for (let i = 0; i < N; i++) {
      const u = i / (N - 1);
      const s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5;
      out[i] = [th0[0] + s * (th1[0] - th0[0]), th0[1] + s * (th1[1] - th0[1])];
    }
    return out;
  }

  _scaleSpeedForLimits(xs, ys, s, v) {
    // Promedia un ciclo virtual y evalúa w/ a; reduce v si excede límites
    const fps = this.motion.fps;
    const N = Math.max(1, Math.floor((s[s.length - 1] / v) * fps));
    const tau = new Float64Array(N);
    for (let i = 0; i < N; i++) tau[i] = (i / fps) * v;
    const sQuery = new Float64Array(N);
    for (let i = 0; i < N; i++) sQuery[i] = tau[i] % s[s.length - 1];
    const x = this._interp1(s, xs, sQuery);
    const y = this._interp1(s, ys, sQuery);
    const th = this._ikPath(x, y, this.motion.elbowUp);
    const { w, a } = this._profiles(th, fps);
    let wMax = 0.0, aMax = 0.0;
    for (let i = 0; i < w.length; i++) wMax = Math.max(wMax, Math.abs(w[i]));
    for (let i = 0; i < a.length; i++) aMax = Math.max(aMax, Math.abs(a[i]));
    const factorW = wMax / Math.max(EPS, this.motion.wMax);
    const factorA = Math.sqrt(aMax / Math.max(EPS, this.motion.aMax));
    const factor = Math.max(1.0, factorW, factorA);
    return v / factor;
  }

  build() {
    // 1) ajustar tamaño al lienzo
    this.tref.ensureMinDiameter(20.0);
    this.tref.fitInsideCanvas();
    const { xs, ys } = this.tref.curveXY(4000);
    if (!this._checkReach(xs, ys)) throw new Error("Trayectoria del trébol inalcanzable con d1,d2 actuales.");

    // 2) arco y punto inicial óptimo
    const s = this._arcLength(xs, ys);
    const L = s[s.length - 1];
    const idx0 = this._closestStartIndexBottom(xs, ys);
    const s0 = s[idx0];
    const x0 = xs[idx0], y0 = ys[idx0];

    // 3) velocidad efectiva (cumple límites)
    const v0 = clamp(this.motion.speed, 0.5, 30.0);
    const vEff = this._scaleSpeedForLimits(xs, ys, s, v0);
    const fps = this.motion.fps;
    const cycles = Math.floor(clamp(this.motion.cycles, 1, 10));
    const Tcycle = L / Math.max(EPS, vEff);
    const Ncycle = Math.max(1, Math.floor(Tcycle * fps));
    const N0 = Math.floor(Math.max(0, this.motion.dwellS) * fps);

    // 3.1) Determinar tiempo/frames de blend para limitar ω/α
    const thPark = [-Math.PI / 2, 0.0];
    const thFirstTmp = this.arm.ikine(x0, y0, this.motion.elbowUp);
    if (!thFirstTmp) throw new Error("Primer punto inalcanzable.");
    const dth1 = Math.abs(thFirstTmp[0] - thPark[0]);
    const dth2 = Math.abs(thFirstTmp[1] - thPark[1]);
    const dth = Math.max(dth1, dth2);
    // T_blend mínimo por velocidad (factor conservador 2.0 para quintic)
    const T_vel = dth / Math.max(EPS, this.motion.wMax) * 2.0;
    // T_blend mínimo por aceleración (aprox conservadora)
    const T_acc = Math.sqrt((10.0 * dth) / Math.max(EPS, this.motion.aMax));
    const T_user = Math.max(0, this.motion.blendS);
    const T_blend = Math.max(T_user, T_vel, T_acc);
    const Nblend = Math.max(1, Math.floor(T_blend * fps));

    const Ntotal = N0 + Nblend + cycles * Ncycle;

    // 4) tiempo
    const t = new Float64Array(Ntotal);
    for (let i = 0; i < Ntotal; i++) t[i] = i / fps;

    // 5) primera vuelta empezando en s0
    const tau = new Float64Array(Ncycle);
    for (let i = 0; i < Ncycle; i++) tau[i] = (i / fps) * vEff;
    const sQuery = new Float64Array(Ncycle);
    for (let i = 0; i < Ncycle; i++) sQuery[i] = (s0 + tau[i]) % L;
    const xcyc = this._interp1(s, xs, sQuery);
    const ycyc = this._interp1(s, ys, sQuery);

    // 6) IK del primer punto y blend quíntico desde parqueo (vertical hacia abajo)
    const thFirst = thFirstTmp;
    const thBlend = this._quinticBlend(thPark, thFirst, Nblend);

    // 7) componer series
    const refXY = new Float64Array(Ntotal * 2);
    const thetas = new Float64Array(Ntotal * 2);
    const [, [xpark, ypark]] = this.arm.fkine(thPark[0], thPark[1]);
    // dwell
    for (let i = 0; i < N0; i++) {
      refXY[i * 2] = xpark; refXY[i * 2 + 1] = ypark;
      thetas[i * 2] = thPark[0]; thetas[i * 2 + 1] = thPark[1];
    }
    // blend
    for (let i = 0; i < Nblend; i++) {
      const th = thBlend[i];
      const [[x1, y1], [x2, y2]] = this.arm.fkine(th[0], th[1]);
      const k = N0 + i;
      refXY[k * 2] = x2; refXY[k * 2 + 1] = y2;
      thetas[k * 2] = th[0]; thetas[k * 2 + 1] = th[1];
    }
    // ciclos
    for (let c = 0; c < cycles; c++) {
      const start = N0 + Nblend + c * Ncycle;
      const end = start + Ncycle;
      // xy
      for (let i = 0; i < Ncycle; i++) {
        refXY[(start + i) * 2] = xcyc[i];
        refXY[(start + i) * 2 + 1] = ycyc[i];
      }
      // thetas
      const thCyc = this._ikPath(xcyc, ycyc, this.motion.elbowUp);
      for (let i = 0; i < Ncycle; i++) {
        thetas[(start + i) * 2] = thCyc[i * 2];
        thetas[(start + i) * 2 + 1] = thCyc[i * 2 + 1];
      }
    }

    const startIdxTraj = N0 + Nblend;
    return {
      t, refXY, thetas, vEff, idx0, startIdxTraj, Ncycle
    };
  }
}


