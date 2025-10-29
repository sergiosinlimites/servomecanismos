#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemelo digital 2R (trébol) + Streaming Serial a Arduino
-------------------------------------------------------
- Genera la trayectoria de una curva rosa ("trébol") dentro de un lienzo 20×20 cm.
- Planifica por longitud de arco para obtener velocidad lineal ~constante.
- Calcula la serie temporal de (θ1, θ2) por IK con continuidad de rama.
- (Opcional) Envía las referencias articulares por Serial a un Arduino que ejecuta PD.
- Lee telemetría desde Arduino y compara "sim vs real" (ángulos y punta).

CLI principal (ejemplos):
  # Simulación pura (animación)
  python gemelo_2R_serial.py

  # Enviar referencias a Arduino en COM3 (Windows) o /dev/ttyACM0 (Linux), guardar telemetría
  python gemelo_2R_serial.py --port=COM3 --baud=115200 --log=run_telemetry.csv --no-anim

  # Cambiar parámetros
  python gemelo_2R_serial.py --n=5 --A=8 --cx=10 --cy=12 --rot=30 --scale=1.2 --v=6 --fps=60 --cycles=2 --elbow=up

Requisitos: numpy, matplotlib, pyserial (solo si usas --port).
"""

from dataclasses import dataclass
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, List, Optional
import csv
import sys
import time

# --- Serial y threading son opcionales si no se usa --port
try:
    import serial  # pyserial
except Exception:
    serial = None
import threading

# ===== Meta =====
__VERSION__ = "2.2-serial"
DEBUG = True  # ponlo en False si no quieres logs

# ==============================
# Utilidades y constantes
# ==============================

CM = 1.0  # unidad: centímetros en este script
EPS = 1e-9

def clamp(x, a, b):
    return max(a, min(b, x))

def angnorm(a: float) -> float:
    """Normaliza ángulo a (-π, π]."""
    return (a + math.pi) % (2 * math.pi) - math.pi

# ==============================
# Parámetros del brazo y canvas
# ==============================

@dataclass
class Canvas:
    """Lienzo de dibujo 20×20 cm, con origen (0,0) abajo-izquierda."""
    width: float = 20.0 * CM
    height: float = 20.0 * CM

    def contains_xy(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (margin <= x <= self.width - margin) and (margin <= y <= self.height - margin)

@dataclass
class ArmParams:
    d1: float
    d2: float
    base: Tuple[float, float] = (-5.0 * CM, -5.0 * CM)  # posición fija de la base (x0, y0)

    def check_reach_requirement(self):
        """
        Requisito de entrega original: garantizar alcance suficiente para cubrir 20×20 cm
        desde una base fuera del lienzo. Esto exige d1 + d2 >= 25√2 ≈ 35.36 cm.
        Ajusta d1/d2 si tu hardware real es menor (p. ej., 12 cm + 12 cm) y reduce A/centro.
        """
        required = 25.0 * math.sqrt(2.0) * CM
        if self.d1 + self.d2 + 1e-6 < required:
            if DEBUG:
                print(f"[WARN] Alcance teórico < 25√2 cm: d1+d2={self.d1+self.d2:.2f} cm. "
                      f"Puedes seguir, pero ajusta centro/amplitud para evitar inalcanzables.")
            # No lanzamos excepción para permitir streaming con hardware real más corto.

# ==============================
# Cinemática brazo 2R
# ==============================

class Arm2R:
    """
    Cinemática directa e inversa de un brazo planar de 2 eslabones (2R).
    θ1: ángulo del primer eslabón respecto a la horizontal positiva (rad)
    θ2: ángulo relativo del segundo eslabón respecto al primero (rad)
    """
    def __init__(self, params: ArmParams):
        self.params = params

    def fkine(self, theta1: float, theta2: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        x0, y0 = self.params.base
        d1, d2 = self.params.d1, self.params.d2
        x1 = x0 + d1 * math.cos(theta1)
        y1 = y0 + d1 * math.sin(theta1)
        x2 = x1 + d2 * math.cos(theta1 + theta2)
        y2 = y1 + d2 * math.sin(theta1 + theta2)
        return (x1, y1), (x2, y2)

    def ikine(self, x: float, y: float, elbow_up: bool = True) -> Optional[Tuple[float, float]]:
        x0, y0 = self.params.base
        dx = x - x0
        dy = y - y0
        d1, d2 = self.params.d1, self.params.d2
        r2 = dx * dx + dy * dy
        r = math.sqrt(max(0.0, r2))
        # Condición de alcanzabilidad
        if r > d1 + d2 + 1e-9 or r < abs(d1 - d2) - 1e-9:
            return None
        c2 = clamp((r2 - d1 * d1 - d2 * d2) / (2.0 * d1 * d2), -1.0, 1.0)
        s2 = math.sqrt(max(0.0, 1.0 - c2 * c2))
        if not elbow_up:
            s2 = -s2
        theta2 = math.atan2(s2, c2)
        k1 = d1 + d2 * c2
        k2 = d2 * s2
        theta1 = math.atan2(dy, dx) - math.atan2(k2, k1)
        return angnorm(theta1), angnorm(theta2)

    def reachable(self, x: float, y: float) -> bool:
        return self.ikine(x, y, True) is not None or self.ikine(x, y, False) is not None

# ==============================
# Generador de trébol (curva rosa)
# ==============================

@dataclass
class RoseSpec:
    n_leaves: int = 3                 # número de hojas (pétalos)
    amplitude: float = 7.0 * CM       # radio máximo “exterior” A
    center: Tuple[float, float] = (10.0 * CM, 10.0 * CM)  # centro en el lienzo
    rotation_deg: float = 0.0         # rotación del trébol
    use_cos: bool = True              # True: r = A cos(kφ), False: r = A sin(kφ)
    # Rosa estilizada que no pasa por el centro
    avoid_center: bool = True         # no cruzar el centro
    alpha: float = 0.5                # r_min = A*alpha, 0<alpha<1

    def k_for_leaves(self) -> int:
        if self.n_leaves <= 0:
            raise ValueError("El número de hojas debe ser ≥ 1.")
        return self.n_leaves if (self.n_leaves % 2 == 1) else self.n_leaves // 2

class RoseGenerator:
    def __init__(self, spec: RoseSpec, canvas: Canvas):
        self.spec = spec
        self.canvas = canvas

    def _polar(self, phi: np.ndarray) -> np.ndarray:
        A = self.spec.amplitude
        k = self.spec.k_for_leaves()
        base = np.cos(k * phi) if self.spec.use_cos else np.sin(k * phi)
        if self.spec.avoid_center:
            alpha = clamp(self.spec.alpha, 0.05, 0.95)
            # (1+base)/2 ∈ [0,1] ⇒ r ∈ [A*alpha, A]
            r = A * (alpha + (1.0 - alpha) * (1.0 + base) / 2.0)
        else:
            r = A * base
        return r

    def curve_xy(self, num_points: int = 3000) -> np.ndarray:
        phi = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)  # evita duplicado
        r = self._polar(phi)
        phi_rot = phi + np.deg2rad(self.spec.rotation_deg)
        xc, yc = self.spec.center
        x = xc + r * np.cos(phi_rot)
        y = yc + r * np.sin(phi_rot)
        return np.stack([x, y], axis=1)

    def fit_inside_canvas(self, margin: float = 0.5 * CM) -> None:
        xc, yc = self.spec.center
        maxA_x = min(xc - margin, self.canvas.width - xc - margin)
        maxA_y = min(yc - margin, self.canvas.height - yc - margin)
        maxA = max(0.0, min(maxA_x, maxA_y))
        if self.spec.amplitude > maxA:
            self.spec.amplitude = maxA
            if DEBUG:
                print(f"[INFO] Ajuste de amplitud para encajar en canvas: A={self.spec.amplitude:.2f} cm")

# ==============================
# Planeador: reparametrización por arco y muestreo temporal
# ==============================

@dataclass
class MotionSpec:
    speed_cm_s: float = 5.0
    fps: int = 60
    cycles: int = 1
    elbow_up: bool = True
    dwell_s: float = 1.0

class TrajectoryPlanner:
    def __init__(self, arm: Arm2R, rose: RoseGenerator, motion: MotionSpec):
        self.arm = arm
        self.rose = rose
        self.motion = motion

    @staticmethod
    def _arc_length(points: np.ndarray) -> np.ndarray:
        d = np.diff(points, axis=0, prepend=points[0:1])
        seg = np.hypot(d[:, 0], d[:, 1])
        s = np.cumsum(seg)
        s[0] = 0.0
        return s

    def _check_loop_reachability(self, xy_loop: np.ndarray) -> bool:
        x0, y0 = self.arm.params.base
        dx = xy_loop[:, 0] - x0
        dy = xy_loop[:, 1] - y0
        r = np.hypot(dx, dy)
        d1, d2 = self.arm.params.d1, self.arm.params.d2
        rmin_ok = np.all(r >= abs(d1 - d2) - 1e-6)
        rmax_ok = np.all(r <= (d1 + d2) + 1e-6)
        return rmin_ok and rmax_ok

    def _try_plan_once(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 1) Curva base y ajuste al lienzo
        self.rose.fit_inside_canvas()
        xy_loop = self.rose.curve_xy(num_points=4000)

        # 2) Alcanzabilidad del loop (completa)
        if not self._check_loop_reachability(xy_loop):
            raise ValueError("La trayectoria del trébol no es alcanzable con los d1/d2/centro/A dados. "
                             "Ajusta A/centro o longitudes.")

        # 3) Reparametrización por arco
        s = self._arc_length(xy_loop)
        L = s[-1]
        if L < EPS:
            raise ValueError("Longitud de trayectoria insuficiente. Revisa los parámetros del trébol.")
        x_interp = lambda squery: np.interp(squery, s, xy_loop[:, 0])
        y_interp = lambda squery: np.interp(squery, s, xy_loop[:, 1])

        # 4) Tiempo
        v = clamp(self.motion.speed_cm_s, 1.0, 10.0)
        cycles = int(clamp(self.motion.cycles, 1, 10))
        fps = int(max(10, self.motion.fps))
        T0 = max(0.0, self.motion.dwell_s)
        N0 = int(T0 * fps)
        T_cycle = L / v
        N_cycle = int(T_cycle * fps)
        N_total = N0 + cycles * N_cycle
        t = np.arange(N_total) / fps

        # 5) Referencia xy con "pose inicial" exacta (izquierda del bounding box, y <= mitad)
        ref_xy = np.zeros((N_total, 2))
        xmin, xmax = float(np.min(xy_loop[:, 0])), float(np.max(xy_loop[:, 0]))
        ymin, ymax = float(np.min(xy_loop[:, 1])), float(np.max(xy_loop[:, 1]))
        y_mid = ymin + 0.5 * (ymax - ymin)
        x_park = xmin - 1.0 * CM
        y_park = y_mid
        # Ajusta si no es alcanzable
        if not self.arm.reachable(x_park, y_park):
            x_park = xmin - 0.2 * CM  # mueve un poco hacia dentro
        ref_xy[:N0, :] = (x_park, y_park)

        for c in range(cycles):
            start = N0 + c * N_cycle
            end = start + N_cycle
            tau = (np.arange(N_cycle) / fps) * v
            s_query = np.mod(tau, L)
            ref_xy[start:end, 0] = x_interp(s_query)
            ref_xy[start:end, 1] = y_interp(s_query)

        # 6) IK punto a punto con continuidad de rama
        thetas = np.zeros((N_total, 2))
        prev = None
        for i in range(N_total):
            x, y = ref_xy[i]
            cand = []
            s_up = self.arm.ikine(x, y, elbow_up=True)
            s_dn = self.arm.ikine(x, y, elbow_up=False)
            if s_up is not None: cand.append(s_up)
            if s_dn is not None: cand.append(s_dn)
            if not cand:
                if DEBUG:
                    print(f"[DEBUG] IK falló en frame {i}: x={x:.3f}, y={y:.3f}")
                raise RuntimeError("Punto inalcanzable durante la planificación temporal.")
            if prev is None or len(cand) == 1:
                sol = cand[0]
            else:
                def angdiff(a,b):
                    d = (a-b + math.pi)%(2*math.pi) - math.pi
                    return d
                sol = min(cand, key=lambda th: abs(angdiff(th[0], prev[0])) + abs(angdiff(th[1], prev[1])))
            thetas[i] = prev = (angnorm(sol[0]), angnorm(sol[1]))

        return t, ref_xy, thetas

    def build_time_series(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            return self._try_plan_once()
        except RuntimeError:
            if DEBUG:
                print("[DEBUG] Reintentando planificación tras reducir amplitud 2%...")
            self.rose.spec.amplitude *= 0.98
            return self._try_plan_once()

# ==============================
# Simulador/Animación
# ==============================

class Simulator:
    def __init__(self, arm: Arm2R, canvas: Canvas,
                 t: np.ndarray, ref_xy: np.ndarray, thetas: np.ndarray,
                 save_csv: Optional[str] = None):
        self.arm = arm
        self.canvas = canvas
        self.t = t
        self.ref_xy = ref_xy
        self.thetas = thetas
        self.save_csv = save_csv

        self.fig, self.ax = plt.subplots(figsize=(6.5, 6.5))
        self._setup_scene()
        if save_csv:
            self._write_csv(save_csv)

    def _setup_scene(self):
        x0, y0 = self.arm.params.base
        self.ax.set_aspect('equal', 'box')
        self.ax.set_xlim(-10, self.canvas.width + 5)
        self.ax.set_ylim(-10, self.canvas.height + 5)
        self.ax.set_xlabel("x [cm]")
        self.ax.set_ylabel("y [cm]")
        self.ax.set_title(f"Gemelo digital 2R — v{__VERSION__}")

        rect = plt.Rectangle((0, 0), self.canvas.width, self.canvas.height,
                             fill=False, lw=2, ls='-')
        self.ax.add_patch(rect)
        self.ax.plot([x0], [y0], marker='o')

        mask = (self.ref_xy[:, 0] >= 0) & (self.ref_xy[:, 0] <= self.canvas.width) & \
               (self.ref_xy[:, 1] >= 0) & (self.ref_xy[:, 1] <= self.canvas.height)
        self.ax.plot(self.ref_xy[mask, 0], self.ref_xy[mask, 1], lw=1, alpha=0.3, label="ref xy (en canvas)")
        self.ax.legend(loc='upper right')

        self.link1_line, = self.ax.plot([], [], lw=3)
        self.link2_line, = self.ax.plot([], [], lw=3)
        self.elbow_dot, = self.ax.plot([], [], 'o')
        self.tip_trace, = self.ax.plot([], [], lw=1)
        self.tip_dot, = self.ax.plot([], [], 'o', ms=4)
        self.text = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes,
                                 va='top', ha='left')
        self.trace_x = []
        self.trace_y = []

        # Pose inicial "park" desde IK del primer frame
        theta1, theta2 = self.thetas[0]
        (x1, y1), (x2, y2) = self.arm.fkine(theta1, theta2)
        self.link1_line.set_data([self.arm.params.base[0], x1], [self.arm.params.base[1], y1])
        self.link2_line.set_data([x1, x2], [y1, y2])
        self.elbow_dot.set_data([x1], [y1])
        self.tip_dot.set_data([x2], [y2])

    def _write_csv(self, path: str):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["t_s", "theta1_rad", "theta2_rad", "x_cm", "y_cm"])
            for ti, th, xy in zip(self.t, self.thetas, self.ref_xy):
                writer.writerow([f"{ti:.6f}", f"{th[0]:.6f}", f"{th[1]:.6f}", f"{xy[0]:.6f}", f"{xy[1]:.6f}"])

    def _frame(self, i: int):
        theta1, theta2 = self.thetas[i]
        (x1, y1), (x2, y2) = self.arm.fkine(theta1, theta2)

        # Actualiza brazo
        self.link1_line.set_data([self.arm.params.base[0], x1], [self.arm.params.base[1], y1])
        self.link2_line.set_data([x1, x2], [y1, y2])
        self.elbow_dot.set_data([x1], [y1])
        self.tip_dot.set_data([x2], [y2])

        # Trazo del extremo cuando está dentro del lienzo
        if 0 <= x2 <= self.canvas.width and 0 <= y2 <= self.canvas.height:
            self.trace_x.append(x2)
            self.trace_y.append(y2)
            self.tip_trace.set_data(self.trace_x, self.trace_y)

        # Texto
        self.text.set_text(f"t = {self.t[i]:5.2f} s\nθ1 = {theta1:+.2f} rad\nθ2 = {theta2:+.2f} rad")

        return (self.link1_line, self.link2_line, self.elbow_dot, self.tip_trace, self.tip_dot, self.text)

    def _init_anim(self):
        return (self.link1_line, self.link2_line, self.elbow_dot, self.tip_trace, self.tip_dot, self.text)

    def animate(self, fps: int):
        self.anim = FuncAnimation(
            self.fig,
            self._frame,
            frames=len(self.t),
            init_func=self._init_anim,
            interval=1000.0 / fps,
            blit=True
        )
        plt.show()

# ==============================
# Serial: Telemetría y streaming de referencias
# ==============================

class TelemetryRecorder:
    def __init__(self):
        # (pc_time_s, arduino_ms, q1, q2, q1_ref, q2_ref, u1, u2)
        self.rows: List[tuple] = []
        self._lock = threading.Lock()
    def add(self, row: tuple):
        with self._lock:
            self.rows.append(row)

def _telemetry_reader(ser, rec: TelemetryRecorder, stop_flag: threading.Event):
    try:
        ser.reset_input_buffer()
    except Exception:
        pass
    while not stop_flag.is_set():
        try:
            line = ser.readline().decode('ascii', errors='ignore').strip()
        except Exception:
            continue
        if not line:
            continue
        if line.startswith('Y,'):
            parts = line.split(',')
            if len(parts) >= 8:
                try:
                    ms = int(parts[1])
                    q1 = float(parts[2]); q2 = float(parts[3])
                    q1r= float(parts[4]); q2r= float(parts[5])
                    u1 = float(parts[6]); u2 = float(parts[7])
                    rec.add( (time.time(), ms, q1, q2, q1r, q2r, u1, u2) )
                except ValueError:
                    pass

def stream_refs_to_arduino(port: str, baud: int, thetas: np.ndarray, fps: int,
                           log_csv_path: Optional[str] = "run_telemetry.csv",
                           warmup_s: float = 0.5):
    if serial is None:
        raise RuntimeError("pyserial no está instalado. Ejecuta: pip install pyserial")
    ser = serial.Serial(port, baudrate=baud, timeout=0.02)
    time.sleep(0.3)
    # Recalibra cero en Arduino y espera
    ser.write(b"Z\n")
    time.sleep(max(0.1, warmup_s))

    rec = TelemetryRecorder()
    stop_flag = threading.Event()
    th = threading.Thread(target=_telemetry_reader, args=(ser, rec, stop_flag), daemon=True)
    th.start()

    dt = 1.0 / float(max(1, fps))
    t0 = time.perf_counter()
    for i, (th1, th2) in enumerate(thetas):
        # Envía referencia actual
        msg = f"R,{th1:.6f},{th2:.6f}\n".encode('ascii')
        try:
            ser.write(msg)
        except Exception:
            pass

        # Espera al siguiente frame (pacing)
        t_next = t0 + (i + 1) * dt
        while True:
            now = time.perf_counter()
            if now >= t_next - 0.0005:
                break
            # lee rápido para no saturar buffer
            try:
                _ = ser.read(0)
            except Exception:
                break

    # Stop y cierre
    try:
        ser.write(b"S\n")
    except Exception:
        pass
    time.sleep(0.2)
    stop_flag.set()
    time.sleep(0.05)
    try:
        ser.close()
    except Exception:
        pass

    # Guarda CSV si se pide
    if log_csv_path:
        with open(log_csv_path, "w", encoding="utf-8", newline="") as f:
            f.write("pc_time_s,arduino_ms,q1,q2,q1_ref,q2_ref,u1,u2\n")
            for r in rec.rows:
                f.write(",".join(str(x) for x in r) + "\n")
    return rec.rows

# ==============================
# Comparación sim vs real
# ==============================

def compare_sim_vs_real(arm: 'Arm2R', t_sim: np.ndarray, thetas_sim: np.ndarray,
                        telemetry_rows: List[tuple]):
    """
    Compara ángulos y punta (x,y) entre simulación y real.
    Mide RMSE (θ1, θ2) y RMSE de punta en [cm].
    """
    if not telemetry_rows:
        print("[WARN] Sin telemetría para comparar.")
        return

    arr = np.array(telemetry_rows, dtype=float)
    t_pc   = arr[:, 0]                 # s (PC)
    t_ms   = arr[:, 1] * 1e-3          # s (Arduino millis)
    q1_r   = arr[:, 2];  q2_r   = arr[:, 3]
    q1_ref = arr[:, 4];  q2_ref = arr[:, 5]

    # Sim -> puntas
    xy_sim = np.zeros((len(thetas_sim), 2))
    for i, (th1, th2) in enumerate(thetas_sim):
        (_, _), (x2, y2) = arm.fkine(th1, th2)
        xy_sim[i] = (x2, y2)

    # Real -> puntas (mapeando ángulos reales)
    xy_real = np.zeros((len(q1_r), 2))
    for i, (th1, th2) in enumerate(zip(q1_r, q2_r)):
        (_, _), (x2, y2) = arm.fkine(th1, th2)
        xy_real[i] = (x2, y2)

    # Interpola sim en el dominio temporal real para métricas
    t_sim_n  = np.linspace(0, 1, len(thetas_sim))
    t_real_n = np.linspace(0, 1, len(q1_r))
    th1_sim_i = np.interp(t_real_n, t_sim_n, thetas_sim[:, 0])
    th2_sim_i = np.interp(t_real_n, t_sim_n, thetas_sim[:, 1])

    # RMSE
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    rmse_th1 = rmse(th1_sim_i, q1_r)
    rmse_th2 = rmse(th2_sim_i, q2_r)

    # Interpola punta también
    xy_sim_i = np.column_stack([
        np.interp(t_real_n, t_sim_n, xy_sim[:, 0]),
        np.interp(t_real_n, t_sim_n, xy_sim[:, 1])
    ])
    rmse_xy = float(np.sqrt(np.mean((xy_sim_i - xy_real) ** 2)))

    print(f"[Métricas] RMSE θ1={rmse_th1:.3f} rad, θ2={rmse_th2:.3f} rad, RMSE punta≈{rmse_xy:.3f} cm")

    # Graficar
    plt.figure(); plt.plot(t_ms, q1_r, label="θ1 real"); plt.plot(t_ms, th1_sim_i, '--', label="θ1 sim (interp)")
    plt.legend(); plt.xlabel("t Arduino [s]"); plt.ylabel("rad"); plt.title("Seguimiento θ1")

    plt.figure(); plt.plot(t_ms, q2_r, label="θ2 real"); plt.plot(t_ms, th2_sim_i, '--', label="θ2 sim (interp)")
    plt.legend(); plt.xlabel("t Arduino [s]"); plt.ylabel("rad"); plt.title("Seguimiento θ2")

    plt.figure(); plt.plot(xy_sim[:,0], xy_sim[:,1], label="punta sim"); plt.plot(xy_real[:,0], xy_real[:,1], '--', label="punta real")
    plt.axis('equal'); plt.legend(); plt.xlabel("x [cm]"); plt.ylabel("y [cm]"); plt.title("Trayectoria punta: sim vs real")
    plt.show()

# ==============================
# Configuración por defecto y ejecutable
# ==============================

def default_setup() -> Tuple[Arm2R, Canvas, TrajectoryPlanner]:
    # OJO: d1/d2 por defecto grandes para cubrir 20×20 desde base (-5,-5).
    # Para hardware real (p. ej. 12 cm + 12 cm), ajusta por CLI: --d1=12 --d2=12
    arm_params = ArmParams(d1=20.0 * CM, d2=18.0 * CM)
    arm_params.check_reach_requirement()
    arm = Arm2R(arm_params)
    canvas = Canvas()

    rose_spec = RoseSpec(
        n_leaves=5,
        amplitude=7.5 * CM,
        center=(10.0 * CM, 12.0 * CM),
        rotation_deg=0.0,
        use_cos=True,
        avoid_center=True,
        alpha=0.5
    )
    rose = RoseGenerator(rose_spec, canvas)

    motion = MotionSpec(
        speed_cm_s=6.0,
        fps=60,
        cycles=1,
        elbow_up=True,
        dwell_s=1.0
    )
    planner = TrajectoryPlanner(arm, rose, motion)
    return arm, canvas, planner

def run_sim(save_csv: Optional[str] = None):
    print(f"[Gemelo 2R] versión {__VERSION__}")
    arm, canvas, planner = default_setup()
    t, ref_xy, thetas = planner.build_time_series()
    sim = Simulator(arm, canvas, t, ref_xy, thetas, save_csv=save_csv)
    sim.animate(fps=planner.motion.fps)

# ==============================
# CLI mínima para cambiar parámetros rápidamente
# ==============================

def parse_cli_and_run(argv: List[str]):
    """
    Argumentos opcionales:
      --d1=cm --d2=cm --n=hojas --A=amplitud_cm --cx=cm --cy=cm
      --rot=grados --scale=factor(max 1.2)
      --v=cm_s --fps=60 --cycles=1 --elbow=up|down --dwell=s
      --no-offset  --alpha=0.5  --csv=ruta.csv
      --port=COM3|/dev/ttyACM0  --baud=115200  --log=telemetria.csv  --no-anim
    """
    arm, canvas, planner = default_setup()
    csv_path = None
    # Nuevos flags
    port = None
    baud = 115200
    log_path = None
    no_anim = False

    for arg in argv:
        if arg.startswith("--d1="):
            planner.arm.params.d1 = float(arg.split("=", 1)[1])
        elif arg.startswith("--d2="):
            planner.arm.params.d2 = float(arg.split("=", 1)[1])
        elif arg.startswith("--n="):
            planner.rose.spec.n_leaves = int(arg.split("=", 1)[1])
        elif arg.startswith("--A="):
            planner.rose.spec.amplitude = float(arg.split("=", 1)[1])
        elif arg.startswith("--cx="):
            cx = float(arg.split("=", 1)[1])
            planner.rose.spec.center = (cx, planner.rose.spec.center[1])
        elif arg.startswith("--cy="):
            cy = float(arg.split("=", 1)[1])
            planner.rose.spec.center = (planner.rose.spec.center[0], cy)
        elif arg.startswith("--rot="):
            # Límite de rotación ±45°
            planner.rose.spec.rotation_deg = clamp(float(arg.split("=", 1)[1]), -45.0, +45.0)
        elif arg.startswith("--scale="):
            s = float(arg.split("=", 1)[1])
            s = clamp(s, 0.1, 1.2)  # ≤ 1.2 como pide la entrega
            planner.rose.spec.amplitude *= s
        elif arg.startswith("--v="):
            planner.motion.speed_cm_s = float(arg.split("=", 1)[1])
        elif arg.startswith("--fps="):
            planner.motion.fps = int(arg.split("=", 1)[1])
        elif arg.startswith("--cycles="):
            planner.motion.cycles = int(arg.split("=", 1)[1])
        elif arg.startswith("--elbow="):
            val = arg.split("=", 1)[1].strip().lower()
            planner.motion.elbow_up = (val != "down")
        elif arg.startswith("--dwell="):
            planner.motion.dwell_s = float(arg.split("=", 1)[1])
        elif arg.startswith("--no-offset"):
            planner.rose.spec.avoid_center = False
        elif arg.startswith("--alpha="):
            planner.rose.spec.alpha = float(arg.split("=", 1)[1])
        elif arg.startswith("--csv="):
            csv_path = arg.split("=", 1)[1]
        elif arg.startswith("--port="):
            port = arg.split("=", 1)[1]
        elif arg.startswith("--baud="):
            baud = int(arg.split("=", 1)[1])
        elif arg.startswith("--log="):
            log_path = arg.split("=", 1)[1]
        elif arg == "--no-anim":
            no_anim = True

    planner.arm.params.check_reach_requirement()
    print(f"[Gemelo 2R] versión {__VERSION__}")
    t, ref_xy, thetas = planner.build_time_series()

    # Si se especifica puerto -> streaming + opcional animación
    if port:
        print(f"[Serial] Enviando referencias a {port} @ {baud} ...")
        rows = stream_refs_to_arduino(port, baud, thetas, fps=planner.motion.fps,
                                      log_csv_path=log_path or "run_telemetry.csv")
        print(f"[Serial] Telemetría capturada: {len(rows)} muestras.")
        # Comparación rápida (puedes comentar si no quieres graficar siempre)
        try:
            compare_sim_vs_real(planner.arm, t, thetas, rows)
        except Exception as e:
            print(f"[WARN] Comparación falló: {e}")
        if not no_anim:
            sim = Simulator(planner.arm, canvas, t, ref_xy, thetas, save_csv=csv_path)
            sim.animate(fps=planner.motion.fps)
    else:
        # comportamiento original
        sim = Simulator(planner.arm, canvas, t, ref_xy, thetas, save_csv=csv_path)
        sim.animate(fps=planner.motion.fps)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        parse_cli_and_run(sys.argv[1:])
    else:
        run_sim(save_csv=None)
