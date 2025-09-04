#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemelo digital: brazo planar 2R que dibuja un trébol (rose curve) en un lienzo de 20×20 cm.
Cumple con:
 - Base del brazo en (-5, -5) cm, a la izquierda y abajo del lienzo.
 - Dos eslabones (d1, d2) y dos motores (θ1 en base, θ2 en la unión).
 - Trayectoria paramétrica en coordenadas polares con seno/coseno para un trébol de n hojas igualmente espaciadas.
 - Velocidad lineal constante a lo largo de la trayectoria (reparametrización por longitud de arco).
 - Posición inicial del brazo plegado a la izquierda, sin tocar el lienzo y sin superar la mitad de su altura.
 - Restricción de alcance: d1 + d2 ≥ 25*sqrt(2) cm para poder cubrir todo el lienzo con la base dada.
 - Animación con matplotlib y registro de datos (t, θ1, θ2, x, y).

Ejecutar directamente:
    python gemelo_digital_2R_trebol.py

Requisitos: numpy, matplotlib
"""

from dataclasses import dataclass
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, List, Optional
import csv
import sys

# ==============================
# Utilidades y constantes
# ==============================

CM = 1.0  # unidad de trabajo: centímetros
EPS = 1e-9

def clamp(x, a, b):
    return max(a, min(b, x))

# ==============================
# Parámetros del brazo y canvas
# ==============================

@dataclass
class Canvas:
    """Lienzo de dibujo 20×20 cm, con origen (0,0) abajo a la izquierda."""
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
        Chequea la condición mínima d1 + d2 ≥ 25*sqrt(2) cm
        Según el enunciado: 5√2 + 20√2 = 25√2 (desde (-5,-5) hasta (20,20)).
        """
        required = 25.0 * math.sqrt(2.0) * CM
        if self.d1 + self.d2 + 1e-6 < required:
            raise ValueError(
                f"Alcance insuficiente: d1 + d2 = {self.d1 + self.d2:.2f} cm < 25√2 ≈ {required:.2f} cm. "
                "Aumenta d1 y/o d2."
            )

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
        """Devuelve posiciones (x1, y1) de la articulación y (x2, y2) del efector final."""
        x0, y0 = self.params.base
        d1, d2 = self.params.d1, self.params.d2

        x1 = x0 + d1 * math.cos(theta1)
        y1 = y0 + d1 * math.sin(theta1)

        x2 = x1 + d2 * math.cos(theta1 + theta2)
        y2 = y1 + d2 * math.sin(theta1 + theta2)

        return (x1, y1), (x2, y2)

    def ikine(self, x: float, y: float, elbow_up: bool = True) -> Optional[Tuple[float, float]]:
        """Resuelve IK para (x, y). Devuelve (θ1, θ2) o None si es inalcanzable."""
        x0, y0 = self.params.base
        dx = x - x0
        dy = y - y0
        d1, d2 = self.params.d1, self.params.d2

        r2 = dx * dx + dy * dy
        r = math.sqrt(r2)

        # Condición de alcanzabilidad del 2R
        if r > d1 + d2 + 1e-9 or r < abs(d1 - d2) - 1e-9:
            return None

        # cos θ2 por ley de cosenos
        c2 = clamp((r2 - d1 * d1 - d2 * d2) / (2.0 * d1 * d2), -1.0, 1.0)
        s2 = math.sqrt(max(0.0, 1.0 - c2 * c2))
        if not elbow_up:
            s2 = -s2
        theta2 = math.atan2(s2, c2)

        # θ1
        k1 = d1 + d2 * c2
        k2 = d2 * s2
        theta1 = math.atan2(dy, dx) - math.atan2(k2, k1)

        # Normaliza a [-π, π] para estabilidad
        theta1 = (theta1 + math.pi) % (2 * math.pi) - math.pi
        theta2 = (theta2 + math.pi) % (2 * math.pi) - math.pi
        return theta1, theta2

    def reachable(self, x: float, y: float) -> bool:
        return self.ikine(x, y, True) is not None or self.ikine(x, y, False) is not None

# ==============================
# Generador de trébol (curva rosa)
# ==============================

@dataclass
class RoseSpec:
    n_leaves: int = 3                 # número de hojas (pétalos) igualmente espaciadas
    amplitude: float = 7.0 * CM       # radio máximo (A)
    center: Tuple[float, float] = (10.0 * CM, 10.0 * CM)  # centro en el lienzo
    rotation_deg: float = 0.0         # rotación del trébol
    use_cos: bool = True              # True: r = A cos(k φ), False: r = A sin(k φ)

    def k_for_leaves(self) -> int:
        # Para obtener exactamente n hojas: k = n si n impar, k = n/2 si n par.
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
        if self.spec.use_cos:
            r = A * np.cos(k * phi)
        else:
            r = A * np.sin(k * phi)
        return r

    def curve_xy(self, num_points: int = 3000) -> np.ndarray:
        """Devuelve puntos (x,y) para una vuelta completa de la curva rosa."""
        phi = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=True)
        r = self._polar(phi)
        phi_rot = phi + np.deg2rad(self.spec.rotation_deg)
        xc, yc = self.spec.center
        x = xc + r * np.cos(phi_rot)
        y = yc + r * np.sin(phi_rot)
        return np.stack([x, y], axis=1)

    def fit_inside_canvas(self, margin: float = 0.5 * CM) -> None:
        """Reduce la amplitud si el trébol se sale del lienzo con el centro actual."""
        xc, yc = self.spec.center
        maxA_x = min(xc - margin, self.canvas.width - xc - margin)
        maxA_y = min(yc - margin, self.canvas.height - yc - margin)
        maxA = max(0.0, min(maxA_x, maxA_y))
        if self.spec.amplitude > maxA:
            self.spec.amplitude = maxA

# ==============================
# Planeador: reparametrización por arco y muestreo temporal
# ==============================

@dataclass
class MotionSpec:
    speed_cm_s: float = 5.0           # velocidad lineal del extremo [cm/s]
    fps: int = 60                     # cuadros por segundo
    cycles: int = 1                   # número de ciclos completos del trébol (1..10)
    elbow_up: bool = True             # modo de codo para IK
    dwell_s: float = 1.0              # pausa inicial (brazo plegado fuera del lienzo)

class TrajectoryPlanner:
    def __init__(self, arm: Arm2R, rose: RoseGenerator, motion: MotionSpec):
        self.arm = arm
        self.rose = rose
        self.motion = motion

    @staticmethod
    def _arc_length(points: np.ndarray) -> np.ndarray:
        d = np.diff(points, axis=0)
        seg = np.hypot(d[:, 0], d[:, 1])
        s = np.concatenate([[0.0], np.cumsum(seg)])
        return s

    def build_time_series(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Devuelve:
          t           : tiempos [s]
          ref_xy(t)   : puntos (x,y) a velocidad constante
          thetas(t)   : ángulos (θ1, θ2) por IK
        """
        # 1) Curva base (una vuelta) y ajuste para que caiga dentro del canvas
        self.rose.fit_inside_canvas()
        xy_loop = self.rose.curve_xy(num_points=4000)

        # 2) Chequeo de alcanzabilidad de TODO el loop
        if not self._check_loop_reachability(xy_loop):
            raise ValueError("La trayectoria del trébol no es alcanzable con los d1/d2 dados. "
                             "Reduce la amplitud, cambia el centro o aumenta longitudes.")

        # 3) Reparametrización por longitud de arco (velocidad constante)
        s = self._arc_length(xy_loop)
        L = s[-1]
        if L < EPS:
            raise ValueError("Longitud de trayectoria insuficiente. Revisa los parámetros del trébol.")
        # Interpoladores s->x, s->y
        x_interp = lambda squery: np.interp(squery, s, xy_loop[:, 0])
        y_interp = lambda squery: np.interp(squery, s, xy_loop[:, 1])

        # 4) Tiempo total por ciclo y serie temporal
        v = clamp(self.motion.speed_cm_s, 1.0, 10.0)
        T_cycle = L / v
        cycles = int(clamp(self.motion.cycles, 1, 10))
        fps = int(max(10, self.motion.fps))

        # Pausa inicial (brazo plegado fuera del lienzo)
        T0 = max(0.0, self.motion.dwell_s)
        N0 = int(T0 * fps)

        # Total frames
        N_cycle = int(T_cycle * fps)
        N_total = N0 + cycles * N_cycle

        t = np.arange(N_total) / fps

        # 5) Construye referencia xy con velocidad constante, repetida 'cycles' veces
        ref_xy = np.zeros((N_total, 2))
        # Mantener fuera del lienzo durante la pausa
        ref_xy[:N0, :] = np.array(self.arm.params.base)  # sin dibujar; solo pose inicial
        # Luego los ciclos
        for c in range(cycles):
            start = N0 + c * N_cycle
            end = start + N_cycle
            tau = (np.arange(N_cycle) / fps) * v  # [0, L)
            s_query = np.mod(tau, L)
            ref_xy[start:end, 0] = x_interp(s_query)
            ref_xy[start:end, 1] = y_interp(s_query)

        # 6) IK para cada punto (elige solución válida)
        thetas = np.zeros((N_total, 2))
        elbow_up = self.motion.elbow_up
        for i in range(N_total):
            x, y = ref_xy[i]
            th = self.arm.ikine(x, y, elbow_up=elbow_up)
            if th is None:
                # intentar solución alternativa
                th = self.arm.ikine(x, y, elbow_up=not elbow_up)
            if th is None:
                raise RuntimeError("Punto inalcanzable durante la planificación temporal.")
            thetas[i, 0], thetas[i, 1] = th

        return t, ref_xy, thetas

    def _check_loop_reachability(self, xy_loop: np.ndarray) -> bool:
        """Verifica alcanzabilidad del loop completo con d1/d2 actuales."""
        # chequeo rápido por distancias min/max al origen del brazo
        x0, y0 = self.arm.params.base
        dx = xy_loop[:, 0] - x0
        dy = xy_loop[:, 1] - y0
        r = np.hypot(dx, dy)
        d1, d2 = self.arm.params.d1, self.arm.params.d2
        rmin_ok = np.all(r >= abs(d1 - d2) - 1e-6)
        rmax_ok = np.all(r <= (d1 + d2) + 1e-6)
        return rmin_ok and rmax_ok

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

        # Estado para animación
        self.fig, self.ax = plt.subplots(figsize=(6.5, 6.5))
        self._setup_scene()

        # Registra CSV si se solicita
        if save_csv:
            self._write_csv(save_csv)

    def _setup_scene(self):
        x0, y0 = self.arm.params.base

        # Límites: muestra la base y el lienzo completo
        self.ax.set_aspect('equal', 'box')
        self.ax.set_xlim(-10, self.canvas.width + 5)
        self.ax.set_ylim(-10, self.canvas.height + 5)
        self.ax.set_xlabel("x [cm]")
        self.ax.set_ylabel("y [cm]")
        self.ax.set_title("Gemelo digital 2R — dibujo de trébol")

        # Dibuja el lienzo 20×20
        rect = plt.Rectangle((0, 0), self.canvas.width, self.canvas.height,
                             fill=False, lw=2, ls='-')
        self.ax.add_patch(rect)

        # Dibuja la base
        self.ax.plot([x0], [y0], marker='o')

        # Dibuja la trayectoria de referencia (solo dentro del lienzo)
        mask = (self.ref_xy[:, 0] >= 0) & (self.ref_xy[:, 0] <= self.canvas.width) & \
               (self.ref_xy[:, 1] >= 0) & (self.ref_xy[:, 1] <= self.canvas.height)
        self.ax.plot(self.ref_xy[mask, 0], self.ref_xy[mask, 1], lw=1, alpha=0.3)

        # Elementos del brazo a animar
        self.link1_line, = self.ax.plot([], [], lw=3)
        self.link2_line, = self.ax.plot([], [], lw=3)
        self.elbow_dot, = self.ax.plot([], [], 'o')
        self.tip_trace, = self.ax.plot([], [], lw=1)
        self.tip_dot, = self.ax.plot([], [], 'o', ms=4)

        # Texto informativo
        self.text = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes,
                                 va='top', ha='left')

        self.trace_x = []
        self.trace_y = []

        # Pose inicial plegada a la izquierda (θ1=π, θ2=0)
        (x1, y1), (x2, y2) = self.arm.fkine(math.pi, 0.0)
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

    def animate(self, fps: int):
        anim = FuncAnimation(self.fig, self._frame, frames=len(self.t), interval=1000.0 / fps, blit=True)
        plt.show()

# ==============================
# Configuración por defecto y ejecutable
# ==============================

def default_setup() -> Tuple[Arm2R, Canvas, TrajectoryPlanner]:
    # ---- Parámetros del brazo (cumple d1 + d2 ≥ 25√2 ≈ 35.36 cm)
    arm_params = ArmParams(d1=20.0 * CM, d2=18.0 * CM)  # puedes cambiarlos
    arm_params.check_reach_requirement()
    arm = Arm2R(arm_params)

    # ---- Lienzo 20×20
    canvas = Canvas()

    # ---- Trébol
    rose_spec = RoseSpec(
        n_leaves=5,                   # número de hojas
        amplitude=7.5 * CM,           # radio A; será ajustado si se sale del lienzo
        center=(10.0 * CM, 12.0 * CM),# centro; "altura" elegida por el usuario
        rotation_deg=0.0,             # rotación global
        use_cos=True
    )
    rose = RoseGenerator(rose_spec, canvas)

    # ---- Movimiento
    motion = MotionSpec(
        speed_cm_s=6.0,               # 1..10 cm/s
        fps=60,
        cycles=1,
        elbow_up=True,
        dwell_s=1.0
    )
    planner = TrajectoryPlanner(arm, rose, motion)
    return arm, canvas, planner

def run_sim(save_csv: Optional[str] = None):
    arm, canvas, planner = default_setup()

    # Planifica
    t, ref_xy, thetas = planner.build_time_series()

    # Simula
    sim = Simulator(arm, canvas, t, ref_xy, thetas, save_csv=save_csv)
    sim.animate(fps=planner.motion.fps)

# ==============================
# CLI mínima para cambiar parámetros rápidamente
# ==============================

def parse_cli_and_run(argv: List[str]):
    """
    Argumentos opcionales:
      --d1=cm --d2=cm --n=hojas --A=amplitud_cm --cx=cm --cy=cm --rot=grados
      --v=cm_s --fps=60 --cycles=1 --elbow=up|down --dwell=s
    """
    # Defaults
    arm, canvas, planner = default_setup()

    # parse
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
            planner.rose.spec.rotation_deg = float(arg.split("=", 1)[1])
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
        elif arg.startswith("--csv="):
            csv_path = arg.split("=", 1)[1]
            # defer writing after plan
        else:
            pass

    # Revalida alcance
    planner.arm.params.check_reach_requirement()

    # Planifica
    t, ref_xy, thetas = planner.build_time_series()

    # Simula
    csv_out = None
    for arg in argv:
        if arg.startswith("--csv="):
            csv_out = arg.split("=", 1)[1]
            break

    sim = Simulator(planner.arm, canvas, t, ref_xy, thetas, save_csv=csv_out)
    sim.animate(fps=planner.motion.fps)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        parse_cli_and_run(sys.argv[1:])
    else:
        run_sim(save_csv=None)
