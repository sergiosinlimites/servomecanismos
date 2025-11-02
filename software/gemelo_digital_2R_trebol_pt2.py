#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemelo digital 2R — trébol interactivo con start/stop, punto inicial óptimo y velocidad constante.

- Trébol: x=(1+M*sin(a*θ+b))*cosθ, y=(1+M*sin(a*θ+b))*sinθ
  Defaults: a=4, b=π/2, M=0.3, scale_cm=7.5 cm, center=(10,10) cm.
- Diámetro mínimo de la figura: 20 cm (se ajusta sin salirse del lienzo 20×20).
- Inicio desde la pose recogida (θ1=π, θ2=0) y desde el **punto más cercano en la mitad inferior del trébol**.
- Velocidad lineal **constante** en el trazo (reparametrización por arco).
- Límites de ω y α mediante *time-scaling* global (la velocidad efectiva puede reducirse).
- UI: sliders a, b(°), M, escala [cm], d1, d2, vel[cm/s], blend[s], ω_max, α_max, cycles(1–10).
- Botones: Start (planifica/arranca), Stop (pausa), Reset (restaura sliders).
- Ventana extra con perfiles θ, ω, α, jerk.
"""

from dataclasses import dataclass
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, TextBox, CheckButtons
from typing import Tuple, Optional
import threading
import time
import csv
try:
    import serial  # pyserial
except Exception:
    serial = None
from datetime import datetime
import os

__VERSION__ = "6.2-start-stop"
CM = 1.0
EPS = 1e-9

def clamp(x, a, b): return max(a, min(b, x))

# ==============================
# Canvas y brazo
# ==============================

@dataclass
class Canvas:
    width: float = 20.0 * CM
    height: float = 20.0 * CM

@dataclass
class ArmParams:
    d1: float
    d2: float
    base: Tuple[float, float] = (-5.0 * CM, -5.0 * CM)
    def check_reach_requirement(self):
        required = 25.0 * math.sqrt(2.0) * CM
        if self.d1 + self.d2 + 1e-6 < required:
            raise ValueError(f"Alcance insuficiente: d1+d2={self.d1+self.d2:.2f} cm < 25√2≈{required:.2f} cm.")

class Arm2R:
    def __init__(self, params: ArmParams): self.params = params
    def fkine(self, t1: float, t2: float):
        x0,y0 = self.params.base; d1,d2 = self.params.d1, self.params.d2
        x1 = x0 + d1*math.cos(t1); y1 = y0 + d1*math.sin(t1)
        x2 = x1 + d2*math.cos(t1+t2); y2 = y1 + d2*math.sin(t1+t2)
        return (x1,y1),(x2,y2)
    def ikine(self, x: float, y: float, elbow_up=True):
        x0,y0 = self.params.base; dx=x-x0; dy=y-y0; d1,d2=self.params.d1,self.params.d2
        r2=dx*dx+dy*dy; r=math.sqrt(r2)
        if r> d1+d2+1e-9 or r< abs(d1-d2)-1e-9: return None
        c2 = clamp((r2-d1*d1-d2*d2)/(2.0*d1*d2), -1.0, 1.0)
        s2 = math.sqrt(max(0.0,1.0-c2*c2));  s2 = s2 if elbow_up else -s2
        t2 = math.atan2(s2,c2); k1=d1+d2*c2; k2=d2*s2
        t1 = math.atan2(dy,dx)-math.atan2(k2,k1)
        t1 = (t1+math.pi)%(2*math.pi)-math.pi; t2 = (t2+math.pi)%(2*math.pi)-math.pi
        return t1,t2

# ==============================
# Trébol (parametrización nueva)
# ==============================

@dataclass
class TrefoilSpec:
    a:int=4; b:float=math.pi/2; M:float=0.3; scale_cm:float=7.5*CM
    center:Tuple[float,float]=(10.0*CM,10.0*CM)  # centrado en el lienzo

class TrefoilGenerator:
    def __init__(self, spec: TrefoilSpec, canvas: Canvas):
        self.spec=spec; self.canvas=canvas
    def _radius(self, th: np.ndarray)->np.ndarray:
        s=self.spec; return s.scale_cm*(1.0 + s.M*np.sin(s.a*th + s.b))
    def curve_xy(self, N:int=4000)->np.ndarray:
        th=np.linspace(0.0,2.0*np.pi,N,endpoint=False); r=self._radius(th)
        xc,yc=self.spec.center; x=xc+r*np.cos(th); y=yc+r*np.sin(th)
        return np.stack([x,y],axis=1)
    def fit_inside_canvas(self, margin:float=0.3*CM):
        """Ajusta scale_cm para no salirse, dejando centro fijo."""
        xc,yc=self.spec.center
        maxR_x=min(xc-margin, self.canvas.width-xc-margin)
        maxR_y=min(yc-margin, self.canvas.height-yc-margin)
        maxR=max(0.0, min(maxR_x,maxR_y))
        allowed = maxR / max(1e-9, 1.0 + self.spec.M)  # r_max = scale*(1+M)
        if self.spec.scale_cm>allowed: self.spec.scale_cm=allowed
    def ensure_min_diameter(self, diameter: float, margin: float = 0.3*CM):
        """Fuerza diámetro externo ≥ 'diameter' si cabe; si no, lo máximo dentro del lienzo."""
        # r_max = scale*(1+M) ≥ diameter/2  ⇒  scale ≥ (diameter/2)/(1+M)
        needed_scale = (diameter/2.0) / max(1e-9, 1.0+self.spec.M)
        self.spec.scale_cm = max(self.spec.scale_cm, needed_scale)
        self.fit_inside_canvas(margin)

# ==============================
# Motion spec con límites y blend
# ==============================

@dataclass
class MotionSpec:
    speed_cm_s: float = 6.0
    fps: int = 60
    cycles: int = 10                 # por defecto 10 ciclos
    elbow_up: bool = True
    dwell_s: float = 1.0
    blend_s: float = 1.0             # aproximación quíntica al primer punto
    w_max: float = 6.0               # límite ω [rad/s]
    a_max: float = 50.0              # límite α [rad/s²]

# ==============================
# Planificador con time-scaling y punto inicial óptimo
# ==============================

class TrajectoryPlanner:
    def __init__(self, arm:Arm2R, tref:TrefoilGenerator, motion:MotionSpec):
        self.arm=arm; self.tref=tref; self.motion=motion

    @staticmethod
    def _arc_length(xy:np.ndarray)->np.ndarray:
        d=np.diff(xy,axis=0,prepend=xy[0:1]); seg=np.hypot(d[:,0], d[:,1])
        s=np.cumsum(seg); s[0]=0.0; return s

    def _check_reach_loop(self, xy:np.ndarray)->bool:
        x0,y0=self.arm.params.base; r=np.hypot(xy[:,0]-x0, xy[:,1]-y0)
        d1,d2=self.arm.params.d1,self.arm.params.d2
        return np.all(r>=abs(d1-d2)-1e-6) and np.all(r<=d1+d2+1e-6)

    def _ik_path(self, xy:np.ndarray, elbow_up:bool)->np.ndarray:
        th=np.zeros((xy.shape[0],2))
        alt=elbow_up
        for i,(x,y) in enumerate(xy):
            sol=self.arm.ikine(x,y,elbow_up=alt)
            if sol is None:
                alt=not alt; sol=self.arm.ikine(x,y,elbow_up=alt)
            if sol is None: raise RuntimeError(f"Punto inalcanzable idx {i}")
            th[i]=sol
        return th

    def _profiles(self, th:np.ndarray, fps:int):
        dt=1.0/fps; w=np.gradient(th,dt,axis=0); a=np.gradient(w,dt,axis=0)
        return w,a

    def _scale_speed_for_limits(self, xy_loop:np.ndarray, s:np.ndarray, v:float)->float:
        """Reduce v si es necesario para cumplir ω/α. Después v se mantiene **constante** en todo el trazo."""
        fps=self.motion.fps
        N=int((s[-1]/v)*fps); N=max(1,N)
        tau=(np.arange(N)/fps)*v
        s_query=np.mod(tau, s[-1])
        x=np.interp(s_query,s,xy_loop[:,0]); y=np.interp(s_query,s,xy_loop[:,1])
        th=self._ik_path(np.stack([x,y],axis=1), self.motion.elbow_up)
        w,a=self._profiles(th,fps)
        w_max_meas=np.max(np.abs(w),axis=0)
        a_max_meas=np.max(np.abs(a),axis=0)
        denom_w=max(EPS,self.motion.w_max); denom_a=max(EPS,self.motion.a_max)
        factor_w=np.max(w_max_meas/denom_w)
        factor_a=np.max(np.sqrt(a_max_meas/denom_a))
        factor=max(1.0, factor_w, factor_a)
        return v/factor

    @staticmethod
    def _quintic_blend(th0:np.ndarray, th1:np.ndarray, N:int)->np.ndarray:
        if N<=1: return np.vstack([th1])
        u=np.linspace(0,1,N); s=10*u**3 - 15*u**4 + 6*u**5
        return th0 + s[:,None]*(th1-th0)

    def _closest_start_index_bottom(self, xy:np.ndarray)->int:
        """Índice de inicio: punto **más cercano** a la punta en parqueo, restringido a y ≤ y_centro."""
        (_, _), (xpark,ypark) = self.arm.fkine(math.pi, 0.0)
        yc = self.tref.spec.center[1]
        mask = xy[:,1] <= yc + 1e-9  # mitad inferior
        if not np.any(mask): mask = np.ones(len(xy), dtype=bool)
        d2 = (xy[:,0]-xpark)**2 + (xy[:,1]-ypark)**2
        d2[~mask] = np.inf
        idx = int(np.argmin(d2))
        return idx

    def build(self):
        # 1) tamaño y ajuste al lienzo
        self.tref.ensure_min_diameter(20.0)
        self.tref.fit_inside_canvas()
        xy_loop=self.tref.curve_xy(N=4000)
        if not self._check_reach_loop(xy_loop):
            raise ValueError("Trayectoria del trébol inalcanzable con d1,d2 actuales.")

        # 2) reparametrización por arco
        s=self._arc_length(xy_loop); L=s[-1]

        # 3) punto inicial óptimo (parte baja + más cercano al parqueo)
        idx0 = self._closest_start_index_bottom(xy_loop)
        s0 = s[idx0]  # desplazamiento inicial sobre el arco
        x0, y0 = xy_loop[idx0]

        # 4) velocidad efectiva (cumple ω/α) y muestreo a velocidad CONSTANTE
        v=clamp(self.motion.speed_cm_s, 0.5, 30.0)
        v=self._scale_speed_for_limits(xy_loop, s, v)  # mantiene v constante después
        fps=self.motion.fps; cycles=int(clamp(self.motion.cycles,1,10))
        T_cycle=L/max(EPS,v); N_cycle=max(1,int(T_cycle*fps))
        N0=int(max(0.0,self.motion.dwell_s)*fps)
        Nblend=max(1, int(max(0.0,self.motion.blend_s)*fps))
        N_total=N0 + Nblend + cycles*N_cycle
        t=np.arange(N_total)/fps

        # 5) referencia xy para un ciclo completo empezando en s0
        tau=(np.arange(N_cycle)/fps)*v
        s_query=np.mod(s0 + tau, L)
        xcyc=np.interp(s_query, s, xy_loop[:,0]); ycyc=np.interp(s_query, s, xy_loop[:,1])

        # 6) IK del primer punto y blend quíntico desde parqueo
        th_park=np.array([math.pi, 0.0])
        th_first=self._ik_path(np.array([[x0, y0]]), self.motion.elbow_up)[0]
        th_blend=self._quintic_blend(th_park, th_first, Nblend)

        # 7) compón las series (dwell -> blend -> ciclos a V constante)
        ref_xy=np.zeros((N_total,2)); thetas=np.zeros((N_total,2))
        (_, _), (xpark,ypark)=self.arm.fkine(th_park[0], th_park[1])
        ref_xy[:N0,:]=(xpark,ypark); thetas[:N0,:]=th_park
        for i in range(Nblend):
            (x1,y1),(x2,y2)=self.arm.fkine(th_blend[i,0], th_blend[i,1])
            ref_xy[N0+i,:]=(x2,y2); thetas[N0+i,:]=th_blend[i,:]
        for c in range(cycles):
            start=N0+Nblend+c*N_cycle; end=start+N_cycle
            ref_xy[start:end,0]=xcyc; ref_xy[start:end,1]=ycyc
            th_cyc=self._ik_path(np.stack([xcyc,ycyc],axis=1), self.motion.elbow_up)
            thetas[start:end,:]=th_cyc

        start_idx_traj = N0 + Nblend  # inicio del primer ciclo (excluye posicionamiento inicial)
        return t, ref_xy, thetas, v, idx0, start_idx_traj, N_cycle  # idx0 devuelto para depuración

# ==============================
# Perfiles (θ, ω, α, jerk)
# ==============================

def angular_profiles(thetas: np.ndarray, fps: int):
    dt=1.0/max(1,fps)
    th1=thetas[:,0]; th2=thetas[:,1]
    w1=np.gradient(th1,dt); w2=np.gradient(th2,dt)
    a1=np.gradient(w1,dt); a2=np.gradient(w2,dt)
    j1=np.gradient(a1,dt); j2=np.gradient(a2,dt)
    return {"theta1":th1,"theta2":th2,"omega1":w1,"omega2":w2,"alpha1":a1,"alpha2":a2,"jerk1":j1,"jerk2":j2}

class ProfilesPlotter:
    def __init__(self):
        self.fig,self.axs=plt.subplots(4,1,figsize=(9,7),sharex=True)
        self.fig.canvas.manager.set_window_title("Perfiles articulares")
        ylabels=["Ángulo θ [°]","Velocidad ω [°/s]","Aceleración α [°/s²]","Jerk dα/dt [°/s³]"]
        for ax,yl in zip(self.axs,ylabels): ax.set_ylabel(yl); ax.grid(True,alpha=0.3)
        self.axs[-1].set_xlabel("Tiempo [s]"); self.lines=None; self.real_lines=None
        
    def update(self, t:np.ndarray, thetas:np.ndarray, fps:int, skip_initial:bool=True,
               start_index: int = None, cycle_frames: int = None):
        """
        Actualiza los perfiles con opción de omitir posicionamiento inicial.
        skip_initial: Si True, excluye los primeros segmentos (dwell + blend)
        """
        P=angular_profiles(thetas,fps)
        
        # Detectar dónde termina el posicionamiento inicial (cambio significativo en velocidad)
        if start_index is not None and 0 <= start_index < len(t):
            idx_start = int(start_index)
        elif skip_initial and len(t) > 120:  # Al menos 2 segundos @ 60fps
            omega_norm = np.sqrt(P["omega1"]**2 + P["omega2"]**2)
            # Buscar donde la velocidad se estabiliza en el trazo
            threshold = np.percentile(omega_norm, 10)  # 10% más bajo
            stable_start = np.where(omega_norm > threshold * 2)[0]
            if len(stable_start) > 0:
                idx_start = max(60, stable_start[0] - 10)  # Con margen
            else:
                idx_start = 0
        else:
            idx_start = 0
        
        # Recortar datos
        if cycle_frames is not None and cycle_frames > 0:
            idx_end = min(len(t), idx_start + int(cycle_frames))
        else:
            idx_end = len(t)
        t_plot = t[idx_start:idx_end]
        
        # Convertir a grados para visualización
        theta1_deg = np.rad2deg(P["theta1"][idx_start:idx_end])
        theta2_deg = np.rad2deg(P["theta2"][idx_start:idx_end])
        omega1_deg = np.rad2deg(P["omega1"][idx_start:idx_end])
        omega2_deg = np.rad2deg(P["omega2"][idx_start:idx_end])
        alpha1_deg = np.rad2deg(P["alpha1"][idx_start:idx_end])
        alpha2_deg = np.rad2deg(P["alpha2"][idx_start:idx_end])
        jerk1_deg = np.rad2deg(P["jerk1"][idx_start:idx_end])
        jerk2_deg = np.rad2deg(P["jerk2"][idx_start:idx_end])
        
        if self.lines is None:
            l1,=self.axs[0].plot(t_plot,theta1_deg,label="θ1"); l2,=self.axs[0].plot(t_plot,theta2_deg,label="θ2")
            l3,=self.axs[1].plot(t_plot,omega1_deg); l4,=self.axs[1].plot(t_plot,omega2_deg)
            l5,=self.axs[2].plot(t_plot,alpha1_deg); l6,=self.axs[2].plot(t_plot,alpha2_deg)
            l7,=self.axs[3].plot(t_plot,jerk1_deg);  l8,=self.axs[3].plot(t_plot,jerk2_deg)
            self.lines=(l1,l2,l3,l4,l5,l6,l7,l8); self.axs[0].legend(loc="upper right")
        else:
            data=[theta1_deg,theta2_deg,omega1_deg,omega2_deg,alpha1_deg,alpha2_deg,jerk1_deg,jerk2_deg]
            for line,y in zip(self.lines,data): line.set_data(t_plot,y)
            for ax in self.axs: ax.relim(); ax.autoscale_view()
        self.fig.canvas.draw_idle()

    def overlay_real(self, t_plot: np.ndarray, theta1_deg: np.ndarray, theta2_deg: np.ndarray):
        if self.real_lines is None:
            lr1,=self.axs[0].plot(t_plot, theta1_deg, 'b--', alpha=0.7, label="θ1 real")
            lr2,=self.axs[0].plot(t_plot, theta2_deg, 'r--', alpha=0.7, label="θ2 real")
            self.real_lines=(lr1,lr2)
            self.axs[0].legend(loc="upper right")
        else:
            self.real_lines[0].set_data(t_plot, theta1_deg)
            self.real_lines[1].set_data(t_plot, theta2_deg)
        for ax in self.axs:
            ax.relim(); ax.autoscale_view()
        self.fig.canvas.draw_idle()

# ==============================
# Análisis de ángulos (caracterización)
# ==============================

class AngleAnalyzer:
    def __init__(self):
        # Usar constrained_layout para evitar solapes entre ejes, colorbars y tabla
        self.fig = plt.figure(figsize=(14, 9), constrained_layout=True)
        self.fig.canvas.manager.set_window_title("Análisis de Ángulos - Caracterización")
        # Reservar más altura para la fila de la tabla para que no invada las gráficas
        gs = self.fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.25])
        
        # Fila 1: Histogramas de θ1 y θ2
        self.ax_hist1 = self.fig.add_subplot(gs[0, 0])
        self.ax_hist2 = self.fig.add_subplot(gs[0, 1])
        self.ax_hist2d = self.fig.add_subplot(gs[0, 2])
        
        # Fila 2: Demandas (velocidad y aceleración)
        self.ax_omega = self.fig.add_subplot(gs[1, 0])
        self.ax_alpha = self.fig.add_subplot(gs[1, 1])
        self.ax_effort = self.fig.add_subplot(gs[1, 2])
        
        # Fila 3: Estadísticas y tabla
        self.ax_stats = self.fig.add_subplot(gs[2, :])
        self.ax_stats.axis('off')
        
    def analyze(self, t: np.ndarray, thetas: np.ndarray, fps: int,
                skip_initial: bool = True, start_index: int = None, cycle_frames: int = None):
        """Analiza y caracteriza los ángulos de la trayectoria."""
        P = angular_profiles(thetas, fps)
        
        # Detectar inicio del trazo (omitir posicionamiento)
        idx_start = 0
        if start_index is not None and 0 <= start_index < len(t):
            idx_start = int(start_index)
        elif skip_initial and len(t) > 120:
            omega_norm = np.sqrt(P["omega1"]**2 + P["omega2"]**2)
            threshold = np.percentile(omega_norm, 10)
            stable_start = np.where(omega_norm > threshold * 2)[0]
            if len(stable_start) > 0:
                idx_start = max(60, stable_start[0] - 10)
        
        # Determinar fin (solo primer ciclo si se solicita)
        if cycle_frames is not None and cycle_frames > 0:
            idx_end = min(len(t), idx_start + int(cycle_frames))
        else:
            idx_end = len(t)
        # Datos en grados para análisis (recortados)
        th1_deg = np.rad2deg(P["theta1"][idx_start:idx_end])
        th2_deg = np.rad2deg(P["theta2"][idx_start:idx_end])
        w1_deg = np.rad2deg(P["omega1"][idx_start:idx_end])
        w2_deg = np.rad2deg(P["omega2"][idx_start:idx_end])
        a1_deg = np.rad2deg(P["alpha1"][idx_start:idx_end])
        a2_deg = np.rad2deg(P["alpha2"][idx_start:idx_end])
        
        # Métricas de demanda
        omega_norm = np.sqrt(w1_deg**2 + w2_deg**2)
        alpha_norm = np.sqrt(a1_deg**2 + a2_deg**2)
        effort = omega_norm + 0.1 * alpha_norm  # Métrica combinada simple
        
        # ===== 1. HISTOGRAMA θ1 =====
        self.ax_hist1.clear()
        counts1, bins1, _ = self.ax_hist1.hist(th1_deg, bins=40, alpha=0.7, color='tab:blue', edgecolor='black')
        self.ax_hist1.set_xlabel('θ₁ [°]', fontsize=10)
        self.ax_hist1.set_ylabel('Frecuencia', fontsize=10)
        self.ax_hist1.set_title('Distribución de θ₁', fontsize=11, fontweight='bold')
        self.ax_hist1.grid(True, alpha=0.3)
        # Marcar ángulo más común
        most_common_idx1 = np.argmax(counts1)
        most_common_th1 = (bins1[most_common_idx1] + bins1[most_common_idx1 + 1]) / 2
        self.ax_hist1.axvline(most_common_th1, color='red', linestyle='--', linewidth=2, label=f'Más común: {most_common_th1:.1f}°')
        self.ax_hist1.legend(fontsize=9)
        
        # ===== 2. HISTOGRAMA θ2 =====
        self.ax_hist2.clear()
        counts2, bins2, _ = self.ax_hist2.hist(th2_deg, bins=40, alpha=0.7, color='tab:orange', edgecolor='black')
        self.ax_hist2.set_xlabel('θ₂ [°]', fontsize=10)
        self.ax_hist2.set_ylabel('Frecuencia', fontsize=10)
        self.ax_hist2.set_title('Distribución de θ₂', fontsize=11, fontweight='bold')
        self.ax_hist2.grid(True, alpha=0.3)
        most_common_idx2 = np.argmax(counts2)
        most_common_th2 = (bins2[most_common_idx2] + bins2[most_common_idx2 + 1]) / 2
        self.ax_hist2.axvline(most_common_th2, color='red', linestyle='--', linewidth=2, label=f'Más común: {most_common_th2:.1f}°')
        self.ax_hist2.legend(fontsize=9)
        
        # ===== 3. HISTOGRAMA 2D (θ1 vs θ2) =====
        self.ax_hist2d.clear()
        h, xedges, yedges, im = self.ax_hist2d.hist2d(th1_deg, th2_deg, bins=30, cmap='YlOrRd')
        self.ax_hist2d.set_xlabel('θ₁ [°]', fontsize=10)
        self.ax_hist2d.set_ylabel('θ₂ [°]', fontsize=10)
        self.ax_hist2d.set_title('Combinaciones θ₁-θ₂', fontsize=11, fontweight='bold')
        cbar = plt.colorbar(im, ax=self.ax_hist2d)
        cbar.set_label('Frecuencia', fontsize=9)
        # Marcar combinación más común
        max_idx = np.unravel_index(np.argmax(h), h.shape)
        combo_th1 = (xedges[max_idx[0]] + xedges[max_idx[0] + 1]) / 2
        combo_th2 = (yedges[max_idx[1]] + yedges[max_idx[1] + 1]) / 2
        self.ax_hist2d.plot(combo_th1, combo_th2, 'b*', markersize=15, label=f'Más común: ({combo_th1:.1f}°, {combo_th2:.1f}°)')
        self.ax_hist2d.legend(fontsize=8)
        
        # ===== 4. DEMANDA DE VELOCIDAD =====
        self.ax_omega.clear()
        self.ax_omega.scatter(th1_deg, th2_deg, c=omega_norm, s=5, cmap='viridis', alpha=0.6)
        cbar_w = plt.colorbar(self.ax_omega.collections[0], ax=self.ax_omega)
        cbar_w.set_label('|ω| [°/s]', fontsize=9)
        self.ax_omega.set_xlabel('θ₁ [°]', fontsize=10)
        self.ax_omega.set_ylabel('θ₂ [°]', fontsize=10)
        self.ax_omega.set_title('Demanda de Velocidad', fontsize=11, fontweight='bold')
        self.ax_omega.grid(True, alpha=0.3)
        # Marcar puntos de mayor velocidad
        top_omega_idx = np.argsort(omega_norm)[-5:]  # Top 5
        self.ax_omega.scatter(th1_deg[top_omega_idx], th2_deg[top_omega_idx], 
                             c='red', s=50, marker='x', linewidths=2, label='Top 5 velocidad')
        self.ax_omega.legend(fontsize=8)
        
        # ===== 5. DEMANDA DE ACELERACIÓN =====
        self.ax_alpha.clear()
        self.ax_alpha.scatter(th1_deg, th2_deg, c=alpha_norm, s=5, cmap='plasma', alpha=0.6)
        cbar_a = plt.colorbar(self.ax_alpha.collections[0], ax=self.ax_alpha)
        cbar_a.set_label('|α| [°/s²]', fontsize=9)
        self.ax_alpha.set_xlabel('θ₁ [°]', fontsize=10)
        self.ax_alpha.set_ylabel('θ₂ [°]', fontsize=10)
        self.ax_alpha.set_title('Demanda de Aceleración', fontsize=11, fontweight='bold')
        self.ax_alpha.grid(True, alpha=0.3)
        top_alpha_idx = np.argsort(alpha_norm)[-5:]
        self.ax_alpha.scatter(th1_deg[top_alpha_idx], th2_deg[top_alpha_idx], 
                             c='red', s=50, marker='x', linewidths=2, label='Top 5 aceleración')
        self.ax_alpha.legend(fontsize=8)
        
        # ===== 6. ESFUERZO COMBINADO =====
        self.ax_effort.clear()
        self.ax_effort.scatter(th1_deg, th2_deg, c=effort, s=5, cmap='coolwarm', alpha=0.6)
        cbar_e = plt.colorbar(self.ax_effort.collections[0], ax=self.ax_effort)
        cbar_e.set_label('Esfuerzo', fontsize=9)
        self.ax_effort.set_xlabel('θ₁ [°]', fontsize=10)
        self.ax_effort.set_ylabel('θ₂ [°]', fontsize=10)
        self.ax_effort.set_title('Esfuerzo Combinado (ω + 0.1α)', fontsize=11, fontweight='bold')
        self.ax_effort.grid(True, alpha=0.3)
        top_effort_idx = np.argsort(effort)[-5:]
        self.ax_effort.scatter(th1_deg[top_effort_idx], th2_deg[top_effort_idx], 
                              c='red', s=50, marker='x', linewidths=2, label='Top 5 esfuerzo')
        self.ax_effort.legend(fontsize=8)
        
        # ===== 7. TABLA DE ESTADÍSTICAS =====
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        # Calcular estadísticas
        stats_data = [
            ['PARÁMETRO', 'θ₁', 'θ₂', 'UNIDAD'],
            ['─' * 30, '─' * 15, '─' * 15, '─' * 10],
            ['Más común (histograma)', f'{most_common_th1:.2f}', f'{most_common_th2:.2f}', '°'],
            ['Combinación más frecuente', f'{combo_th1:.2f}', f'{combo_th2:.2f}', '°'],
            ['Promedio', f'{np.mean(th1_deg):.2f}', f'{np.mean(th2_deg):.2f}', '°'],
            ['Desv. estándar', f'{np.std(th1_deg):.2f}', f'{np.std(th2_deg):.2f}', '°'],
            ['Rango (max-min)', f'{np.ptp(th1_deg):.2f}', f'{np.ptp(th2_deg):.2f}', '°'],
            ['Mínimo', f'{np.min(th1_deg):.2f}', f'{np.min(th2_deg):.2f}', '°'],
            ['Máximo', f'{np.max(th1_deg):.2f}', f'{np.max(th2_deg):.2f}', '°'],
            ['', '', '', ''],
            ['Velocidad máxima |ω|', f'{np.max(np.abs(w1_deg)):.2f}', f'{np.max(np.abs(w2_deg)):.2f}', '°/s'],
            ['Velocidad promedio |ω|', f'{np.mean(np.abs(w1_deg)):.2f}', f'{np.mean(np.abs(w2_deg)):.2f}', '°/s'],
            ['Aceleración máxima |α|', f'{np.max(np.abs(a1_deg)):.2f}', f'{np.max(np.abs(a2_deg)):.2f}', '°/s²'],
            ['Aceleración promedio |α|', f'{np.mean(np.abs(a1_deg)):.2f}', f'{np.mean(np.abs(a2_deg)):.2f}', '°/s²'],
            ['', '', '', ''],
            ['CONDICIONES MÁS DEMANDANTES:', '', '', ''],
            ['Mayor velocidad en', f'{th1_deg[np.argmax(omega_norm)]:.2f}', f'{th2_deg[np.argmax(omega_norm)]:.2f}', '°'],
            ['Mayor aceleración en', f'{th1_deg[np.argmax(alpha_norm)]:.2f}', f'{th2_deg[np.argmax(alpha_norm)]:.2f}', '°'],
            ['Mayor esfuerzo en', f'{th1_deg[np.argmax(effort)]:.2f}', f'{th2_deg[np.argmax(effort)]:.2f}', '°'],
        ]
        
        table = self.ax_stats.table(cellText=stats_data, cellLoc='left', loc='center',
                                    colWidths=[0.4, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        # Reducir escala vertical para evitar que la tabla invada filas superiores
        table.scale(1, 1.3)
        
        # Estilo de header
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Estilo de secciones
        for row_idx in [15, ]:  # Filas de subtítulos
            for col_idx in range(4):
                table[(row_idx, col_idx)].set_facecolor('#E8F5E9')
                table[(row_idx, col_idx)].set_text_props(weight='bold')
        
        self.fig.canvas.draw_idle()

# ==============================
# Simulador + UI Start/Stop
# ==============================

class Simulator:
    def __init__(self, arm:Arm2R, canvas:Canvas):
        self.arm=arm; self.canvas=canvas
        self.fig,self.ax=plt.subplots(figsize=(7,7))
        self.fig.canvas.manager.set_window_title(f"Gemelo digital 2R — v{__VERSION__}")
        plt.subplots_adjust(bottom=0.42)
        self._setup_scene()
        self.anim=None; self.running=False

    def _setup_scene(self):
        x0,y0=self.arm.params.base
        self.ax.set_aspect('equal','box'); self.ax.set_xlim(-10,self.canvas.width+5); self.ax.set_ylim(-10,self.canvas.height+5)
        self.ax.set_xlabel("x [cm]"); self.ax.set_ylabel("y [cm]"); self.ax.set_title("Gemelo digital 2R")
        rect=plt.Rectangle((0,0),self.canvas.width,self.canvas.height,fill=False,lw=2,ls='-'); self.ax.add_patch(rect)
        self.ax.plot([x0],[y0],'o')
        (self.traj_line,) = self.ax.plot([],[],lw=1,alpha=0.3)
        (self.link1_line,) = self.ax.plot([],[],lw=3)
        (self.link2_line,) = self.ax.plot([],[],lw=3)
        (self.elbow_dot,) = self.ax.plot([],[],'o')
        (self.tip_trace,) = self.ax.plot([],[],lw=1)
        (self.tip_dot,)   = self.ax.plot([],[],'o',ms=4)
        self.text = self.ax.text(0.02,0.98,"",transform=self.ax.transAxes,va='top',ha='left')
        self.status = self.ax.text(0.5,-0.12,"",transform=self.ax.transAxes,ha='center',va='top',color="tab:red")
        self.trace_x=[]; self.trace_y=[]

    def load(self, t:np.ndarray, ref_xy:np.ndarray, thetas:np.ndarray, fps:int):
        self.t=t; self.ref_xy=ref_xy; self.thetas=thetas; self.fps=fps
        mask=(ref_xy[:,0]>=0)&(ref_xy[:,0]<=self.canvas.width)&(ref_xy[:,1]>=0)&(ref_xy[:,1]<=self.canvas.height)
        self.traj_line.set_data(ref_xy[mask,0], ref_xy[mask,1]); self.trace_x.clear(); self.trace_y.clear()
        (x1,y1),(x2,y2)=self.arm.fkine(math.pi,0.0)
        self.link1_line.set_data([self.arm.params.base[0],x1],[self.arm.params.base[1],y1])
        self.link2_line.set_data([x1,x2],[y1,y2]); self.elbow_dot.set_data([x1],[y1]); self.tip_dot.set_data([x2],[y2])
        self.fig.canvas.draw_idle()

    def _frame(self,i):
        t1,t2=self.thetas[i]; (x1,y1),(x2,y2)=self.arm.fkine(t1,t2)
        self.link1_line.set_data([self.arm.params.base[0],x1],[self.arm.params.base[1],y1])
        self.link2_line.set_data([x1,x2],[y1,y2]); self.elbow_dot.set_data([x1],[y1]); self.tip_dot.set_data([x2],[y2])
        if 0<=x2<=self.canvas.width and 0<=y2<=self.canvas.height:
            self.trace_x.append(x2); self.trace_y.append(y2); self.tip_trace.set_data(self.trace_x,self.trace_y)
        self.text.set_text(f"t={self.t[i]:5.2f} s\nθ1={t1:+.2f} rad\nθ2={t2:+.2f} rad")
        return (self.link1_line,self.link2_line,self.elbow_dot,self.tip_trace,self.tip_dot,self.text)

    def _init_anim(self):
        return (self.link1_line,self.link2_line,self.elbow_dot,self.tip_trace,self.tip_dot,self.text)

    def start(self):
        if self.anim is not None: del self.anim
        self.anim=FuncAnimation(self.fig,self._frame,frames=len(self.t),
                                init_func=self._init_anim, interval=1000.0/self.fps, blit=True, repeat=False)
        self.running=True

    def stop(self):
        if self.anim is not None and self.running:
            self.anim.event_source.stop()
        self.running=False

# ==============================
# Serial streaming a Arduino + Telemetría
# ==============================

class SerialStreamer:
    def __init__(self, port: str, baud: int, out_dir: str):
        self.port = port
        self.baud = baud
        self.out_dir = out_dir
        self.thread = None
        self._stop = threading.Event()
        self.telemetry_rows = []  # (pc_time_s, arduino_ms, q1, q2, q1_ref, q2_ref, u1, u2)
        self._ser = None

    def start(self, t: np.ndarray, thetas: np.ndarray) -> bool:
        if serial is None:
            return False
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.02)
        except Exception:
            return False
        self._stop.clear()
        self.thread = threading.Thread(target=self._run, args=(t, thetas), daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self._ser is not None:
            try:
                self._ser.write(b'S\n')
                self._ser.flush()
            except Exception:
                pass
            try:
                self._ser.close()
            except Exception:
                pass

    def _run(self, t: np.ndarray, thetas: np.ndarray):
        ser = self._ser
        # Cero rápido
        try:
            ser.write(b'Z\n'); ser.flush()
        except Exception:
            pass
        t0 = time.perf_counter()
        t_csv0 = float(t[0])
        ms0 = None
        buf = b''
        i = 0
        n = len(t)
        while not self._stop.is_set() and i < n:
            # timing
            tgt = (float(t[i]) - t_csv0)
            while not self._stop.is_set() and (time.perf_counter() - t0) < (tgt - 1e-4):
                # leer telemetría mientras esperamos
                try:
                    chunk = ser.read(128)
                    if chunk:
                        buf += chunk
                        while b'\n' in buf:
                            line, buf = buf.split(b'\n', 1)
                            line = line.decode('ascii', errors='ignore').strip()
                            if not line:
                                continue
                            if line.startswith('Y'):
                                parts = line.split(',')
                                if len(parts) >= 8:
                                    try:
                                        ar_ms = int(parts[1])
                                    except Exception:
                                        continue
                                    if ms0 is None:
                                        ms0 = ar_ms
                                    try:
                                        q1 = float(parts[2]); q2 = float(parts[3])
                                        q1r= float(parts[4]); q2r= float(parts[5])
                                        u1 = float(parts[6]); u2 = float(parts[7])
                                    except Exception:
                                        continue
                                    pc_time = time.perf_counter() - t0
                                    self.telemetry_rows.append((pc_time, ar_ms, q1, q2, q1r, q2r, u1, u2))
                except Exception:
                    pass
                time.sleep(0.0005)
            # enviar comando R
            try:
                cmd = f"R,{thetas[i,0]:.6f},{thetas[i,1]:.6f}\n".encode('ascii')
                ser.write(cmd)
            except Exception:
                pass
            i += 1

    def write_log(self, filepath: str):
        if not self.telemetry_rows:
            return
        header = ['pc_time_s','arduino_ms','q1','q2','q1_ref','q2_ref','u1','u2']
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(self.telemetry_rows)
        except Exception:
            pass

# ==============================
# Guardado de datos
# ==============================

def save_trajectory_data(t: np.ndarray, thetas: np.ndarray, 
                         arm_params: ArmParams, tref_spec: TrefoilSpec, 
                         motion: MotionSpec, v_eff: float, idx0: int,
                         start_index: int, cycle_frames: int,
                         output_dir: str = "data/trayectorias"):
    """Guarda parámetros en TXT y serie temporal θ₁,θ₂ en CSV."""
    # Asegurar ruta absoluta relativa al archivo actual (no al CWD)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir_abs = os.path.join(base_dir, output_dir)
    os.makedirs(out_dir_abs, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1) Archivo TXT con parámetros legibles
    txt_path = os.path.join(out_dir_abs, f"config_{timestamp}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("CONFIGURACIÓN DE TRAYECTORIA 2R - TRÉBOL\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("PARÁMETROS DEL TRÉBOL:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Número de hojas (lóbulos): {tref_spec.a}\n")
        f.write(f"  Parámetro b [grados]: {np.rad2deg(tref_spec.b):.2f}°\n")
        f.write(f"  Parámetro b [radianes]: {tref_spec.b:.4f} rad\n")
        f.write(f"  Parámetro M (modulación): {tref_spec.M:.3f}\n")
        f.write(f"  Escala [cm]: {tref_spec.scale_cm:.2f} cm\n")
        f.write(f"  Centro [cm]: ({tref_spec.center[0]:.2f}, {tref_spec.center[1]:.2f})\n\n")
        
        f.write("PARÁMETROS DEL BRAZO:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Longitud eslabón 1 (d1): {arm_params.d1:.2f} cm\n")
        f.write(f"  Longitud eslabón 2 (d2): {arm_params.d2:.2f} cm\n")
        f.write(f"  Base del brazo: ({arm_params.base[0]:.2f}, {arm_params.base[1]:.2f}) cm\n")
        f.write(f"  Alcance total (d1+d2): {arm_params.d1 + arm_params.d2:.2f} cm\n\n")
        
        f.write("PARÁMETROS DE MOVIMIENTO:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Velocidad lineal deseada: {motion.speed_cm_s:.2f} cm/s\n")
        f.write(f"  Velocidad lineal efectiva: {v_eff:.2f} cm/s\n")
        f.write(f"  Frecuencia de muestreo (fps): {motion.fps} Hz\n")
        f.write(f"  Número de ciclos: {motion.cycles}\n")
        f.write(f"  Configuración de codo: {'up' if motion.elbow_up else 'down'}\n")
        f.write(f"  Tiempo de blend inicial: {motion.blend_s:.2f} s\n")
        f.write(f"  Tiempo de espera (dwell): {motion.dwell_s:.2f} s\n\n")
        
        f.write("LÍMITES ARTICULARES:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  ω máxima: {motion.w_max:.2f} rad/s\n")
        f.write(f"  α máxima: {motion.a_max:.2f} rad/s²\n\n")
        
        f.write("INFORMACIÓN DE LA TRAYECTORIA:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Duración total: {t[-1]:.3f} s\n")
        f.write(f"  Número total de puntos: {len(t)}\n")
        f.write(f"  Índice de inicio en curva: {idx0}\n\n")
        
        f.write("CONVENCIÓN DE ÁNGULOS:\n")
        f.write("-" * 40 + "\n")
        f.write("  θ₁: ángulo del eslabón 1 medido desde la horizontal (eje +X)\n")
        f.write("  θ₂: ángulo relativo del eslabón 2 respecto al eslabón 1\n")
        f.write("       (θ₂ = 0 cuando los eslabones están colineales)\n\n")
        
        f.write("ARCHIVOS GENERADOS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Configuración: config_{timestamp}.txt\n")
        f.write(f"  Datos CSV: trajectory_{timestamp}.csv\n\n")
        f.write("=" * 60 + "\n")
    
    # 2) Archivo CSV con serie temporal
    csv_path = os.path.join(out_dir_abs, f"trajectory_{timestamp}.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("# Trayectoria 2R - Referencias angulares\n")
        f.write(f"# Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# theta1: ángulo eslabón 1 desde horizontal [rad]\n")
        f.write("# theta2: ángulo relativo eslabón 2 [rad]\n")
        # Metadatos para visualización (no afectan el consumo en Arduino)
        f.write(f"# start_index={start_index}\n")
        f.write(f"# cycle_frames={cycle_frames}\n")
        f.write(f"# fps={motion.fps}\n")
        f.write("time_s,theta1_rad,theta2_rad\n")
        for i in range(len(t)):
            f.write(f"{t[i]:.6f},{thetas[i,0]:.6f},{thetas[i,1]:.6f}\n")
    
    return txt_path, csv_path

# ==============================
# App principal
# ==============================

class App:
    def __init__(self):
        self.canvas=Canvas()
        self.arm_params=ArmParams(d1=20.0*CM,d2=18.0*CM); self.arm_params.check_reach_requirement()
        self.arm=Arm2R(self.arm_params)
        self.tref_spec=TrefoilSpec(a=4,b=math.pi/2,M=0.3,scale_cm=7.5*CM,center=(10.0*CM,10.0*CM))
        self.tref=TrefoilGenerator(self.tref_spec,self.canvas)
        self.motion=MotionSpec()  # cycles=10 por defecto
        # Serial
        self.serial_port = os.environ.get('SERIAL_PORT', 'COM3')
        self.serial_baud = int(os.environ.get('SERIAL_BAUD', '115200'))
        self.serial_enabled = True
        self.streamer: Optional[SerialStreamer] = None

        self.sim=Simulator(self.arm,self.canvas)
        self.prof=ProfilesPlotter()
        self.analyzer=AngleAnalyzer()  # Nueva ventana de análisis
        
        # Almacenar última trayectoria planificada
        self.last_t = None
        self.last_thetas = None
        self.last_v_eff = None
        self.last_idx0 = None
        self.last_start_idx = None
        
        self._build_controls()
        self._try_plan_and_load()  # plan inicial (no arranca animación)

    def _build_controls(self):
        fig=self.sim.fig
        # Bloque 1: trébol
        ax_a     = fig.add_axes([0.10,0.35,0.32,0.03])
        ax_bdeg  = fig.add_axes([0.10,0.30,0.32,0.03])
        ax_M     = fig.add_axes([0.10,0.25,0.32,0.03])
        ax_scale = fig.add_axes([0.10,0.20,0.32,0.03])
        # Bloque 2: brazo y movimiento
        ax_d1    = fig.add_axes([0.58,0.35,0.32,0.03])
        ax_d2    = fig.add_axes([0.58,0.30,0.32,0.03])
        ax_v     = fig.add_axes([0.58,0.25,0.32,0.03])
        ax_blend = fig.add_axes([0.58,0.20,0.32,0.03])
        # Bloque 3: límites y ciclos (ajustado para dejar espacio a controles Serial)
        ax_wmax  = fig.add_axes([0.10,0.17,0.32,0.03])
        ax_amax  = fig.add_axes([0.58,0.17,0.32,0.03])
        ax_cycles= fig.add_axes([0.34,0.12,0.32,0.03])
        # Botones (ligero ajuste de tamaño)
        ax_start = fig.add_axes([0.16,0.02,0.20,0.045])
        ax_stop  = fig.add_axes([0.40,0.02,0.20,0.045])
        ax_reset = fig.add_axes([0.64,0.02,0.20,0.045])

        # Serial controls (TextBox para puerto/baudios y Check para habilitar)
        ax_port = fig.add_axes([0.10,0.07,0.22,0.035])
        ax_baud = fig.add_axes([0.36,0.07,0.22,0.035])
        ax_chk  = fig.add_axes([0.62,0.07,0.26,0.04])

        self.s_a     = Slider(ax_a,    "a (lóbulos)", 1, 12, valinit=self.tref_spec.a, valstep=1)
        self.s_b_deg = Slider(ax_bdeg, "b (°)", 0, 360, valinit=np.rad2deg(self.tref_spec.b))
        self.s_M     = Slider(ax_M,    "M", 0.0, 0.95, valinit=self.tref_spec.M)
        self.s_scale = Slider(ax_scale,"escala [cm]", 1.0, 12.0, valinit=self.tref_spec.scale_cm)

        self.s_d1    = Slider(ax_d1,   "d1 [cm]", 10.0, 35.0, valinit=self.arm_params.d1)
        self.s_d2    = Slider(ax_d2,   "d2 [cm]", 10.0, 35.0, valinit=self.arm_params.d2)
        self.s_v     = Slider(ax_v,    "vel [cm/s]", 0.5, 20.0, valinit=self.motion.speed_cm_s)
        self.s_blend = Slider(ax_blend,"blend [s]", 0.0, 3.0, valinit=self.motion.blend_s)

        self.s_wmax  = Slider(ax_wmax, "ω_max [rad/s]", 0.5, 12.0, valinit=self.motion.w_max)
        self.s_amax  = Slider(ax_amax, "α_max [rad/s²]", 5.0, 200.0, valinit=self.motion.a_max)
        self.s_cycles= Slider(ax_cycles,"cycles", 1, 10, valinit=self.motion.cycles, valstep=1)

        self.btn_start = Button(ax_start,"Start", color="0.88", hovercolor="0.82")
        self.btn_stop  = Button(ax_stop, "Stop",  color="0.88", hovercolor="0.82")
        self.btn_reset = Button(ax_reset,"Reset", color="0.88", hovercolor="0.82")

        self.txt_port = TextBox(ax_port, "Puerto ", initial=str(self.serial_port))
        self.txt_baud = TextBox(ax_baud, "Baudios ", initial=str(self.serial_baud))
        self.chk_serial = CheckButtons(ax_chk, ["Serial"], [self.serial_enabled])

        self.btn_start.on_clicked(self._on_start)
        self.btn_stop.on_clicked(self._on_stop)
        self.btn_reset.on_clicked(self._on_reset)

        def _on_port_submit(text):
            self.serial_port = text.strip() or self.serial_port
        def _on_baud_submit(text):
            try:
                self.serial_baud = int(text.strip())
            except Exception:
                pass
        def _on_chk(label):
            self.serial_enabled = not self.serial_enabled
        self.txt_port.on_submit(_on_port_submit)
        self.txt_baud.on_submit(_on_baud_submit)
        self.chk_serial.on_clicked(_on_chk)

    def _read_sliders_into_state(self):
        self.tref_spec.a=int(self.s_a.val)
        self.tref_spec.b=np.deg2rad(self.s_b_deg.val)
        self.tref_spec.M=float(self.s_M.val)
        self.tref_spec.scale_cm=float(self.s_scale.val)
        self.arm_params.d1=float(self.s_d1.val)
        self.arm_params.d2=float(self.s_d2.val)
        self.motion.speed_cm_s=float(self.s_v.val)
        self.motion.blend_s=float(self.s_blend.val)
        self.motion.w_max=float(self.s_wmax.val)
        self.motion.a_max=float(self.s_amax.val)
        self.motion.cycles=int(self.s_cycles.val)

    def _try_plan_and_load(self) -> bool:
        try:
            self.arm_params.check_reach_requirement()
            planner=TrajectoryPlanner(self.arm, self.tref, self.motion)
            t, ref_xy, thetas, v_eff, idx0, start_idx, cycle_frames = planner.build()
            
            # Guardar datos de la planificación
            self.last_t = t
            self.last_thetas = thetas
            self.last_v_eff = v_eff
            self.last_idx0 = idx0
            self.last_start_idx = start_idx
            self.last_cycle_frames = cycle_frames
            
            # Simulación: cargar la secuencia completa (incluye dwell + blend + todas las vueltas)
            self.sim.load(t, ref_xy, thetas, self.motion.fps)
            self.prof.update(t, thetas, self.motion.fps, skip_initial=True,
                             start_index=self.last_start_idx, cycle_frames=self.last_cycle_frames)
            self.analyzer.analyze(t, thetas, self.motion.fps, skip_initial=True,
                                  start_index=self.last_start_idx, cycle_frames=self.last_cycle_frames)
            self.sim.status.set_text(f"OK (v efectiva={v_eff:.2f} cm/s, inicio idx={idx0})")
            return True
        except Exception as e:
            self.sim.status.set_text(str(e))
            return False

    def _on_start(self, _evt):
        self.sim.stop()
        self._read_sliders_into_state()
        if self._try_plan_and_load():
            # Guardar archivos TXT y CSV con la trayectoria
            try:
                txt_path, csv_path = save_trajectory_data(
                    self.last_t, self.last_thetas,
                    self.arm_params, self.tref_spec, self.motion,
                    self.last_v_eff, self.last_idx0,
                    self.last_start_idx, self.last_cycle_frames
                )
                print(f"✓ Archivos guardados:")
                print(f"  - Configuración: {txt_path}")
                print(f"  - Trayectoria CSV: {csv_path}")
                try:
                    self.sim.status.set_text(f"Guardado: {os.path.basename(txt_path)}, {os.path.basename(csv_path)}")
                except Exception:
                    pass
            except Exception as e:
                print(f"⚠ Error al guardar archivos: {e}")
            # Iniciar streaming Serial → Arduino (si está habilitado)
            if self.serial_enabled:
                telem_path = os.path.join(os.path.dirname(txt_path), os.path.basename(txt_path).replace('config_', 'telemetry_').replace('.txt', '.csv'))
                self.streamer = SerialStreamer(self.serial_port, self.serial_baud, os.path.dirname(txt_path))
                started = self.streamer.start(self.last_t, self.last_thetas)
                if not started:
                    self.sim.status.set_text("Serial NO iniciado (revise puerto/pyserial)")
                else:
                    self.sim.status.set_text(f"Serial {self.serial_port}@{self.serial_baud} activo…")
            self.sim.start()

    def _on_stop(self, _evt):
        self.sim.stop()
        # Detener streamer y registrar telemetría + overlay
        if self.streamer is not None:
            self.streamer.stop()
            # Guardar telemetría
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'trayectorias')
            os.makedirs(out_dir, exist_ok=True)
            telem_file = os.path.join(out_dir, f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            self.streamer.write_log(telem_file)
            # Overlay de θ real en la primera vuelta
            if self.streamer.telemetry_rows and self.last_start_idx is not None and self.last_cycle_frames is not None:
                tel = np.array(self.streamer.telemetry_rows)
                tel_t = tel[:,0]  # pc_time_s relativo
                tel_q1 = tel[:,2]; tel_q2 = tel[:,3]
                # Segmento sim a graficar
                idx0 = int(self.last_start_idx)
                idx1 = int(min(len(self.last_t), idx0 + int(self.last_cycle_frames)))
                t_seg = self.last_t[idx0:idx1]
                t_rel = t_seg - t_seg[0]
                # Re-muestrear telemetría sobre t_rel
                # Asegurar monotonicidad
                if len(tel_t) >= 2 and (tel_t[-1] - tel_t[0]) > 1e-6:
                    q1r = np.interp(t_rel, tel_t - tel_t[0], tel_q1)
                    q2r = np.interp(t_rel, tel_t - tel_t[0], tel_q2)
                    self.prof.overlay_real(t_rel, np.rad2deg(q1r), np.rad2deg(q2r))
        self.streamer = None

    def _on_reset(self, _evt):
        for s in (self.s_a, self.s_b_deg, self.s_M, self.s_scale,
                  self.s_d1, self.s_d2, self.s_v, self.s_blend, self.s_wmax, self.s_amax, self.s_cycles):
            s.reset()
        self.sim.stop()
        self._read_sliders_into_state()
        self._try_plan_and_load()

    def start(self):
        plt.show()

# ==============================
# Arranque
# ==============================

if __name__ == "__main__":
    App().start()
