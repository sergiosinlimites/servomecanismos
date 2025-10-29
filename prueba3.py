#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemelo digital 2R — Trébol interactivo (25×25), inicio plegado “a la izquierda”
y aproximación en dos etapas: (1) solo θ1 → derecha, (2) solo θ2 → punto más cercano.
Velocidad de dibujo constante.

Parametrización:
    r(θ) = E · R · (1 + M·sin(aθ + b))
    x(θ) = r(θ)·cos θ
    y(θ) = r(θ)·sin θ
Con sliders/inputs para a, b(°), M, R[cm], E.
"""

from dataclasses import dataclass
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, TextBox
from typing import Tuple, Optional, Callable

__VERSION__ = "7.7-two-stage-approach"
CM = 1.0
EPS = 1e-9

def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))

def wrap_pi(x: float) -> float:
    return (x + math.pi) % (2*math.pi) - math.pi

# ==============================
# Rangos UI
# ==============================

@dataclass
class UIRangeSpec:
    a_min: int = 1;     a_max: int = 12
    b_min: float = 0.0; b_max: float = 360.0
    M_min: float = 0.10; M_max: float = 0.40
    R_min: float = 5.0;  R_max: float = 15.0
    E_min: float = 1.0;  E_max: float = 1.20
    d1_min: float = 10.0; d1_max: float = 35.0
    d2_min: float = 10.0; d2_max: float = 35.0
    v_min: float = 0.5;  v_max: float = 20.0
    blend_min: float = 0.0; blend_max: float = 3.0
    wmax_min: float = 0.5; wmax_max: float = 12.0
    amax_min: float = 5.0; amax_max: float = 120.0
    cycles_min: int = 1;  cycles_max: int = 10

def get_ui_ranges() -> UIRangeSpec:
    return UIRangeSpec()

# ==============================
# Canvas y brazo
# ==============================

@dataclass
class Canvas:
    width: float = 25.0 * CM
    height: float = 25.0 * CM

@dataclass
class ArmParams:
    d1: float
    d2: float
    base: Tuple[float, float] = (-5.0 * CM, -5.0 * CM)
    def check_reach_requirement(self) -> None:
        required = 30.0 * math.sqrt(2.0) * CM * 0.8
        if self.d1 + self.d2 + 1e-6 < required:
            raise ValueError(
                f"Alcance bajo: d1+d2={self.d1+self.d2:.2f} cm. "
                f"Sug.: ≥ {required:.1f} cm para 25×25."
            )

class Arm2R:
    def __init__(self, params: ArmParams):
        self.params = params
    def fkine(self, t1: float, t2: float) -> Tuple[Tuple[float,float], Tuple[float,float]]:
        x0,y0 = self.params.base; d1,d2 = self.params.d1, self.params.d2
        x1 = x0 + d1*math.cos(t1); y1 = y0 + d1*math.sin(t1)
        x2 = x1 + d2*math.cos(t1+t2); y2 = y1 + d2*math.sin(t1+t2)
        return (x1,y1),(x2,y2)
    def ikine(self, x: float, y: float, elbow_up: bool=True) -> Optional[Tuple[float,float]]:
        x0,y0 = self.params.base; dx=x-x0; dy=y-y0; d1,d2=self.params.d1,self.params.d2
        r2=dx*dx+dy*dy; r=math.sqrt(r2)
        if r> d1+d2+1e-9 or r< abs(d1-d2)-1e-9: return None
        c2 = clamp((r2-d1*d1-d2*d2)/(2.0*d1*d2), -1.0, 1.0)
        s2 = math.sqrt(max(0.0,1.0-c2*c2));  s2 = s2 if elbow_up else -s2
        t2 = math.atan2(s2,c2); k1=d1+d2*c2; k2=d2*s2
        t1 = math.atan2(dy,dx)-math.atan2(k2,k1)
        return wrap_pi(t1), wrap_pi(t2)

# ==============================
# Trébol
# ==============================

@dataclass
class TrefoilSpec:
    a:int=4
    b:float=math.pi/2
    M:float=0.3
    R_cm:float=8.0*CM
    E:float=1.0
    center:Tuple[float,float]=(12.5*CM, 12.5*CM)

class TrefoilGenerator:
    def __init__(self, spec: TrefoilSpec, canvas: Canvas):
        self.spec=spec; self.canvas=canvas
    def _radius(self, th: np.ndarray) -> np.ndarray:
        s=self.spec
        return s.E * s.R_cm * (1.0 + s.M*np.sin(s.a*th + s.b))
    def curve_xy(self, N:int=4000) -> np.ndarray:
        th=np.linspace(0.0,2.0*np.pi,N,endpoint=False); r=self._radius(th)
        xc,yc=self.spec.center; x=xc+r*np.cos(th); y=yc+r*np.sin(th)
        return np.stack([x, y], axis=1)
    def fit_inside_canvas(self, margin:float=0.3*CM) -> None:
        xc,yc=self.spec.center
        maxR_x=min(xc-margin, self.canvas.width-xc-margin)
        maxR_y=min(yc-margin, self.canvas.height-yc-margin)
        maxR=max(0.0, min(maxR_x,maxR_y))
        denom = max(1e-9, self.spec.E*(1.0+self.spec.M))
        R_allowed = maxR / denom
        if self.spec.R_cm > R_allowed:
            self.spec.R_cm = R_allowed
    def ensure_min_diameter(self, diameter: float, margin: float = 0.3*CM) -> None:
        needed_R = (diameter/2.0) / max(1e-9, self.spec.E*(1.0+self.spec.M))
        self.spec.R_cm = max(self.spec.R_cm, needed_R)
        self.fit_inside_canvas(margin)

# ==============================
# Movimiento y límites
# ==============================

@dataclass
class MotionSpec:
    speed_cm_s: float = 6.0
    fps: int = 60
    cycles: int = 10
    elbow_up: bool = True
    dwell_s: float = 1.0
    blend_s: float = 1.0
    w_max: float = 6.0
    a_max: float = 50.0

# ==============================
# Planificador
# ==============================

class TrajectoryPlanner:
    """Genera (t, ref_xy, thetas) cumpliendo v constante en dibujo y límites ω/α."""
    def __init__(self, arm:Arm2R, tref:TrefoilGenerator, motion:MotionSpec):
        self.arm=arm; self.tref=tref; self.motion=motion

    @staticmethod
    def _arc_length(xy:np.ndarray) -> np.ndarray:
        d=np.diff(xy,axis=0,prepend=xy[0:1]); seg=np.hypot(d[:,0], d[:,1])
        s=np.cumsum(seg); s[0]=0.0; return s

    def _check_reach_loop(self, xy:np.ndarray) -> bool:
        x0,y0=self.arm.params.base; r=np.hypot(xy[:,0]-x0, xy[:,1]-y0)
        d1,d2=self.arm.params.d1,self.arm.params.d2
        return np.all(r>=abs(d1-d2)-1e-6) and np.all(r<=d1+d2+1e-6)

    def _ik_path(self, xy:np.ndarray, elbow_up:bool) -> np.ndarray:
        th=np.zeros((xy.shape[0],2)); alt=elbow_up
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

    def _scale_speed_for_limits(self, xy_loop:np.ndarray, s:np.ndarray, v:float) -> float:
        fps=self.motion.fps
        N=max(1,int((s[-1]/max(EPS,v))*fps))
        tau=(np.arange(N)/fps)*v
        s_query=np.mod(tau, s[-1])
        x=np.interp(s_query,s,xy_loop[:,0]); y=np.interp(s_query,s,xy_loop[:,1])
        th=self._ik_path(np.stack([x,y],axis=1), self.motion.elbow_up)
        w,a=self._profiles(th,fps)
        w_max_meas=np.max(np.abs(w)); a_max_meas=np.max(np.abs(a))
        factor_w=w_max_meas/max(EPS,self.motion.w_max)
        factor_a=math.sqrt(a_max_meas/max(EPS,self.motion.a_max))
        factor=max(1.0, factor_w, factor_a)
        return v/factor

    def _park_pose(self) -> Tuple[float,float]:
        """
        Parqueo “plegada a la izquierda”:
            θ1 = π
            γ = 0 si d2 ≤ d1;  γ = −arccos(d1/d2) si d2 > d1
            θ2 = wrap(γ − θ1)
        """
        d1=self.arm.params.d1; d2=self.arm.params.d2
        theta1 = math.pi
        if d2 <= d1:
            gamma = 0.0
        else:
            gamma = -math.acos(clamp(d1/d2, -1.0, 1.0))
        theta2 = wrap_pi(gamma - theta1)
        return theta1, theta2

    # ---------- utilidades de blend ----------
    @staticmethod
    def _quintic_blend(th0:np.ndarray, th1:np.ndarray, N:int) -> np.ndarray:
        if N<=1: return np.vstack([th1])
        u=np.linspace(0,1,N); s=10*u**3 - 15*u**4 + 6*u**5
        return th0 + s[:,None]*(th1-th0)

    @staticmethod
    def _quintic_scalar(a0:float, a1:float, N:int)->np.ndarray:
        if N<=1: return np.array([a1])
        u=np.linspace(0,1,N); s=10*u**3 - 15*u**4 + 6*u**5
        return a0 + s*(a1-a0)

    # ---------- selección de inicio para etapa 2 (θ1 fijo, solo θ2) ----------
    def _start_index_with_t1_fixed(self, xy:np.ndarray, t1_fixed:float) -> int:
        """
        Con θ1=t1_fixed (codo en (x1,y1) fijo), el tip se mueve sobre un círculo de radio d2.
        Elegimos el punto del trébol en la **mitad inferior** (y ≤ y_centro) que minimiza
        |dist((x,y),(x1,y1)) − d2|, es decir, el más cercano alcanzable con SOLO θ2.
        """
        (x1,y1), _ = self.arm.fkine(t1_fixed, 0.0)   # codo solo depende de θ1
        d2 = self.arm.params.d2
        yc = self.tref.spec.center[1]
        mask = (xy[:,1] <= yc + 1e-9)
        if not np.any(mask): mask = np.ones(len(xy), dtype=bool)
        r = np.hypot(xy[:,0]-x1, xy[:,1]-y1)
        err = np.abs(r - d2)
        err[~mask] = np.inf
        return int(np.argmin(err))

    def build(self):
        """
        Devuelve:
            t (N) [s], ref_xy (Nx2) [cm], thetas (Nx2) [rad],
            v_eff (cm/s), idx_start (int), draw_start_i (int).
        """
        # 1) Trébol y alcanzabilidad
        self.tref.ensure_min_diameter(20.0)
        self.tref.fit_inside_canvas()
        xy_loop=self.tref.curve_xy(N=4000)
        if not self._check_reach_loop(xy_loop):
            raise ValueError("Trayectoria del trébol inalcanzable con d1/d2 actuales.")

        # 2) Parámetros de velocidad constante
        s=self._arc_length(xy_loop); L=s[-1]
        v=clamp(self.motion.speed_cm_s, 0.5, 30.0)
        v=self._scale_speed_for_limits(xy_loop, s, v)
        fps=self.motion.fps; cycles=int(clamp(self.motion.cycles,1,10))

        # 3) Dwell + reparto del blend en 3 fases (θ1 solo, θ2 solo, micro-ajuste)
        N0=int(max(0.0,self.motion.dwell_s)*fps)
        Nblend=max(1, int(max(0.0,self.motion.blend_s)*fps))
        # proporciones: 50% θ1-solo, 40% θ2-solo, 10% micro-ajuste (ambas)
        N1=max(1,int(0.50*Nblend))
        N2=max(1,int(0.40*Nblend))
        N3=max(1, Nblend - N1 - N2)

        # 4) Poses
        th_park=np.array(self._park_pose())           # (π, ...), link1 ← ; link2 →
        t1_target = 0.0                               # “hacia la derecha” (1er–4º cuadrante)
        # índice de inicio cuando θ1 está fijo en t1_target y solo mueve θ2:
        idx_start = self._start_index_with_t1_fixed(xy_loop, t1_target)
        s0 = s[idx_start]
        x0, y0 = xy_loop[idx_start]

        # 5) Etapa 1: solo θ1 (θ2 fijo = θ2_park)
        t1_seq_1 = self._quintic_scalar(th_park[0], t1_target, N1)
        t2_seq_1 = np.full(N1, th_park[1])

        # 6) Etapa 2: con θ1=t1_target fijo, calcula θ2_hit que deja el tip sobre el
        #    punto del círculo más cercano al punto (x0,y0) elegido del trébol.
        (x1,y1), _ = self.arm.fkine(t1_target, 0.0)
        vdir = np.array([x0-x1, y0-y1])
        if np.hypot(*vdir) < 1e-9:
            ang_dir = 0.0
        else:
            ang_dir = math.atan2(vdir[1], vdir[0])
        # el tip buscado en la etapa 2 está en la dirección de vdir a distancia d2:
        t2_hit = wrap_pi(ang_dir - t1_target)
        t1_seq_2 = np.full(N2, t1_target)
        t2_seq_2 = self._quintic_scalar(th_park[1], t2_hit, N2)

        # 7) Micro-ajuste (opcional): llevar (θ1,θ2) exactamente a la IK del punto inicial
        th_first = np.array(self._ik_path(np.array([[x0, y0]]), self.motion.elbow_up)[0])
        th_start_approx = np.array([t1_target, t2_hit])
        th_seq_3 = self._quintic_blend(th_start_approx, th_first, N3)

        # 8) Cronograma total y trayectoria a v constante desde s0
        T_cycle=L/max(EPS,v); N_cycle=max(1,int(T_cycle*fps))
        N_total=N0 + N1 + N2 + N3 + cycles*N_cycle
        t=np.arange(N_total)/fps

        tau=(np.arange(N_cycle)/fps)*v
        s_query=np.mod(s0 + tau, L)
        xcyc=np.interp(s_query, s, xy_loop[:,0]); ycyc=np.interp(s_query, s, xy_loop[:,1])

        # 9) Componer series
        ref_xy=np.zeros((N_total,2)); thetas=np.zeros((N_total,2))

        # dwell
        for i in range(N0):
            thetas[i,:]=th_park
            (_, _), (x2,y2)=self.arm.fkine(th_park[0], th_park[1])
            ref_xy[i,:]=(x2,y2)

        # etapa 1
        off=N0
        for k in range(N1):
            t1=t1_seq_1[k]; t2=t2_seq_1[k]
            thetas[off+k,:]=(t1,t2)
            (_, _), (x2,y2)=self.arm.fkine(t1,t2)
            ref_xy[off+k,:]=(x2,y2)

        # etapa 2
        off+=N1
        for k in range(N2):
            t1=t1_seq_2[k]; t2=t2_seq_2[k]
            thetas[off+k,:]=(t1,t2)
            (_, _), (x2,y2)=self.arm.fkine(t1,t2)
            ref_xy[off+k,:]=(x2,y2)

        # micro-ajuste
        off+=N2
        for k in range(N3):
            t1,t2=th_seq_3[k]
            thetas[off+k,:]=(t1,t2)
            (_, _), (x2,y2)=self.arm.fkine(t1,t2)
            ref_xy[off+k,:]=(x2,y2)

        # ciclos a v constante
        off+=N3
        for c in range(cycles):
            start=off+c*N_cycle; end=start+N_cycle
            ref_xy[start:end,0]=xcyc; ref_xy[start:end,1]=ycyc
            th_cyc=self._ik_path(np.stack([xcyc,ycyc],axis=1), self.motion.elbow_up)
            thetas[start:end,:]=th_cyc

        draw_start_i = N0 + N1 + N2 + N3
        return t, ref_xy, thetas, v, idx_start, draw_start_i

# ==============================
# Perfiles
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
        ylabels=["Ángulo θ [°]","Velocidad ω [rad/s]","Aceleración α [rad/s²]","Jerk dα/dt [rad/s³]"]
        for ax,yl in zip(self.axs,ylabels): ax.set_ylabel(yl); ax.grid(True,alpha=0.3)
        self.axs[-1].set_xlabel("Tiempo [s]"); self.lines=None
    def update(self, t:np.ndarray, thetas:np.ndarray, fps:int) -> None:
        P=angular_profiles(thetas,fps)
        th1_deg = np.rad2deg(P["theta1"])
        th2_deg = np.rad2deg(P["theta2"])
        if self.lines is None:
            l1,=self.axs[0].plot(t, th1_deg, label="θ1"); l2,=self.axs[0].plot(t, th2_deg, label="θ2")
            l3,=self.axs[1].plot(t, P["omega1"]); l4,=self.axs[1].plot(t, P["omega2"])
            l5,=self.axs[2].plot(t, P["alpha1"]); l6,=self.axs[2].plot(t, P["alpha2"])
            l7,=self.axs[3].plot(t, P["jerk1"]);  l8,=self.axs[3].plot(t, P["jerk2"])
            self.lines=(l1,l2,l3,l4,l5,l6,l7,l8); self.axs[0].legend(loc="upper right")
        else:
            data=[th1_deg, th2_deg, P["omega1"], P["omega2"], P["alpha1"], P["alpha2"], P["jerk1"], P["jerk2"]]
            for line,y in zip(self.lines,data): line.set_data(t,y)
            for ax in self.axs: ax.relim(); ax.autoscale_view()
        self.fig.canvas.draw_idle()

# ==============================
# Simulador
# ==============================

class Simulator:
    def __init__(self, arm:Arm2R, canvas:Canvas):
        self.arm=arm; self.canvas=canvas
        self.fig=plt.figure(figsize=(12,7))
        self.fig.canvas.manager.set_window_title(f"Gemelo digital 2R — v{__VERSION__}")
        self.ax = self.fig.add_axes([0.06, 0.08, 0.45, 0.84])
        self._setup_scene()
        self.anim=None; self.running=False
        self.ax_status = self.fig.add_axes([0.56, 0.88, 0.38, 0.08])
        self.ax_status.axis("off")
        self.status_text = self.ax_status.text(0.02,0.5,"",ha="left",va="center",color="tab:red",fontsize=9)
        self.draw_start_i = 0
        self.trace_x=[]; self.trace_y=[]

    def _setup_scene(self) -> None:
        x0,y0=self.arm.params.base
        self.ax.set_aspect('equal','box')
        self.ax.set_xlim(-25, 25); self.ax.set_ylim(-25, 25)
        self.ax.set_xlabel("x [cm]"); self.ax.set_ylabel("y [cm]"); self.ax.set_title("Gemelo digital 2R")
        rect=plt.Rectangle((0,0),self.canvas.width,self.canvas.height,fill=False,lw=2,ls='-'); self.ax.add_patch(rect)
        self.ax.plot([x0],[y0],'o')
        (self.traj_line,) = self.ax.plot([],[],lw=1,alpha=0.3)
        (self.link1_line,) = self.ax.plot([],[],lw=3,color="C2")
        (self.link2_line,) = self.ax.plot([],[],lw=3,color="C3")
        (self.elbow_dot,) = self.ax.plot([],[],'o')
        (self.tip_trace,) = self.ax.plot([],[],lw=1)
        (self.tip_dot,)   = self.ax.plot([],[],'o',ms=4)
        self.text = self.ax.text(0.02,0.98,"",transform=self.ax.transAxes,va='top',ha='left')

    def set_status(self, msg: str) -> None:
        self.status_text.set_text(msg); self.fig.canvas.draw_idle()

    def load(self, t:np.ndarray, ref_xy:np.ndarray, thetas:np.ndarray, fps:int, draw_start_i:int) -> None:
        self.t=t; self.ref_xy=ref_xy; self.thetas=thetas; self.fps=fps; self.draw_start_i=int(draw_start_i)
        mask=(ref_xy[:,0]>=0)&(ref_xy[:,0]<=self.canvas.width)&(ref_xy[:,1]>=0)&(ref_xy[:,1]<=self.canvas.height)
        self.traj_line.set_data(ref_xy[mask,0], ref_xy[mask,1])
        self.trace_x.clear(); self.trace_y.clear()
        self._draw_pose(self.thetas[0,0], self.thetas[0,1])
        self.fig.canvas.draw_idle()

    def _draw_pose(self, t1:float, t2:float) -> None:
        (x1,y1),(x2,y2)=self.arm.fkine(t1,t2)
        self.link1_line.set_data([self.arm.params.base[0],x1],[self.arm.params.base[1],y1])
        self.link2_line.set_data([x1,x2],[y1,y2]); self.elbow_dot.set_data([x1],[y1]); self.tip_dot.set_data([x2],[y2])

    def _frame(self,i: int):
        t1,t2=self.thetas[i]; (x1,y1),(x2,y2)=self.arm.fkine(t1,t2)
        self.link1_line.set_data([self.arm.params.base[0],x1],[self.arm.params.base[1],y1])
        self.link2_line.set_data([x1,x2],[y1,y2]); self.elbow_dot.set_data([x1],[y1]); self.tip_dot.set_data([x2],[y2])
        # traza solo desde el inicio del ciclo (evita ver la aproximación)
        if i >= self.draw_start_i and 0<=x2<=self.canvas.width and 0<=y2<=self.canvas.height:
            self.trace_x.append(x2); self.trace_y.append(y2); self.tip_trace.set_data(self.trace_x,self.trace_y)
        self.text.set_text(f"t={self.t[i]:5.2f} s\nθ1={t1:+.2f} rad\nθ2={t2:+.2f} rad")
        if i == len(self.t)-1: self.running=False
        return (self.link1_line,self.link2_line,self.elbow_dot,self.tip_trace,self.tip_dot,self.text)

    def _init_anim(self):
        return (self.link1_line,self.link2_line,self.elbow_dot,self.tip_trace,self.tip_dot,self.text)

    def start(self) -> None:
        if self.anim is not None: del self.anim
        self.anim=FuncAnimation(self.fig,self._frame,frames=len(self.t),
                                init_func=self._init_anim, interval=1000.0/self.fps, blit=True, repeat=False)
        self.running=True

    def stop(self) -> None:
        if self.anim is not None and self.running:
            self.anim.event_source.stop()
        self.running=False

# ==============================
# App (UI derecha)
# ==============================

class App:
    def __init__(self):
        self.ranges = get_ui_ranges()
        self.canvas=Canvas()
        self.arm_params=ArmParams(d1=20.0*CM,d2=18.0*CM); self.arm_params.check_reach_requirement()
        self.arm=Arm2R(self.arm_params)
        self.tref_spec=TrefoilSpec(a=4,b=math.pi/2,M=0.3,R_cm=8.0*CM,E=1.0,
                                   center=(self.canvas.width/2, self.canvas.height/2))
        self.tref=TrefoilGenerator(self.tref_spec,self.canvas)
        self.motion=MotionSpec()
        self.sim=Simulator(self.arm,self.canvas)
        self.prof=ProfilesPlotter()
        self._build_controls_two_cols()
        self._try_plan_and_load()

    @staticmethod
    def _bind_slider_text(slider: Slider, textbox: TextBox,
                          cast: Callable[[float], float], fmt: str,
                          minv: float, maxv: float) -> None:
        updating=[False]
        def on_slider(val):
            if updating[0]: return
            updating[0]=True; textbox.set_val(fmt.format(val)); updating[0]=False
        def on_submit(text):
            try: v=cast(float(text))
            except Exception: v=slider.val
            v=clamp(v, minv, maxv)
            if updating[0]: return
            updating[0]=True; slider.set_val(v); textbox.set_val(fmt.format(v)); updating[0]=False
        slider.on_changed(on_slider); textbox.on_submit(on_submit); textbox.set_val(fmt.format(slider.val))

    def _build_controls_two_cols(self) -> None:
        left, bottom, width, height = 0.56, 0.06, 0.38, 0.80
        row_h, gap = 0.045, 0.012
        def add_row(y, label, vmin, vmax, vinit, is_int=False):
            ax_slider = self.sim.fig.add_axes([left, y, width*0.74, row_h])
            ax_text   = self.sim.fig.add_axes([left+width*0.76, y, width*0.20, row_h])
            s = Slider(ax_slider, label, vmin, vmax, valinit=vinit, valstep=1 if is_int else None)
            tb = TextBox(ax_text, ""); self._bind_slider_text(s, tb, (int if is_int else float),
                                                              ("{:.0f}" if is_int else "{:.2f}"), vmin, vmax)
            return s, tb, y-(row_h+gap)
        y = bottom + height - row_h; r = self.ranges
        self.s_a, self.tb_a,     y = add_row(y, "a (lóbulos)", r.a_min, r.a_max, 4, True)
        self.s_b, self.tb_b,     y = add_row(y, "b (°)",       r.b_min, r.b_max, 180.0, False)
        self.s_M, self.tb_M,     y = add_row(y, "M",           r.M_min, r.M_max, 0.30, False)
        self.s_R, self.tb_R,     y = add_row(y, "R [cm]",      r.R_min, r.R_max, 8.0, False)
        self.s_E, self.tb_E,     y = add_row(y, "E (escala)",  r.E_min, r.E_max, 1.00, False)
        self.s_d1,self.tb_d1,    y = add_row(y, "d1 [cm]", r.d1_min, r.d1_max, 20.0, False)
        self.s_d2,self.tb_d2,    y = add_row(y, "d2 [cm]", r.d2_min, r.d2_max, 18.0, False)
        self.s_v, self.tb_v,     y = add_row(y, "vel [cm/s]", r.v_min, r.v_max, 6.0, False)
        self.s_blend,self.tb_bl, y = add_row(y, "blend [s]",  r.blend_min, r.blend_max, 1.0, False)
        self.s_wmax,self.tb_wm,  y = add_row(y, "ω_max [rad/s]", r.wmax_min, r.wmax_max, 6.0, False)
        self.s_amax,self.tb_am,  y = add_row(y, "α_max [rad/s²]", r.amax_min, r.amax_max, 50.0, False)
        self.s_cycles,self.tb_c, y = add_row(y, "cycles", r.cycles_min, r.cycles_max, 10, True)
        btn_h = 0.06
        self.ax_start = self.sim.fig.add_axes([left, 0.015, width*0.28, btn_h])
        self.ax_stop  = self.sim.fig.add_axes([left+width*0.36, 0.015, width*0.28, btn_h])
        self.ax_reset = self.sim.fig.add_axes([left+width*0.72, 0.015, width*0.28, btn_h])
        self.btn_start = Button(self.ax_start,"Start", color="0.88", hovercolor="0.82")
        self.btn_stop  = Button(self.ax_stop, "Stop",  color="0.88", hovercolor="0.82")
        self.btn_reset = Button(self.ax_reset,"Reset", color="0.88", hovercolor="0.82")
        self.btn_start.on_clicked(self._on_start); self.btn_stop.on_clicked(self._on_stop); self.btn_reset.on_clicked(self._on_reset)

    def _read_sliders_into_state(self) -> None:
        self.tref_spec.a=int(self.s_a.val); self.tref_spec.b=np.deg2rad(self.s_b.val)
        self.tref_spec.M=float(self.s_M.val); self.tref_spec.R_cm=float(self.s_R.val); self.tref_spec.E=float(self.s_E.val)
        self.arm_params.d1=float(self.s_d1.val); self.arm_params.d2=float(self.s_d2.val)
        self.motion.speed_cm_s=float(self.s_v.val); self.motion.blend_s=float(self.s_blend.val)
        self.motion.w_max=float(self.s_wmax.val); self.motion.a_max=float(self.s_amax.val)
        self.motion.cycles=int(self.s_cycles.val)

    def _stats_text(self, v_eff: float, idx0: int) -> str:
        s = self.tref_spec
        dia = 2.0 * s.E * s.R_cm * (1.0 + s.M)
        return (
            f"OK — v={v_eff:.2f} cm/s, inicio idx={idx0}\n"
            f"Trébol: diámetro≈{dia:.2f} cm, a={s.a}, b={np.rad2deg(s.b):.1f}°, "
            f"M={s.M:.2f}, E={s.E:.2f}, R={s.R_cm:.2f} cm\n"
            f"Brazo: d1={self.arm_params.d1:.2f} cm, d2={self.arm_params.d2:.2f} cm; "
            f"Canvas: {self.canvas.width:.0f}×{self.canvas.height:.0f} cm; "
            f"cycles={self.motion.cycles}"
        )

    def _try_plan_and_load(self) -> bool:
        try:
            self.arm_params.check_reach_requirement()
            planner=TrajectoryPlanner(self.arm, self.tref, self.motion)
            t, ref_xy, thetas, v_eff, idx0, draw_start_i = planner.build()
            self.sim.load(t, ref_xy, thetas, self.motion.fps, draw_start_i)
            self.prof.update(t, thetas, self.motion.fps)
            self.sim.set_status(self._stats_text(v_eff, idx0))
            return True
        except Exception as e:
            self.sim.set_status(str(e))
            return False

    def _on_start(self, _evt) -> None:
        self.sim.stop(); self._read_sliders_into_state()
        if self._try_plan_and_load(): self.sim.start()

    def _on_stop(self, _evt) -> None:
        self.sim.stop()

    def _on_reset(self, _evt) -> None:
        for s in (self.s_a, self.s_b, self.s_M, self.s_R, self.s_E,
                  self.s_d1, self.s_d2, self.s_v, self.s_blend,
                  self.s_wmax, self.s_amax, self.s_cycles):
            s.reset()
        self.sim.stop(); self._read_sliders_into_state(); self._try_plan_and_load()

    def start(self) -> None:
        plt.show()

# ==============================
# Arranque
# ==============================

if __name__ == "__main__":
    App().start()
