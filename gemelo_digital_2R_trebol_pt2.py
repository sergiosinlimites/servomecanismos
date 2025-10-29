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
from matplotlib.widgets import Slider, Button
from typing import Tuple, Optional

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

        return t, ref_xy, thetas, v, idx0  # idx0 devuelto para depuración

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
        ylabels=["Ángulo θ [rad]","Velocidad ω [rad/s]","Aceleración α [rad/s²]","Jerk dα/dt [rad/s³]"]
        for ax,yl in zip(self.axs,ylabels): ax.set_ylabel(yl); ax.grid(True,alpha=0.3)
        self.axs[-1].set_xlabel("Tiempo [s]"); self.lines=None
    def update(self, t:np.ndarray, thetas:np.ndarray, fps:int):
        P=angular_profiles(t,thetas.shape[0]//(t[-1]/(1.0/fps)) if t[-1]>0 else fps) if False else angular_profiles(thetas,fps)
        if self.lines is None:
            l1,=self.axs[0].plot(t,P["theta1"],label="θ1"); l2,=self.axs[0].plot(t,P["theta2"],label="θ2")
            l3,=self.axs[1].plot(t,P["omega1"]); l4,=self.axs[1].plot(t,P["omega2"])
            l5,=self.axs[2].plot(t,P["alpha1"]); l6,=self.axs[2].plot(t,P["alpha2"])
            l7,=self.axs[3].plot(t,P["jerk1"]);  l8,=self.axs[3].plot(t,P["jerk2"])
            self.lines=(l1,l2,l3,l4,l5,l6,l7,l8); self.axs[0].legend(loc="upper right")
        else:
            data=[P["theta1"],P["theta2"],P["omega1"],P["omega2"],P["alpha1"],P["alpha2"],P["jerk1"],P["jerk2"]]
            for line,y in zip(self.lines,data): line.set_data(t,y)
            for ax in self.axs: ax.relim(); ax.autoscale_view()
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

        self.sim=Simulator(self.arm,self.canvas)
        self.prof=ProfilesPlotter()
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
        # Bloque 3: límites y ciclos
        ax_wmax  = fig.add_axes([0.10,0.14,0.32,0.03])
        ax_amax  = fig.add_axes([0.58,0.14,0.32,0.03])
        ax_cycles= fig.add_axes([0.34,0.09,0.32,0.03])
        # Botones
        ax_start = fig.add_axes([0.18,0.02,0.18,0.05])
        ax_stop  = fig.add_axes([0.41,0.02,0.18,0.05])
        ax_reset = fig.add_axes([0.64,0.02,0.18,0.05])

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

        self.btn_start.on_clicked(self._on_start)
        self.btn_stop.on_clicked(self._on_stop)
        self.btn_reset.on_clicked(self._on_reset)

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
            t, ref_xy, thetas, v_eff, idx0 = planner.build()
            self.sim.load(t, ref_xy, thetas, self.motion.fps)
            self.prof.update(t, thetas, self.motion.fps)
            self.sim.status.set_text(f"OK (v efectiva={v_eff:.2f} cm/s, inicio idx={idx0})")
            return True
        except Exception as e:
            self.sim.status.set_text(str(e))
            return False

    def _on_start(self, _evt):
        self.sim.stop()
        self._read_sliders_into_state()
        if self._try_plan_and_load():
            self.sim.start()

    def _on_stop(self, _evt):
        self.sim.stop()

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
