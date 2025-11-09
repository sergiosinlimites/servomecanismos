import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline

# -- 1. Definición de los Parámetros del Robot y del Entorno --

# Longitudes de los eslabones en mm
L1 = 235
L2 = 165

# Posición base del robot en mm
BASE_X = -170
BASE_Y = -170

# Parámetros del Encoder AS5600 (12-bit)
POSICIONES_ENCODER = 4096
RESOLUCION_ANGULAR_ENCODER = 360.0 / POSICIONES_ENCODER

# Parámetros de los motores
VELOCIDAD_MAX_RPM = 120.0
VELOCIDAD_MAX_GRADOS_S = (VELOCIDAD_MAX_RPM / 60.0) * 360.0

# -- 2. Modelo Cinemático del Brazo Robótico (con "Codo Abajo") --

class BrazoRobotico2DOF:
    def __init__(self, l1, l2, base_x=0, base_y=0):
        self.l1 = l1
        self.l2 = l2
        self.base_x = base_x
        self.base_y = base_y
        self.max_reach = l1 + l2

    def forward_kinematics(self, theta1, theta2):
        theta1_rad = np.deg2rad(theta1)
        theta2_rad = np.deg2rad(theta2)
        x = self.base_x + self.l1 * np.cos(theta1_rad) + self.l2 * np.cos(theta1_rad + theta2_rad)
        y = self.base_y + self.l1 * np.sin(theta1_rad) + self.l2 * np.sin(theta1_rad + theta2_rad)
        return x, y

    def inverse_kinematics(self, x, y):
        # Trasladar al sistema de coordenadas local del robot
        x_local = x - self.base_x
        y_local = y - self.base_y
        
        d_sq = x_local**2 + y_local**2
        if d_sq > (self.l1 + self.l2)**2 * 0.999 or d_sq < (self.l1 - self.l2)**2 * 1.001:
            return None, None
        cos_theta2 = (d_sq - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)

        # Configuración "Codo Abajo"
        theta2_rad = np.arccos(cos_theta2)

        k1 = self.l1 + self.l2 * np.cos(theta2_rad)
        k2 = self.l2 * np.sin(theta2_rad)
        theta1_rad = np.arctan2(y_local, x_local) - np.arctan2(k2, k1)
        theta1_deg = np.rad2deg(theta1_rad)
        theta2_deg = np.rad2deg(theta2_rad)
        return theta1_deg, theta2_deg

# -- 3. Simulación del Encoder --

def simular_encoder(angulo_ideal, resolucion_grados):
    pulsos = np.floor(angulo_ideal / resolucion_grados)
    angulo_cuantizado = pulsos * resolucion_grados
    return angulo_cuantizado

# -- 4. Generación de la Trayectoria del Trébol de 4 Pétalos --

def generar_trayectoria_trebol_4petalos(num_puntos=2000):
    """
    Genera trayectoria del trébol de 4 pétalos usando interpolación spline cúbica
    basado en puntos de control y simetría rotacional.
    """
    # --- Datos originales (mitad del pétalo) en mm ---
    x = np.array([82, 84, 88, 100, 112, 100, 88, 78])
    y = np.array([0, 6, 9, 15, 40, 65, 71, 78])
    
    # --- Centro de simetría (línea de 45°) ---
    x_ref = y.copy()
    y_ref = x.copy()
    
    # --- Unir ambas mitades ---
    x_total = np.concatenate([x, x_ref[::-1]])
    y_total = np.concatenate([y, y_ref[::-1]])
    
    # --- Normalización del parámetro ---
    t = np.arange(len(x_total))
    puntos_por_petalo = num_puntos // 4
    tt = np.linspace(0, len(x_total) - 1, puntos_por_petalo, endpoint=False)
    
    # --- Interpolación spline cúbica ---
    cs_x = CubicSpline(t, x_total, bc_type='natural')
    cs_y = CubicSpline(t, y_total, bc_type='natural')
    
    xx = cs_x(tt)
    yy = cs_y(tt)
    
    # --- Replicar los 4 pétalos rotando ---
    x_completo = []
    y_completo = []
    
    for k in range(4):
        ang = k * 90  # rotaciones: 0°, 90°, 180°, 270°
        ang_rad = np.deg2rad(ang)
        
        # Matriz de rotación
        cos_ang = np.cos(ang_rad)
        sin_ang = np.sin(ang_rad)
        
        x_rotado = cos_ang * xx - sin_ang * yy
        y_rotado = sin_ang * xx + cos_ang * yy
        
        x_completo.extend(x_rotado)
        y_completo.extend(y_rotado)
    
    return np.array(x_completo), np.array(y_completo)

# -- 5. Generación de Trayectoria mediante Giro de Articulaciones --

def generar_trayectoria_por_giro_articulaciones(robot, q1_inicial, q2_inicial, q1_final, q2_final, num_puntos=100):
    """
    Genera una trayectoria interpolando directamente los ángulos de las articulaciones
    en lugar de interpolar puntos cartesianos.
    
    Args:
        robot: Instancia del robot
        q1_inicial, q2_inicial: Ángulos iniciales (grados)
        q1_final, q2_final: Ángulos finales (grados)
        num_puntos: Número de puntos en la trayectoria
    
    Returns:
        x_tray, y_tray: Coordenadas cartesianas de la trayectoria
        q1_tray, q2_tray: Ángulos de las articulaciones en la trayectoria
    """
    # Interpolar los ángulos linealmente
    q1_tray = np.linspace(q1_inicial, q1_final, num_puntos)
    q2_tray = np.linspace(q2_inicial, q2_final, num_puntos)
    
    # Calcular las posiciones cartesianas usando cinemática directa
    x_tray = []
    y_tray = []
    
    for q1, q2 in zip(q1_tray, q2_tray):
        x, y = robot.forward_kinematics(q1, q2)
        x_tray.append(x)
        y_tray.append(y)
    
    return np.array(x_tray), np.array(y_tray), q1_tray, q2_tray

# -- 6. Proceso Principal de Simulación --

robot = BrazoRobotico2DOF(L1, L2, BASE_X, BASE_Y)

print("Generando secuencia de inicio con giros de 90°...")

# Definir posición inicial: Eslabones alineados hacia abajo
# Punto cartesiano: (-153, -565)
punto_inicio_cartesiano = (-153, -565)

# Calcular los ángulos para la posición inicial
q1_inicial, q2_inicial = robot.inverse_kinematics(punto_inicio_cartesiano[0], punto_inicio_cartesiano[1])

if q1_inicial is None:
    print(f"ERROR: Punto inicial fuera de alcance: {punto_inicio_cartesiano}")
    exit()

print(f"Posición inicial: {punto_inicio_cartesiano}")
print(f"Ángulos iniciales: q1={q1_inicial:.2f}°, q2={q2_inicial:.2f}°")

# Definir posición final del giro: Inicio del trébol (82, 0)
punto_final_cartesiano = (82, 0)

# Calcular los ángulos para la posición final
q1_final, q2_final = robot.inverse_kinematics(punto_final_cartesiano[0], punto_final_cartesiano[1])

if q1_final is None:
    print(f"ERROR: Punto final fuera de alcance: {punto_final_cartesiano}")
    exit()

print(f"Posición final del giro: {punto_final_cartesiano}")
print(f"Ángulos finales: q1={q1_final:.2f}°, q2={q2_final:.2f}°")

# Calcular el cambio de ángulos
delta_q1 = q1_final - q1_inicial
delta_q2 = q2_final - q2_inicial

print(f"\nCambios angulares:")
print(f"Δq1 = {delta_q1:.2f}°")
print(f"Δq2 = {delta_q2:.2f}°")

# Generar la trayectoria del giro interpolando en el espacio de articulaciones
num_puntos_giro = 150
x_giro, y_giro, q1_giro, q2_giro = generar_trayectoria_por_giro_articulaciones(
    robot, q1_inicial, q2_inicial, q1_final, q2_final, num_puntos_giro
)

print(f"Trayectoria de giro generada con {num_puntos_giro} puntos")

# -- 6.2. Generar trayectoria del trébol --
print("\nGenerando trayectoria del trébol...")
x_trebol, y_trebol = generar_trayectoria_trebol_4petalos(num_puntos=1600)

# -- 6.3. Combinar trayectoria de giro + trayectoria del trébol --
x_ideal = np.concatenate([x_giro, x_trebol])
y_ideal = np.concatenate([y_giro, y_trebol])

# -- 6.4. Calcular cinemática inversa para toda la trayectoria --
puntos_reales_x, puntos_reales_y = [], []
angulos_reales_q1, angulos_reales_q2 = [], []

print(f"\nProcesando {len(x_ideal)} puntos totales (giro + trébol)...")
puntos_validos = 0
puntos_invalidos = 0

# Para la parte del giro, ya tenemos los ángulos calculados
for i in range(len(q1_giro)):
    q1_real = simular_encoder(q1_giro[i], RESOLUCION_ANGULAR_ENCODER)
    q2_real = simular_encoder(q2_giro[i], RESOLUCION_ANGULAR_ENCODER)
    angulos_reales_q1.append(q1_real)
    angulos_reales_q2.append(q2_real)
    x_real, y_real = robot.forward_kinematics(q1_real, q2_real)
    puntos_reales_x.append(x_real)
    puntos_reales_y.append(y_real)
    puntos_validos += 1

# Para la parte del trébol, calcular cinemática inversa
for i in range(len(x_trebol)):
    xi, yi = x_trebol[i], y_trebol[i]
    q1_ideal, q2_ideal = robot.inverse_kinematics(xi, yi)
    if q1_ideal is not None:
        q1_real = simular_encoder(q1_ideal, RESOLUCION_ANGULAR_ENCODER)
        q2_real = simular_encoder(q2_ideal, RESOLUCION_ANGULAR_ENCODER)
        angulos_reales_q1.append(q1_real)
        angulos_reales_q2.append(q2_real)
        x_real, y_real = robot.forward_kinematics(q1_real, q2_real)
        puntos_reales_x.append(x_real)
        puntos_reales_y.append(y_real)
        puntos_validos += 1
    else:
        puntos_invalidos += 1

print(f"Puntos válidos: {puntos_validos}")
print(f"Puntos fuera de alcance: {puntos_invalidos}")
print(f"Alcance máximo del robot: {L1 + L2} mm")
print(f"Puntos de giro: {len(q1_giro)}")
print(f"Puntos del trébol: {len(x_trebol)}")

# -- 7. Visualización y Animación --

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_aspect('equal')

# Ajustar límites para mostrar todo
margen = 50
ax.set_xlim(BASE_X - margen, BASE_X + L1 + L2 + margen)
ax.set_ylim(BASE_Y - 450, BASE_Y + L1 + L2 + margen)

ax.set_title("Simulación Trébol con Giro de 90° - Brazo Robótico 2DOF", 
             fontsize=14, fontweight='bold')
ax.set_xlabel("Eje X (mm)", fontsize=12)
ax.set_ylabel("Eje Y (mm)", fontsize=12)
ax.grid(True, alpha=0.3)

# Marcar el centro del dibujo (0, 0)
ax.plot(0, 0, 'go', markersize=8, label='Centro del dibujo (0,0)', zorder=5)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

# Marcar la base del robot
ax.plot(BASE_X, BASE_Y, 'bs', markersize=10, label=f'Base del robot ({BASE_X},{BASE_Y})', zorder=5)

# Marcar puntos clave
ax.plot(punto_inicio_cartesiano[0], punto_inicio_cartesiano[1], 'mo', markersize=10, 
        label=f'Inicio: {punto_inicio_cartesiano}', zorder=5)
ax.plot(punto_final_cartesiano[0], punto_final_cartesiano[1], 'co', markersize=10, 
        label=f'Inicio trébol: {punto_final_cartesiano}', zorder=5)

# Dibujar trayectorias ideales
ax.plot(x_giro, y_giro, 'b--', label='Trayectoria de Giro', alpha=0.5, linewidth=2)
ax.plot(x_trebol, y_trebol, 'g--', label='Trayectoria Ideal del Trébol', alpha=0.4, linewidth=1.5)

# Círculo de alcance máximo
workspace_circle = plt.Circle((BASE_X, BASE_Y), L1 + L2, color='blue', fill=False, 
                              linestyle=':', label=f'Alcance Máximo ({L1+L2} mm)', linewidth=2)
ax.add_artist(workspace_circle)

# Elementos animados
linea_eslabon1, = ax.plot([], [], 'o-', lw=6, color='black', markersize=10, label='Eslabón 1')
linea_eslabon2, = ax.plot([], [], 'o-', lw=6, color='red', markersize=8, label='Eslabón 2')
trayectoria_dibujada, = ax.plot([], [], 'r-', label='Trayectoria Real (con Encoder)', lw=2.5)
efector_final, = ax.plot([], [], 'k*', markersize=15, label='Efector Final')

# Texto para mostrar ángulos actuales (posición ajustada abajo a la izquierda)
texto_angulos = ax.text(0.02, 0.12, '', transform=ax.transAxes, 
                        verticalalignment='bottom', fontsize=10, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

def init():
    linea_eslabon1.set_data([], [])
    linea_eslabon2.set_data([], [])
    trayectoria_dibujada.set_data([], [])
    efector_final.set_data([], [])
    texto_angulos.set_text('')
    return linea_eslabon1, linea_eslabon2, trayectoria_dibujada, efector_final, texto_angulos

def animate(i):
    q1 = angulos_reales_q1[i]
    q2 = angulos_reales_q2[i]
    
    # Base del robot
    x0, y0 = BASE_X, BASE_Y
    
    # Posición del codo (final del eslabón 1)
    x1 = BASE_X + L1 * np.cos(np.deg2rad(q1))
    y1 = BASE_Y + L1 * np.sin(np.deg2rad(q1))
    
    # Posición del efector final
    x2, y2 = robot.forward_kinematics(q1, q2)
    
    linea_eslabon1.set_data([x0, x1], [y0, y1])
    linea_eslabon2.set_data([x1, x2], [y1, y2])
    trayectoria_dibujada.set_data(puntos_reales_x[:i+1], puntos_reales_y[:i+1])
    efector_final.set_data([x2], [y2])
    
    # Actualizar texto con ángulos actuales
    fase = "GIRO" if i < len(q1_giro) else "TRÉBOL"
    texto_angulos.set_text(f'Fase: {fase}\nq1: {q1:.1f}°\nq2: {q2:.1f}°')
    
    return linea_eslabon1, linea_eslabon2, trayectoria_dibujada, efector_final, texto_angulos

ax.legend(loc='upper right', fontsize=8, ncol=2)

# Crear la animación
print("\nGenerando animación...")
ani = FuncAnimation(fig, animate, frames=len(puntos_reales_x), 
                   init_func=init, blit=True, interval=30, repeat=True)

# Mostrar la animación en ventana interactiva
print("Mostrando animación (cierra la ventana para terminar)...")
plt.tight_layout()
plt.show()

