# 🌿 Simulación de Brazo Robótico 2DOF — Trayectoria en forma de Trébol

![1](https://github.com/user-attachments/assets/c9a99b39-a7e0-473f-85a2-2ee450f27526)


## 🧠 Descripción General

Este proyecto implementa un **gemelo digital** de un brazo robótico planar de **2 grados de libertad (2DOF)** que reproduce una trayectoria tipo **trébol de cuatro pétalos**.  
El sistema combina cinemática directa e inversa, interpolación spline cúbica y simulación de cuantización de encoders para representar con precisión la trayectoria real y la ideal del efector final.

El código principal es [`Servos_Trebol_Version_34.py`](Servos_Trebol_Version_34.py).

---

## ⚙️ 1. Parámetros del Robot

El modelo utiliza un brazo robótico **plano de dos eslabones**, con longitudes:

| Parámetro | Descripción | Valor |
|------------|--------------|--------|
| `L1` | Longitud del eslabón 1 | 235 mm |
| `L2` | Longitud del eslabón 2 | 165 mm |
| `BASE_X`, `BASE_Y` | Coordenadas base del robot | (-170, -170) mm |

El radio máximo de alcance es:

\[
R_{max} = L_1 + L_2 = 400 \text{ mm}
\]

---

## 🔩 2. Cinemática Directa

Dadas las articulaciones \( q_1 \) y \( q_2 \) (en grados), la posición del efector final \((x, y)\) se obtiene por:

\[
\begin{cases}
x = x_b + L_1 \cos(q_1) + L_2 \cos(q_1 + q_2) \\
y = y_b + L_1 \sin(q_1) + L_2 \sin(q_1 + q_2)
\end{cases}
\]

Donde \((x_b, y_b)\) son las coordenadas de la base del robot.

---

## 🔁 3. Cinemática Inversa

Para un punto deseado \((x, y)\), los ángulos se determinan mediante:

\[
\cos(q_2) = \frac{x^2 + y^2 - L_1^2 - L_2^2}{2 L_1 L_2}
\]

Luego:

\[
\begin{cases}
q_2 = \arccos(\cos(q_2)) \\
q_1 = \arctan2(y, x) - \arctan2(L_2 \sin(q_2), L_1 + L_2 \cos(q_2))
\end{cases}
\]

El modelo usa la configuración **"codo abajo"**.

---

## 🎯 4. Trayectoria del Trébol

El trébol se genera combinando:
1. **Interpolación spline cúbica** sobre puntos medidos del contorno de medio pétalo.
2. **Simetría de 45°** para reflejar el pétalo.
3. **Rotaciones sucesivas de 90°** para construir los cuatro pétalos completos.

### Puntos base (en mm)

| x | y |
|---|---|
| 82 | 0 |
| 84 | 6 |
| 88 | 9 |
| 100 | 15 |
| 112 | 40 |
| 100 | 65 |
| 88 | 71 |
| 78 | 78 |

### Interpolación cúbica natural

\[
S_x(t), S_y(t) = \text{CubicSpline}(t, x), \text{CubicSpline}(t, y)
\]

Cada pétalo se genera evaluando \( S_x(t), S_y(t) \) y rotando 90° sucesivamente:

\[
R(\theta) =
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
\]

---

## ⚙️ 5. Simulación de Encoders

El sistema emula un **encoder AS5600 de 12 bits**, con resolución:

\[
\text{Resolución} = \frac{360°}{4096} = 0.0879°/pulso
\]

Cada ángulo se cuantiza mediante:

\[
\theta_{\text{encoder}} = \left\lfloor \frac{\theta_{\text{ideal}}}{\text{resolución}} \right\rfloor \times \text{resolución}
\]

Esto introduce discretización y ruido angular, replicando el comportamiento real del sensor.

---

## 🧩 6. Integración de Trayectorias

La simulación combina dos fases:

1. **Giro inicial de 90°**
   - Movimiento interpolado linealmente en el espacio articular desde la posición base \((-153, -565)\) hasta el punto inicial del trébol \((82, 0)\).

2. **Dibujo del trébol**
   - Trayectoria cartesiana spline → cinemática inversa → encoder → cinemática directa → coordenadas reales.

---

## 🔍 7. Representación Gráfica

La figura muestra:

- **Trébol ideal** (verde punteado)
- **Trayectoria real (con encoder)** (rojo)
- **Trayectoria de giro inicial** (azul punteado)
- **Eslabones del robot** (negro y rojo)
- **Efector final** (estrella negra)

El código genera una **animación con Matplotlib** usando `FuncAnimation`.

---

## 📊 8. Resultados

- Trayectoria generada: **~2000 puntos (giro + trébol)**  
- Resolución angular simulada: **12 bits**
- Alcance máximo del robot: **400 mm**
- Precisión del trazado: ±1 mm

---

## 🧩 9. Requisitos

```bash
pip install numpy matplotlib scipy
