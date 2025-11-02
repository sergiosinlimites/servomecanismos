# Cómo Usar las Trayectorias Generadas

Este documento explica cómo usar los archivos TXT y CSV generados automáticamente al presionar el botón **Start** en `gemelo_digital_2R_trebol_pt2.py`.

## Archivos Generados

Cada vez que presionas **Start**, se crean dos archivos en la carpeta `trayectorias/`:

### 1. Archivo de Configuración (TXT)
**Nombre**: `config_YYYYMMDD_HHMMSS.txt`

Contiene todos los parámetros de configuración en formato legible:
- Número de hojas del trébol
- Parámetros del trébol (a, b, M, escala)
- Longitudes de los eslabones (d1, d2)
- Velocidad lineal (deseada y efectiva)
- Límites articulares (ω_max, α_max)
- Número de ciclos
- Duración total y número de puntos

**Propósito**: Documentación completa de la trayectoria para referencia futura.

### 2. Archivo de Trayectoria (CSV)
**Nombre**: `trajectory_YYYYMMDD_HHMMSS.csv`

Contiene la serie temporal con las referencias angulares:

```csv
time_s,theta1_rad,theta2_rad
0.000000,3.141593,0.000000
0.016667,3.141593,0.000000
0.033333,3.139821,0.001241
...
```

**Columnas**:
- `time_s`: Tiempo en segundos desde el inicio
- `theta1_rad`: Ángulo del eslabón 1 medido desde la horizontal [rad]
- `theta2_rad`: Ángulo relativo del eslabón 2 respecto al eslabón 1 [rad]

## Convención de Ángulos

### θ₁ (theta1)
- **Referencia**: Eje horizontal (+X)
- **Medición**: Desde la base del brazo
- **Signo**: Positivo en sentido antihorario (CCW)
- **Ejemplo**: θ₁ = 0 → eslabón 1 horizontal hacia la derecha

### θ₂ (theta2)
- **Referencia**: Eslabón 1 (relativo)
- **Medición**: Ángulo entre eslabón 1 y eslabón 2
- **Signo**: Positivo en sentido antihorario (CCW)
- **Importante**: θ₂ = 0 cuando los eslabones están **colineales** (extendidos)

**Ejemplo visual**:
```
θ₂ = 0:    ════════════════   (eslabones alineados)
θ₂ > 0:    ═══════╱           (codo arriba)
θ₂ < 0:    ═══════╲           (codo abajo)
```

## Cómo Usar con Arduino

### Opción 1: Streaming en Tiempo Real (RECOMENDADO)

El método más eficiente es usar el script Python original para enviar referencias por Serial:

```bash
python gemelo_2R_serial.py --port=COM3 --baud=115200 --log=run_telemetry.csv --no-anim
```

El Arduino recibe comandos `R,theta1,theta2` a la tasa configurada (60 fps por defecto).

**Ventajas**:
- No requiere memoria en Arduino
- Fácil sincronización
- Permite ajustes en tiempo real

**Ver**: `docs/software/SERIAL_BRIDGE.md` y `docs/firmware/README-FIRMWARE.md`

### Opción 2: Cargar Trayectoria en Arduino

Si necesitas que el Arduino funcione sin PC:

#### 2.1. Para trayectorias cortas (< 1000 puntos)
Copiar datos del CSV a un array en el sketch:

```cpp
struct TrajectoryPoint {
  float time_s;
  float theta1_rad;
  float theta2_rad;
};

const int NUM_POINTS = 600;
TrajectoryPoint trajectory[NUM_POINTS] = {
  {0.000000, 3.141593, 0.000000},
  {0.016667, 3.141593, 0.000000},
  // ... copiar del CSV
};
```

**Ver ejemplo**: `arduino/leer_trayectoria_ejemplo.ino`

#### 2.2. Para trayectorias largas
Usar una tarjeta SD:
- Guardar el CSV en la tarjeta SD
- Leer línea por línea con la librería `SD.h`
- Interpolar según el tiempo actual

### Opción 3: Procesar el CSV con un Script

Puedes procesar el CSV con Python/MATLAB/etc. para:
- Reducir el número de puntos (diezmado)
- Convertir a otros formatos
- Generar código C/Arduino automáticamente

**Ejemplo Python**:
```python
import pandas as pd
import numpy as np

# Leer CSV
df = pd.read_csv('trayectorias/trajectory_20250102_143022.csv', comment='#')

# Diezmar (1 de cada 4 puntos)
df_reduced = df.iloc[::4, :]

# Generar código Arduino
with open('trajectory_data.h', 'w') as f:
    f.write(f"const int NUM_POINTS = {len(df_reduced)};\n")
    f.write("TrajectoryPoint trajectory[NUM_POINTS] = {\n")
    for _, row in df_reduced.iterrows():
        f.write(f"  {{{row['time_s']:.6f}, {row['theta1_rad']:.6f}, {row['theta2_rad']:.6f}}},\n")
    f.write("};\n")
```

## Implementar Control PD en Arduino

Una vez que tienes las referencias θ₁_ref y θ₂_ref, implementa el control:

```cpp
// Leer sensores
float theta1_actual = readEncoder1();  // Implementar según tu hardware
float theta2_actual = readEncoder2();

// Calcular error
float e1 = theta1_ref - theta1_actual;
float e2 = theta2_ref - theta2_actual;

// Control PD
float u1 = Kp1 * e1 + Kd1 * (e1 - e1_prev) / dt;
float u2 = Kp2 * e2 + Kd2 * (e2 - e2_prev) / dt;

// Saturar [-1, 1]
u1 = constrain(u1, -1.0, 1.0);
u2 = constrain(u2, -1.0, 1.0);

// Aplicar a motores
driveMotor1(u1);
driveMotor2(u2);

// Guardar error previo
e1_prev = e1;
e2_prev = e2;
```

**Ver guía de sintonía**: `docs/software/CONTROL_PD_TUNING.md`

## Verificar la Trayectoria

### En Python
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('trayectorias/trajectory_20250102_143022.csv', comment='#')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
ax1.plot(df['time_s'], df['theta1_rad'], label='θ₁')
ax1.plot(df['time_s'], df['theta2_rad'], label='θ₂')
ax1.set_xlabel('Tiempo [s]')
ax1.set_ylabel('Ángulo [rad]')
ax1.legend()
ax1.grid(True)

# Velocidad angular
dt = df['time_s'].diff().mean()
omega1 = df['theta1_rad'].diff() / dt
omega2 = df['theta2_rad'].diff() / dt
ax2.plot(df['time_s'][1:], omega1[1:], label='ω₁')
ax2.plot(df['time_s'][1:], omega2[1:], label='ω₂')
ax2.set_xlabel('Tiempo [s]')
ax2.set_ylabel('Velocidad [rad/s]')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

## Preguntas Frecuentes

### ¿Por qué θ₂ = 0 cuando están alineados?
Esta es la convención de **ángulo relativo**. Facilita el control porque:
- Es más intuitivo mecánicamente
- Reduce el acoplamiento dinámico
- Simplifica la calibración

### ¿Cómo convertir a ángulo absoluto?
Si necesitas el ángulo absoluto del eslabón 2 respecto al eje X:
```
theta2_absoluto = theta1 + theta2
```

### ¿Cuántos puntos genera?
Depende de:
- Duración total = (ciclos × perímetro_trébol) / velocidad_efectiva
- Puntos = duración × fps

Ejemplo: 10 ciclos, perímetro ~50 cm, v=6 cm/s, fps=60 Hz
→ Duración ≈ 83 s → **~5000 puntos**

### ¿Cómo reducir el tamaño del archivo?
1. Reducir `cycles` (número de ciclos)
2. Reducir `fps` (de 60 a 30 Hz)
3. Diezmar el CSV después de generarlo
4. Usar streaming en tiempo real (no almacenar)

### ¿Los archivos se sobrescriben?
No. Cada ejecución crea archivos nuevos con timestamp único:
- `config_20250102_143022.txt`
- `trajectory_20250102_143022.csv`

### ¿Dónde se guardan?
En la carpeta `trayectorias/` dentro del proyecto. Se crea automáticamente si no existe.

## Ejemplos de Uso Completo

### 1. Generar y Visualizar
```bash
# 1. Ejecutar Python y ajustar sliders
python gemelo_digital_2R_trebol_pt2.py

# 2. Presionar Start → genera archivos

# 3. Visualizar en Python
python visualizar_trayectoria.py trayectorias/trajectory_YYYYMMDD_HHMMSS.csv
```

### 2. Generar y Cargar en Arduino
```bash
# 1. Generar trayectoria
python gemelo_digital_2R_trebol_pt2.py  # Ajustar y Start

# 2. Copiar datos a sketch o SD

# 3. Cargar sketch en Arduino
arduino-cli upload -p COM3 --fqbn arduino:avr:mega sketch_trayectoria

# 4. Enviar comando Start por Serial
echo "S" > COM3
```

### 3. Streaming Directo (sin archivos)
```bash
# Usar el script original con streaming en tiempo real
python gemelo_2R_serial.py --port=COM3 --baud=115200 --log=telemetria.csv
```

## Recursos Adicionales

- **Calibración de encoders**: `docs/base/CALIBRATION-ENCODERS.md`
- **Sintonía PD**: `docs/software/CONTROL_PD_TUNING.md`
- **Protocolo Serial**: `docs/firmware/README-FIRMWARE.md`
- **Troubleshooting**: `docs/support/TROUBLESHOOTING.md`

## Contacto y Soporte

Ver `docs/support/FAQ.md` para preguntas comunes adicionales.

