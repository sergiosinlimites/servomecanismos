# GUÍA RÁPIDA: Generación de Trayectorias para Arduino

Esta guía te muestra paso a paso cómo generar y usar las trayectorias del brazo 2R.

## 📋 Paso 1: Verificar Instalación

```bash
# Desde la carpeta del proyecto
pip install -r requirements.txt
```

Deberías tener instalado:
- numpy
- matplotlib
- pandas
- pyserial

## 🎮 Paso 2: Ejecutar el Simulador Interactivo

```bash
python gemelo_digital_2R_trebol_pt2.py
```

Se abrirán dos ventanas:
1. **Gemelo digital 2R**: Simulación visual con sliders
2. **Perfiles articulares**: Gráficas de θ, ω, α, jerk

## ⚙️ Paso 3: Ajustar Parámetros

Usa los sliders para configurar:

### Parámetros del Trébol:
- **a (lóbulos)**: Número de hojas del trébol (1-12)
- **b (°)**: Rotación del trébol (0-360°)
- **M**: Modulación de la forma (0-0.95)
- **escala [cm]**: Tamaño del trébol (1-12 cm)

### Parámetros del Brazo:
- **d1 [cm]**: Longitud del eslabón 1 (10-35 cm)
- **d2 [cm]**: Longitud del eslabón 2 (10-35 cm)

### Parámetros de Movimiento:
- **vel [cm/s]**: Velocidad lineal de la punta (0.5-20 cm/s)
- **blend [s]**: Tiempo de aproximación inicial (0-3 s)

### Límites Articulares:
- **ω_max [rad/s]**: Velocidad angular máxima (0.5-12 rad/s)
- **α_max [rad/s²]**: Aceleración angular máxima (5-200 rad/s²)
- **cycles**: Número de ciclos a ejecutar (1-10)

## ▶️ Paso 4: Generar y Guardar Trayectoria

1. **Ajusta los parámetros** con los sliders
2. **Presiona el botón "Start"**
3. Se guardan automáticamente:
   ```
   trayectorias/
   ├── config_YYYYMMDD_HHMMSS.txt        ← Parámetros legibles
   └── trajectory_YYYYMMDD_HHMMSS.csv    ← Datos θ₁, θ₂
   ```
4. En la consola verás:
   ```
   ✓ Archivos guardados:
     - Configuración: trayectorias\config_20250102_143022.txt
     - Trayectoria CSV: trayectorias\trajectory_20250102_143022.csv
   ```

## 📊 Paso 5: Visualizar la Trayectoria

```bash
python visualizar_trayectoria.py trayectorias/trajectory_20250102_143022.csv
```

Se mostrará:
- Gráficas de ángulos, velocidades, aceleraciones
- Estadísticas completas (rangos, máximos, etc.)

## 🔄 Paso 6: Convertir para Arduino

### Opción A: Usar Streaming (RECOMENDADO)
Envía las referencias en tiempo real por Serial:

```bash
python gemelo_2R_serial.py --port=COM3 --baud=115200 --log=telemetria.csv
```

### Opción B: Cargar en Arduino
Convierte el CSV a código C++:

```bash
python csv_to_arduino.py trayectorias/trajectory_20250102_143022.csv -o trajectory_data.h -d 2
```

Parámetros:
- `-o trajectory_data.h`: Archivo de salida
- `-d 2`: Diezmar por 2 (reduce puntos a la mitad)

El archivo `trajectory_data.h` contendrá:
```cpp
const int NUM_TRAJECTORY_POINTS = 2500;
const TrajectoryPoint trajectory[NUM_TRAJECTORY_POINTS] PROGMEM = {
  {0.000000, 3.141593, 0.000000},
  {0.033333, 3.141593, 0.000000},
  // ...
};
```

## 🤖 Paso 7: Usar en Arduino

### Método 1: Streaming desde PC (Recomendado)
Ver: `docs/firmware/README-FIRMWARE.md`

El Arduino recibe comandos `R,theta1,theta2` y ejecuta PD:
```cpp
// En el loop del Arduino (ya implementado en el sketch)
if (Serial.available()) {
  // Leer comando R,theta1,theta2
  // Ejecutar control PD por junta
  // Enviar telemetría Y,...
}
```

### Método 2: Array en Arduino
Incluye el header generado:
```cpp
#include "trajectory_data.h"

void setup() {
  // Inicializar motores y sensores
}

void loop() {
  // Leer punto actual según tiempo
  TrajectoryPoint point = getTrajectoryPoint(currentIndex);
  
  // Ejecutar control PD
  float e1 = point.theta1_rad - readEncoder1();
  float e2 = point.theta2_rad - readEncoder2();
  
  float u1 = Kp1 * e1 + Kd1 * (e1 - e1_prev) / dt;
  float u2 = Kp2 * e2 + Kd2 * (e2 - e2_prev) / dt;
  
  driveMotor1(u1);
  driveMotor2(u2);
}
```

Ver ejemplo completo: `arduino/leer_trayectoria_ejemplo.ino`

## 📖 Convención de Ángulos

**IMPORTANTE**: Entiende bien la convención para evitar errores:

### θ₁ (theta1)
```
     ^ y
     |
     |      /  ← θ₁ = 45°
     |    /
     |  /
     |/________> x
    base
```
- Medido desde el eje horizontal (+X)
- Positivo en sentido antihorario (CCW)

### θ₂ (theta2)
```
θ₂ = 0   (colineales):    ═══════════════

θ₂ > 0   (codo arriba):   ═══════╱

θ₂ < 0   (codo abajo):    ═══════╲
```
- Ángulo RELATIVO entre eslabón 1 y eslabón 2
- θ₂ = 0 cuando están alineados (extendidos)
- Positivo cuando el codo está arriba (CCW)

**Conversión a ángulo absoluto** (si lo necesitas):
```
theta2_absoluto = theta1 + theta2
```

## 🎯 Ejemplo Completo

```bash
# 1. Ejecutar simulador
python gemelo_digital_2R_trebol_pt2.py

# 2. Ajustar parámetros:
#    - a = 4 (trébol de 4 hojas)
#    - d1 = 20 cm, d2 = 18 cm
#    - vel = 6 cm/s
#    - cycles = 3

# 3. Presionar Start → genera archivos

# 4. Visualizar
python visualizar_trayectoria.py trayectorias/trajectory_20250102_143022.csv

# 5. (Opcional) Convertir para Arduino
python csv_to_arduino.py trayectorias/trajectory_20250102_143022.csv -o trajectory_data.h -d 2

# 6. Cargar en Arduino y ejecutar
```

## ⚠️ Problemas Comunes

### "Trayectoria inalcanzable"
- Aumenta d1 + d2 (debe ser > 25√2 ≈ 35.4 cm)
- Reduce la escala del trébol
- Verifica que el centro esté en (10, 10)

### "Archivos muy grandes"
- Reduce `cycles` (de 10 a 3)
- Reduce `fps` (de 60 a 30 Hz)
- Usa `-d 4` al convertir (diezmar por 4)
- Usa streaming en lugar de arrays

### "Arduino sin memoria"
- Usa PROGMEM (ya incluido en el conversor)
- Aumenta el diezmado `-d 4` o `-d 8`
- Considera usar tarjeta SD
- **Mejor**: usa streaming desde PC

## 📚 Documentación Adicional

- **Guía completa de trayectorias**: `docs/COMO_USAR_TRAYECTORIAS.md`
- **Control PD y sintonía**: `docs/software/CONTROL_PD_TUNING.md`
- **Calibración de encoders**: `docs/base/CALIBRATION-ENCODERS.md`
- **Troubleshooting**: `docs/support/TROUBLESHOOTING.md`
- **FAQ**: `docs/support/FAQ.md`

## 💡 Consejos

1. **Empieza simple**: Prueba primero con `cycles=1` y `vel=3 cm/s`
2. **Verifica alcance**: d1 + d2 debe ser suficiente (≥ 35.4 cm)
3. **Usa streaming**: Es más flexible que cargar arrays en Arduino
4. **Calibra primero**: Comando `Z` para establecer ceros articulares
5. **Sintoniza PD**: Empieza con Kp bajo, Kd muy bajo, y sube gradualmente

## 🚀 Siguiente Paso

Después de generar la trayectoria:
1. Calibrar encoders/potenciómetros (comando `Z`)
2. Sintonizar control PD (comando `P,Kp1,Kd1,Kp2,Kd2`)
3. Ejecutar trayectoria y registrar telemetría
4. Analizar RMSE y ajustar parámetros

¡Éxito con tu proyecto! 🎉

