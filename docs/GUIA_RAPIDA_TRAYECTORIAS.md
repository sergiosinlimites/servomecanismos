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
python software/gemelo_digital_2R_trebol_pt2.py
```

Se abrirán tres ventanas:
1. **Gemelo digital 2R**: Simulación visual con sliders (muestra todo)
2. **Perfiles articulares**: Gráficas en grados (solo primera vuelta, sin inicial)
3. **Análisis de ángulos**: Histogramas/mapas/tabla (solo primera vuelta, sin inicial)

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
   data/trayectorias/
   ├── config_YYYYMMDD_HHMMSS.txt        ← Parámetros legibles
   └── trajectory_YYYYMMDD_HHMMSS.csv    ← Datos θ₁, θ₂
   ```

## 📊 Paso 5: Visualizar la Trayectoria

```bash
python software/visualizar_trayectoria.py data/trayectorias/trajectory_20250102_143022.csv
```

Por defecto muestra TODO (incluye inicial y todas las vueltas). Opciones:
- `--exclude-initial`
- `--first-cycle`
- `--no-analysis`

## 🔄 Paso 6: Enviar a Arduino (Streaming)

```bash
python software/stream_csv_to_serial.py --port COM3 --baud 115200 \
  --csv data/trayectorias/trajectory_20250102_143022.csv
```

- El Arduino debe correr `arduino/control_2R_serial.ino`
- Protocolo: `R,theta1,theta2` en radianes (más `P,`, `Z`, `S`)

## 🔧 Paso 7 (Opcional): Convertir para Arduino

```bash
python software/csv_to_arduino.py data/trayectorias/trajectory_20250102_143022.csv \
  -o trajectory_data.h -d 2
```

## 📖 Convención de Ángulos

- θ₁: desde la horizontal (+X) [rad]
- θ₂: relativo al eslabón 1 (0=colineales) [rad]

## ⚠️ Problemas Comunes

- "Trayectoria inalcanzable": sube d1+d2 o reduce escala
- "Archivos muy grandes": baja cycles/fps o usa `--first-cycle`
- "Arduino sin memoria": usa streaming o diezmado alto


