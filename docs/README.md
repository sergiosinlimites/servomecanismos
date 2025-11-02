# Gemelo Digital 2R — Trébol con Streaming a Arduino (NO BORRAR)

Este repositorio implementa un **gemelo digital** de un brazo planar 2R que dibuja un trébol (curva rosa) sobre un lienzo virtual de **20×20 cm** y permite enviar **referencias articulares (θ₁, θ₂)** a un **Arduino** para seguimiento con **control PD** por junta. También captura **telemetría** y compara **simulación vs real**.

## Objetivos
- Generar una trayectoria tipo **curva rosa** (“trébol”) con **velocidad lineal ~ constante**.
- Planificar la serie temporal de **ángulos articulares** con **IK** y continuidad de rama (evita flips).
- Enviar referencias a Arduino por **Serial** (115200 baudios) y ejecutar **PD** por junta.
- Registrar **telemetría** y obtener **métricas (RMSE)** y gráficas de **seguimiento**.

## Estructura
```
.
├─ README.md
├─ QUICKSTART.md
├─ ROADMAP.md
├─ CHANGELOG.md
├─ LICENSE.md
├─ docs/
│  └─ CALIBRATION-ENCODERS.md
├─ firmware/
│  ├─ README-FIRMWARE.md
│  ├─ SERIAL_PROTOCOL.md
│  ├─ CONTROL_PD_TUNING.md
│  ├─ BUILD_FLASH.md
│  └─ TELEMETRY_FORMAT.md
├─ software/
│  ├─ README-SOFTWARE.md
│  ├─ INSTALL.md
│  ├─ SIMULATION.md
│  ├─ SERIAL_BRIDGE.md
│  ├─ ANALYSIS.md
│  └─ DATA_LOGGING.md
└─ support/
   ├─ TROUBLESHOOTING.md
   └─ FAQ.md
```

## Archivos clave
- **`gemelo_digital_2R_trebol_pt2.py`** — Simulador interactivo con UI (guarda TXT y CSV al presionar Start).
- **`gemelo_2R_serial.py`** — Script principal (simulación + serial + análisis en tiempo real).
- **`visualizar_trayectoria.py`** — Herramienta para visualizar archivos CSV generados.
- **`csv_to_arduino.py`** — Conversor de CSV a arrays de Arduino (PROGMEM).
- **Sketch Arduino** — Ver **firmware/** y **SERIAL_PROTOCOL.md**.

## Cómo correr (vista rápida)

### Opción 1: Simulador Interactivo (genera archivos TXT/CSV)
```bash
python gemelo_digital_2R_trebol_pt2.py
```
- Ajusta parámetros con sliders
- Presiona **Start** → guarda automáticamente:
  - `trayectorias/config_YYYYMMDD_HHMMSS.txt` (parámetros legibles)
  - `trayectorias/trajectory_YYYYMMDD_HHMMSS.csv` (serie temporal θ₁, θ₂)
- Ver guía completa: **`docs/COMO_USAR_TRAYECTORIAS.md`**

### Opción 2: Streaming en Tiempo Real a Arduino
1. Python ≥ 3.9. Instala dependencias: `pip install numpy matplotlib pyserial pandas`.
2. Carga el sketch en Arduino (IDE o CLI). Ver **firmware/BUILD_FLASH.md**.
3. Ejecuta:
   ```bash
   python gemelo_2R_serial.py --port=COM3 --baud=115200 --log=run_telemetry.csv --no-anim
   ```
4. Para solo simular: `python gemelo_2R_serial.py`

### Herramientas Adicionales
```bash
# Visualizar trayectoria guardada
python visualizar_trayectoria.py trayectorias/trajectory_20250102_143022.csv

# Convertir CSV a código Arduino
python csv_to_arduino.py trayectorias/trajectory_20250102_143022.csv -o trajectory_data.h -d 2
```

## Licencia
Ver **LICENSE.md**.
