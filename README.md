# Proyecto 2R Trefoil — Estructura y Uso

## Estructura del Proyecto
```
.
├─ README.md                      # Este documento
├─ requirements.txt               # Dependencias Python
├─ data/
│  └─ trayectorias/               # Salidas TXT/CSV (ignoradas en git)
├─ software/                      # Código Python
│  ├─ gemelo_digital_2R_trebol_pt2.py     # Simulador interactivo (genera CSV/TXT)
│  ├─ visualizar_trayectoria.py           # Análisis/visualización de CSV
│  ├─ stream_csv_to_serial.py             # Streaming CSV → Arduino (R,th1,th2)
│  └─ csv_to_arduino.py                   # Conversión CSV → arrays Arduino (PROGMEM)
├─ arduino/                       # Firmware y ejemplos Arduino
│  ├─ control_2R_serial.ino                # Control P por Serial (radianes)
│  ├─ leer_trayectoria_ejemplo.ino         # Ejemplo básico con array
│  └─ 2r-joint-tracker/2r-joint-tracker.ino
└─ docs/                          # Documentación
   ├─ README.md
   ├─ GUIA_RAPIDA_TRAYECTORIAS.md
   ├─ NUEVAS_FUNCIONALIDADES.md
   └─ CAMBIOS_IMPLEMENTADOS.md
```

## Flujo de Trabajo
1) Ejecutar simulador y generar CSV/TXT
```bash
python software/gemelo_digital_2R_trebol_pt2.py
# Start → guarda en data/trayectorias/
```

2) Visualizar (todo por defecto; banderas para recortes)
```bash
python software/visualizar_trayectoria.py data/trayectorias/trajectory_YYYYMMDD_HHMMSS.csv
# Opciones:
#   --exclude-initial    Oculta posicionamiento inicial
#   --first-cycle        Solo primera vuelta
#   --no-analysis        Oculta ventana de caracterización
```

3) Enviar CSV a Arduino por Serial (radianes)
```bash
python software/stream_csv_to_serial.py --port COM3 --baud 115200 \
  --csv data/trayectorias/trajectory_YYYYMMDD_HHMMSS.csv
# Opcional: --first-cycle-only
```

4) (Opcional) Convertir CSV → arrays Arduino
```bash
python software/csv_to_arduino.py data/trayectorias/trajectory_YYYYMMDD_HHMMSS.csv \
  -o trajectory_data.h -d 2
```

## Streaming Python → Arduino
- Sketch: `arduino/control_2R_serial.ino`
- Protocolo (texto):
  - `R,theta1,theta2` en radianes
  - `P,Kp1,Kp2` para ganancias
  - `Z` cero rápido; `S` stop
- Telemetría (para logging):
  - `Y,<ms>,<q1>,<q2>,<q1_ref>,<q2_ref>,<u1>,<u2>`

## Instalación
```bash
pip install -r requirements.txt
```

## Notas
- Los CSV/TXT se guardan en `data/trayectorias/` y no se versionan.
- Las gráficas del simulador muestran solo la primera vuelta (sin posicionamiento). La animación sí muestra todo.
- El visor por defecto muestra todo (incluye posicionamiento y todas las vueltas), ajustable por flags.


