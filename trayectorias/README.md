# Carpeta de Trayectorias

Esta carpeta contiene los archivos de trayectoria generados automáticamente por `gemelo_digital_2R_trebol_pt2.py`.

## Archivos Generados

Cada vez que presionas **Start** en el simulador, se crean dos archivos:

### 1. Configuración (TXT)
**Formato**: `config_YYYYMMDD_HHMMSS.txt`

Contiene todos los parámetros de configuración en formato legible:
- Parámetros del trébol (hojas, modulación, escala)
- Parámetros del brazo (d1, d2, base)
- Parámetros de movimiento (velocidad, fps, ciclos)
- Límites articulares (ω_max, α_max)
- Información de la trayectoria (duración, puntos)
- Convención de ángulos

**Uso**: Documentación y referencia

### 2. Trayectoria (CSV)
**Formato**: `trajectory_YYYYMMDD_HHMMSS.csv`

Contiene la serie temporal con referencias angulares:
```
time_s,theta1_rad,theta2_rad
0.000000,3.141593,0.000000
0.016667,3.141593,0.000000
...
```

**Uso**: 
- Alimentar a Arduino
- Análisis posterior
- Visualización

## Archivos de Ejemplo

- `ejemplo_config.txt`: Ejemplo de archivo de configuración
- `ejemplo_trajectory.csv`: Ejemplo de archivo CSV (primeras líneas)

## Cómo Usar

### Ver trayectoria:
```bash
python visualizar_trayectoria.py trayectorias/trajectory_YYYYMMDD_HHMMSS.csv
```

### Convertir para Arduino:
```bash
python csv_to_arduino.py trayectorias/trajectory_YYYYMMDD_HHMMSS.csv -o trajectory_data.h -d 2
```

### Streaming directo a Arduino:
```bash
python gemelo_2R_serial.py --port=COM3 --baud=115200 --log=telemetria.csv
```

## Convención de Ángulos

- **θ₁**: Ángulo del eslabón 1 desde horizontal (eje +X)
- **θ₂**: Ángulo relativo del eslabón 2 (0 = colineales)

Ver `docs/COMO_USAR_TRAYECTORIAS.md` para más detalles.

## Notas

- Los archivos NO se sobrescriben (timestamp único)
- La carpeta se crea automáticamente al presionar Start
- Puedes eliminar archivos antiguos manualmente si lo deseas

