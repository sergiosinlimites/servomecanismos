# Puente Serial (Python ↔ Arduino)

## Envío de referencias a fps
Se envía `R,θ1,θ2` en rad a la tasa indicada (por defecto 60 fps). El firmware mantiene el último setpoint si existe jitter USB.

## Warm-up y comandos
- Al iniciar, se manda `Z` (cero rápido).
- Al finalizar, `S` (stop).

## Puertos comunes
- Windows: COM3, COM4, ...
- Linux: /dev/ttyACM0, /dev/ttyUSB0

## Ejemplos
```bash
python gemelo_2R_serial.py --port=COM3 --baud=115200 --log=run_telemetry.csv --no-anim
python gemelo_2R_serial.py --n=5 --A=8 --cx=10 --cy=12 --rot=30 --scale=1.2 --v=6 --fps=60 --cycles=2 --elbow=up
```

## Notas
- Si no hay pyserial, instala: `pip install pyserial`.
- Si saturas el puerto, baja fps o sube el timeout del Serial.
