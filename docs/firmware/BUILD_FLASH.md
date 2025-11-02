# Compilar y Cargar el Firmware

## Opción A — Arduino IDE
1. Abre el sketch.
2. Placa: UNO o MEGA (según tu hardware).
3. Puerto: COMx (Win) o /dev/ttyACMx (Linux).
4. Subir.

## Opción B — CLI (arduino-cli)
```bash
arduino-cli compile --fqbn arduino:avr:uno path/al/sketch
arduino-cli upload  --fqbn arduino:avr:uno -p COM3 path/al/sketch
```
Ajusta `--fqbn` y `-p` a tu placa/puerto.
