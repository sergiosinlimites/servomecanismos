# Software (Python) — Gemelo Digital 2R

El script `gemelo_2R_serial.py` integra:
- Generador de trébol (rosa) que encaja en 20×20 cm.
- Planificador por longitud de arco (v ~ cte), con “pose inicial” a la izquierda del bounding box.
- IK con continuidad de rama.
- Animación con Matplotlib.
- Streaming Serial a Arduino y lectura de telemetría.
- Comparación sim vs real (RMSE, gráficas).

Ver también: `SIMULATION.md`, `SERIAL_BRIDGE.md`, `ANALYSIS.md`.
