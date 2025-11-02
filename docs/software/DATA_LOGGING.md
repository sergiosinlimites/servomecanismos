# Registro de Datos (CSV)

## Simulación (opcional)
El simulador puede guardar: `t_s, theta1_rad, theta2_rad, x_cm, y_cm`.

## Telemetría (--log)
El puente serial guarda: `pc_time_s, arduino_ms, q1, q2, q1_ref, q2_ref, u1, u2`.

### Recomendaciones
- Usa nombres con fecha/hora: `run_YYYYMMDD_HHMM.csv`.
- Acompaña cada CSV con los parámetros usados (n, A, rot, v, fps, cycles, elbow).
