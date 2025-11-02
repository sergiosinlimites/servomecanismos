# Formato de Telemetría

Líneas `Y,...` a ~50 Hz:

```
Y,<millis>,<q1>,<q2>,<q1_ref>,<q2_ref>,<u1>,<u2>
```

- `millis` : tiempo desde reset (ms)
- `q1,q2` : ángulos medidos (rad)
- `q1_ref,q2_ref` : referencias (rad)
- `u1,u2` : esfuerzos normalizados (aprox) en [-1,1]

## CSV generado
Cuando corres Python con `--log=archivo.csv`, se guarda:
```
pc_time_s,arduino_ms,q1,q2,q1_ref,q2_ref,u1,u2
```
