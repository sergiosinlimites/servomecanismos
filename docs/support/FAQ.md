# FAQ

**¿Qué unidades se usan?**
Radianes para ángulos; centímetros para la geometría del lienzo; mapeo ADC→rad en el firmware.

**¿Puedo cambiar el número de hojas del trébol?**
Sí: `--n=3|5|...` y puedes rotar con `--rot` (limitado a ±45°) y escalar con `--scale ≤ 1.2`.

**¿Por qué se usa “avoid_center”?**
Para que el trazo no pase por el origen y evitar singularidades/alcanza-bilidad complicadas.

**¿Cómo calibro rápido los ceros?**
Comando `Z` (o el script lo envía al iniciar). Luego verifica signos.

**¿Dónde veo mis datos?**
Telemetría en el CSV pasado por `--log`. Las gráficas aparecen al final del script (si no usas `--no-anim`).

**¿Se puede usar PID?**
Sí, pero el PD es suficiente para este caso. Si añades feedforward de velocidad, puedes bajar ganancias y mejorar el seguimiento.
