# Sintonía del Control PD

## Síntomas y acciones
- Lento / con error en régimen → Sube **Kp** poco a poco.
- Sobrepico u oscilación → Baja **Kp** o sube **Kd** ligeramente.
- Vibración de alta frecuencia → Baja **Kd** y/o filtra la derivada (ver abajo).
- Dirección incorrecta → Invierte signo de `u` o el pin **DIR**.

## Derivada sobre medición (opcional, más robusto al ruido)
En lugar de `Kd*(e - e_prev)/Ts`, usa:
```
q_dot = beta*(q - q_prev)/Ts + (1-beta)*q_dot_prev   # 0<beta<=1
u = Kp*e - Kd*q_dot
```
Con `beta≈0.5` reduces ruido. Documenta el coeficiente elegido.
