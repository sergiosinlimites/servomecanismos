# Firmware Arduino — PD por junta

El sketch ejecuta un lazo a **100 Hz** (Ts=10 ms), lee encoders/pots, calcula el error por junta y aplica un **PD** con saturación a `[-1,1]` (normalizado para PWM).

## Componentes
- Lectura: `readJoint1/2()` → ADC a rad y normalización (−π,π].
- Control: `u = Kp*e + Kd*(e - e_prev)/Ts`.
- Actuación: `driveMotor(pwm, dir, u)` (PWM 0..255 y dirección).
- Serial:
  - Comandos: `R,q1,q2` (rad), `S`, `Z`, `P,Kp1,Kd1,Kp2,Kd2`.
  - Telemetría: `Y,<ms>,<q1>,<q2>,<q1_ref>,<q2_ref>,<u1>,<u2>` ~ 50 Hz.

## Recomendaciones
- Empezar con **Kp** moderado y **Kd** bajo para evitar vibración.
- Asegurar **GND común** entre Arduino, drivers y fuentes.
- Si un eje gira “al revés”: invierte `DIR` o el signo de `u` para ese motor.
