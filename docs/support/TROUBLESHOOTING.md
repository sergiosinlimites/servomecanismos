# Troubleshooting

## No se mueve
- Verifica puerto correcto (--port) y baudios (115200).
- Revisa fuente de los drivers y GND común.
- Envía Z manualmente y referencias pequeñas: `R,0.1,0.0`.

## Gira al revés
- Invierte pin DIR o multiplica u por −1 para ese motor.

## Vibra mucho
- Baja Kd y/o Kp. Agrega derivada sobre medición con filtro.

## Se satura (no llega a la referencia)
- Sube Kp gradualmente o mejora la fuente/driver/motor.
- Reduce velocidad (v) o baja carga mecánica.

## Serial inestable
- Baja fps, eleva timeout del puerto, evita otros procesos pesados.
