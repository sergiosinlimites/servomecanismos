# Calibración de Encoders / Potenciómetros

Esta guía alinea los **cero articulares** y valida el mapeo **ADC → radianes**.

## 1) Cero rápido con comando `Z`
- Con el brazo en una postura conocida (por ejemplo, alineado a marcas mecánicas), envía `Z` por Serial o deja que el script Python lo haga automáticamente al iniciar el streaming.
- El firmware toma las lecturas actuales como **θ₁ = 0** y **θ₂ = 0** y guarda los offsets `off1`, `off2`.

## 2) Verificar rango y sentido
- Mueve manualmente cada junta y observa en el Monitor Serie la telemetría `Y,...`.
- Asegúrate de que el ángulo crece con el giro esperado. Si no:
  - Invierte el **sentido** en `driveMotor(...)` (cambia el signo de `u`), o
  - Invierte el cable del pin **DIR** del driver (seguro y rápido).

## 3) Normalización angular
El firmware normaliza cada lectura a **(−π, π]** para evitar saltos al cruzar ±π.

## 4) Ganancia ADC→rad
El mapeo base es `RAD_PER_ADC = 2π / 1023`. Si usas un encoder con otra escala:
- Ajusta la constante o convierte a pulsos por vuelta → radianes.
- Si hay **desalineación mecánica**, re‑haz `Z` con el brazo en la postura de referencia.

## 5) Prueba final
- Envía referencias pequeñas y verifica que **e = q_ref − q** se acerca a 0 sin oscilaciones.
- Guarda una corrida de telemetría y revisa θ₁/θ₂ y la punta en las gráficas.
