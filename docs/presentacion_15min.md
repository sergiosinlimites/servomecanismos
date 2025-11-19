## Presentación (15 minutos) — Manipulador 2R con trayectoria tipo trébol

Duración total: 15:00
Audiencia: Profesor y jurados del curso de Servomecanismos
Formato sugerido: 11 diapositivas (última incluye cierre + Q&A)


## 1) Portada — 00:30
- **Mensaje clave**: Equipo, curso y proyecto.
- **Añadir**: Título, integrantes, docente, semestre.
- **Evidencias**: Logo UNAL (`escudo_unal.png`), foto del prototipo si está disponible.

- Contenido (conciso):
  - Proyecto Académico: Manipulador 2R con seguimiento de trayectoria tipo trébol.
  - Equipo: Juan Beltrán, Sergio Bolaños, Nicolás Garzón, David Pirateque, Óscar Siabato, Jonatan Valero.
  - Docente: Víctor H. Grisales — 2025-2S.

- Imágenes sugeridas:
  - `escudo_unal.png` al centro; si hay, foto del prototipo de fondo tenue.


## 2) Objetivo y alcance — 01:30
- **Mensaje clave**: Qué problema resolvemos y qué entregamos.
- **Añadir**:
  - Objetivo: seguir automáticamente una trayectoria tipo trébol con un 2R.
  - Alcance: planificación, gemelo digital, control en HW, UI para visualización/telemetría.
  - Restricciones de operación (área de 224 mm, velocidad 1–10 cm/s, escala ≤1.2, rotación ±45°).
- **Evidencias**: Fragmento de requerimientos del documento del curso.

- Contenido (conciso):
  - Construir un 2R que dibuje un trébol estilizado respetando área circunscrita de 224 mm.
  - Velocidad nominal constante (1–10 cm/s), escala hasta 1.2× y rotación ±45°.
  - Entregables: hardware 2R, control (PID/PD), gemelo digital y UI web con gráficas/telemetría/export.

- Imágenes sugeridas:
  - Bloque con “Objetivo / Restricciones / Entregables” (iconos simples).


## 3) Requerimientos clave — 01:15
- **Mensaje clave**: Criterios medibles de éxito.
- **Añadir**:
  - Base fija, 2 eslabones (R–R), seguimiento sobre extremo distal.
  - Postura inicial “recogida”, fase de aproximación, multiciclo (hasta 10).
  - Gráficas en tiempo real y posibilidad de exportar.
- **Evidencias**: Lista corta y clara con íconos o checks.

- Contenido (conciso):
  - 2R planar: dos articulaciones rotacionales, dibujo en el extremo del eslabón 2.
  - Inicio recogido por debajo de media altura y a la izquierda del cuadrado; aproximación inicial.
  - Ejecutar hasta 10 ciclos del trébol con seguimiento rápido y preciso.
  - Visualización en tiempo real desde PC y exportación de datos.

- Imágenes sugeridas:
  - Mini-viñeta del cuadrado circunscrito y posición inicial relativa.


## 4) Arquitectura del sistema — 01:15
- **Mensaje clave**: Vista de bloques de la solución.
- **Añadir**:
  - Bloques: Trayectoria → IK + planificación → Referencias articulares → Arduino (PID/PD) → Motores → Encoders → Telemetría → UI Web.
  - Comunicación serie PC–Arduino.
- **Evidencias**: Diagrama de bloques (simple y grande).

- Contenido (conciso):
  - Trayectoria (spline) → IK 2R → perfiles θ(t) con límites de ω/α y blend inicial.
  - PC envía referencias por serie (“R, th1, th2”); Arduino ejecuta PID/PD a 50 ms.
  - Encoders devuelven telemetría; UI web despliega perfiles y estadísticas.

- Imágenes sugeridas:
  - Diagrama de bloques con flechas (PC/UI, Serie, Arduino, Potencia, Motores, Encoders).


## 5) Trayectoria: polar vs spline — 01:45
- **Mensaje clave**: Elegimos splines por control local de forma.
- **Añadir**:
  - Ecuación polar: r = 1 + M sin(aθ + b); ventajas y limitaciones (afecta toda la forma, difícil ajuste local).
  - Enfoque final: puntos de control + spline cúbica (suavidad C2, precisión CAD).
  - Construcción de pétalo, simetrías y rotaciones para 4 hojas.
- **Evidencias**: Figura polar (`Trayectoria_Polar.png`) y por puntos de control (`Trayectoria_Puntos_Control.png`).

- Contenido (conciso):
  - Probamos r(θ)=1+M sin(aθ+b) (cardioide/“trébol”): rápida pero poco control local.
  - Limitaciones: parámetros modifican toda la forma; difícil casar con CAD/espacio de trabajo.
  - Solución: puntos de control (8 puntos) + spline cúbica natural; reflejo y 3 rotaciones (4 hojas).
  - Trayectoria final suave (C2), fiel a cotas y orientable/escaleable.

- Imágenes sugeridas:
  - `Trayectoria_Polar.png` vs `Trayectoria_Puntos_Control.png` (lado a lado).


## 6) Diseño mecánico y actuadores — 01:45
- **Mensaje clave**: Se adaptó el diseño a motores disponibles.
- **Añadir**:
  - M1: Pittman GM9413 con reductora; soporte por chumaceras para aliviar par gravitacional.
  - M2: Faulhaber 2342 con reducción; eslabón 2 en tubo EMT aligerado.
  - Materiales: acrílico + PETG + aluminio; base en MDF.
- **Evidencias**: Foto del conjunto o esquema (`Esquema del mecanismo.png` si disponible).

- Contenido (conciso):
  - L1=235 mm, L2=165 mm; Rmax=400 mm cubre cuadrado 224 mm.
  - M1 (Pittman + reductora 19.7:1): par continuo 0.318 N·m; soporte en chumaceras reduce par gravitacional efectivo.
  - M2 (Faulhaber + reducción): mueve eslabón 2 (tubo EMT) con amplio margen.
  - Estructura: acrílico 5 mm, PETG estructural, aluminio; base MDF.

- Imágenes sugeridas:
  - `Esquema del mecanismo.png` o foto del montaje; resaltar chumaceras y motores.


## 7) Sensores y electrónica — 01:15
- **Mensaje clave**: Medición y potencia para control robusto.
- **Añadir**:
  - Encoders ópticos con OPB800 (180 ranuras, 2°/paso) o sensores absolutos si se usaron.
  - Puente H L298N; fuentes separadas 12/24 V; tierras comunes.
  - Cableado y protección (diodos, desacoplo).
- **Evidencias**: Foto de módulos, breve tabla de especificaciones.

- Contenido (conciso):
  - Sensado: discos 180 ranuras + OPB800 → cuantización 2°; lectura por polling/INT según sección.
  - Potencia: L298N (2 A/canal); PWM desde Arduino (12 bits); 24 V (M1) y 12 V (M2), GND común.
  - Buenas prácticas: diodos de rueda libre y desacoplos; separación potencia/señal.

- Imágenes sugeridas:
  - Fotos de OPB800/discos y módulo L298N; esquema simple de conexiones/tierras.


## 8) Cinemática y planificación — 01:15
- **Mensaje clave**: IK 2R + perfiles temporales suaves.
- **Añadir**:
  - Fórmulas de cinemática directa e inversa (en una sola lámina, tipografía grande).
  - Planificación con control de ω/α máximas y blend inicial desde “parqueo”.
  - Concepto de punto de inicio óptimo sobre la curva.
- **Evidencias**: Diagrama con variables y ángulos (simple).

- Contenido (conciso):
  - FD: x= x_b + L1 cos θ1 + L2 cos(θ1+θ2), y análogo; IK por ley de cosenos (solución codo abajo).
  - Planificación: limitar |ω| y |α|; determinar T_blend por límites y distancia desde parqueo (−90°, 0°).
  - Elegir índice de inicio cercano al “punto de parqueo” en mitad inferior del trébol para suavidad.

- Imágenes sugeridas:
  - Diagrama 2R (θ1, θ2, L1, L2) con flechas; recuadro con ecuaciones grandes.


## 9) Control PID/PD y comunicación — 01:30
- **Mensaje clave**: Estrategia por articulación y lazo completo.
- **Añadir**:
  - M1: PID con anti-windup y límites asimétricos; M2: PD con banda muerta.
  - Periodo de control (p. ej., 50 ms), discretización de encoders.
  - Protocolo serie: comandos R, Z, S; envío/recepción; tuning Kp/Ki/Kd.
- **Evidencias**: Tabla de ganancias iniciales y cómo se ajustaron en pruebas.

- Contenido (conciso):
  - M1: PID posicional con saturación del integrador y límite descendente menor (protege reductora).
  - M2: PD con banda muerta de 2°; sin I por baja carga/holgura.
  - Lazo: 50 ms; comandos serie: Z (zero), R (referencia θ1,θ2), S (stop); ajuste de K en caliente.

- Imágenes sugeridas:
  - Diagrama del lazo con puntos de saturación/banda muerta; tabla breve de K iniciales.


## 10) Gemelo digital y UI Web — 01:30
- **Mensaje clave**: Visualización, perfiles y telemetría.
- **Añadir**:
  - Canvas 2D del brazo, pan/zoom, trazo de punta.
  - Gráficas Plotly: θ, ω, α, jerk; histogramas y mapas 2D; overlay de telemetría real.
  - Exportables: configuración, trayectoria, telemetría.
- **Evidencias**: Capturas de `web/index.html` con dos columnas de gráficas y tarjetas.

- Contenido (conciso):
  - Simulación 2D en canvas (20×20 cm), vista ajustable y trazo del efector.
  - Perfiles θ/ω/α/jerk y análisis (histogramas, mapas 2D de |ω|, |α|, esfuerzo).
  - Superposición de telemetría real sobre referencias; exportar TXT/CSV.

- Imágenes sugeridas:
  - Capturas de la UI (sección simulación y “Sección 3: Gráficas” en 2 columnas).


## 11) Resultados — 02:00
- **Mensaje clave**: Evidencia de desempeño frente a requisitos.
- **Añadir**:
  - Comparativa simulación ideal vs cuantizada (2°) — figura (`2.jpeg` si corresponde).
  - Métricas: |ω| máx, |α| máx, error típico vs referencia, cobertura del trébol, ciclos.
  - Demostración breve (si el tiempo/condición lo permite) o video corto.
- **Evidencias**: Capturas de perfiles y tablas de stats desde la UI.

- Contenido (conciso):
  - Con encoder ideal: seguimiento casi perfecto; con cuantización 2°: trayectoria “dienteada” controlada.
  - Requisitos: área 224 mm cubierta, multiciclo (≤10), velocidad dentro de 1–10 cm/s.
  - Valores característicos mostrados en UI: picos de |ω|/|α| dentro de límites configurados.

- Imágenes sugeridas:
  - `2.jpeg` (comparativa ideal vs 2°); capturas de plots y tabla de estadísticas de la UI.


## 12) Conclusiones, lecciones y futuro (+ Q&A) — 01:15
- **Mensaje clave**: Qué logramos, qué aprendimos y próximos pasos.
- **Añadir**:
  - Conclusiones: cumplimiento de requisitos, utilidad del gemelo digital y UI.
  - Lecciones: ventaja de splines vs polar en ajuste, importancia de soporte mecánico para el par.
  - Futuro: mayor resolución de sensores, control feedforward, límites de torque, trayectorias adicionales.
  - Cierre y preguntas.
- **Evidencias**: 3–4 bullets de alto impacto (sin párrafos largos).

- Contenido (conciso):
  - Logros: 2R funcional; planificación y control que cumplen requisitos; UI web útil para prueba y diagnóstico.
  - Lecciones: splines dan control local y reproducen CAD; chumaceras clave para reducir par en M1.
  - Futuro: sensores de mayor resolución (o absolutos), feedforward/limitadores de torque, librería de trayectorias.
  - Gracias — Preguntas.

- Imágenes sugeridas:
  - Mini-grid de 3 imágenes: prototipo, UI, trazo del trébol.


### Notas de apoyo (opcional para presenter view)
- Llevar un “demo script” breve: planificar → start → detener → exportar y mostrar plots.
- Si hay riesgos con la demo en vivo, usar videos/gifs o capturas preparadas.
- Mantener fórmulas grandes y pocas: priorizar gráficos y esquemas.
- Si falta tiempo: priorizar slides 2, 4, 5, 9, 10 y 11.


