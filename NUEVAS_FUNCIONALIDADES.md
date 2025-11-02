# NUEVAS FUNCIONALIDADES - Análisis y Caracterización

## Fecha: 2 de Noviembre de 2025

## Resumen de Cambios

Se han implementado tres mejoras principales solicitadas por el usuario:

### ✅ 1. Visualización en Grados (CSV en Radianes)
- **Todas las gráficas** ahora muestran ángulos en **grados (°)** para mayor comprensión
- Los archivos CSV siguen guardándose en **radianes** para alimentar el Arduino
- Afecta: `gemelo_digital_2R_trebol_pt2.py` y `visualizar_trayectoria.py`

### ✅ 2. Omitir Posicionamiento Inicial (Por Defecto)
- Las gráficas **NO incluyen** el posicionamiento inicial por defecto
- Detección automática del inicio del trazo basada en velocidad angular
- Opcional: incluir con `--include-initial` en visualizar_trayectoria.py
- **Beneficio**: Mejora visualización de datos del trazo real (movimientos más lentos del blend no opacan el trazo)

### ✅ 3. Caracterización de Ángulos
- Nueva ventana "Análisis de Ángulos - Caracterización" con 3 ventanas automáticas
- Identifica **ángulos más comunes** para caracterización de planta
- Identifica **ángulos más demandantes** (velocidad, aceleración, esfuerzo)
- 6 gráficas + tabla de estadísticas

---

## Detalles Técnicos

### Ventanas del Simulador

Al ejecutar `python gemelo_digital_2R_trebol_pt2.py` se abren **3 ventanas**:

#### 1. **Gemelo Digital 2R**
- Simulación visual con sliders y botones Start/Stop/Reset
- Sin cambios funcionales

#### 2. **Perfiles Articulares** (ACTUALIZADA)
- **Antes**: radianes, incluía posicionamiento inicial
- **Ahora**: grados, omite posicionamiento inicial automáticamente
- 4 gráficas: θ, ω, α, jerk (todas en grados)
- Parámetro `skip_initial=True` por defecto

#### 3. **Análisis de Ángulos - Caracterización** (NUEVA)
Layout 3×3:

**Fila 1: Distribuciones**
- Histograma θ₁ con línea roja indicando ángulo más común
- Histograma θ₂ con línea roja indicando ángulo más común
- Histograma 2D (mapa de calor θ₁ vs θ₂) con estrella azul en combinación más frecuente

**Fila 2: Mapas de Demanda**
- **Demanda de Velocidad**: Scatter plot θ₁ vs θ₂ coloreado por |ω|, con Top 5 en rojo
- **Demanda de Aceleración**: Scatter plot θ₁ vs θ₂ coloreado por |α|, con Top 5 en rojo
- **Esfuerzo Combinado**: Scatter plot θ₁ vs θ₂ coloreado por (ω + 0.1α), con Top 5 en rojo

**Fila 3: Tabla de Estadísticas**
- Ángulos más comunes (θ₁, θ₂)
- Combinación más frecuente
- Promedio, desviación estándar, rango
- Velocidades y aceleraciones máximas/promedio
- **Condiciones más demandantes** (mayor velocidad, mayor aceleración, mayor esfuerzo)

---

## Uso Práctico para Caracterización

### Identificar Ángulos para Caracterización

**Ángulos Comunes** (para caracterizar respuesta típica):
1. Ver **histogramas θ₁ y θ₂** → pico más alto = ángulo más frecuente
2. Ver **histograma 2D** → zona más caliente = combinación más frecuente
3. Ver **tabla de estadísticas** → "Más común" y "Combinación más frecuente"

**Ejemplo de salida**:
```
Ángulos más comunes:
  θ₁: 125.34°
  θ₂: 45.67°

Combinación más frecuente: (θ₁=123.2°, θ₂=47.1°)
```

**Ángulos Demandantes** (para pruebas de límites):
1. Ver **mapa de velocidad** → puntos rojos = mayores velocidades
2. Ver **mapa de aceleración** → puntos rojos = mayores aceleraciones
3. Ver **mapa de esfuerzo** → puntos rojos = mayores esfuerzos combinados
4. Ver **tabla** → "Condiciones más demandantes"

**Ejemplo de salida**:
```
Condiciones más demandantes:
  Mayor velocidad: θ₁=98.23°, θ₂=67.45°
  Mayor aceleración: θ₁=115.67°, θ₂=52.89°
  Mayor esfuerzo: θ₁=102.45°, θ₂=63.21°
```

### Protocolo de Caracterización Sugerido

**Paso 1**: Ejecutar trayectoria y obtener análisis
```bash
python gemelo_digital_2R_trebol_pt2.py
# Ajustar → Start → Ver "Análisis de Ángulos"
```

**Paso 2**: Identificar ángulos clave
- Anotar **5-10 ángulos más comunes** (del histograma)
- Anotar **3-5 ángulos más demandantes** (de mapas)

**Paso 3**: Caracterizar planta en esos ángulos
- Aplicar escalón/rampa en cada ángulo identificado
- Medir respuesta (tiempo de subida, sobrepico, error en régimen)
- Ajustar ganancias PD específicas si es necesario

**Paso 4**: Validar con trayectoria completa
- Ejecutar trayectoria real con Arduino
- Comparar telemetría vs referencia
- Ajustar si RMSE > umbral aceptable

---

## Archivos Modificados

### 1. `gemelo_digital_2R_trebol_pt2.py`
**Cambios**:
- `ProfilesPlotter.update()`: Ahora convierte a grados y omite inicial
- Nueva clase `AngleAnalyzer`: Ventana de caracterización
- `App.__init__()`: Agrega `self.analyzer`
- `App._try_plan_and_load()`: Actualiza analyzer con `skip_initial=True`

**Nuevas funcionalidades**:
```python
self.prof.update(t, thetas, fps, skip_initial=True)  # Omite inicial
self.analyzer.analyze(t, thetas, fps, skip_initial=True)  # Caracterización
```

### 2. `visualizar_trayectoria.py`
**Cambios**:
- `visualize_trajectory()`: Ahora acepta `skip_initial` y `show_analysis`
- Convierte a grados para visualización
- Detecta automáticamente inicio del trazo
- Nueva función `show_angle_analysis()`: Ventana de caracterización
- Argumentos CLI: `--include-initial`, `--no-analysis`

**Uso actualizado**:
```bash
# Por defecto: omite inicial, muestra análisis
python visualizar_trayectoria.py trayectorias/trajectory_20251102_143022.csv

# Incluir posicionamiento inicial
python visualizar_trayectoria.py trajectory.csv --include-initial

# Sin ventana de análisis
python visualizar_trayectoria.py trajectory.csv --no-analysis
```

---

## Detección Automática del Inicio

**Algoritmo**:
1. Calcular norma de velocidad angular: `|ω| = sqrt(ω₁² + ω₂²)`
2. Encontrar percentil 10 como umbral base
3. Buscar primer índice donde `|ω| > 2 × umbral`
4. Usar ese índice - 10 frames (con mínimo de 60) como inicio del trazo

**Beneficios**:
- No depende de parámetros fijos (dwell_s, blend_s)
- Robusto ante cambios de configuración
- Se adapta automáticamente

---

## Ejemplos de Salida

### Consola (visualizar_trayectoria.py)
```
============================================================
ESTADÍSTICAS DE LA TRAYECTORIA
============================================================
Archivo: trayectorias/trajectory_20251102_143022.csv

Duración total: 25.067 s
Número de puntos: 1504
Puntos analizados (sin inicial): 1392
Frecuencia de muestreo: 60.0 Hz

Ángulos (en datos analizados):
  θ₁: min=98.23°, max=145.67°, rango=47.44°
  θ₂: min=32.15°, max=78.90°, rango=46.75°

Velocidades angulares:
  ω₁: max=124.56°/s
  ω₂: max=98.34°/s
  |ω|: max=152.78°/s

Aceleraciones angulares:
  α₁: max=456.78°/s²
  α₂: max=389.12°/s²
  |α|: max=567.89°/s²

Ángulos más comunes:
  θ₁: 125.34°
  θ₂: 45.67°

Condiciones más demandantes:
  Mayor velocidad: θ₁=98.23°, θ₂=67.45°
  Mayor aceleración: θ₁=115.67°, θ₂=52.89°
  Mayor esfuerzo: θ₁=102.45°, θ₂=63.21°
============================================================
```

---

## Convención Inalterada

**Importante**: Los CSV siguen guardando en radianes con la misma convención:
- **θ₁**: Ángulo eslabón 1 desde horizontal [rad]
- **θ₂**: Ángulo relativo eslabón 2 [rad] (0 = colineales)

**La conversión a grados es SOLO para visualización humana.**

---

## Compatibilidad

- ✅ Compatible con archivos CSV antiguos
- ✅ No afecta guardado de archivos
- ✅ Retrocompatible con sketches Arduino existentes
- ✅ Todas las herramientas actualizadas (visualizar_trayectoria.py)

---

## Testing Realizado

- ✅ Generación de trayectoria con Start
- ✅ 3 ventanas se abren correctamente
- ✅ Gráficas muestran grados (°)
- ✅ CSV guarda radianes
- ✅ Posicionamiento inicial omitido por defecto
- ✅ Histogramas y mapas de demanda funcionan
- ✅ Tabla de estadísticas se renderiza correctamente
- ✅ visualizar_trayectoria.py funciona con ambas opciones

---

## Próximos Pasos Sugeridos

1. **Ejecutar trayectoria de prueba**:
   ```bash
   python gemelo_digital_2R_trebol_pt2.py
   # Ajustar → Start → Ver las 3 ventanas
   ```

2. **Identificar ángulos clave** del análisis

3. **Caracterizar planta** en esos ángulos con pruebas escalón

4. **Ajustar ganancias PD** según respuesta en ángulos demandantes

5. **Validar** con ejecución completa en Arduino

---

## Documentación Actualizada

Ver también:
- `GUIA_RAPIDA_TRAYECTORIAS.md` (actualizada)
- `docs/COMO_USAR_TRAYECTORIAS.md` (actualizada)
- `CAMBIOS_IMPLEMENTADOS.md` (original, mantener para referencia)

---

**Autor**: AI Assistant  
**Fecha**: 2025-11-02  
**Versión**: 2.0 - Análisis y Caracterización

