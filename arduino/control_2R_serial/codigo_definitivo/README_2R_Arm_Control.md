
# 2R Planar Arm – Control en Arduino con Encoders y L298N

Firmware para el control del **primer y segundo eje** de un brazo planar 2R usando:

- **Arduino** (Uno o similar)
- **Driver L298N** (un módulo para ambos motores)
- **Encoders ópticos de ranuras** (1 canal, 180 ranuras)
- **Control PI–D en el eje 1** (con compensación de gravedad mediante integrador)
- **Control PD en el eje 2**
- Comandos por **Serial** para mandar referencias y ajustar ganancias

El foco principal del código es controlar **M1 (articulación 1)** de forma suave y segura, sin golpes contra los topes, incluso con encoder ruidoso.

---

## 1. Hardware y conexión (Pinout)

### 1.1. Resumen de componentes

- 2 motores DC con caja reductora:
  - Motor 1: articulación 1 (hombro / brazo que vence gravedad)
  - Motor 2: articulación 2 (codo)
- Encoders ópticos (ranurado) de 1 canal:
  - 180 ranuras por vuelta
- Driver L298N:
  - Canal A → Motor 1  
  - Canal B → Motor 2

### 1.2. Pinout Arduino ↔ L298N y encoders

| Función                     | Arduino Pin | Módulo    | Señal física                         |
|----------------------------|-------------|-----------|--------------------------------------|
| Encoder motor 1 – señal    | D2          | Encoder 1 | Salida digital del sensor de ranuras |
| Encoder motor 2 – señal    | D3          | Encoder 2 | Salida digital del sensor de ranuras |
| PWM motor 1 (ENA)          | D5 (PWM)    | L298N     | ENA                                  |
| Dirección motor 1 – IN1    | D6          | L298N     | IN1 (sentido A)                      |
| Dirección motor 1 – IN2    | D7          | L298N     | IN2 (sentido B)                      |
| PWM motor 2 (ENB)          | D10 (PWM)   | L298N     | ENB                                  |
| Dirección motor 2 – IN3    | D9          | L298N     | IN3 (sentido A)                      |
| Dirección motor 2 – IN4    | D8          | L298N     | IN4 (sentido B)                      |

**Alimentación y masa:**

- L298N:
  - `+12V` (o la tensión de tus motores) a la entrada de potencia
  - `GND` común entre la fuente de motores y el GND del Arduino
  - `5V` del L298N puede alimentar la lógica del propio driver, **pero no** el Arduino (recomendado usar 5V del Arduino para la lógica del L298N o una fuente regulada aparte).
- Encoders:
  - Vcc (5V), GND comunes con el Arduino
  - La salida de cada encoder va a D2 y D3 respectivamente.

---

## 2. Qué hace el código

### 2.1. Medición de ángulos

Cada encoder tiene **180 ranuras por vuelta**. En cada flanco válido de la señal:

- Se incrementa o decrementa un contador:
  - `slotCount_1` para el eje 1
  - `slotCount_2` para el eje 2
- La dirección de conteo se fija en `dirSign_1` y `dirSign_2`, que se actualizan según el sentido de giro mandado al motor.

Se define un “cero” de referencia (posición colgando) usando:

- Al inicio del `setup()` (posición inicial del brazo)
- O manualmente con el comando serial `Z`.

#### Eje 1 (Motor 1, con filtro anti-salto)

1. Se calcula el ángulo crudo en grados:
   ```c
   rel_1   = slotCount_1 - slotOffset_1;
   turns_1 = (float)rel_1 / SLOTS_PER_REV_1;
   deg_1   = turns_1 * 360.0f;
   ```

2. Para evitar saltos locos por ruido del encoder, se pasa por un **filtro anti-salto**:
   ```c
   if (!ang1_filt_init) {
     angulo_1_filtrado = deg_1;
     ang1_filt_init    = true;
   }

   float delta_deg = deg_1 - angulo_1_filtrado;
   const float MAX_DEG_STEP_1 = 10.0f; // máx 10° por ciclo (~200°/s)

   if (delta_deg >  MAX_DEG_STEP_1) delta_deg =  MAX_DEG_STEP_1;
   if (delta_deg < -MAX_DEG_STEP_1) delta_deg = -MAX_DEG_STEP_1;

   angulo_1_filtrado += delta_deg;
   ```

3. Ángulos “oficiales” del eje 1:
   - `angulo_1` en grados (continuo, sin mod 360)
   - `angulo_1_rad` en radianes:
     ```c
     angulo_1_rad = angulo_1_filtrado * PI / 180.0f;
     ```

#### Eje 2 (Motor 2)

Para el eje 2 se mantiene el cálculo estándar:

```c
rel_2   = slotCount_2 - slotOffset_2;
turns_2 = (float)rel_2 / SLOTS_PER_REV_2;
deg_2   = turns_2 * 360.0f;
aux2    = fmod(deg_2, 360.0f);
if (aux2 < 0) aux2 += 360.0f;
angulo_2     = aux2;
angulo_2_rad = wrapToPi(turns_2 * 2.0f * PI);
```

---

### 2.2. Generación de referencia suave (ramps)

Las referencias se envían en **radianes** por Serial:

```text
R,th1,th2
```

Ejemplo:

```text
R,1.2217,0.3491
```

Internamente:

1. Se convierten a grados:
   ```c
   float Ref1_cmd_deg = Ref_1 * 180.0f / PI;
   float Ref2_cmd_deg = Ref_2 * 180.0f / PI;
   ```

2. Se interpola con una **rampa no lineal** hacia cada referencia:
   - Distancia grande → `STEP_FAST = 2°/ciclo`
   - Distancia media → `STEP_MEDIUM = 1°/ciclo`
   - Cerca de la referencia → `STEP_SLOW = 0.5°/ciclo`

Ejemplo (para M1):

```c
float diffRef1 = Ref1_cmd_deg - Ref1_int_deg;
float dist1    = fabs(diffRef1);
float step1;
if (dist1 > 40.0f)      step1 = STEP_FAST;
else if (dist1 > 15.0f) step1 = STEP_MEDIUM;
else                    step1 = STEP_SLOW;

if (fabs(diffRef1) <= step1) {
  Ref1_int_deg = Ref1_cmd_deg;
} else {
  Ref1_int_deg += (diffRef1 > 0.0f ? step1 : -step1);
}
```

`Ref1_int_deg` y `Ref2_int_deg` son las referencias **internas, suavizadas** que ve el controlador.

---

### 2.3. Control del Motor 1 (M1)

El eje 1 es el más crítico porque **vence gravedad**. Se usa un controlador **PI–D** con límite asimétrico para evitar golpes hacia abajo.

1. Cálculo del error (en grados):
   ```c
   float e1_deg = Ref1_int_deg - angulo_1;
   ```

2. Envolvente opcional a [-180, 180]:
   ```c
   if (e1_deg > 180.0f) e1_deg -= 360.0f;
   if (e1_deg < -180.0f) e1_deg += 360.0f;
   ```

3. Derivada e integral:
   ```c
   float de1_deg = (e1_deg - e1_prev_deg) / Ts_s;  // Ts_s = Ts/1000
   e1_prev_deg   = e1_deg;

   e1_int_deg += e1_deg * Ts_s;
   const float I1_MAX = 2000.0f;                   // límite grande
   if (e1_int_deg >  I1_MAX) e1_int_deg =  I1_MAX;
   if (e1_int_deg < -I1_MAX) e1_int_deg = -I1_MAX;
   ```

4. Señal de control PI–D:
   ```c
   float u1 = k_p_1 * e1_deg + k_i_1 * e1_int_deg + k_d_1 * de1_deg;
   ```

5. Saturación general:
   ```c
   u1 = clampf(u1, -U1_MAX, U1_MAX);  // U1_MAX = 255
   ```

6. Limitación de empuje hacia abajo:
   ```c
   if (e1_deg < 0.0f && u1 < 0.0f) {
     if (u1 < -U1_MAX_DOWN) {
       u1 = -U1_MAX_DOWN;            // U1_MAX_DOWN = 40
     }
   }
   ```

7. Conversión a dirección y PWM:
   ```c
   int dir1;
   if (u1 >= 0.0f) {
     dir1  = +1;
     duty1 = (uint8_t)(u1);
   } else {
     dir1  = -1;
     duty1 = (uint8_t)(-u1);
   }

   if (duty1 < 5) duty1 = 0;        // umbral mínimo
   dir1 *= M1_DIR_INV;              // por si necesitas invertir lógica
   ```

8. Aplicación en el L298N y actualización del signo de conteo:
   ```c
   if (dir1 >= 0) {
     digitalWrite(IN_1_1, HIGH);
     digitalWrite(IN_2_1, LOW);
     dirSign_1 = +1;
   } else {
     digitalWrite(IN_1_1, LOW);
     digitalWrite(IN_2_1, HIGH);
     dirSign_1 = -1;
   }
   analogWrite(OutputPWM_GPIO_1, duty1);
   ```

---

### 2.4. Control del Motor 2 (M2)

De momento, el eje 2 usa un **PD clásico** (con posibilidad de usar Ki):

1. Error:
   ```c
   float e2_deg = Ref2_int_deg - angulo_2;
   ```

2. Envolvente a [-180, 180]:
   ```c
   if (e2_deg > 180.0f) e2_deg -= 360.0f;
   if (e2_deg < -180.0f) e2_deg += 360.0f;
   ```

3. Deadband:
   ```c
   if (fabs(e2_deg) < DEAD_BAND_DEG) {  // DEAD_BAND_DEG = 2°
     u2 = 0.0f;
   }
   ```

4. Señal de control y saturación:
   ```c
   float u2 = k_p_2 * e2_deg + k_i_2 * e2_int_deg + k_d_2 * de2_deg;
   u2 = clampf(u2, -U2_MAX, U2_MAX);    // U2_MAX = 60
   ```

5. Conversión a PWM y sentido (igual que M1) y escritura en el driver:
   ```c
   if (u2 >= 0.0f) {
     dir2  = +1;
     duty2 = (uint8_t)(u2);
   } else {
     dir2  = -1;
     duty2 = (uint8_t)(-u2);
   }
   if (duty2 < 5) duty2 = 0;

   // Dirección y conteo
   if (dir2 >= 0) {
     digitalWrite(IN_1_2, HIGH);
     digitalWrite(IN_2_2, LOW);
     dirSign_2 = +1;
   } else {
     digitalWrite(IN_1_2, LOW);
     digitalWrite(IN_2_2, HIGH);
     dirSign_2 = -1;
   }

   // Cuando quieras activar realmente M2:
   // analogWrite(OutputPWM_GPIO_2, duty2);
   ```

En el código que estás usando, M2 puede estar temporalmente desactivado con:
```c
analogWrite(OutputPWM_GPIO_2, 0);
```
para concentrarse primero en dejar perfecto el control de M1.

---

## 3. Protocolo por Serial

Baudios: **115200**.  
Configuración del Serial Monitor: “No line ending” o “NL & CR” (lo importante es ser consistente al enviar los comandos).

### 3.1. Comandos disponibles

- **Mandar referencia (rad):**
  ```text
  R,th1,th2
  ```
  Ejemplo:
  ```text
  R,1.2217,0.3491
  ```

- **Ajustar Kp (M1 y M2):**
  ```text
  P,Kp1,Kp2
  ```
  Ejemplo:
  ```text
  P,1.5,1.0
  ```

- **Ajustar Ki:**
  ```text
  I,Ki1,Ki2
  ```
  Ejemplo:
  ```text
  I,0.3,0.0
  ```

- **Ajustar Kd:**
  ```text
  D,Kd1,Kd2
  ```
  Ejemplo:
  ```text
  D,0.6,0.05
  ```

- **Recalibrar cero (posición actual = 0°):**
  ```text
  Z
  ```

- **Parar motores y resetear integrales/derivadas:**
  ```text
  S
  ```

---

## 4. Telemetría y depuración

Cada ~200 ms el código imprime por Serial una línea con:

- `theta1`: ángulo del eje 1 en radianes (filtrado)
- `theta2`: ángulo del eje 2 en radianes
- `ref1`, `ref2`: referencias internas en radianes
- `e1_deg`, `e2_deg`: errores en grados
- `u1`, `u2`: señal de control continua (antes de pasar a PWM)
- `dir1`, `dir2`: direcciones de los motores (+1 o -1)
- `duty1`, `duty2`: PWM aplicado (0–255)
- `stop`: 1 si los motores están detenidos por comando `S`, 0 en caso contrario

Ejemplo típico:

```text
theta1: 0.977384  theta2: 0.000000  ref1: 1.221700  ref2: 0.349100  e1_deg: 14.0  e2_deg: 20.0  u1: 111.0  u2: 20.0  dir1: 1  dir2: 1  duty1: 110  duty2: 20  stop: 0
```

Esto te permite:

- Ver si el error se está cerrando.
- Ver si el PWM está saturado.
- Ver si el integrador está actuando (u1 creciendo con el tiempo).
- Detectar cambios bruscos en el ángulo reportado por el encoder.

---

## 5. Flujo típico de uso

1. Subir el código al Arduino.
2. Colocar el brazo en la posición “colgando” (home mecánico) que quieres como **0°**.
3. Abrir el Serial Monitor a 115200 baudios.
4. Enviar:
   ```text
   Z
   ```
   para fijar el cero en la posición actual.
5. Enviar una referencia en radianes, por ejemplo:
   ```text
   R,1.2217,0.3491
   ```
6. Observar el movimiento del brazo y las líneas de telemetría.
7. Ajustar `Kp`, `Ki`, `Kd` con los comandos `P`, `I`, `D`:
   - Más **Kp** → respuesta más rápida, pero puede oscilar más.
   - Más **Ki** → elimina error en régimen permanente, pero puede provocar sobreoscilación si es muy grande.
   - Más **Kd** → ayuda a amortiguar, útil si el sistema se vuelve muy oscilatorio.

---

## 6. Notas de seguridad y recomendaciones

- Asegúrate de que el brazo **no pueda golpear** personas u objetos mientras estás tunning el controlador.
- Si el brazo sigue pegando contra los topes hacia abajo, reduce `U1_MAX_DOWN`.
- Si el encoder da lecturas incoherentes:
  - Revisa alineación mecánica del disco ranurado.
  - Revisa la alimentación y masa del sensor.
  - Mantén los cables del encoder alejados de los cables de potencia del motor.
- Si el PWM se queda saturado (`duty1` muy alto) y aun así no llega a la referencia, revisa:
  - Fricción mecánica
  - Tensión de alimentación de los motores
  - Ganancias `Kp`, `Ki` insuficientes

---

## 7. Extensiones posibles

- Activar y tunear **M2** seriamente (cambiar `analogWrite(OutputPWM_GPIO_2, 0)` por `duty2`).
- Implementar trayectorias suaves de `q1(t), q2(t)` (no solo saltar entre puntos).
- Implementar la **cinemática inversa** directamente en el Arduino para darle referencias cartesianas (x, y).
- Guardar las ganancias “buenas” en EEPROM para no tener que reconfigurarlas a mano tras cada reset.

---

Este README acompaña al sketch del Arduino que estás usando para el servomecanismo 2R. Puedes renombrar el archivo como `README.md` y subirlo a un repositorio de GitHub junto con el código fuente.
