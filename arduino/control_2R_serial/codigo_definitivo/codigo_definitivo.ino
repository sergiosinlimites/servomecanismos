// ================== PINES ADAPTADOS A TU HARDWARE ==================
// ENCODERS
#define Sensor_1 2                  // Pin del encoder motor 1
#define Sensor_2 3                  // Pin del encoder motor 2

// MOTORES L298
// Motor 1
#define OutputPWM_GPIO_1 5          // ENA motor 1 (L298)
#define IN_1_1 6                    // IN1 motor 1
#define IN_2_1 7                    // IN2 motor 1

// Motor 2
#define OutputPWM_GPIO_2 10         // ENB motor 2 (L298)
#define IN_1_2 9                    // IN3 motor 2
#define IN_2_2 8                    // IN4 motor 2

// ================== CONFIG PWM / CONTROL ==================
#define PWM_MAX        255          // analogWrite 0–255

// Estos quedan definidos pero YA NO se usan directamente en el control.
#define MIN_DUTY_1_HOLD    90       // PWM para sostener brazo 1 en posición
#define MIN_DUTY_1_MOVE_L  200      // PWM mínimo para mover M1 si ángulo < 100°
#define MIN_DUTY_1_MOVE_H  190      // PWM mínimo para mover M1 si ángulo >= 100°

#define MIN_DUTY_2      5           // PWM mínimo motor 2
#define MAX_DUTY_2      30          // PWM máximo motor 2

// Deadband alrededor de la referencia (la uso en M2)
#define DEAD_BAND_DEG   2.0f        

// Saturaciones de esfuerzo del controlador (en “unidades PWM” aprox)
const float U1_MAX      = 255.0f;   // Máximo esfuerzo M1 (general)
const float U1_MAX_DOWN = 40.0f;    // Máximo empuje HACIA ABAJO en M1 (limitado)
const float U2_MAX      = 60.0f;    // Máximo esfuerzo M2

// +1 = como está ahora, -1 = invierte sentido lógico eje 1 (por hardware)
#define M1_DIR_INV     (+1)

// ================== TIEMPOS ==================
unsigned long previousMillis = 0;
const unsigned long Ts = 50;        // Periodo de control = 50 ms (~20 Hz)

// ================== MEDICIÓN ÁNGULOS ==================
float angulo_1 = 0.0f;              // Ángulo motor 1 en grados (filtrado, continuo)
float angulo_2 = 0.0f;              // Ángulo motor 2 en grados [0,360)
float angulo_1_rad = 0.0f;          // Ángulo motor 1 en radianes
float angulo_2_rad = 0.0f;          // Ángulo motor 2 en radianes [-pi,pi]

// Filtro simple para el ángulo 1 (anti-salto)
float angulo_1_filtrado = 0.0f;
bool  ang1_filt_init    = false;

// Offsets (posición "cero" en ranuras)
volatile int32_t slotOffset_1 = 0;
volatile int32_t slotOffset_2 = 0;

// ================== REFERENCIAS / CONTROL ==================
// Referencia comandada (radianes, lo que mandas por Serial: R,th1,th2)
float Ref_1 = 0.0f;
float Ref_2 = 0.0f;

// Referencia INTERNA en grados (se mueve suave hacia la comandada)
float Ref1_int_deg = 0.0f;
float Ref2_int_deg = 0.0f;
bool  refIntInit   = false;

// Rampa de referencia (suave, para no pegar brincos)
const float STEP_FAST   = 2.0f;     // lejos
const float STEP_MEDIUM = 1.0f;     // medio
const float STEP_SLOW   = 0.5f;     // cerca

// ===== Ganancias PID (valores de arranque; ajustables por serial) =====
float k_p_1 = 1.0f;                 // motor 1
float k_p_2 = 1.0f;                 // motor 2

float k_i_1 = 0.3f;                 // integral M1
float k_i_2 = 0.0f;                 // integral M2

float k_d_1 = 0.6f;                 // derivativo motor 1
float k_d_2 = 0.05f;                // derivativo motor 2

// Errores previos para el término D
float e1_prev_deg = 0.0f;
float e2_prev_deg = 0.0f;

// Integrales del error
float e1_int_deg = 0.0f;
float e2_int_deg = 0.0f;

// Flag de STOP
bool stopAll = false;

// ================== ENCODERS ==================
const uint16_t SLOTS_PER_REV_1  = 180;
const uint16_t SLOTS_PER_REV_2  = 180;
const bool     COUNT_ON_FALL_1  = true;
const bool     COUNT_ON_FALL_2  = true;
const uint32_t DEBOUNCE_US      = 150;

// Contadores con signo
volatile int32_t slotCount_1 = 0;
volatile int32_t slotCount_2 = 0;
volatile uint32_t lastUs_1   = 0;
volatile uint32_t lastUs_2   = 0;

// Dirección ACTUAL que está comandando cada motor (+1 o -1)
volatile int8_t dirSign_1 = +1;
volatile int8_t dirSign_2 = +1;

// ================== DEBUG / TELEMETRÍA ==================
unsigned long lastPrint = 0;
const unsigned long PRINT_MS = 50;

uint8_t duty1 = 0;
uint8_t duty2 = 0;

// Variables de debug
float dbg_e1_deg = 0.0f;
float dbg_e2_deg = 0.0f;
int   dbg_dir1   = 0;
int   dbg_dir2   = 0;
float dbg_u1     = 0.0f;
float dbg_u2     = 0.0f;

// ================== UTILIDADES ==================
static inline float clampf(float x, float a, float b) {
  return x < a ? a : (x > b ? b : x);
}

float wrapToPi(float a) {
  while (a > PI)  a -= 2.0f * PI;
  while (a <= -PI) a += 2.0f * PI;
  return a;
}

// ================== ISRs ENCÓDERS ==================
void isrSlot1() {
  uint32_t now_1 = micros();
  if (now_1 - lastUs_1 < DEBOUNCE_US) return;
  slotCount_1 += dirSign_1;
  lastUs_1 = now_1;
}

void isrSlot2() {
  uint32_t now_2 = micros();
  if (now_2 - lastUs_2 < DEBOUNCE_US) return;
  slotCount_2 += dirSign_2;
  lastUs_2 = now_2;
}

// ================== COMANDOS SERIALES ==================
void leerComandosSerial() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == 'R') {
      float a = Serial.parseFloat();
      float b = Serial.parseFloat();

      noInterrupts();
      Ref_1 = a;
      Ref_2 = b;
      stopAll     = false;
      refIntInit  = false;   // reenganchar referencia interna
      e1_prev_deg = 0.0f;
      e2_prev_deg = 0.0f;
      e1_int_deg  = 0.0f;
      e2_int_deg  = 0.0f;
      interrupts();

      // Consumir cualquier campo extra hasta fin de línea (p. ej., timestamp enviado por el simulador)
      while (Serial.available() > 0) {
        char d = Serial.read();
        if (d == '\n' || d == '\r') break;
      }

      // (silenciar prints de cambio de referencias para no saturar el puerto)
    }
    else if (c == 'P') {
      float p1 = Serial.parseFloat();
      float p2 = Serial.parseFloat();

      noInterrupts();
      k_p_1 = p1;
      k_p_2 = p2;
      interrupts();

      Serial.print(F("Nuevos Kp -> k_p_1="));
      Serial.print(k_p_1, 4);
      Serial.print(F(", k_p_2="));
      Serial.println(k_p_2, 4);
    }
    else if (c == 'I') {
      float i1 = Serial.parseFloat();
      float i2 = Serial.parseFloat();

      noInterrupts();
      k_i_1 = i1;
      k_i_2 = i2;
      e1_int_deg = 0.0f;
      e2_int_deg = 0.0f;
      interrupts();

      Serial.print(F("Nuevos Ki -> k_i_1="));
      Serial.print(k_i_1, 4);
      Serial.print(F(", k_i_2="));
      Serial.println(k_i_2, 4);
    }
    else if (c == 'D') {
      float d1 = Serial.parseFloat();
      float d2 = Serial.parseFloat();

      noInterrupts();
      k_d_1 = d1;
      k_d_2 = d2;
      interrupts();

      Serial.print(F("Nuevos Kd -> k_d_1="));
      Serial.print(k_d_1, 4);
      Serial.print(F(", k_d_2="));
      Serial.println(k_d_2, 4);
    }
    else if (c == 'Z') {
      noInterrupts();
      slotOffset_1 = slotCount_1;
      slotOffset_2 = slotCount_2;
      refIntInit   = false;
      e1_prev_deg  = 0.0f;
      e2_prev_deg  = 0.0f;
      e1_int_deg   = 0.0f;
      e2_int_deg   = 0.0f;
      // reinicio filtro de ángulo
      ang1_filt_init    = false;
      angulo_1_filtrado = 0.0f;
      interrupts();

      Serial.println(F("Cero recalibrado (Z)."));
    }
    else if (c == 'S') {
      stopAll = true;
      analogWrite(OutputPWM_GPIO_1, 0);
      analogWrite(OutputPWM_GPIO_2, 0);
      e1_prev_deg  = 0.0f;
      e2_prev_deg  = 0.0f;
      e1_int_deg   = 0.0f;
      e2_int_deg   = 0.0f;
      Serial.println(F("Motores detenidos (S)."));
    }
  }
}

// ================== CONTROL Y CÁLCULO DE ÁNGULOS ==================
void controlStep() {
  unsigned long now = millis();
  if (now - previousMillis < Ts) {
    leerComandosSerial();
    return;
  }
  previousMillis = now;

  float Ts_s = (float)Ts / 1000.0f;   // periodo en segundos

  // Actualizar ángulos desde los encoders
  int32_t slots_1, slots_2;
  noInterrupts();
  slots_1 = slotCount_1;
  slots_2 = slotCount_2;
  interrupts();

  // ----- Encoder 1 (con filtro anti-salto) -----
  int32_t rel_1   = slots_1 - slotOffset_1;
  float   turns_1 = (float)rel_1 / (float)SLOTS_PER_REV_1;
  float   deg_1   = turns_1 * 360.0f;   // sin mod 360, ángulo "crudo"

  // Inicializar el filtro la primera vez
  if (!ang1_filt_init) {
    angulo_1_filtrado = deg_1;
    ang1_filt_init    = true;
  }

  float delta_deg = deg_1 - angulo_1_filtrado;

  // Máximo cambio razonable de ángulo por ciclo (ajústalo si hace falta)
  const float MAX_DEG_STEP_1 = 10.0f; // 10° cada 50 ms ~ 200°/s

  if (delta_deg >  MAX_DEG_STEP_1) delta_deg =  MAX_DEG_STEP_1;
  if (delta_deg < -MAX_DEG_STEP_1) delta_deg = -MAX_DEG_STEP_1;

  angulo_1_filtrado += delta_deg;

  // Ángulos "oficiales" de M1
  angulo_1     = angulo_1_filtrado;               // grados (continuo)
  angulo_1_rad = angulo_1_filtrado * PI / 180.0f; // radianes

  // ----- Encoder 2 (como antes) -----
  int32_t rel_2   = slots_2 - slotOffset_2;
  float   turns_2 = (float)rel_2 / (float)SLOTS_PER_REV_2;
  float   deg_2   = turns_2 * 360.0f;
  float aux2 = fmod(deg_2, 360.0f);
  if (aux2 < 0) aux2 += 360.0f;
  angulo_2     = aux2;
  angulo_2_rad = wrapToPi(turns_2 * 2.0f * PI);

  // Inicializar referencia interna al ángulo actual la primera vez
  if (!refIntInit) {
    Ref1_int_deg = angulo_1;
    Ref2_int_deg = angulo_2;
    refIntInit   = true;
  }

  if (stopAll) {
    duty1 = 0;
    duty2 = 0;
    analogWrite(OutputPWM_GPIO_1, 0);
    analogWrite(OutputPWM_GPIO_2, 0);
  } else {
    // ---------- Usar referencias directamente (sin escalonado) ----------
    float Ref1_cmd_deg = Ref_1 * 180.0f / PI;
    float Ref2_cmd_deg = Ref_2 * 180.0f / PI;
    Ref1_int_deg = Ref1_cmd_deg;
    Ref2_int_deg = Ref2_cmd_deg;

    // ----- Control en grados usando referencia interna -----
    float e1_deg = Ref1_int_deg - angulo_1;
    float e2_deg = Ref2_int_deg - angulo_2;

    // Envolvente [-180,180] (para rangos menores a 180° no afecta)
    if (e1_deg > 180.0f) e1_deg -= 360.0f;
    if (e1_deg < -180.0f) e1_deg += 360.0f;
    if (e2_deg > 180.0f) e2_deg -= 360.0f;
    if (e2_deg < -180.0f) e2_deg += 360.0f;

    dbg_e1_deg = e1_deg;
    dbg_e2_deg = e2_deg;

    // ===== Derivadas del error (para D) =====
    float de1_deg = (e1_deg - e1_prev_deg) / Ts_s;
    float de2_deg = (e2_deg - e2_prev_deg) / Ts_s;
    e1_prev_deg = e1_deg;
    e2_prev_deg = e2_deg;

    // ===== Integrales del error (para I) =====
    e1_int_deg += e1_deg * Ts_s;
    e2_int_deg += e2_deg * Ts_s;

    // AUMENTAMOS EL MÁXIMO DEL INTEGRADOR PARA M1
    const float I1_MAX = 2000.0f;   // << antes 300.0f
    const float I2_MAX = 200.0f;

    if (e1_int_deg >  I1_MAX) e1_int_deg =  I1_MAX;
    if (e1_int_deg < -I1_MAX) e1_int_deg = -I1_MAX;

    if (e2_int_deg >  I2_MAX) e2_int_deg =  I2_MAX;
    if (e2_int_deg < -I2_MAX) e2_int_deg = -I2_MAX;

    // ====================== MOTOR 1: PI-D con limitación de empuje hacia abajo ======================
    float u1 = k_p_1 * e1_deg + k_i_1 * e1_int_deg + k_d_1 * de1_deg;

    // Saturación general
    u1 = clampf(u1, -U1_MAX, U1_MAX);

    // Si estamos por ENCIMA de la referencia (e1 < 0) y el control va hacia abajo (u1 < 0),
    // limitamos el empuje negativo para que no se lance con toda la fuerza.
    if (e1_deg < 0.0f && u1 < 0.0f) {
      if (u1 < -U1_MAX_DOWN) {
        u1 = -U1_MAX_DOWN;
      }
    }

    int dir1;
    if (u1 >= 0.0f) {
      dir1  = +1;
      duty1 = (uint8_t)(u1);
    } else {
      dir1  = -1;
      duty1 = (uint8_t)(-u1);
    }

    // Pequeño umbral para evitar ruidito de PWM muy bajo
    if (duty1 < 5) duty1 = 0;

    // aplicar inversión lógica si hace falta
    dir1 *= M1_DIR_INV;

    dbg_dir1 = dir1;
    dbg_u1   = u1;

    // ====================== MOTOR 2: PD clásico con deadband ======================
    float u2 = k_p_2 * e2_deg + k_i_2 * e2_int_deg + k_d_2 * de2_deg;

    if (fabs(e2_deg) < DEAD_BAND_DEG) {
      u2 = 0.0f;
    }

    u2 = clampf(u2, -U2_MAX, U2_MAX);

    int dir2;
    if (u2 >= 0.0f) {
      dir2  = +1;
      duty2 = (uint8_t)(u2);
    } else {
      dir2  = -1;
      duty2 = (uint8_t)(-u2);
    }

    if (duty2 < 5) duty2 = 0;

    dbg_dir2 = dir2;
    dbg_u2   = u2;

    // --------- Dirección motor 1 + actualizar signo para el encoder ---------
    if (dir1 >= 0) {
      digitalWrite(IN_1_1, HIGH);
      digitalWrite(IN_2_1, LOW);
      noInterrupts();
      dirSign_1 = +1;
      interrupts();
    } else {
      digitalWrite(IN_1_1, LOW);
      digitalWrite(IN_2_1, HIGH);
      noInterrupts();
      dirSign_1 = -1;
      interrupts();
    }

    // --------- Dirección motor 2 + actualizar signo para el encoder ---------
    if (dir2 >= 0) {
      digitalWrite(IN_1_2, HIGH);
      digitalWrite(IN_2_2, LOW);
      noInterrupts();
      dirSign_2 = +1;
      interrupts();
    } else {
      digitalWrite(IN_1_2, LOW);
      digitalWrite(IN_2_2, HIGH);
      noInterrupts();
      dirSign_2 = -1;
      interrupts();
    }

    // Si solo estás probando M1, puedes dejar M2 apagado:
    analogWrite(OutputPWM_GPIO_1, duty1);
    analogWrite(OutputPWM_GPIO_2, 0);  // o duty2 cuando quieras usarlo
  }

  // ----- Telemetría -----
  if (now - lastPrint >= PRINT_MS) {
    lastPrint = now;
    // Telemetría compacta para el simulador:
    // Y,<millis>,<q1>,<q2>,<q1_ref>,<q2_ref>,<u1>,<u2>
    float ref1_int_rad = Ref1_int_deg * PI / 180.0f;
    float ref2_int_rad = Ref2_int_deg * PI / 180.0f;
    float u1_norm = clampf(dbg_u1 / U1_MAX, -1.0f, 1.0f);
    float u2_norm = clampf(dbg_u2 / U2_MAX, -1.0f, 1.0f);

    Serial.print('Y'); Serial.print(',');
    Serial.print(millis()); Serial.print(',');
    Serial.print(angulo_1_rad, 6); Serial.print(',');
    Serial.print(angulo_2_rad, 6); Serial.print(',');
    Serial.print(ref1_int_rad, 6); Serial.print(',');
    Serial.print(ref2_int_rad, 6); Serial.print(',');
    Serial.print(u1_norm, 6); Serial.print(',');
    Serial.println(u2_norm, 6);
  }

  leerComandosSerial();
}

// ================== SETUP / LOOP ==================
void setup() {
  Serial.begin(115200);

  pinMode(Sensor_1, INPUT_PULLUP);
  pinMode(Sensor_2, INPUT_PULLUP);

  pinMode(IN_1_1, OUTPUT);
  pinMode(IN_2_1, OUTPUT);
  pinMode(IN_1_2, OUTPUT);
  pinMode(IN_2_2, OUTPUT);
  pinMode(OutputPWM_GPIO_1, OUTPUT);
  pinMode(OutputPWM_GPIO_2, OUTPUT);

  if (COUNT_ON_FALL_1) {
    attachInterrupt(digitalPinToInterrupt(Sensor_1), isrSlot1, FALLING);
  } else {
    attachInterrupt(digitalPinToInterrupt(Sensor_1), isrSlot1, RISING);
  }

  if (COUNT_ON_FALL_2) {
    attachInterrupt(digitalPinToInterrupt(Sensor_2), isrSlot2, FALLING);
  } else {
    attachInterrupt(digitalPinToInterrupt(Sensor_2), isrSlot2, RISING);
  }

  delay(50);

  // Cero inicial (posición colgando)
  noInterrupts();
  slotOffset_1 = slotCount_1;
  slotOffset_2 = slotCount_2;
  dirSign_1    = +1;
  dirSign_2    = +1;
  refIntInit   = false;
  e1_prev_deg  = 0.0f;
  e2_prev_deg  = 0.0f;
  e1_int_deg   = 0.0f;
  e2_int_deg   = 0.0f;
  ang1_filt_init    = false;
  angulo_1_filtrado = 0.0f;
  interrupts();

  Serial.println(F("Sistema listo (M1 con PI-D, limitación hacia abajo y filtro anti-salto en encoder)."));
  Serial.println(F("Comandos:"));
  Serial.println(F("  Z                -> recalibrar cero"));
  Serial.println(F("  R,th1,th2        -> referencias en radianes"));
  Serial.println(F("  P,Kp1,Kp2        -> ajustar Kp"));
  Serial.println(F("  I,Ki1,Ki2        -> ajustar Ki"));
  Serial.println(F("  D,Kd1,Kd2        -> ajustar Kd"));
  Serial.println(F("  S                -> stop"));
}

void loop() {
  controlStep();
}
