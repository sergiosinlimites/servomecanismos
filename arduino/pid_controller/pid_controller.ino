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

// Valores mínimos medidos para que arranque cada motor
#define MIN_DUTY_1     90           // PWM mínimo motor 1 para moverse
#define MIN_DUTY_2     5            // PWM mínimo motor 2 para moverse
#define MAX_DUTY_2     30           // PWM máximo motor 2 (tú lo mediste así)

#define DEAD_BAND_DEG  1.0f         // si |error| < 1°, lo consideramos en posición

// +1 = como está ahora, -1 = invierte sentido lógico eje 1 (por hardware)
#define M1_DIR_INV     (+1)

// ================== TIEMPOS ==================
unsigned long previousMillis = 0;
const unsigned long Ts = 50;        // Periodo de control = 50 ms (~20 Hz)

// ================== MEDICIÓN ÁNGULOS ==================
float angulo_1 = 0.0f;              // Ángulo motor 1 en grados [0,360)
float angulo_2 = 0.0f;              // Ángulo motor 2 en grados [0,360)
float angulo_1_rad = 0.0f;          // Ángulo motor 1 en radianes [-pi,pi]
float angulo_2_rad = 0.0f;          // Ángulo motor 2 en radianes [-pi,pi]

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

// Rampa de referencia (valores máximos de paso por ciclo)
const float STEP_FAST   = 3.0f;     // lejos
const float STEP_MEDIUM = 2.0f;     // medio
const float STEP_SLOW   = 1.0f;     // cerca

// ====== Ganancias PID (iniciales, LAS VAS A TUNEAR) ======
float k_p_1 = 2.0f, k_i_1 = 0.5f,  k_d_1 = 0.1f;    // Motor 1
float k_p_2 = 3.0f, k_i_2 = 0.8f,  k_d_2 = 0.15f;   // Motor 2

// ====== Estados del PID ======
float e1_prev_deg = 0.0f;
float e2_prev_deg = 0.0f;
float i1_term     = 0.0f;   // integral motor 1
float i2_term     = 0.0f;   // integral motor 2

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
const unsigned long PRINT_MS = 200;

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

// ================== RESET ESTADOS PID ==================
void resetPID() {
  e1_prev_deg = 0.0f;
  e2_prev_deg = 0.0f;
  i1_term     = 0.0f;
  i2_term     = 0.0f;
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
      stopAll   = false;
      refIntInit = false;   // para volver a enganchar la referencia interna
      resetPID();
      interrupts();

      Serial.print(F("Nuevas refs -> Ref_1="));
      Serial.print(Ref_1, 4);
      Serial.print(F(" rad, Ref_2="));
      Serial.print(Ref_2, 4);
      Serial.println(F(" rad"));
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
      i1_term = 0.0f;
      i2_term = 0.0f;
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
      refIntInit   = false;  // re-inicializar ref interna en la nueva posición
      resetPID();
      interrupts();

      Serial.println(F("Cero recalibrado (Z)."));
    }
    else if (c == 'S') {
      stopAll = true;
      analogWrite(OutputPWM_GPIO_1, 0);
      analogWrite(OutputPWM_GPIO_2, 0);
      resetPID();
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

  float Ts_s = (float)Ts / 1000.0f;

  // Actualizar ángulos desde los encoders
  int32_t slots_1, slots_2;
  noInterrupts();
  slots_1 = slotCount_1;
  slots_2 = slotCount_2;
  interrupts();

  // ----- Encoder 1 -----
  int32_t rel_1 = slots_1 - slotOffset_1;
  float   turns_1 = (float)rel_1 / (float)SLOTS_PER_REV_1;
  float   deg_1   = turns_1 * 360.0f;
  float aux1 = fmod(deg_1, 360.0f);
  if (aux1 < 0) aux1 += 360.0f;
  angulo_1     = aux1;
  angulo_1_rad = wrapToPi(turns_1 * 2.0f * PI);

  // ----- Encoder 2 -----
  int32_t rel_2 = slots_2 - slotOffset_2;
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
    analogWrite(OutputPWM_GPIO_1, 0);
    analogWrite(OutputPWM_GPIO_2, 0);
  } else {
    // ---------- Generador de referencia suave (en grados) ----------
    float Ref1_cmd_deg = Ref_1 * 180.0f / PI;
    float Ref2_cmd_deg = Ref_2 * 180.0f / PI;

    // Rampa no lineal para motor 1
    float diff1 = Ref1_cmd_deg - Ref1_int_deg;
    float dist1 = fabs(diff1);
    float step1;
    if (dist1 > 40.0f)      step1 = STEP_FAST;    // lejos
    else if (dist1 > 15.0f) step1 = STEP_MEDIUM;  // medio
    else                    step1 = STEP_SLOW;    // cerca

    if (fabs(diff1) <= step1) {
      Ref1_int_deg = Ref1_cmd_deg;
    } else {
      Ref1_int_deg += (diff1 > 0.0f ? step1 : -step1);
    }

    // Rampa no lineal para motor 2
    float diff2 = Ref2_cmd_deg - Ref2_int_deg;
    float dist2 = fabs(diff2);
    float step2;
    if (dist2 > 40.0f)      step2 = STEP_FAST;
    else if (dist2 > 15.0f) step2 = STEP_MEDIUM;
    else                    step2 = STEP_SLOW;

    if (fabs(diff2) <= step2) {
      Ref2_int_deg = Ref2_cmd_deg;
    } else {
      Ref2_int_deg += (diff2 > 0.0f ? step2 : -step2);
    }

    // ----- Errores en grados (usando referencia interna) -----
    float e1_deg = Ref1_int_deg - angulo_1;
    float e2_deg = Ref2_int_deg - angulo_2;

    // Envolvente opcional [-180,180] para evitar caminos largos
    if (e1_deg > 180.0f) e1_deg -= 360.0f;
    if (e1_deg < -180.0f) e1_deg += 360.0f;
    if (e2_deg > 180.0f) e2_deg -= 360.0f;
    if (e2_deg < -180.0f) e2_deg += 360.0f;

    dbg_e1_deg = e1_deg;
    dbg_e2_deg = e2_deg;

    // ===== PID MOTOR 1 =====
    float u1 = 0.0f;
    if (fabs(e1_deg) < DEAD_BAND_DEG) {
      // Cerca de la referencia: no actuamos, reseteamos integral
      i1_term = 0.0f;
      u1      = 0.0f;
    } else {
      // Integral con anti-windup sencillo
      i1_term += e1_deg * Ts_s;
      if (k_i_1 > 1e-6f) {
        float i1_max = PWM_MAX / k_i_1;
        i1_term = clampf(i1_term, -i1_max, i1_max);
      }

      float de1 = (e1_deg - e1_prev_deg) / Ts_s;
      e1_prev_deg = e1_deg;

      u1 = k_p_1 * e1_deg + k_i_1 * i1_term + k_d_1 * de1;
    }

    // Saturación de u1
    u1 = clampf(u1, -PWM_MAX, PWM_MAX);
    dbg_u1 = u1;

    // Dirección y PWM motor 1
    int dir1 = (u1 >= 0.0f) ? 1 : -1;
    dir1 *= M1_DIR_INV;
    dbg_dir1 = dir1;

    float mag1 = fabs(u1);
    if (mag1 < 1.0f) {
      duty1 = 0;
    } else {
      if (mag1 < MIN_DUTY_1) mag1 = MIN_DUTY_1;       // vencer fricción
      if (mag1 > PWM_MAX)    mag1 = PWM_MAX;
      duty1 = (uint8_t)mag1;
    }

    // ===== PID MOTOR 2 =====
    float u2 = 0.0f;
    if (fabs(e2_deg) < DEAD_BAND_DEG) {
      i2_term = 0.0f;
      u2      = 0.0f;
    } else {
      i2_term += e2_deg * Ts_s;
      if (k_i_2 > 1e-6f) {
        float i2_max = PWM_MAX / k_i_2;
        i2_term = clampf(i2_term, -i2_max, i2_max);
      }

      float de2 = (e2_deg - e2_prev_deg) / Ts_s;
      e2_prev_deg = e2_deg;

      u2 = k_p_2 * e2_deg + k_i_2 * i2_term + k_d_2 * de2;
    }

    // Saturación de u2
    u2 = clampf(u2, -PWM_MAX, PWM_MAX);
    dbg_u2 = u2;

    int dir2 = (u2 >= 0.0f) ? 1 : -1;
    dbg_dir2 = dir2;

    float mag2 = fabs(u2);
    if (mag2 < 1.0f) {
      duty2 = 0;
    } else {
      // Escalamos al rango [MIN_DUTY_2, MAX_DUTY_2]
      if (mag2 > PWM_MAX) mag2 = PWM_MAX;
      float norm2 = mag2 / PWM_MAX;  // 0..1
      float d2f = MIN_DUTY_2 + norm2 * (MAX_DUTY_2 - MIN_DUTY_2);
      if (d2f > MAX_DUTY_2) d2f = MAX_DUTY_2;
      duty2 = (uint8_t)d2f;
    }

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

    analogWrite(OutputPWM_GPIO_1, duty1);
    analogWrite(OutputPWM_GPIO_2, duty2);
  }

  // ----- Telemetría -----
  if (now - lastPrint >= PRINT_MS) {
    lastPrint = now;
    Serial.print("theta1: ");    Serial.print(angulo_1_rad, 6);
    Serial.print("  theta2: ");  Serial.print(angulo_2_rad, 6);

    float ref1_int_rad = Ref1_int_deg * PI / 180.0f;
    float ref2_int_rad = Ref2_int_deg * PI / 180.0f;

    Serial.print("  ref1: ");    Serial.print(ref1_int_rad, 6);
    Serial.print("  ref2: ");    Serial.print(ref2_int_rad, 6);

    Serial.print("  e1_deg: ");  Serial.print(dbg_e1_deg, 1);
    Serial.print("  e2_deg: ");  Serial.print(dbg_e2_deg, 1);

    Serial.print("  u1: ");      Serial.print(dbg_u1, 1);
    Serial.print("  u2: ");      Serial.print(dbg_u2, 1);

    Serial.print("  dir1: ");    Serial.print(dbg_dir1);
    Serial.print("  dir2: ");    Serial.print(dbg_dir2);

    Serial.print("  duty1: ");   Serial.print(duty1);
    Serial.print("  duty2: ");   Serial.print(duty2);

    Serial.print("  stop: ");    Serial.println(stopAll ? 1 : 0);
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
  resetPID();
  interrupts();

  Serial.println(F("Sistema listo (PID completo + referencia interna suave)."));
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
