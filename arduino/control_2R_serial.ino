// Control 2R por referencias seriales (radianes) con control proporcional
// HW supuesto: Arduino UNO (Timer1 PWM 12-bit en pines 9 y 10) + DIR en pines 7 y 8
// Encoders de ranuras en pines 2 y 3 (interrupciones externas)

#include <Arduino.h>

// ==== Configuración de pines ====
#define ENC1_PIN           2
#define ENC2_PIN           3
#define M1_PWM_PIN         9    // OC1A
#define M2_PWM_PIN         10   // OC1B
#define M1_DIR_PIN         7
#define M2_DIR_PIN         8

// ==== PWM 12-bit en Timer1 ====
#define PWM_MAX            4095
static inline void setupPWM12() {
  DDRB |= _BV(PB1) | _BV(PB2);               // pins 9,10 como salida
  TCCR1A = _BV(COM1A1) | _BV(COM1B1) | _BV(WGM11);  // no-inverting, modo 14
  TCCR1B = _BV(WGM13) | _BV(WGM12) | _BV(CS10);     // no prescaling
  ICR1   = 0x0FFF;                                 // TOP 12-bit
}
static inline void analogWrite12(uint8_t pin, uint16_t val) {
  if (val > PWM_MAX) val = PWM_MAX;
  switch (pin) {
    case 9:  OCR1A = val; break;
    case 10: OCR1B = val; break;
  }
}

// ==== Encoders (ranuras) ====
const uint16_t SLOTS_PER_REV_1 = 180;
const uint16_t SLOTS_PER_REV_2 = 180;
const bool COUNT_ON_FALL_1 = true;
const bool COUNT_ON_FALL_2 = true;
const uint32_t DEBOUNCE_US = 150;

volatile uint32_t slotCount1 = 0;
volatile uint32_t slotCount2 = 0;
volatile uint32_t lastUs1 = 0;
volatile uint32_t lastUs2 = 0;

// Offsets de cero (en ranuras)
volatile int32_t slotOffset1 = 0;
volatile int32_t slotOffset2 = 0;

void isrSlot1() {
  uint32_t now = micros();
  if (now - lastUs1 < DEBOUNCE_US) return;
  slotCount1++;
  lastUs1 = now;
}
void isrSlot2() {
  uint32_t now = micros();
  if (now - lastUs2 < DEBOUNCE_US) return;
  slotCount2++;
  lastUs2 = now;
}

// ==== Utilidades ====
static inline float clampf(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }
static inline float wrapToPi(float a) {
  while (a >  PI) a -= 2.0f*PI;
  while (a <= -PI) a += 2.0f*PI;
  return a;
}

// Lee ángulo en radianes (normalizado a (-π, π]) a partir de ranuras
float readJointRad(uint8_t joint) {
  uint32_t cnt; int32_t off; uint16_t spr;
  if (joint == 1) { noInterrupts(); cnt = slotCount1; off = slotOffset1; spr = SLOTS_PER_REV_1; interrupts(); }
  else            { noInterrupts(); cnt = slotCount2; off = slotOffset2; spr = SLOTS_PER_REV_2; interrupts(); }
  if (spr == 0) return 0.0f;
  int32_t rel = ((int32_t)cnt - off);
  // ángulo absoluto reducido al intervalo de una vuelta
  float turns = (float)(rel % spr) / (float)spr;  // [-spr..spr] -> [-1,1)
  float ang = turns * (2.0f * PI);
  return wrapToPi(ang);
}

// ==== Control proporcional ====
volatile float Kp1 = 0.8f;
volatile float Kp2 = 0.8f;

// Referencias en radianes (desde Python)
volatile float q1_ref = 0.0f;
volatile float q2_ref = 0.0f;

// ==== Serial protocolo simple ====
// "R,theta1,theta2"  (radianes)
// "P,Kp1,Kp2"
// "Z"  (cero rápido)
// "S"  (stop PWM)

char lineBuf[64];
uint8_t lineLen = 0;

void processLine() {
  lineBuf[lineLen] = '\0';
  if (lineLen == 0) return;

  if (lineBuf[0] == 'R') {
    // R,th1,th2
    float a, b; char c;
    if (sscanf(lineBuf, "R%c%f%c%f", &c, &a, &c, &b) >= 5) {
      noInterrupts(); q1_ref = a; q2_ref = b; interrupts();
    }
  } else if (lineBuf[0] == 'P') {
    float p1, p2; char c;
    if (sscanf(lineBuf, "P%c%f%c%f", &c, &p1, &c, &p2) >= 5) {
      noInterrupts(); Kp1 = p1; Kp2 = p2; interrupts();
    }
  } else if (lineBuf[0] == 'Z') {
    // toma los conteos actuales como cero (offset)
    noInterrupts(); slotOffset1 = (int32_t)slotCount1; slotOffset2 = (int32_t)slotCount2; interrupts();
  } else if (lineBuf[0] == 'S') {
    analogWrite12(M1_PWM_PIN, 0);
    analogWrite12(M2_PWM_PIN, 0);
  }
}

void pollSerial() {
  while (Serial.available()) {
    char ch = (char)Serial.read();
    if (ch == '\n' || ch == '\r') {
      processLine();
      lineLen = 0;
    } else {
      if (lineLen < sizeof(lineBuf) - 1) lineBuf[lineLen++] = ch;
    }
  }
}

// ==== Bucle a 100 Hz ====
const uint32_t Ts_ms = 10;
uint32_t lastTickMs = 0;

// Telemetría cada 50 ms
const uint32_t TEL_MS = 50;
uint32_t lastTel = 0;

void setup() {
  pinMode(M1_DIR_PIN, OUTPUT);
  pinMode(M2_DIR_PIN, OUTPUT);
  pinMode(M1_PWM_PIN, OUTPUT);
  pinMode(M2_PWM_PIN, OUTPUT);

  pinMode(ENC1_PIN, INPUT_PULLUP);
  pinMode(ENC2_PIN, INPUT_PULLUP);

  if (COUNT_ON_FALL_1) attachInterrupt(digitalPinToInterrupt(ENC1_PIN), isrSlot1, FALLING);
  else                 attachInterrupt(digitalPinToInterrupt(ENC1_PIN), isrSlot1, RISING);
  if (COUNT_ON_FALL_2) attachInterrupt(digitalPinToInterrupt(ENC2_PIN), isrSlot2, FALLING);
  else                 attachInterrupt(digitalPinToInterrupt(ENC2_PIN), isrSlot2, RISING);

  setupPWM12();
  analogWrite12(M1_PWM_PIN, 0);
  analogWrite12(M2_PWM_PIN, 0);

  Serial.begin(115200);
  while(!Serial) { ; }
  delay(50);

  // Cero inicial (equivalente a 'Z')
  noInterrupts(); slotOffset1 = (int32_t)slotCount1; slotOffset2 = (int32_t)slotCount2; interrupts();
}

static inline void driveMotor(uint8_t pwmPin, uint8_t dirPin, float u_norm) {
  // u_norm ∈ [-1,1]
  u_norm = clampf(u_norm, -1.0f, 1.0f);
  digitalWrite(dirPin, (u_norm >= 0.0f) ? HIGH : LOW);
  uint16_t duty = (uint16_t)(fabs(u_norm) * (float)PWM_MAX + 0.5f);
  analogWrite12(pwmPin, duty);
}

void loop() {
  pollSerial();

  uint32_t now = millis();
  if (now - lastTickMs >= Ts_ms) {
    lastTickMs = now;

    // Medición
    float q1 = readJointRad(1);
    float q2 = readJointRad(2);

    // Control P con envoltura angular
    float e1 = wrapToPi(q1_ref - q1);
    float e2 = wrapToPi(q2_ref - q2);

    // Ganancia fija (se puede actualizar con 'P,')
    float u1 = Kp1 * e1;  // [-? , ?]
    float u2 = Kp2 * e2;

    // Normalización burda: si |u|>=1 satura
    driveMotor(M1_PWM_PIN, M1_DIR_PIN, clampf(u1, -1.0f, 1.0f));
    driveMotor(M2_PWM_PIN, M2_DIR_PIN, clampf(u2, -1.0f, 1.0f));

    // Telemetría (formato compatible con scripts Python)
    if (now - lastTel >= TEL_MS) {
      lastTel = now;
      Serial.print('Y'); Serial.print(',');
      Serial.print(now); Serial.print(',');
      Serial.print(q1, 6); Serial.print(',');
      Serial.print(q2, 6); Serial.print(',');
      Serial.print(q1_ref, 6); Serial.print(',');
      Serial.print(q2_ref, 6); Serial.print(',');
      Serial.print(clampf(u1, -1.0f, 1.0f), 3); Serial.print(',');
      Serial.println(clampf(u2, -1.0f, 1.0f), 3);
    }
  }
}


