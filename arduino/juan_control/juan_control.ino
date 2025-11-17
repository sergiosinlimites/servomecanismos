// ======================= CONFIGURACIÓN HARDWARE (TU MONTAJE) =======================

// Encoders (OPB800) - MODO POLLING
#define Sensor    2      // Encoder motor 1
#define Sensor_2  3      // Encoder motor 2

// L298
#define OutputPWM_GPIO    5    // ENA L298 (motor 1)  -> PWM
#define OutputPWM_GPIO_2 10    // ENB L298 (motor 2)  -> PWM

// Pines de dirección L298
const uint8_t PIN_IN1 = 6;   // IN1 motor 1
const uint8_t PIN_IN2 = 7;   // IN2 motor 1
const uint8_t PIN_IN3 = 9;   // IN3 motor 2
const uint8_t PIN_IN4 = 8;   // IN4 motor 2

// Inversión lógica por si un eje quedó “al revés” de lo que queremos (+1 o -1)
#define M1_DIR_INV (+1)
#define M2_DIR_INV (+1)

// PWM 8 bits (analogWrite)
#define pwmMax 255

// Límites de PWM según tu montaje
#define MIN_DUTY_1 200   // motor 1: mínimo para vencer peso cerca de 0°
#define MAX_DUTY_1 255

#define MIN_DUTY_2 10    // motor 2
#define MAX_DUTY_2 40

// Deadband de error angular (en grados)
#define DEAD_BAND_1 2.0f
#define DEAD_BAND_2 2.0f

// Unidades para el control PID (escala interna)
#define Uunits 100       // "máximo" teórico de u_n para normalizar

// ======================= TIEMPOS =======================
unsigned long pTime = 0;
unsigned long dTime = 0;
long previousMillis  = 0;  // Para la función del bucle principal
long Ts = 10;              // Tiempo de muestreo en ms (control cada 10 ms)
long previousMillis2 = 0;

// ======================= SERIAL AVANZADO =======================
const byte numChars = 32;
char receivedChars[numChars];
boolean newData = false;

// ======================= VARIABLES DE MEDICIÓN =======================
float angulo    = 0.0;  // ángulo motor 1 [deg]
float angulo_2  = 0.0;  // ángulo motor 2 [deg]

// ======================= REFERENCIAS =======================
float Ref   = 10.0;   // referencia motor 1 (grados)
float Ref_2 = 80.0;   // referencia motor 2 (grados)

// ======================= SALIDAS PWM =======================
unsigned int pwmDuty   = 0;  // motor 1
unsigned int pwmDuty_2 = 0;  // motor 2

// ======================= PID MOTOR 1 =======================
float k_p = 1.0;     
float k_i = 26.4;
float k_d = 0.008;
int   N   = 100;

float e_n      = 0.0, e_n_1      = 0.0;
float u_n      = 0.0, u_p        = 0.0, u_i       = 0.0, u_d       = 0.0;
float u_n_1_i  = 0.0, u_n_1_d    = 0.0, u_n_1     = 0.0;

// Tal cual en el código de tus compas
float ts = 0.0010;  // [s]

// ======================= PID MOTOR 2 =======================
float k_p_2 = 0.1;    
float k_i_2 = 0.1;
float k_d_2 = 0.001;

float e_n_2     = 0.0, e_n_1_2   = 0.0;
float u_n_2     = 0.0, u_p_2     = 0.0, u_i_2     = 0.0, u_d_2     = 0.0;
float u_n_1_i_2 = 0.0, u_n_1_d_2 = 0.0, u_n_1_2   = 0.0;

// ======================= ENCODERS (POLLING) =======================
const uint16_t SLOTS_PER_REV  = 180;   // # de ranuras por vuelta
const bool     COUNT_ON_FALL  = true;  // true: FALLING, false: RISING
const bool     COUNT_ON_FALL_2 = true;

int32_t slotCount   = 0;
int32_t slotCount_2 = 0;

uint32_t lastUs   = 0;
uint32_t lastUs_2 = 0;
const uint32_t DEBOUNCE_US = 100;

uint8_t  lastState   = HIGH;
uint8_t  lastState_2 = HIGH;

uint32_t lastPrint   = 0;
uint32_t lastPrint_2 = 0;

// IMPORTANTE: signo de la dirección REAL que estamos mandando al motor
volatile int8_t dirSign_1 = +1;
volatile int8_t dirSign_2 = +1;

// ======================= LECTURA ENCODER 1 (POLLING) =======================
void leerEncoder() {
  uint8_t currentState = digitalRead(Sensor);

  if (currentState != lastState) {
    uint32_t now = micros();

    if (now - lastUs >= DEBOUNCE_US) {
      if (COUNT_ON_FALL) {
        // FALLING: HIGH -> LOW
        if (lastState == HIGH && currentState == LOW) {
          slotCount += dirSign_1;   // AHORA usa la dirección real, NO el error
          lastUs = now;
        }
      } else {
        // RISING: LOW -> HIGH
        if (lastState == LOW && currentState == HIGH) {
          slotCount += dirSign_1;
          lastUs = now;
        }
      }
    }
    lastState = currentState;
  }
}

// ======================= LECTURA ENCODER 2 (POLLING) =======================
void leerEncoder_2() {
  uint8_t currentState_2 = digitalRead(Sensor_2);

  if (currentState_2 != lastState_2) {
    uint32_t now_2 = micros();

    if (now_2 - lastUs_2 >= DEBOUNCE_US) {
      if (COUNT_ON_FALL_2) {
        if (lastState_2 == HIGH && currentState_2 == LOW) {
          slotCount_2 += dirSign_2;
          lastUs_2 = now_2;
        }
      } else {
        if (lastState_2 == LOW && currentState_2 == HIGH) {
          slotCount_2 += dirSign_2;
          lastUs_2 = now_2;
        }
      }
    }
    lastState_2 = currentState_2;
  }
}

// ======================= FUNCIÓN DE CONTROL (ambos motores) =======================
void calibracion(void) {
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= Ts) {
    previousMillis = currentMillis;

    const float degPerSlot = 360.0f / (float)SLOTS_PER_REV;

    // ------------------ MOTOR 1 ------------------
    int32_t slots = slotCount;
    angulo = slots * degPerSlot;   // puede ser negativo

    // Error (Ref en grados)
    e_n = Ref - angulo;

    // PID discreto (tal cual lo tenían tus compas)
    u_p = k_p * e_n;
    u_i = (k_i * ts * e_n_1) + u_n_1_i;
    u_d = (k_d * N * e_n) - (k_d * N * e_n_1) - (N * ts * u_n_1_d) + u_n_1_d;
    u_n = u_p + u_i + u_d;

    // Actualizar términos anteriores
    e_n_1    = e_n;
    u_n_1_i  = u_i;
    u_n_1_d  = u_d;
    u_n_1    = u_n;

    // Sentido de giro según error (aplicando inversión lógica si hace falta)
    int dirLogic1 = (e_n >= 0.0f) ? 1 : -1;
    dirLogic1 *= M1_DIR_INV;

    if (dirLogic1 >= 0) {
      digitalWrite(PIN_IN1, HIGH);
      digitalWrite(PIN_IN2, LOW);
      dirSign_1 = +1;
    } else {
      digitalWrite(PIN_IN1, LOW);
      digitalWrite(PIN_IN2, HIGH);
      dirSign_1 = -1;
    }

    // Magnitud de PWM a partir de |u_n|
    float U_tl = fabs(u_n);
    if (U_tl > Uunits) U_tl = Uunits;

    if (fabs(e_n) < DEAD_BAND_1) {
      pwmDuty = 0;   // dentro de la ventana muerta
    } else {
      float u_norm = U_tl / Uunits;  // 0..1
      float duty_f = (float)MIN_DUTY_1 + u_norm * ((float)MAX_DUTY_1 - (float)MIN_DUTY_1);
      if (duty_f > (float)MAX_DUTY_1) duty_f = (float)MAX_DUTY_1;
      if (duty_f < (float)MIN_DUTY_1) duty_f = (float)MIN_DUTY_1;
      pwmDuty = (unsigned int)duty_f;
    }

    analogWrite(OutputPWM_GPIO, pwmDuty);

    // ------------------ MOTOR 2 ------------------
    int32_t slots_2 = slotCount_2;
    angulo_2 = slots_2 * degPerSlot;

    e_n_2 = Ref_2 - angulo_2;

    u_p_2 = k_p_2 * e_n_2;
    u_i_2 = (k_i_2 * ts * e_n_1_2) + u_n_1_i_2;
    u_d_2 = (k_d_2 * N * e_n_2) - (k_d_2 * N * e_n_1_2) - (N * ts * u_n_1_d_2) + u_n_1_d_2;
    u_n_2 = u_p_2 + u_d_2 + u_i_2;

    e_n_1_2    = e_n_2;
    u_n_1_i_2  = u_i_2;
    u_n_1_d_2  = u_d_2;
    u_n_1_2    = u_n_2;

    int dirLogic2 = (e_n_2 >= 0.0f) ? 1 : -1;
    dirLogic2 *= M2_DIR_INV;

    if (dirLogic2 >= 0) {
      digitalWrite(PIN_IN3, HIGH);
      digitalWrite(PIN_IN4, LOW);
      dirSign_2 = +1;
    } else {
      digitalWrite(PIN_IN3, LOW);
      digitalWrite(PIN_IN4, HIGH);
      dirSign_2 = -1;
    }

    float U_tl_2 = fabs(u_n_2);
    if (U_tl_2 > Uunits) U_tl_2 = Uunits;

    if (fabs(e_n_2) < DEAD_BAND_2) {
      pwmDuty_2 = 0;
    } else {
      float u_norm2 = U_tl_2 / Uunits;
      float duty_f2 = (float)MIN_DUTY_2 + u_norm2 * ((float)MAX_DUTY_2 - (float)MIN_DUTY_2);
      if (duty_f2 > (float)MAX_DUTY_2) duty_f2 = (float)MAX_DUTY_2;
      if (duty_f2 < (float)MIN_DUTY_2) duty_f2 = (float)MIN_DUTY_2;
      pwmDuty_2 = (unsigned int)duty_f2;
    }

    analogWrite(OutputPWM_GPIO_2, pwmDuty_2);

    // --------- LOG POR SERIAL ---------
    Serial.print("Tiempo: ");
    Serial.print(millis());
    Serial.print(", ang1: ");
    Serial.print(angulo);
    Serial.print(", Ref1: ");
    Serial.print(Ref);
    Serial.print(", PWM1: ");
    Serial.print((pwmDuty * 100.0) / pwmMax);
    Serial.print("%, err1: ");
    Serial.print(e_n);

    Serial.print("  ||  ang2: ");
    Serial.print(angulo_2);
    Serial.print(", Ref2: ");
    Serial.print(Ref_2);
    Serial.print(", PWM2: ");
    Serial.print((pwmDuty_2 * 100.0) / pwmMax);
    Serial.print("%, err2: ");
    Serial.print(e_n_2);
    Serial.print(", U_n2: ");
    Serial.println(u_n_2);
  }

  // Entrada serial avanzada (solo cambia Ref_2, como en el original)
  recvWithStartEndMarkers();  
  if (newData == true) {
    parseData();
    newData = false;
  }
}

// ======================= SERIAL AVANZADO =======================
void recvWithStartEndMarkers() {
  static boolean recvInProgress = false;
  static byte ndx = 0;
  char startMarker = '<';
  char endMarker   = '>';
  char rc;

  while (Serial.available() > 0 && newData == false) {
    rc = Serial.read();

    if (recvInProgress == true) {
      if (rc != endMarker) {
        receivedChars[ndx] = rc;
        ndx++;
        if (ndx >= numChars) {
          ndx = numChars - 1;
        }
      } else {
        receivedChars[ndx] = '\0';
        recvInProgress = false;
        ndx = 0;
        newData = true;
      }
    } else if (rc == startMarker) {
      recvInProgress = true;
    }
  }
}

void parseData() {
  // De momento solo cambia Ref_2 como en el código original
  Ref_2 = atof(receivedChars); 
  Serial.print("Nueva referencia motor 2: ");
  Serial.println(Ref_2);
}

// ======================= SETUP =======================
void setup() {
  Serial.begin(115200);

  pinMode(Sensor,   INPUT_PULLUP);
  pinMode(Sensor_2, INPUT_PULLUP);

  pinMode(PIN_IN1, OUTPUT);
  pinMode(PIN_IN2, OUTPUT);
  pinMode(PIN_IN3, OUTPUT);
  pinMode(PIN_IN4, OUTPUT);

  // Estado inicial (motor 1 y 2 hacia "sentido positivo lógico")
  digitalWrite(PIN_IN1, HIGH);
  digitalWrite(PIN_IN2, LOW);
  digitalWrite(PIN_IN3, HIGH);
  digitalWrite(PIN_IN4, LOW);
  dirSign_1 = +1;
  dirSign_2 = +1;

  // PWM inicial a 0
  analogWrite(OutputPWM_GPIO,   0);
  analogWrite(OutputPWM_GPIO_2, 0);

  // Estados iniciales de los encoders
  lastState   = digitalRead(Sensor);
  lastState_2 = digitalRead(Sensor_2);

  Serial.println(F("PID (compas) adaptado a tu montaje:"));
  Serial.println(F(" - Pines cambiados a tus L298/encoders"));
  Serial.println(F(" - Encoder suma/resta según dirSign, no según error"));
  Serial.println(F(" - MIN/MAX PWM ajustados a tu brazo pesado"));
  Serial.println(F("Envía <valor> por Serial para cambiar Ref_2 (en grados, ej: <90>)"));
  delay(2000);
}

// ======================= LOOP =======================
void loop() {
  // Leer encoders constantemente
  leerEncoder();
  leerEncoder_2();

  // Ejecutar control cada Ts
  calibracion();
}
