// ======= 2R Tracker (Arduino UNO/MEGA) =======
// Recibe θ1, θ2 [rad] por Serial y ejecuta control PD por articulación.
// Telemetría: Y,<ms>,<q1>,<q2>,<q1_ref>,<q2_ref>,<u1>,<u2>

#include <Arduino.h>

// ---- Pines (ajusta si usas otro driver) ----
const int ENC1_PIN = A1;     // pot/encoder 1 (0..1023 -> 0..2π)
const int ENC2_PIN = A2;     // pot/encoder 2
const int M1_PWM   = 9;
const int M1_DIR   = 8;
const int M2_PWM   = 10;
const int M2_DIR   = 7;

// ---- Geometría (no usada en control, solo referencia) ----
const float L1 = 0.12f;  // [m]
const float L2 = 0.12f;  // [m]

// ---- Mapeo analógico->ángulo (rad) ----
const float RAD_PER_ADC1 = (2.0f * PI) / 1023.0f;
const float RAD_PER_ADC2 = (2.0f * PI) / 1023.0f;
float off1 = 0.0f;  // offset cero junta 1
float off2 = 0.0f;  // offset cero junta 2

// ---- Control PD ----
const float Ts_ms = 10.0f;                 // período de muestreo [ms]
const float Ts    = Ts_ms / 1000.0f;       // [s]
float Kp1 = 18.0f, Kd1 = 0.9f;             // ganas por defecto
float Kp2 = 18.0f, Kd2 = 0.9f;
float e1_prev = 0, e2_prev = 0;

// ---- Referencias por Serial ----
volatile float q1_ref = 0.0f, q2_ref = 0.0f;
volatile bool  has_ref = false;
volatile bool  stop_cmd = false;

// ---- Telemetría ----
const int LOG_DIV = 2; // 100 Hz / 2 = 50 Hz
int log_div_cnt = 0;

// ---- Utils ----
static inline float clampf(float x, float lo, float hi){
  return x < lo ? lo : (x > hi ? hi : x);
}

// ---- Encoders (o potenciómetros) ----
float readJoint1(){
  int raw = analogRead(ENC1_PIN);
  float ang = raw * RAD_PER_ADC1 - off1;
  while(ang >  PI) ang -= 2*PI;
  while(ang < -PI) ang += 2*PI;
  return ang;
}
float readJoint2(){
  int raw = analogRead(ENC2_PIN);
  float ang = raw * RAD_PER_ADC2 - off2;
  while(ang >  PI) ang -= 2*PI;
  while(ang < -PI) ang += 2*PI;
  return ang;
}

// ---- Actuación de motor con comando en [-1,1] ----
void driveMotor(int pwmPin, int dirPin, float u){
  u = clampf(u, -1.0f, 1.0f);
  int pwm = (int)(fabs(u) * 255.0f);
  digitalWrite(dirPin, (u >= 0) ? HIGH : LOW);
  analogWrite(pwmPin, pwm);
}

// ---- Parser de líneas simples "R,..." "S" "Z" "P,..." ----
void calibrateZero(){
  int r1 = analogRead(ENC1_PIN);
  int r2 = analogRead(ENC2_PIN);
  off1 = r1 * RAD_PER_ADC1;
  off2 = r2 * RAD_PER_ADC2;
}

void parseLine(char* s){
  if(s[0]=='R'){ // R,<q1>,<q2>
    float a1,a2;
    if(sscanf(s+1,",%f,%f",&a1,&a2)==2){
      q1_ref = a1; q2_ref = a2; has_ref = true;
    }
  } else if(s[0]=='S'){
    stop_cmd = true;
  } else if(s[0]=='Z'){
    calibrateZero();
  } else if(s[0]=='P'){
    float kp1,kd1,kp2,kd2;
    if(sscanf(s+1,",%f,%f,%f,%f",&kp1,&kd1,&kp2,&kd2)==4){
      Kp1=kp1; Kd1=kd1; Kp2=kp2; Kd2=kd2;
    }
  }
}

void serialService(){
  static const size_t BUF_SZ = 64;
  static char buf[BUF_SZ];
  static uint8_t idx = 0;
  while(Serial.available()){
    char c = (char)Serial.read();
    if(c=='\n' || c=='\r'){
      if(idx>0){
        buf[idx]=0;
        parseLine(buf);
      }
      idx=0;
    } else {
      if(idx<BUF_SZ-1) buf[idx++]=c;
    }
  }
}

void setup(){
  pinMode(M1_PWM, OUTPUT); pinMode(M1_DIR, OUTPUT);
  pinMode(M2_PWM, OUTPUT); pinMode(M2_DIR, OUTPUT);
  pinMode(ENC1_PIN, INPUT); pinMode(ENC2_PIN, INPUT);

  Serial.begin(115200);
  delay(400);

  calibrateZero();
  e1_prev = e2_prev = 0.0f;
  stop_cmd = false;

  Serial.println(F("2R PD listo; esperando referencias R,q1,q2 (rad)."));
}

void loop(){
  static unsigned long lastTick = 0;
  unsigned long now = millis();

  serialService();

  if(now - lastTick < (unsigned long)Ts_ms) return;
  lastTick = now;

  // STOP seguro si lo piden
  if(stop_cmd){
    driveMotor(M1_PWM, M1_DIR, 0);
    driveMotor(M2_PWM, M2_DIR, 0);
    return;
  }

  // Lectura actual
  float q1 = readJoint1();
  float q2 = readJoint2();

  // Si aún no hay referencia recibida, mantén motores quietos
  if(!has_ref){
    driveMotor(M1_PWM, M1_DIR, 0);
    driveMotor(M2_PWM, M2_DIR, 0);
  } else {
    // Control PD
    float e1 = q1_ref - q1;
    float e2 = q2_ref - q2;

    // Derivada (diferencia hacia atrás)
    float u1 = Kp1*e1 + Kd1*(e1 - e1_prev)/Ts;
    float u2 = Kp2*e2 + Kd2*(e2 - e2_prev)/Ts;
    e1_prev = e1; e2_prev = e2;

    // Normalización simple a [-1,1]
    const float U_NORM = 1.0f; // ajusta si falta/sobra par
    driveMotor(M1_PWM, M1_DIR, clampf(u1/U_NORM, -1, 1));
    driveMotor(M2_PWM, M2_DIR, clampf(u2/U_NORM, -1, 1));
  }

  // Telemetría ~50 Hz
  if((log_div_cnt++ % LOG_DIV)==0){
    Serial.print(F("Y,"));
    Serial.print(now);
    Serial.print(F(","));
    Serial.print(q1,6); Serial.print(F(",")); Serial.print(q2,6); Serial.print(F(","));
    Serial.print(q1_ref,6); Serial.print(F(",")); Serial.print(q2_ref,6); Serial.print(F(","));
    // Opcional: reporta últimos u1,u2 (estimados con e_prev ya actualizado)
    // Para exactitud, podrías recomputar u1,u2 o guardarlos
    Serial.print( (double)Kp1*e1_prev,6 ); Serial.print(F(",")); // parte proporcional aprox
    Serial.println( (double)Kp2*e2_prev,6 );
  }
}
