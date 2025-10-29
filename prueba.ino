// ======= 2R Trefoil Tracker (Arduino UNO/MEGA) =======
// Control PD por articulación + generador de trayectoria con v ~constante

// ---- Pines (ajusta si usas otro driver) ----
const int ENC1_PIN = A1;     // pot/encoder 1 (0..1023 -> 0..2π)
const int ENC2_PIN = A2;     // pot/encoder 2
const int M1_PWM   = 9;
const int M1_DIR   = 8;
const int M2_PWM   = 10;
const int M2_DIR   = 7;

// ---- Parámetros del robot 2R (m) ----
const float L1 = 0.15;   // Longitud eslabón 1 (12 cm por defecto)
const float L2 = 0.15;   // Longitud eslabón 2

// ---- Mapeo analógico->ángulo (rad) ----
// Supuesto: 0..1023 ≈ 0..2π. Ajusta ganancia/offset tras calibrar.
const float RAD_PER_ADC1 = (2.0 * PI) / 1023.0;
const float RAD_PER_ADC2 = (2.0 * PI) / 1023.0;
float off1 = 0.0;  // offset cero de junta 1 (se fija en setup)
float off2 = 0.0;  // offset cero de junta 2

// ---- Control PD ----
const float Ts_ms = 10.0;                 // período de muestreo [ms]
const float Ts = Ts_ms / 1000.0;          // [s]
const float Kp1 = 18.0, Kd1 = 0.9;       // ganancias junta 1
const float Kp2 = 18.0, Kd2 = 0.9;       // ganancias junta 2
float e1_prev = 0, e2_prev = 0;

// ---- Parámetros del trébol ----
// r(θ) = E*R * (1 + M * sin(aθ + b))
// x = r cos(θ + ψ), y = r sin(θ + ψ)
float E   = 1.00;      // escala [1, 1.2]
float Rb  = 0.10;      // “radio base” (10 cm -> lado mínimo ≈ 20 cm)
float M   = 0.25;      // magnitud lóbulos [0.1..0.4]
int   a   = 4;          // nº lóbulos (ej. 3, 4…)
float b   = 0.0;       // rotación de lóbulos [0..2π]
float psi = 0.0;       // rotación global de la figura (±45° -> ±π/4)

// ---- Velocidad de la punta ----
float v_tip = 0.05;     // [m/s] (5 cm/s dentro de 1..10 cm/s)

// ---- Seguimiento de ciclos ----
float theta_curve = 0.0;
unsigned long lastTick = 0;
int cycles_done = 0;
const int MAX_CYCLES = 5; // pon 10 si quieres agotar el requisito

// ---- Utilidades ----
float clamp(float x, float lo, float hi){ return x < lo ? lo : (x > hi ? hi : x); }

void setup(){
  pinMode(M1_PWM, OUTPUT); pinMode(M1_DIR, OUTPUT);
  pinMode(M2_PWM, OUTPUT); pinMode(M2_DIR, OUTPUT);
  pinMode(ENC1_PIN, INPUT); pinMode(ENC2_PIN, INPUT);

  Serial.begin(115200);
  delay(500);

  // Calibración rápida de cero: toma lectura inicial como θ=0
  int r1 = analogRead(ENC1_PIN);
  int r2 = analogRead(ENC2_PIN);
  off1 = r1 * RAD_PER_ADC1;
  off2 = r2 * RAD_PER_ADC2;

  lastTick = millis();

  Serial.println(F("2R Trefoil Tracker listo."));
}

// --- Lectura de juntas (sustituible por tu encoder real) ---
float readJoint1(){
  int raw = analogRead(ENC1_PIN);
  float ang = raw * RAD_PER_ADC1 - off1;
  // Pasa a rango [-π, π] para evitar saltos
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

// --- Actuación de motor con comando en [-1,1] ---
void driveMotor(int pwmPin, int dirPin, float u){
  u = clamp(u, -1.0, 1.0);
  int pwm = (int)(fabs(u) * 255.0);
  digitalWrite(dirPin, (u >= 0) ? HIGH : LOW);
  analogWrite(pwmPin, pwm);
}

// --- Generador del trébol y su derivada geométrica ---
void trefoil(float th, float& x, float& y, float& s_prime){
  // ρ(θ)
  float rho = E * Rb * (1.0 + M * sinf(a*th + b));
  // dρ/dθ
  float drho = E * Rb * (M * a * cosf(a*th + b));
  // Rotación global ψ
  float cg = cosf(th + psi), sg = sinf(th + psi);
  x = rho * cg;
  y = rho * sg;

  // |d r / dθ| = sqrt( (dρ)^2 + ρ^2 )
  s_prime = sqrtf(drho*drho + rho*rho);
}

// --- IK 2R (elige codo: +1 arriba, -1 abajo) ---
bool ik2R(float x, float y, float& th1, float& th2, int elbow = +1){
  float r2 = x*x + y*y;
  float c2 = (r2 - L1*L1 - L2*L2) / (2.0*L1*L2);
  c2 = clamp(c2, -1.0, 1.0);
  float s2 = elbow * sqrtf(fmax(0.0, 1.0 - c2*c2));
  th2 = atan2f(s2, c2);
  float k1 = L1 + L2 * c2;
  float k2 = L2 * s2;
  th1 = atan2f(y, x) - atan2f(k2, k1);
  return true;
}

// --- Fase de aproximación a la trayectoria (rámpa 1 s) ---
void approachTo(float q1_ref, float q2_ref){
  static float alpha = 0.0;
  alpha += Ts / 1.0; // 1 s de rampa
  if(alpha > 1.0) alpha = 1.0;

  float q1 = readJoint1();
  float q2 = readJoint2();
  float e1 = (q1_ref - q1) * alpha;
  float e2 = (q2_ref - q2) * alpha;

  float u1 = Kp1*e1 + Kd1*(e1 - e1_prev)/Ts;
  float u2 = Kp2*e2 + Kd2*(e2 - e2_prev)/Ts;
  e1_prev = e1; e2_prev = e2;

  // Normaliza a [-1,1] suponiendo 1 rad ≈ 100% PWM (ajusta si falta par)
  driveMotor(M1_PWM, M1_DIR, clamp(u1, -1, 1));
  driveMotor(M2_PWM, M2_DIR, clamp(u2, -1, 1));
}

void loop(){
  unsigned long now = millis();
  if(now - lastTick < (unsigned long)Ts_ms) return;
  lastTick = now;

  if(cycles_done >= MAX_CYCLES){
    // Stop seguro
    driveMotor(M1_PWM, M1_DIR, 0);
    driveMotor(M2_PWM, M2_DIR, 0);
    return;
  }

  // 1) Generar punto deseado (x,y) con v_tip ~ constante
  float x_d, y_d, sprime;
  trefoil(theta_curve, x_d, y_d, sprime);

  // Calcular paso de θ para lograr v_tip
  float dtheta = (v_tip * Ts) / fmax(1e-5, sprime);
  theta_curve += dtheta;

  // Contar ciclos (cada 2π)
  if(theta_curve >= 2.0f*PI){
    theta_curve -= 2.0f*PI;
    cycles_done++;
  }

  // 2) IK -> referencias de articulación
  float q1_ref, q2_ref;
  ik2R(x_d, y_d, q1_ref, q2_ref, +1); // +1: codo "arriba" (cámbialo si lo necesitas)

  // 3) Lectura actual
  float q1 = readJoint1();
  float q2 = readJoint2();

  // 4) Control PD por junta
  float e1 = q1_ref - q1;
  float e2 = q2_ref - q2;

  float u1 = Kp1*e1 + Kd1*(e1 - e1_prev)/Ts;
  float u2 = Kp2*e2 + Kd2*(e2 - e2_prev)/Ts;
  e1_prev = e1; e2_prev = e2;

  // 5) Salida a motores (normaliza a [-1,1] con factor simple)
  const float U_NORM = 1.0; // si falta par, sube Kp/Kd o mejora U_NORM con feedforward
  driveMotor(M1_PWM, M1_DIR, clamp(u1/U_NORM, -1, 1));
  driveMotor(M2_PWM, M2_DIR, clamp(u2/U_NORM, -1, 1));

  // 6) Telemetría básica
  static int div = 0;
  if((div++ % 20) == 0){ // cada ~0.2 s
    Serial.print("xd,yd: "); Serial.print(x_d,3); Serial.print(", "); Serial.print(y_d,3);
    Serial.print(" | q1: "); Serial.print(q1,3); Serial.print(" -> "); Serial.print(q1_ref,3);
    Serial.print(" | q2: "); Serial.print(q2,3); Serial.print(" -> "); Serial.println(q2_ref,3);
  }
}