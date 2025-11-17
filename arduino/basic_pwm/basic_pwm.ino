// ===== MOTORES =====
#define enA_1 5
#define in1_1 6
#define in2_1 7

#define enA_2 10
#define in1_2 9
#define in2_2 8

// ===== ENCODERS OPB810W51Z =====
#define encoder1Pin 2   // Colector (blanco) del encoder del Motor 1
#define encoder2Pin 3   // Colector (blanco) del encoder del Motor 2

volatile long encoder1Pulses = 0;  // contador de pulsos encoder 1
volatile long encoder2Pulses = 0;  // contador de pulsos encoder 2

// Misma resolución para los dos encoders
const float PULSES_PER_REV = 180.0;  // ranuras por vuelta

// ISR para encoder 1 (pin 2)
void encoder1ISR() {
  encoder1Pulses++;
}

// ISR para encoder 2 (pin 3)
void encoder2ISR() {
  encoder2Pulses++;
}

void setup() {
  Serial.begin(115200);

  // ===== MOTOR 1 =====
  pinMode(enA_1, OUTPUT);
  pinMode(in1_1, OUTPUT);
  pinMode(in2_1, OUTPUT);
  digitalWrite(in1_1, HIGH);   // dirección fija por ahora
  digitalWrite(in2_1, LOW);

  // ===== MOTOR 2 =====
  pinMode(enA_2, OUTPUT);
  pinMode(in1_2, OUTPUT);
  pinMode(in2_2, OUTPUT);
  digitalWrite(in1_2, HIGH);   // dirección fija por ahora
  digitalWrite(in2_2, LOW);

  // ===== ENCODER 1 =====
  pinMode(encoder1Pin, INPUT_PULLUP);  // emisor a GND, colector al pin
  attachInterrupt(
    digitalPinToInterrupt(encoder1Pin),
    encoder1ISR,
    FALLING              // cuenta flanco HIGH->LOW
  );

  // ===== ENCODER 2 =====
  pinMode(encoder2Pin, INPUT_PULLUP);
  attachInterrupt(
    digitalPinToInterrupt(encoder2Pin),
    encoder2ISR,
    FALLING              // mismo criterio de conteo
  );
}

void loop() {
  // Control de motores (pon aquí el PWM que quieras para probar)
  analogWrite(enA_1, 190);   // PWM motor 1
  analogWrite(enA_2, 10);   // PWM motor 2

  // Cada 100 ms mostramos cuántos pulsos llegaron en cada encoder
  static unsigned long lastPrint = 0;
  static long lastPulses1 = 0;
  static long lastPulses2 = 0;

  unsigned long now = millis();
  if (now - lastPrint >= 100) {
    lastPrint = now;

    // Leer contadores de forma atómica
    noInterrupts();
    long pulsesNow1 = encoder1Pulses;
    long pulsesNow2 = encoder2Pulses;
    interrupts();

    long deltaPulses1 = pulsesNow1 - lastPulses1;
    long deltaPulses2 = pulsesNow2 - lastPulses2;

    lastPulses1 = pulsesNow1;
    lastPulses2 = pulsesNow2;

    // Cálculo de RPM para cada motor
    // Periodo de muestreo = 0.1 s (100 ms)
    float rpm1 = 0.0;
    float rpm2 = 0.0;

    if (PULSES_PER_REV > 0) {
      float revsPerSecond1 = (float)deltaPulses1 / PULSES_PER_REV / 0.1;
      float revsPerSecond2 = (float)deltaPulses2 / PULSES_PER_REV / 0.1;
      rpm1 = revsPerSecond1 * 60.0;
      rpm2 = revsPerSecond2 * 60.0;
    }

    Serial.print("M1 Pulsos totales: ");
    Serial.print(pulsesNow1);
    Serial.print(" | M1 Pulsos en 100 ms: ");
    Serial.print(deltaPulses1);
    Serial.print(" | M1 RPM aprox: ");
    Serial.println(rpm1);

    Serial.print("M2 Pulsos totales: ");
    Serial.print(pulsesNow2);
    Serial.print(" | M2 Pulsos en 100 ms: ");
    Serial.print(deltaPulses2);
    Serial.print(" | M2 RPM aprox: ");
    Serial.println(rpm2);

    Serial.println("-----");
  }
}
