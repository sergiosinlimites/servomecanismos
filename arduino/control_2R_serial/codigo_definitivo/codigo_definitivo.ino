// Definir variables
#define Sensor 2               
#define Sensor_2 3
#define Sensor_anal A0
#define Sensor_anal_2 A1
#define OutputPWM_GPIO 9           // Pin de salida PWM para el control de la hélice
#define OutputPWM_GPIO_2 10         // Pin de salida PWM para el control de la hélice
#define pwmRes 12                  // Resolución del PWM (12 bits)
#define pwmMax 4095                // Valor máximo para el PWM (4095 para 12 bits)

// Variables para la conversión y salida
#define Uunits 100                 // Unidades para la salida de control (u) [mA]

// Variables de tiempo de ejecución
unsigned long pTime = 0;
unsigned long dTime = 0;
long previousMillis = 0;          // Para la función del bucle principal
long Ts = 10;                   // Tiempo de muestreo en ms
long previousMillis2 = 0;         // Para funciones auxiliares
bool up = true;
int i = 0;

// Advanced Serial Input Variables
const byte numChars = 32;
char receivedChars[numChars];
boolean newData = false;

// Variables de medición        // Valor leído del sensor analógico
float angulo = 0.0;
float angulo_2 = 0.0;

// Variables de control del sistema
float Ref = 40;                // angulo de referencia  
float Ref_2 = 40;       
float U_t = 0.0;                 // Salida de control (PWM)
float U_t_2 = 0.0;
unsigned int pwmDuty = 0;        // Ciclo de trabajo del PWM
unsigned int pwmDuty_2 = 0;        // Ciclo de trabajo del PWM


// Variables para el controlador PID motor 1
float k_p = 1;    
float k_i = 2;
float k_d = 0.008;

float e_n = 0.0, e_n_1 = 0.0;
float u_n = 0.0 , u_p = 0.0, u_i = 0.0, u_d = 0.0,  u_n_1_i = 0.0, u_n_1_d = 0.0, u_n_1 = 0.0;
float ts = 0.0010;

// Variables para el controlador PID motor 2
float k_p_2 = 0.4;    
float k_i_2 = 1;
float k_d_2 = 0.008;
float e_n_2 = 0.0, e_n_1_2 = 0.0;
float u_n_2 = 0.0 , u_p_2 = 0.0, u_i_2 = 0.0, u_d_2 = 0.0,  u_n_1_i_2 = 0.0, u_n_1_d_2 = 0.0, u_n_1_2 = 0.0;

int N = 100;


//Variables sentido de giro motor - DIRECCIÓN FIJA (NO CAMBIA)
const uint8_t PIN_IN1 = 4;  // Dirección motor 1
const uint8_t PIN_IN2 = 5;  // Dirección motor 1
const uint8_t PIN_IN3 = 6;  // Dirección motor 2
const uint8_t PIN_IN4 = 7;  // Dirección motor 2

bool sentidoAdelante = true; // Variable para controlar el sentido actual

//Variables medición de angulos - MODO POLLING
const uint16_t SLOTS_PER_REV  = 180;      // # de ranuras por vuelta
const bool     COUNT_ON_FALL  = true;     // true: cuenta flanco FALLING; false: RISING
const bool     COUNT_ON_FALL_2  = true; 

int32_t slotCount = 0;                    // Contador slots motor 1
int32_t slotCount_2 = 0;                  // Contador slots motor 2
uint32_t lastUs = 0;                      // para antirruido
uint32_t lastUs_2 = 0;
const uint32_t DEBOUNCE_US = 800;         // ignora cambios más rápidos que esto

uint8_t lastState = HIGH;                 // estado anterior del pin (inicia en HIGH por pullup)
uint8_t lastState_2 = HIGH;  
uint32_t lastPrint = 0;
uint32_t lastPrint_2 = 0;

float sensorValue_mag = 0;
float angulo_mag = 0;

float sensorValue_mag_2 = 0;
float angulo_mag_2 = 0;

// Debug: últimas lecturas crudas y deltas tras aplicar offset/unwrapping
int dbg_raw0 = 0, dbg_delta0 = 0, dbg_offset0 = 830;
int dbg_raw1 = 0, dbg_delta1 = 0, dbg_offset1 = 455;

// Control extra Motor 2: manejo de inversión suave
int8_t errSign2_prev = 0;                 // 1 o -1 según signo del error anterior
int flipCooldown2 = 0;                    // cuenta muestras tras invertir
const int FLIP_COOLDOWN_SAMPLES = 20;     // ~200 ms si Ts=10 ms
const float FLIP_PWM_FRAC = 0.35f;        // limitar PWM al 35% durante cooldown

// Función para leer encoder por polling - SIN cambio de dirección
void leerEncoder() {
  // ===== DETECCIÓN DE FLANCOS POR POLLING =====
  uint8_t currentState = digitalRead(Sensor);
  
  // Detecta cambio de estado
  if (currentState != lastState) {
    uint32_t now = micros();
    
    // Antirrebote: ignora cambios muy rápidos
    if (now - lastUs >= DEBOUNCE_US) {
      // Verifica el tipo de flanco que queremos contar
      if (COUNT_ON_FALL) {
        // Flanco descendente: lastState=HIGH, currentState=LOW
        if (lastState == HIGH && currentState == LOW) {
          // SIMPLIFICADO: Solo incrementa (dirección física fija)
          slotCount++;  
          lastUs = now;
        }
      } else {
        // Flanco ascendente: lastState=LOW, currentState=HIGH
        if (lastState == LOW && currentState == HIGH) {
          slotCount++;  
          lastUs = now;
        }
      }
    }
    
    lastState = currentState;  // Actualiza el estado anterior
  }
}

void leerEncoder_2() {
  // ===== DETECCIÓN DE FLANCOS POR POLLING =====
  uint8_t currentState_2 = digitalRead(Sensor_2);
  
  // Detecta cambio de estado
  if (currentState_2 != lastState_2) {
    uint32_t now_2 = micros();
    
    // Antirrebote: ignora cambios muy rápidos
    if (now_2 - lastUs_2 >= DEBOUNCE_US) {
      // Verifica el tipo de flanco que queremos contar
      if (COUNT_ON_FALL_2) {
        // Flanco descendente: lastState=HIGH, currentState=LOW
        if (lastState_2 == HIGH && currentState_2 == LOW) {
          // SIMPLIFICADO: Solo incrementa (dirección física fija)
          slotCount_2++;  
          lastUs_2 = now_2;
        }
      } else {
        // Flanco ascendente: lastState=LOW, currentState=HIGH
        if (lastState_2 == LOW && currentState_2 == HIGH) {
          slotCount_2++;  
          lastUs_2 = now_2;
        }
      }
    }
    
    lastState_2 = currentState_2;  // Actualiza el estado anterior
  }
}

// Función de calibración
void calibracion(void) {
    unsigned long currentMillis = millis(); // Actualizar el tiempo actual
    if (currentMillis - previousMillis >= Ts) {
        previousMillis = currentMillis;

//--------------------MOTOR 1-----------------------------------------------
        // Calcular ángulo
        int32_t slots = slotCount;
        
        // Conversión a grados
        float degPerSlot = 360.0f / (float)SLOTS_PER_REV;
        angulo = slots * degPerSlot;

        // Calcular error (Motor 1 usa Ref; Motor 2 usa Ref_2)
        e_n = Ref - angulo_mag;

        // Lectura AS5600 con “unwrapping” respecto a offset para continuidad 0–360°
        {
          int raw0 = analogRead(Sensor_anal);        // 0..1023
          const int offset0 = 830;                   // offset ya calibrado
          int delta0 = raw0 - offset0;               // delta relativo al cero propio
          if (delta0 < 0) delta0 += 1024;            // envolver para continuidad (0..1023)
          sensorValue_mag = (float)delta0;
          angulo_mag = (sensorValue_mag / 1023.0f) * 360.0f; // 0..360° continuo desde nuestro cero
          // debug
          dbg_raw0 = raw0; dbg_delta0 = delta0; dbg_offset0 = offset0;
        }

        // Cálculo del control PID
        u_p = k_p * e_n;
        u_i = (k_i * ts * e_n_1) + u_n_1_i;
        u_d = (k_d * N * e_n) - (k_d * N * e_n_1) - (N * ts * u_n_1_d) + u_n_1_d;
        u_n = u_p + u_d + u_i;

        // Control PWM - CORREGIDO: Usa la salida del PID
        U_t = u_n;
        float U_tl = min(max( u_n , 0), Uunits);
        pwmDuty = int((U_tl / Uunits) * pwmMax);
        analogWriteADJ(OutputPWM_GPIO, pwmDuty);

        // Actualizar valores anteriores
        e_n_1 = e_n;
        u_n_1_i = u_i;
        u_n_1_d = u_d;
        
//----------------------------------------------------------------------------------

//-----------------------------Motor 2----------------------------------------------
        int32_t slots_2 = slotCount_2;
        
        // Conversión a grados
        float degPerSlot_2 = 360.0f / (float)SLOTS_PER_REV;
        angulo_2 = slots_2 * degPerSlot_2;

        // Calcular error
        e_n_2 = Ref_2 - angulo_mag_2;

        // Lectura AS5600 #2 con unwrapping similar (consistencia)
        {
          int raw1 = analogRead(Sensor_anal_2);      // 0..1023
          const int offset1 = 455;                   // offset ya calibrado
          int delta1 = raw1 - offset1;
          if (delta1 < 0) delta1 += 1024;
          sensorValue_mag_2 = (float)delta1;
          angulo_mag_2 = (sensorValue_mag_2 / 1023.0f) * 360.0f;
          // debug
          dbg_raw1 = raw1; dbg_delta1 = delta1; dbg_offset1 = offset1;
        }

        // Dirección y control para Motor 2 con fricción: si el error es negativo,
        // invertimos la polaridad y tratamos el error como positivo (magnitud).
        int currSign2 = (e_n_2 >= 0.0f) ? 1 : -1;
        if (currSign2 >= 0) {
            // Adelante
            digitalWrite(PIN_IN3, HIGH);
            digitalWrite(PIN_IN4, LOW);
        } else {
            // Reversa
            digitalWrite(PIN_IN3, LOW);
            digitalWrite(PIN_IN4, HIGH);
        }
        // Detectar cambio de signo del error para suavizar (evitar sobrecorrección)
        if (errSign2_prev != 0 && currSign2 != errSign2_prev) {
            // Resetear integrador y derivada, iniciar ventana de PWM reducido
            u_n_1_i_2 = 0.0f;
            u_n_1_d_2 = 0.0f;
            flipCooldown2 = FLIP_COOLDOWN_SAMPLES;
        }
        errSign2_prev = currSign2;

        // Magnitud del error para el controlador (siempre positiva)
        float e2_mag = (e_n_2 >= 0.0f) ? e_n_2 : -e_n_2;

        // Cálculo del control (PID/PI) usando error positivo (magnitud)
        u_p_2 = k_p_2 * e2_mag;
        u_i_2 = (k_i_2 * ts * e_n_1_2) + u_n_1_i_2;
        u_d_2 = (k_d_2 * N * e2_mag) - (k_d_2 * N * e_n_1_2) - (N * ts * u_n_1_d_2) + u_n_1_d_2;
        // PID completo: suma P + I + D
        u_n_2 = u_p_2 + u_i_2 + u_d_2;
  
        // Magnitud del esfuerzo desde el control (valor absoluto) y limitación
        U_t_2 = u_n_2;
        float mag2 = (u_n_2 >= 0.0f) ? u_n_2 : -u_n_2;
        float U_tl_2 = min(mag2, (float)Uunits);
        // Si acabamos de invertir sentido, recortar agresividad por unas muestras
        if (flipCooldown2 > 0) {
            float cap = (float)Uunits * FLIP_PWM_FRAC;
            if (U_tl_2 > cap) U_tl_2 = cap;
            flipCooldown2--;
        }
        pwmDuty_2 = int((U_tl_2 / Uunits) * pwmMax);
        analogWriteADJ(OutputPWM_GPIO_2, pwmDuty_2);

        // Actualizar estados con la magnitud (consistente con el control)
        e_n_1_2 = e2_mag;
        u_n_1_i_2 = u_i_2;
        u_n_1_d_2 = u_d_2;

        // Enviar datos al monitor serial (solo refs, ángulos, errores y PWM)
        Serial.print("Ref1: ");
        Serial.print(Ref);
        Serial.print(", Ref2: ");
        Serial.print(Ref_2);
        Serial.print(", Ang1: ");
        Serial.print(angulo_mag);
        Serial.print(", Ang2: ");
        Serial.print(angulo_mag_2);
        Serial.print(", Err1: ");
        Serial.print(e_n);
        Serial.print(", Err2: ");
        Serial.print(e_n_2);
        Serial.print(", PWM1%: ");
        Serial.print((pwmDuty * 100.0) / pwmMax);
        Serial.print(", PWM2%: ");
        Serial.println((pwmDuty_2 * 100.0) / pwmMax);
    }

    // Procesar comandos de la PC (R,theta1,theta2[,t] y S para parar)
    pollSerial();
}

// Configuración del PWM
void setupPWMadj() {
    DDRB |= _BV(PB1) | _BV(PB2);        /* set pins as outputs */
    TCCR1A = _BV(COM1A1) | _BV(COM1B1)  /* non-inverting PWM */
        | _BV(WGM11);                   /* mode 14: fast PWM, TOP=ICR1 */
    TCCR1B = _BV(WGM13) | _BV(WGM12)
        | _BV(CS10);                    /* no prescaling */
    ICR1 = 0x0fff;                      /* TOP counter value - SETS RESOLUTION/FREQUENCY */
}

// Versión de analogWrite() de 12 bits
void analogWriteADJ(uint8_t pin, uint16_t val) {
    switch (pin) {
        case 9: OCR1A = val; break;
        case 10: OCR1B = val; break;
    }
}

// Nuevo parser de líneas: R,th1,th2[,t] y S
char serBuf[64];
uint8_t serIdx = 0;

void handleSerialLine(char *line) {
    // Ignorar espacios iniciales
    while (*line == ' ' || *line == '\t') line++;
    if (*line == 'S') {
        // Stop inmediato: PWM a 0
        analogWriteADJ(OutputPWM_GPIO, 0);
        analogWriteADJ(OutputPWM_GPIO_2, 0);
        Serial.println(F("ACK S"));
        return;
    }
    if (*line == 'R') {
        // Formato: R,th1_deg,th2_deg[,pc_time]
        // Avanza sobre 'R' y coma
        char *p = line;
        // Busca primera coma
        p = strchr(p, ',');
        if (!p) return;
        p++; // después de primera coma
        // th1
        char *p2 = strchr(p, ',');
        if (!p2) return;
        *p2 = '\0';
        float th1_deg = atof(p);
        // th2
        char *p3 = p2 + 1;
        char *p4 = strchr(p3, ',');
        if (p4) *p4 = '\0'; // pc_time opcional ignorado
        float th2_deg = atof(p3);
        // Ya vienen en grados
        Ref  = th1_deg;
        Ref_2 = th2_deg;
        Serial.print(F("ACK R deg: "));
        Serial.print(Ref);
        Serial.print(F(","));
        Serial.println(Ref_2);
        return;
    }
}

void pollSerial() {
    while (Serial.available() > 0) {
        char c = (char)Serial.read();
        if (c == '\r') continue;
        if (c == '\n') {
            serBuf[serIdx] = '\0';
            if (serIdx > 0) handleSerialLine(serBuf);
            serIdx = 0;
        } else {
            if (serIdx < sizeof(serBuf) - 1) serBuf[serIdx++] = c;
        }
    }
}

void setup() {
    Serial.begin(115200); // Iniciar la comunicación serial a alta velocidad
    
    // Configuración de entrada analógica
    pinMode(Sensor, INPUT_PULLUP); // Configurar el pin del sensor como entrada
    pinMode(PIN_IN1, OUTPUT);
    pinMode(PIN_IN2, OUTPUT);

    pinMode(Sensor_2, INPUT_PULLUP); // Configurar el pin del sensor como entrada
    pinMode(PIN_IN3, OUTPUT);
    pinMode(PIN_IN4, OUTPUT);

    // Inicializar sentido del motor: DIRECCIÓN FIJA (NO CAMBIA)
    // Motor 1
    digitalWrite(PIN_IN1, HIGH);
    digitalWrite(PIN_IN2, LOW);
    sentidoAdelante = true;

    // Motor 2
    digitalWrite(PIN_IN3, HIGH);
    digitalWrite(PIN_IN4, LOW);

    // Lee el estado inicial del pin para el encoder por polling
    lastState = digitalRead(Sensor);
    lastState_2 = digitalRead(Sensor_2);

    // Configuración del PWM
    setupPWMadj();
    analogWriteADJ(OutputPWM_GPIO, 0); // Iniciar en 0
    analogWriteADJ(OutputPWM_GPIO_2, 0); // Iniciar en 0

    Serial.println(F("Sistema de control PID con encoder OPB800 (POLLING MODE)"));
    Serial.println(F("180 ranuras -> 2.0 grados/ranura"));
    Serial.println(F("DIRECCION FIJA - SIN cambio automatico de giro"));
    Serial.println(F("Comandos: R,th1_deg,th2_deg[,t]  |  S (stop)\n"));

    delay(2000); // Esperar 2 segundos antes de iniciar
}

void loop() {
    // Leer encoder continuamente
    leerEncoder();
    leerEncoder_2();

    // Ejecutar función de control
    calibracion();
}