// Definir variables
#define Sensor_1 2                  // Pin del encoder motor 1
#define Sensor_2 3                  // Pin del encoder motro 2
#define OutputPWM_GPIO_1 9          // Pin de salida PWM para el control motor 1
#define IN_1_1 4  
#define IN_2_1 5         
#define OutputPWM_GPIO_2 10           // Pin de salida PWM para el control motor 2
#define IN_1_2 7                     // Pin dirección motor 2 (corregido)
#define IN_2_2 8                     // Pin dirección motor 2 (corregido) 
#define pwmRes 12                  // Resolución del PWM (12 bits)
#define pwmMax 4095                // Valor máximo para el PWM (4095 para 12 bits)

// Variables para la conversión y salida
#define Uunits 100                 // Unidades para la salida de control (u) [mA]

// Variables de tiempo de ejecución
unsigned long pTime = 0;
unsigned long dTime = 0;
long previousMillis = 0;          // Para la función del bucle principal
long Ts = 1000;                   // Tiempo de muestreo en ms
long previousMillis2 = 0;         // Para funciones auxiliares
bool up = true;
int i = 0;
float ts = 0.010;

// Protocolo Serial (formato compatible con Python)
char lineBuf[64];
uint8_t lineLen = 0;

// Variables de medición
float encoder_1 = 0.0;         // Valor leído del sensor analógico
float encoder_2 = 0.0;   
float angulo_1 = 0.0;          // Ángulo motor 1 en grados
float angulo_2 = 0.0;          // Ángulo motor 2 en grados
float angulo_1_rad = 0.0;      // Ángulo motor 1 en radianes
float angulo_2_rad = 0.0;      // Ángulo motor 2 en radianes

// Offsets de cero (en ranuras)
volatile int32_t slotOffset_1 = 0;
volatile int32_t slotOffset_2 = 0;

// Variables de control del sistema
float Ref_1 = 0;                // angulo de referencia motor 1 (radianes)
float Ref_1_Fut = 0;
float Ref_2 = 0;                 // angulo de referencia motor 2 (radianes)   
float Ref_2_Fut = 0 ;    
float U_t_1 = 0.0;                 // Salida de control motor 1 (PWM)
float U_t_2 = 0.0;                 // Salida de control motro 2 (PWM)
unsigned int pwmDuty_1 = 0;        // Ciclo de trabajo del PWM motor 1
unsigned int pwmDuty_1_1 = 0;
unsigned int pwmDuty_2 = 0;        // Ciclo de trabajo del PWM motor 2
unsigned int pwmDuty_2_2 = 0;

// Variables para el controlador PID
float k_p_1 = 0.3; // proporcional motor 1 
float k_p_2 = 0.3; // Proporcional motor 2  
float e_n_1 = 0.0, e_n_1_1 = 0.0; //error control motor 1
float e_n_2 = 0.0, e_n_2_1 = 0.0; //error control motor 2
float u_n_1 = 0.0 , u_p_1 = 0.0, u_n_1_1 = 0.0; // motor 1
float u_n_2 = 0.0 , u_p_2 = 0.0, u_n_2_1 = 0.0; // motor 2

// Variables de control
unsigned long pwmStartTime = 0;   // Variable para el tiempo de inicio del PWM constante
unsigned long pwmDuration = 40000; // Duración del PWM constante en milisegundos (3 segundos en este caso)
bool pwmConstant = true;    

// Variables medida sensor
const uint16_t SLOTS_PER_REV_1  = 180;      // # de ranuras por vuelta
const uint16_t SLOTS_PER_REV_2  = 180;
const bool     COUNT_ON_FALL_1  = true;     // true: cuenta flanco FALLING; false: RISING
const bool     COUNT_ON_FALL_2  = true; 
const uint32_t PRINT_EVERY_MS = 100;      // período de impresión

volatile uint32_t slotCount_1 = 0;          // ranuras acumuladas (flancos contados)
volatile uint32_t slotCount_2 = 0;
volatile uint32_t lastUs_1 = 0;             // para antirruido
volatile uint32_t lastUs_2 = 0; 
const uint32_t DEBOUNCE_US = 150;         // ignora cambios más rápidos que esto

uint32_t lastPrint = 0;

// Telemetría
const uint32_t TEL_MS = 50;      // Período de telemetría (50 ms = ~20 Hz)
uint32_t lastTel = 0;

// Función de utilidad para envolver ángulo a [-π, π]
float wrapToPi(float a) {
    while (a > PI) a -= 2.0f * PI;
    while (a <= -PI) a += 2.0f * PI;
    return a;
}

// Función de calibración
void calibracion(void) {
    unsigned long currentMillis = millis(); // Actualizar el tiempo actual
    if (currentMillis - previousMillis >= Ts) {
        previousMillis = currentMillis;
        
        // Convertir referencias de radianes a grados para cálculo de error local
        float Ref_1_deg = Ref_1 * (180.0 / PI);
        float Ref_2_deg = Ref_2 * (180.0 / PI);
        
        // Error en grados (para mantener compatibilidad con el control actual)
        e_n_1 = Ref_1_deg - angulo_1;
        e_n_2 = Ref_2_deg - angulo_2;

        //ENCODER 1
         noInterrupts();
            uint32_t slots_1 = slotCount_1;
         interrupts();

         // Ángulo absoluto encoder 1 (0.. <360)
         // Cálculo del ángulo absoluto del encoder 1 a partir de los impulsos (slotCount_1)
         // 1. Calcula la posición relativa respecto al valor de referencia (offset de cero)
         int32_t rel_1 = ((int32_t)slots_1 - slotOffset_1);

         // 2. Realiza un módulo para mantener la cuenta de ranuras dentro del rango [0, SLOTS_PER_REV_1)
         //    Esto asegura que el ángulo mapee correctamente en una vuelta completa, incluso con conteo negativo
         int32_t mod_1 = (SLOTS_PER_REV_1 == 0) ? 0 : ((rel_1 % (int32_t)SLOTS_PER_REV_1 + (int32_t)SLOTS_PER_REV_1) % (int32_t)SLOTS_PER_REV_1);

         // 3. Convierte el resultado a uint32_t para usarlo en el cálculo de ángulo
         uint32_t slotsMod_1 = (uint32_t)mod_1;

         // 4. Calcula los grados que representa cada ranura
         float degPerSlot_1 = 360.0 / (float)SLOTS_PER_REV_1;

         // 5. Calcula el ángulo absoluto (en grados) dentro de una vuelta [0, 360)
         angulo_1 = slotsMod_1 * degPerSlot_1;

         // 6. Calcula el ángulo "extendido" (en radianes, para telemetría), permitiendo vueltas múltiples
         //    Se normaliza entre -pi y pi usando wrapToPi, útil para control y visualización continua
         float turns_1 = (float)rel_1 / (float)SLOTS_PER_REV_1;
         angulo_1_rad = wrapToPi(turns_1 * (2.0f * PI));

         noInterrupts();
            uint32_t slots_2 = slotCount_2;
         interrupts();

         // Ángulo absoluto encoder 2 (0.. <360)
         int32_t rel_2 = ((int32_t)slots_2 - slotOffset_2);
         // Manejar módulo con signo correctamente
         int32_t mod_2 = (SLOTS_PER_REV_2 == 0) ? 0 : ((rel_2 % (int32_t)SLOTS_PER_REV_2 + (int32_t)SLOTS_PER_REV_2) % (int32_t)SLOTS_PER_REV_2);
         uint32_t slotsMod_2 = (uint32_t)mod_2;
         float degPerSlot_2 = 360.0 / (float)SLOTS_PER_REV_2;
         angulo_2 = slotsMod_2 * degPerSlot_2;
         // Ángulo en radianes (para telemetría)
         float turns_2 = (float)rel_2 / (float)SLOTS_PER_REV_2;
         angulo_2_rad = wrapToPi(turns_2 * (2.0f * PI));

        
          u_p_1 = k_p_1*e_n_1;
          u_n_1 = u_p_1;

          u_p_2 = k_p_2*e_n_2;
          u_n_2 = u_p_2;

        if (Ref_1>Ref_1_Fut){
        digitalWrite(IN_1_1, LOW);
        digitalWrite(IN_2_1, HIGH);
        float U_tl_1 = min(max((u_n_1), 0), Uunits); // Control motor 1 saturado
        pwmDuty_1 = int((U_tl_1 / Uunits) * pwmMax); // Convertir a ciclo de trabajo PWM
        analogWriteADJ(OutputPWM_GPIO_1, pwmDuty_1); // Escribir el valor de PWM en el pin
        } else {
        digitalWrite(IN_1_1, HIGH);
        digitalWrite(IN_2_1, LOW);
        float U_tl_1 = min(max((u_n_1), 0), Uunits); // Control motor 1 saturado
        pwmDuty_1 = int((U_tl_1 / Uunits) * pwmMax); // Convertir a ciclo de trabajo PWM
        analogWriteADJ(OutputPWM_GPIO_1, pwmDuty_1); // Escribir el valor de PWM en el pin
        }


        if (Ref_2>Ref_2_Fut){
        digitalWrite(IN_1_2, HIGH);
        digitalWrite(IN_2_2, LOW);
        float U_tl_2 = min(max((u_n_2), 0), Uunits); // Control motor 2 saturado
        pwmDuty_2 = int((U_tl_2 / Uunits) * pwmMax); // Convertir a ciclo de trabajo PWM
        analogWriteADJ(OutputPWM_GPIO_2, pwmDuty_2); // Escribir el valor de PWM en el pin
        }else {
        digitalWrite(IN_1_2, LOW);
        digitalWrite(IN_2_2, HIGH);  // Corregido: era IN_1_2 dos veces
        float U_tl_2 = min(max((u_n_2), 0), Uunits); // Control motor 2 saturado
        pwmDuty_2 = int((U_tl_2 / Uunits) * pwmMax); // Convertir a ciclo de trabajo PWM
        analogWriteADJ(OutputPWM_GPIO_2, pwmDuty_2); // Escribir el valor de PWM en el pin
        }
        
           e_n_1_1 = e_n_1;
           e_n_2_1 = e_n_2;
        
        // Calcular esfuerzos normalizados [-1, 1] para telemetría
        float u1_norm = clampf(u_n_1 / Uunits, -1.0f, 1.0f);
        float u2_norm = clampf(u_n_2 / Uunits, -1.0f, 1.0f);
        
        // Telemetría en formato Y,millis,q1,q2,q1_ref,q2_ref,u1,u2
        if (currentMillis - lastTel >= TEL_MS) {
            lastTel = currentMillis;
            Serial.print('Y'); Serial.print(',');
            Serial.print(currentMillis); Serial.print(',');
            Serial.print(angulo_1_rad, 6); Serial.print(',');
            Serial.print(angulo_2_rad, 6); Serial.print(',');
            Serial.print(Ref_1, 6); Serial.print(',');
            Serial.print(Ref_2, 6); Serial.print(',');
            Serial.print(u1_norm, 3); Serial.print(',');
            Serial.println(u2_norm, 3);
        }
    }

    // Procesar comandos seriales
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

// Función de utilidad clamp
static inline float clampf(float x, float a, float b) { 
    return x < a ? a : (x > b ? b : x); 
}

// Protocolo Serial: procesar línea recibida
void processLine() {
    lineBuf[lineLen] = '\0';
    if (lineLen == 0) return;

    if (lineBuf[0] == 'R') {
        // R,theta1,theta2 (radianes)
        float a, b; 
        if (sscanf(lineBuf, "R,%f,%f", &a, &b) == 2) {
            noInterrupts(); 
            Ref_1 = a; 
            Ref_2 = b; 
            interrupts();
        }
    } else if (lineBuf[0] == 'P') {
        // P,Kp1,Kp2 (ajustar ganancias)
        float p1, p2; 
        if (sscanf(lineBuf, "P,%f,%f", &p1, &p2) == 2) {
            noInterrupts(); 
            k_p_1 = p1; 
            k_p_2 = p2; 
            interrupts();
        }
    } else if (lineBuf[0] == 'Z') {
        // Calibración: tomar conteos actuales como cero (offset)
        noInterrupts(); 
        slotOffset_1 = (int32_t)slotCount_1; 
        slotOffset_2 = (int32_t)slotCount_2; 
        interrupts();
    } else if (lineBuf[0] == 'S') {
        // Stop: detener PWM
        analogWriteADJ(OutputPWM_GPIO_1, 0);
        analogWriteADJ(OutputPWM_GPIO_2, 0);
    }
}

// Polling de Serial (llamar desde loop)
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

void isrSlot1() {
  uint32_t now_1 = micros();
  if (now_1 - lastUs_1 < DEBOUNCE_US) return; // filtro rápido contra rebotes/ruido
  slotCount_1++;
  lastUs_1 = now_1;
}

void isrSlot2() {
  uint32_t now_2 = micros();
  if (now_2 - lastUs_2 < DEBOUNCE_US) return; // filtro rápido contra rebotes/ruido
  slotCount_2++;
  lastUs_2 = now_2;
}



void setup() {
    Serial.begin(115200); // Iniciar la comunicación serial a 115200 baudios (compatible con Python)
    
    // Configuración de entrada analógica
    pinMode(Sensor_1, INPUT_PULLUP); // Configurar el pin del encoder_1 como entrada
    pinMode(Sensor_2, INPUT_PULLUP); // Configurar el pin del Encoder_2 como entrada
    pinMode(IN_1_1, OUTPUT); 
    pinMode(IN_2_1, OUTPUT); 
    pinMode(IN_1_2, OUTPUT); 
    pinMode(IN_2_2, OUTPUT); 

       // Elige flanco
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

    // Configuración del PWM
    setupPWMadj();
    analogWriteADJ(OutputPWM_GPIO_1, 0);
    analogWriteADJ(OutputPWM_GPIO_2, 0);

    while(!Serial) { ; }  // Esperar conexión serial
    delay(50);
    
    // Calibración inicial (equivalente a comando 'Z')
    noInterrupts(); 
    slotOffset_1 = (int32_t)slotCount_1; 
    slotOffset_2 = (int32_t)slotCount_2; 
    interrupts();
}



void loop() {
    calibracion();
}