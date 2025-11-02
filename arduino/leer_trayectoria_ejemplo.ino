/*
 * Ejemplo: Leer trayectoria desde archivo CSV generado por Python
 * 
 * Este sketch muestra cómo leer y usar los datos de trayectoria
 * guardados en trajectory_YYYYMMDD_HHMMSS.csv
 * 
 * NOTA: Este es un ejemplo conceptual. En la práctica, deberías:
 * 1) Copiar los datos del CSV a un array en el código, o
 * 2) Usar una tarjeta SD para leer el archivo, o
 * 3) Recibir los datos por Serial desde Python (método recomendado)
 */

// Ejemplo de estructura para almacenar puntos de trayectoria
struct TrajectoryPoint {
  float time_s;
  float theta1_rad;
  float theta2_rad;
};

// Ejemplo: primeros puntos de una trayectoria
// (copiar del CSV generado por Python)
const int NUM_POINTS = 5;
TrajectoryPoint trajectory[NUM_POINTS] = {
  {0.000000, 3.141593, 0.000000},  // Pose inicial (parqueo)
  {0.016667, 3.141593, 0.000000},  // Durante dwell
  {0.033333, 3.141593, 0.000000},
  {0.050000, 3.139821, 0.001241},  // Comienza blend
  {0.066667, 3.134561, 0.004964}
};

int currentIndex = 0;
unsigned long startTime = 0;
bool trajectoryActive = false;

void setup() {
  Serial.begin(115200);
  Serial.println("Arduino - Reproductor de Trayectoria 2R");
  Serial.println("Comandos:");
  Serial.println("  S - Start (iniciar trayectoria)");
  Serial.println("  P - Pause (pausar)");
  Serial.println("  R - Reset (volver al inicio)");
  Serial.println();
  
  // Configurar pines de motores (ejemplo)
  // pinMode(MOTOR1_PWM, OUTPUT);
  // pinMode(MOTOR1_DIR, OUTPUT);
  // pinMode(MOTOR2_PWM, OUTPUT);
  // pinMode(MOTOR2_DIR, OUTPUT);
}

void loop() {
  // Leer comandos por Serial
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    handleCommand(cmd);
  }
  
  // Ejecutar trayectoria si está activa
  if (trajectoryActive) {
    updateTrajectory();
  }
  
  delay(10);  // 100 Hz
}

void handleCommand(char cmd) {
  switch(cmd) {
    case 'S':
    case 's':
      startTime = millis();
      currentIndex = 0;
      trajectoryActive = true;
      Serial.println("Trayectoria iniciada");
      break;
      
    case 'P':
    case 'p':
      trajectoryActive = false;
      Serial.println("Trayectoria pausada");
      break;
      
    case 'R':
    case 'r':
      trajectoryActive = false;
      currentIndex = 0;
      Serial.println("Trayectoria reiniciada");
      break;
  }
}

void updateTrajectory() {
  // Tiempo actual desde el inicio
  float elapsedTime = (millis() - startTime) / 1000.0;
  
  // Buscar el punto correspondiente al tiempo actual
  while (currentIndex < NUM_POINTS - 1 && 
         elapsedTime >= trajectory[currentIndex + 1].time_s) {
    currentIndex++;
  }
  
  // Si llegamos al final, detener
  if (currentIndex >= NUM_POINTS - 1) {
    trajectoryActive = false;
    Serial.println("Trayectoria completada");
    return;
  }
  
  // Obtener referencias actuales
  float theta1_ref = trajectory[currentIndex].theta1_rad;
  float theta2_ref = trajectory[currentIndex].theta2_rad;
  
  // Imprimir para debug
  Serial.print("t=");
  Serial.print(elapsedTime, 3);
  Serial.print(" θ1=");
  Serial.print(theta1_ref, 4);
  Serial.print(" θ2=");
  Serial.println(theta2_ref, 4);
  
  // AQUÍ: Implementar control PD
  // float theta1_actual = readEncoder1();  // Leer encoder/pot
  // float theta2_actual = readEncoder2();
  // 
  // float e1 = theta1_ref - theta1_actual;
  // float e2 = theta2_ref - theta2_actual;
  // 
  // float u1 = Kp1 * e1 + Kd1 * (e1 - e1_prev) / dt;
  // float u2 = Kp2 * e2 + Kd2 * (e2 - e2_prev) / dt;
  // 
  // driveMotor1(u1);
  // driveMotor2(u2);
}

/*
 * MÉTODO RECOMENDADO: Recibir por Serial en tiempo real desde Python
 * 
 * En lugar de almacenar todo el array, el Arduino puede recibir
 * comandos del tipo:
 *   R,theta1,theta2
 * 
 * desde Python a 60 fps, y ejecutar control PD en cada ciclo.
 * 
 * Ver: docs/firmware/README-FIRMWARE.md y docs/software/SERIAL_BRIDGE.md
 */

