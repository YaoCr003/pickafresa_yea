// --- Pines ---
const int sensorPin = 34;   // Entrada analógica (sensor de presión)
const int pwmPin = 18;      // Salida PWM

// --- Controlador PI ---
float Kp = 2.0;        // Ganancia proporcional
float Ki = 0.5;        // Ganancia integral
float setpoint = 70.0; // Presión deseada (ejemplo: 50 kPa)

// --- Variables ---
float pressure = 0.0;
float error = 0.0;
float integral = 0.0;
float controlSignal = 0.0;
int pwmValue = 0;

// --- PWM ---
const int freq = 5000;
const int pwmChannel = 0;
const int resolution = 8; // 8 bits (0–255)

// --- Sensor ---
const float vMin = 0.5;   // Voltaje a 0 psi
const float vMax = 4.5;   // Voltaje a presión máxima
const float pMin = 0.0;   // Presión mínima
const float pMax = 100.0; // Presión máxima del sensor

// --- Tiempo ---
unsigned long prevTime = 0;
float dt = 0.0;

void setup() {
  Serial.begin(115200);
  ledcSetup(pwmChannel, freq, resolution);
  ledcAttachPin(pwmPin, pwmChannel);

  Serial.println("Tiempo(s),Presion(Pa),PWM,Error");
}

void loop() {
  unsigned long now = millis();
  dt = (now - prevTime) / 10.0; // Tiempo en segundos
  if (dt <= 0) dt = 0.001;
  prevTime = now;

  // --- Leer el sensor ---
  int rawValue = analogRead(sensorPin);
  float voltage = (rawValue / 4095.0) * 3.3; // Voltaje leído

  // Convertir a presión (lineal)
  pressure = (voltage - vMin) * (pMax - pMin) / (vMax - vMin);
  if (pressure < 0) pressure = 0;
  if (pressure > pMax) pressure = pMax;

  // --- Calcular PI ---
  error = setpoint - pressure;
  integral += error * dt;
  controlSignal = Kp * error + Ki * integral;

  // Limitar PWM
  if (controlSignal > 255) controlSignal = 255;
  if (controlSignal < 0) controlSignal = 0;

  pwmValue = (int)controlSignal;
  ledcWrite(pwmChannel, pwmValue);

  // --- Imprimir datos ---
  Serial.print(now / 1000.0, 2);
  Serial.print(",");
  Serial.print(pressure, 2);
  Serial.print(",");
  Serial.print(pwmValue);
  Serial.print(",");
  Serial.println(error, 2);

  delay(500); // cada 0.5 segundos
}


/* Impresion en el Monitor Serial

Tiempo(s),Presion(Pa),PWM,Error
0.50,12.45,240,37.55
1.00,35.70,180,14.30
1.50,48.20,130,1.80
2.00,50.10,125,-0.10
 */
