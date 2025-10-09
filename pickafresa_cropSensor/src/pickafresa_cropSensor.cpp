// ================
// YEA Pickafresa System Crop Sensor Firmware
// Version: Beta v1.2

// Developed by: Team YEA
// @YaoCr003
// @a01705488-rgb
// @aldrick-t
// 
// Developed for: Pickafresa Project
// 
// Tested and calibrated for:
// Hardware: ESP32 (ESP32-DOIT DevKit V1), DHT22, 3x Capacitive Soil Moisture Sensors v1.2, 10k LDR
//
// Firmware features: (Beta v1.2)
// - Reads temperature and humidity from DHT22
// - Reads substrate moisture from 3 capacitive soil moisture sensors
// - Reads ambient light level from LDR
// - Connects to WiFi and MQTT broker
// - Publishes sensor data to MQTT topics
// - Subscribes to MQTT topic for data requests
// - Buffers data and applies outlier filtering before publishing
// - Alerts via MQTT on sensor errors
// 
// AUGUST 2025 - DECEMBER 2025
// ================


#include <Arduino.h>
#include <DHT.h>
#include <PubSubClient.h>
#include <WiFi.h>

#ifndef WIFI_SSID
  #include "secrets.h"
#endif

// DHT Definitions
#define DHTPIN 12       // Pin for DHT22 
#define DHTTYPE DHT22   // DHT Sensor type

#define LED_BUILTIN 2   // Pin for built-in LED

//MQTT Topics
#define PUB_TEMP "sensor/temperature"
#define PUB_AMBIH "sensor/ambientH"
#define PUB_SUBSH "sensor/substrateM"
#define PUB_LIGHT "sensor/light"

#define PUB_ALERT "alerts"

#define SUB_REQ "request/data"

// Pin Definitions
const int shs0Pin = 32; // Pin for substrate humidity sensor 0
const int shs1Pin = 33;  // Pin for substrate humidity sensor 1
const int shs2Pin = 34; // Pin for substrate humidity sensor 2
const int ldrPin = 35;  // Pin para el LDR (Light Dependent Resistor)

// ESP32 ADC Config (12-bit ADC)
const float VCC = 3.3;         // ESP-32 VCC
const float ADC_RES = 4095.0;  // ADC resolution

// Substate humidity sensor calibration values (all measured at 12-bit ADC, submerged halfway)
const int shs0Air = 2650;       // Air calibration value for sensor 0
const int shs0Dry = 2430;       // Dry calibration value for sensor 0
const int shs0Moist = 1660;     // Moist calibration value for sensor 0
const int shs0Saturated = 1350; // Saturated calibration value for sensor 0
const int shs0Water = 1090;     // Water calibration value for
const int ldrNoLight = 100;   // No light calibration value for LDR (Near complete darkness)
const int ldrMidLight = 1500; // Mid light calibration value for LDR (Indoor lighting)
const int ldrLight = 3000;  // Full light calibration value for LDR (Flashlight very close)

// Data storage configuration
const int DATA_BUFFER_SIZE = 50;  // Store last 50 readings (50 minutes at 60s intervals)
const unsigned long RECORDING_INTERVAL = 60000; // 60 seconds in milliseconds

// Circular buffers for sensor data
float tempBuffer[DATA_BUFFER_SIZE];
float humidBuffer[DATA_BUFFER_SIZE];
float substrateBuffer[DATA_BUFFER_SIZE]; // Average of 3 substrate sensors
float lightBuffer[DATA_BUFFER_SIZE];
int bufferIndex = 0;
int bufferCount = 0; // Tracks how many valid entries we have

// DHT Config
DHT dht(DHTPIN, DHTTYPE);

WiFiClient espClient;
PubSubClient client(espClient);

unsigned long lastMsg = 0;
unsigned long lastSensorRead = 0;

// Function declarations
void recordSensorData();
float filterOutliersAndAverage(float* buffer, int count);
void publishFilteredData();
void clearBuffers();
void mqttCallback(char* topic, byte* payload, unsigned int length);

// BUILTIN LED Basic Flash function
void ledFlash(byte times, int delayTime) {
  for (byte i = 0; i < times; i++) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(delayTime);
    digitalWrite(LED_BUILTIN, LOW);
    delay(delayTime);
  }
}

void sensorCalib() {
  return;
}

// Publish alert message in format "level|type|message"
void publishAlert(const char* level, const char* type, const char* message) {
  char alertMsg[128];
  snprintf(alertMsg, sizeof(alertMsg), "%s|%s|%s", level, type, message);
  client.publish(PUB_ALERT, alertMsg);
  Serial.print("ALERT published: ");
  Serial.println(alertMsg);
}

// MQTT callback function
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  if (strcmp(topic, SUB_REQ) == 0) {
    char reqTimestamp[24];
    unsigned int n = (length < sizeof(reqTimestamp) - 1) ? length : (sizeof(reqTimestamp) - 1);
    memcpy(reqTimestamp, payload, n);
    reqTimestamp[n] = '\0';

    Serial.print("SUB_REQ received: ");
    Serial.println(reqTimestamp);

    publishFilteredData();
  }
}

// Record current sensor readings into circular buffers
void recordSensorData() {
  // Read raw sensors
  int substrateHumid0 = analogRead(shs0Pin);
  int substrateHumid1 = analogRead(shs1Pin);
  int substrateHumid2 = analogRead(shs2Pin);
  float ambientHumid = dht.readHumidity();
  float ambientTemp = dht.readTemperature();
  int lecturaLDR = analogRead(ldrPin);

  // Sensor error detection
  bool sensorError = false;
  // ADC returns -1 or 4095 (max) for disconnected sensors on ESP32
  if (substrateHumid0 < 1000 || substrateHumid1 < 1000 || substrateHumid2 < 1000 ||
      isnan(ambientHumid) || isnan(ambientTemp) ||
      lecturaLDR <= 0 || lecturaLDR >= 4095) {
    sensorError = true;
  }

  int substrateHumid0Perc = 0, substrateHumid1Perc = 0, substrateHumid2Perc = 0, lightPerc = 0;
  float avgSubstrate = 0.0;
  float safeTemp = 0.0, safeHumid = 0.0;

  if (sensorError) {
    // Set all readings to zero
    substrateHumid0Perc = 0;
    substrateHumid1Perc = 0;
    substrateHumid2Perc = 0;
    avgSubstrate = 0.0;
    lightPerc = 0;
    safeTemp = 0.0;
    safeHumid = 0.0;
    publishAlert("error", "SENSOR", "One or more sensors disconnected or invalid");
  } else {
    // Convert to percentages
    substrateHumid0Perc = constrain(map(substrateHumid0, shs0Air, shs0Water, 0, 100), 0, 100);
    substrateHumid1Perc = constrain(map(substrateHumid1, shs0Air, shs0Water, 0, 100), 0, 100);
    substrateHumid2Perc = constrain(map(substrateHumid2, shs0Air, shs0Water, 0, 100), 0, 100);
    avgSubstrate = (substrateHumid0Perc + substrateHumid1Perc + substrateHumid2Perc) / 3.0;
    lightPerc = constrain(map(lecturaLDR, ldrNoLight, ldrLight, 0, 100), 0, 100);
    safeTemp = ambientTemp;
    safeHumid = ambientHumid;
  }

  // Store in circular buffers
  tempBuffer[bufferIndex] = safeTemp;
  humidBuffer[bufferIndex] = safeHumid;
  substrateBuffer[bufferIndex] = avgSubstrate;
  lightBuffer[bufferIndex] = lightPerc;

  // Update buffer management
  bufferIndex = (bufferIndex + 1) % DATA_BUFFER_SIZE;
  if (bufferCount < DATA_BUFFER_SIZE) {
    bufferCount++;
  }

  Serial.print("Recorded data [index ");
  Serial.print(bufferIndex);
  Serial.print(", count ");
  Serial.print(bufferCount);
  Serial.print("]: T=");
  Serial.print(safeTemp);
  Serial.print(", H=");
  Serial.print(safeHumid);
  Serial.print(", S=");
  Serial.print(avgSubstrate);
  Serial.print(", L=");
  Serial.println(lightPerc);
}

// Filter outliers using IQR method and return average of remaining values
float filterOutliersAndAverage(float* buffer, int count) {
  if (count <= 2) {
    // Not enough data for outlier detection, just average
    float sum = 0;
    for (int i = 0; i < count; i++) {
      sum += buffer[i];
    }
    return sum / count;
  }

  // Create a sorted copy for quartile calculation
  float sorted[DATA_BUFFER_SIZE];
  for (int i = 0; i < count; i++) {
    sorted[i] = buffer[i];
  }

  // Simple bubble sort
  for (int i = 0; i < count - 1; i++) {
    for (int j = 0; j < count - i - 1; j++) {
      if (sorted[j] > sorted[j + 1]) {
        float temp = sorted[j];
        sorted[j] = sorted[j + 1];
        sorted[j + 1] = temp;
      }
    }
  }

  // Calculate quartiles
  int q1_idx = count / 4;
  int q3_idx = (3 * count) / 4;
  float q1 = sorted[q1_idx];
  float q3 = sorted[q3_idx];
  float iqr = q3 - q1;
  float lowerBound = q1 - 1.5 * iqr;
  float upperBound = q3 + 1.5 * iqr;

  // Average values within bounds
  float sum = 0;
  int validCount = 0;
  for (int i = 0; i < count; i++) {
    if (buffer[i] >= lowerBound && buffer[i] <= upperBound) {
      sum += buffer[i];
      validCount++;
    }
  }

  return validCount > 0 ? sum / validCount : sorted[count / 2]; // Return median if all are outliers
}

// Clear all data buffers and reset counters
void clearBuffers() {
  bufferIndex = 0;
  bufferCount = 0;
  Serial.println("DATA BUFFERS CLEARED");
}

// Publish filtered and averaged sensor data
void publishFilteredData() {
  if (bufferCount == 0) {
    Serial.println("NO DATA AVAILABLE: PUBLISH NONE");
    return;
  }

  // Filter and average each sensor type
  float avgTemp = filterOutliersAndAverage(tempBuffer, bufferCount);
  float avgHumid = filterOutliersAndAverage(humidBuffer, bufferCount);
  float avgSubstrate = filterOutliersAndAverage(substrateBuffer, bufferCount);
  float avgLight = filterOutliersAndAverage(lightBuffer, bufferCount);

  // Publish to MQTT topics
  char tempStr[16], humidStr[16], substrateStr[16], lightStr[16];
  dtostrf(avgTemp, 0, 2, tempStr);
  dtostrf(avgHumid, 0, 2, humidStr);
  dtostrf(avgSubstrate, 0, 1, substrateStr);
  dtostrf(avgLight, 0, 1, lightStr);

  client.publish(PUB_TEMP, tempStr);
  client.publish(PUB_AMBIH, humidStr);
  client.publish(PUB_SUBSH, substrateStr);
  client.publish(PUB_LIGHT, lightStr);

  Serial.print("Published filtered averages - T:");
  Serial.print(avgTemp);
  Serial.print(", H:");
  Serial.print(avgHumid);
  Serial.print(", S:");
  Serial.print(avgSubstrate);
  Serial.print(", L:");
  Serial.print(avgLight);
  Serial.print(" (from ");
  Serial.print(bufferCount);
  Serial.println(" samples)");
  
  // Clear buffers after publishing to ensure fresh data for next request
  clearBuffers();
}



//MQTT Reconnect
void mqtt_reconnect() {
  while (!client.connected()) {
    if (client.connect("ESP32Client", MQTT_USER, MQTT_PASS)) { //mqtt_user, mqtt_pass
      Serial.println("MQTT connected!");
      client.subscribe(SUB_REQ);
    } else {
      delay(2000);
    }
    yield();
  }
}

void setup() {
  // Serial for debugging
  Serial.begin(115200);
  // WiFi setup
  Serial.print("Connecting to ");
  Serial.println(WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 15000) {
    delay(300);
    Serial.print('.');
    ledFlash(1, 100);
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    Serial.print("IP: "); Serial.println(WiFi.localIP());
    ledFlash(3, 100);
  } else {
    Serial.println("\nWiFi connect failed.");
    ledFlash(5, 500);
  }
  // Initialize DHT sensor
  dht.begin();

  //Configure pin modes
  pinMode(LED_BUILTIN, OUTPUT);
  // Pin modes for sensors are preset to ADC input by default

  // Configure ADC resolution
  analogReadResolution(12); // Ensure 12-bit ADC resolution

  // Flash LED to indicate setup complete (acts as delay)
  ledFlash(10, 100);

  // Sensor calibration
  sensorCalib();
  
  //Establish connection to MQTT broker
  client.setServer(MQTT_BROKER, MQTT_PORT);
  client.setCallback(mqttCallback);
  mqtt_reconnect();
  

  Serial.println("Pickafresa Crop Sensor Initialized");
  
}


void loop() {
  unsigned long currentTime = millis();

  // Record sensor data every RECORDING_INTERVAL (60 seconds)
  if (currentTime - lastSensorRead >= RECORDING_INTERVAL) {
    recordSensorData();
    lastSensorRead = currentTime;
  }

  // Verify MQTT connection and keep alive
  if (!client.connected()) {
    mqtt_reconnect();
  }
  client.loop();

  // Optional: Still show real-time readings for debugging (less frequent)
  if (currentTime - lastMsg >= 10000) { // Every 10 seconds
    lastMsg = currentTime;
    
    // Quick real-time readings for debug
    int substrateHumid0 = analogRead(shs0Pin);
    int substrateHumid1 = analogRead(shs1Pin);
    Serial.print("Substrate: "); Serial.println(substrateHumid1);
    int substrateHumid2 = analogRead(shs2Pin);
    float ambientHumid = dht.readHumidity();
    float ambientTemp = dht.readTemperature();
    int lecturaLDR = analogRead(ldrPin);

    int substrateHumid0Perc = constrain(map(substrateHumid0, shs0Air, shs0Water, 0, 100), 0, 100);
    int substrateHumid1Perc = constrain(map(substrateHumid1, shs0Air, shs0Water, 0, 100), 0, 100);
    int substrateHumid2Perc = constrain(map(substrateHumid2, shs0Air, shs0Water, 0, 100), 0, 100);
    int lightPerc = constrain(map(lecturaLDR, ldrNoLight, ldrLight, 0, 100), 0, 100);

    Serial.print("Real-time - SH: ");
    Serial.print(substrateHumid0Perc); Serial.print(", ");
    Serial.print(substrateHumid1Perc); Serial.print(", ");
    Serial.print(substrateHumid2Perc); Serial.print("% | ");
    Serial.print("T: "); Serial.print(ambientTemp); Serial.print("°C | ");
    Serial.print("H: "); Serial.print(ambientHumid); Serial.print("% | ");
    Serial.print("L: "); Serial.print(lightPerc); Serial.println("%");
  }
  

  delay(100); // Small delay to prevent overwhelming the system
}