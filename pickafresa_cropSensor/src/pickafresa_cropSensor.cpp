// ================
// YEA Pickafresa System Crop Sensor Firmware
// Version: Beta v1.3.1

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
// Firmware features: (Beta v1.3.1)
// - Reads temperature and humidity from DHT22
// - Reads substrate moisture from 3 capacitive soil moisture sensors
// - Reads ambient light level from LDR
// - Connects to WiFi and MQTT broker
// - Publishes sensor data to MQTT topics
// - Subscribes to MQTT topic for data requests
// - Buffers data and applies outlier filtering before publishing
// - Alerts via MQTT on sensor errors
// - Debug stream via MQTT & serial console
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

#define SUB_DEBUG "debug"
#define PUB_DEBUG_STREAM "debug_stream"

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
const unsigned long RECORDING_INTERVAL = 10000; // 60 seconds in milliseconds

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

// Debug stream state
bool debugStreamActive = false;

// Function declarations
void recordSensorData();
bool checkSensorErrors(int substrateHumid0, int substrateHumid1, int substrateHumid2, 
                      float ambientTemp, float ambientHumid, int lecturaLDR,
                      int& substrateHumid0Perc, int& substrateHumid1Perc, int& substrateHumid2Perc,
                      int& lightPerc, float& safeTemp, float& safeHumid);
float filterOutliersAndAverage(float* buffer, int count);
void publishFilteredData();
void clearBuffers();
void mqttCallback(char* topic, byte* payload, unsigned int length);

// BUILTIN LED Basic Flash function
void ledFlash(byte times, int onTime, int offTime) {
  for (byte i = 0; i < times; i++) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(onTime);
    digitalWrite(LED_BUILTIN, LOW);
    delay(offTime);
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
  if (debugStreamActive) {
    client.publish(PUB_DEBUG_STREAM, alertMsg);
  }
  Serial.print("ALERT published: ");
  Serial.println(alertMsg);
  ledFlash(3, 500, 100);
}

// Check sensor errors and convert raw readings to percentages
// Returns true if any sensor error is detected, false otherwise
// Modifies the percentage and safe value parameters by reference
bool checkSensorErrors(int substrateHumid0, int substrateHumid1, int substrateHumid2, 
                      float ambientTemp, float ambientHumid, int lecturaLDR,
                      int& substrateHumid0Perc, int& substrateHumid1Perc, int& substrateHumid2Perc,
                      int& lightPerc, float& safeTemp, float& safeHumid) {
  bool sensorError = false;
  
  // Initialize output parameters
  substrateHumid0Perc = 0;
  substrateHumid1Perc = 0;
  substrateHumid2Perc = 0;
  lightPerc = 0;
  safeTemp = 0.0;
  safeHumid = 0.0;

  // Check substrate sensor 0
  if (substrateHumid0 < 1000 || substrateHumid0 > 3000) {
    substrateHumid0Perc = 0;
    sensorError = true;
    publishAlert("error", "SENSOR", "Substrate sensor 0 disconnected or invalid");
  } else {
    substrateHumid0Perc = constrain(map(substrateHumid0, shs0Air, shs0Water, 0, 100), 0, 100);
  }

  // Check substrate sensor 1
  if (substrateHumid1 < 1000 || substrateHumid1 > 3000) {
    substrateHumid1Perc = 0;
    sensorError = true;
    publishAlert("error", "SENSOR", "Substrate sensor 1 disconnected or invalid");
  } else {
    substrateHumid1Perc = constrain(map(substrateHumid1, shs0Air, shs0Water, 0, 100), 0, 100);
  }

  // Check substrate sensor 2
  if (substrateHumid2 < 1000 || substrateHumid2 > 3000) {
    substrateHumid2Perc = 0;
    sensorError = true;
    publishAlert("error", "SENSOR", "Substrate sensor 2 disconnected or invalid");
  } else {
    substrateHumid2Perc = constrain(map(substrateHumid2, shs0Air, shs0Water, 0, 100), 0, 100);
  }

  // Check DHT22 sensor
  if (isnan(ambientHumid) || isnan(ambientTemp)) {
    safeTemp = 0.0;
    safeHumid = 0.0;
    sensorError = true;
    publishAlert("error", "SENSOR", "DHT22 sensor disconnected or invalid");
  } else {
    safeTemp = ambientTemp;
    safeHumid = ambientHumid;
  }

  // Check LDR sensor
  if (lecturaLDR <= 0 || lecturaLDR >= 4095) {
    lightPerc = 0;
    sensorError = true;
    publishAlert("error", "SENSOR", "LDR sensor disconnected or invalid");
  } else {
    lightPerc = constrain(map(lecturaLDR, ldrNoLight, ldrLight, 0, 100), 0, 100);
  }

  return sensorError;
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
    if (debugStreamActive) {
      char ackMsg[64];
      snprintf(ackMsg, sizeof(ackMsg), "SUB_REQ received: %s", reqTimestamp);
      client.publish(PUB_DEBUG_STREAM, ackMsg);
    }

    // Publish filtered data (debug info will be sent only if debugStreamActive)
    publishFilteredData();
  } else if (strcmp(topic, SUB_DEBUG) == 0) {
    char msg[16];
    unsigned int n = (length < sizeof(msg) - 1) ? length : (sizeof(msg) - 1);
    memcpy(msg, payload, n);
    msg[n] = '\0';
    Serial.print("DEBUG topic received: ");
    Serial.println(msg);
    if (strcmp(msg, "debug") == 0 && !debugStreamActive) {
      debugStreamActive = true;
      char startMsg[64];
      snprintf(startMsg, sizeof(startMsg), "Start Debug Stream on from %s IP", WiFi.localIP().toString().c_str());
      client.publish(PUB_DEBUG_STREAM, startMsg);
      Serial.println(startMsg);
    } else if ((strcmp(msg, "stop") == 0 || strcmp(msg, "close") == 0) && debugStreamActive) {
      // Publish closed message while still active, then deactivate
      client.publish(PUB_DEBUG_STREAM, "Debug Stream closed.");
      Serial.println("Debug Stream closed.");
      debugStreamActive = false;
    }
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

  // Variables for processed sensor data
  int substrateHumid0Perc, substrateHumid1Perc, substrateHumid2Perc, lightPerc;
  float safeTemp, safeHumid;

  // Check sensor errors and get processed values
  bool sensorError = checkSensorErrors(substrateHumid0, substrateHumid1, substrateHumid2,
                                      ambientTemp, ambientHumid, lecturaLDR,
                                      substrateHumid0Perc, substrateHumid1Perc, substrateHumid2Perc,
                                      lightPerc, safeTemp, safeHumid);

  // Calculate average substrate moisture from valid sensors
  float avgSubstrate = (substrateHumid0Perc + substrateHumid1Perc + substrateHumid2Perc) / 3.0;

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
  if (debugStreamActive) {
    char recordedMsg[160];
    snprintf(recordedMsg, sizeof(recordedMsg),
             "Recorded data [index %d, count %d]: T=%.2f, H=%.2f, S=%.1f, L=%d",
             bufferIndex, bufferCount, safeTemp, safeHumid, avgSubstrate, lightPerc);
    client.publish(PUB_DEBUG_STREAM, recordedMsg);
  }
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
  if (debugStreamActive) {
    client.publish(PUB_DEBUG_STREAM, "DATA BUFFERS CLEARED");
  }
}

// Publish filtered and averaged sensor data
void publishFilteredData() {
  if (bufferCount == 0) {
    Serial.println("NO DATA AVAILABLE: PUBLISH NONE");
    if (debugStreamActive) {
      client.publish(PUB_DEBUG_STREAM, "NO DATA AVAILABLE: PUBLISH NONE");
    }
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

  // Concise published summary (mirror on Serial and debug stream when active)
  {
    String publishedInfo = "Published averages - T:" + String(avgTemp,2) +
                           ", H:" + String(avgHumid,2) +
                           ", S:" + String(avgSubstrate,1) +
                           ", L:" + String(avgLight,1) +
                           " (" + String(bufferCount) + " samples)";
    Serial.println(publishedInfo);
    if (debugStreamActive) {
      client.publish(PUB_DEBUG_STREAM, publishedInfo.c_str());
    }
  }

  // Print/publish buffer contents for debug
  String debugInfo = "Averages - T:" + String(avgTemp,2) + ", H:" + String(avgHumid,2) + ", S:" + String(avgSubstrate,1) + ", L:" + String(avgLight,1) + " (" + String(bufferCount) + " samples)\n";
  debugInfo += "Raw buffers: T[";
  for (int i = 0; i < bufferCount; i++) debugInfo += String(tempBuffer[i],2) + (i<bufferCount-1?",":"");
  debugInfo += "] H[";
  for (int i = 0; i < bufferCount; i++) debugInfo += String(humidBuffer[i],2) + (i<bufferCount-1?",":"");
  debugInfo += "] S[";
  for (int i = 0; i < bufferCount; i++) debugInfo += String(substrateBuffer[i],1) + (i<bufferCount-1?",":"");
  debugInfo += "] L[";
  for (int i = 0; i < bufferCount; i++) debugInfo += String(lightBuffer[i],1) + (i<bufferCount-1?",":"");
  debugInfo += "]";

  Serial.println(debugInfo);
  if (debugStreamActive) {
    client.publish(PUB_DEBUG_STREAM, debugInfo.c_str());
  }

  // Clear buffers after publishing to ensure fresh data for next request
  clearBuffers();
  ledFlash(2, 100, 50); // Quick flash to indicate publish
}



//MQTT Reconnect
void mqtt_reconnect() {
  while (!client.connected()) {
    if (client.connect("ESP32Client", MQTT_USER, MQTT_PASS)) { //mqtt_user, mqtt_pass
      Serial.println("MQTT connected!");
      client.subscribe(SUB_REQ);
      client.subscribe(SUB_DEBUG);
      ledFlash(3, 100, 50);
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
    ledFlash(1, 200, 0);
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    Serial.print("IP: "); Serial.println(WiFi.localIP());
    ledFlash(3, 100, 50);
  } else {
    Serial.println("\nWiFi connect failed.");
    ledFlash(5, 750, 250);
  }
  // Initialize DHT sensor
  dht.begin();

  //Configure pin modes
  pinMode(LED_BUILTIN, OUTPUT);
  // Pin modes for sensors are preset to ADC input by default

  // Configure ADC resolution
  analogReadResolution(12); // Ensure 12-bit ADC resolution

  
  // Sensor calibration
  sensorCalib();
  
  //Establish connection to MQTT broker
  client.setServer(MQTT_BROKER, MQTT_PORT);
  client.setCallback(mqttCallback);
  mqtt_reconnect();
  
  // Flash LED to indicate setup complete (acts as delay)
  ledFlash(10, 100, 50);

  Serial.println("Pickafresa Crop Sensor Initialized");
  
}


void loop() {
  unsigned long currentTime = millis();
  // Raw substate sensor readings
  int substrateHumid0 = analogRead(shs0Pin);
  int substrateHumid1 = analogRead(shs1Pin);
  int substrateHumid2 = analogRead(shs2Pin);

  // DHT Sensor Readings (Temperature and Humidity)
  float ambientHumid = dht.readHumidity();
  float ambientTemp = dht.readTemperature();


  // Raw LDR reading
  int lecturaLDR = analogRead(ldrPin);
  float voltajeLDR = (lecturaLDR * VCC) / ADC_RES;

  // Substrate humidity calculation (in percentage)
  int substrateHumid0Perc = map(substrateHumid0, shs0Air, shs0Water, 0, 100);
  substrateHumid0Perc = constrain(substrateHumid0Perc, 0, 100);
  int substrateHumid1Perc = map(substrateHumid1, shs0Air, shs0Water, 0, 100);
  substrateHumid1Perc = constrain(substrateHumid1Perc, 0, 100);
  int substrateHumid2Perc = map(substrateHumid2, shs0Air, shs0Water, 0, 100);
  substrateHumid2Perc = constrain(substrateHumid2Perc, 0, 100);


  // Debug serial prints
  Serial.print("SH 0: "); Serial.print(substrateHumid0); Serial.print(" ("); Serial.print(substrateHumid0Perc); Serial.print("%) | ");
  Serial.print("SH 1: "); Serial.print(substrateHumid1); Serial.print(" ("); Serial.print(substrateHumid1Perc); Serial.print("%) | ");
  Serial.print("SH 2: "); Serial.print(substrateHumid2); Serial.print(" ("); Serial.print(substrateHumid2Perc); Serial.print("%) | ");

  Serial.print("AH: "); Serial.print(ambientHumid); Serial.print("% | ");
  Serial.print("AT: "); Serial.print(ambientTemp); Serial.print("°C | ");
  Serial.print("LT: "); Serial.print(lecturaLDR); Serial.print(" ("); Serial.print(voltajeLDR); Serial.println(" V)");

  
  // Light intensity calculation (in lux)

  // Light intensity calculation in percetage
  int lightPerc = map(lecturaLDR, ldrNoLight, ldrLight, 0, 100);
  lightPerc = constrain(lightPerc, 0, 100);


  // Show light intensity percentage on serial monitor
  Serial.print("LT%: "); Serial.print(lightPerc); Serial.print("%");

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
    int substrateHumid2 = analogRead(shs2Pin);
    float ambientHumid = dht.readHumidity();
    float ambientTemp = dht.readTemperature();
    int lecturaLDR = analogRead(ldrPin);

    int substrateHumid0Perc = constrain(map(substrateHumid0, shs0Air, shs0Water, 0, 100), 0, 100);
    int substrateHumid1Perc = constrain(map(substrateHumid1, shs0Air, shs0Water, 0, 100), 0, 100);
    int substrateHumid2Perc = constrain(map(substrateHumid2, shs0Air, shs0Water, 0, 100), 0, 100);
    int lightPerc = constrain(map(lecturaLDR, ldrNoLight, ldrLight, 0, 100), 0, 100);

    //check for sensor errors
    checkSensorErrors(substrateHumid0, substrateHumid1, substrateHumid2,
                      ambientTemp, ambientHumid, lecturaLDR,
                      substrateHumid0Perc, substrateHumid1Perc, substrateHumid2Perc,
                      lightPerc, ambientTemp, ambientHumid);


    char debugMsg[128];
    snprintf(debugMsg, sizeof(debugMsg),
      "Real-time - SH: %d (%d), %d (%d), %d (%d)%% | T: %.2f°C | H: %.2f%% | L: %d (%d)%%",
      substrateHumid0Perc, substrateHumid0,
      substrateHumid1Perc, substrateHumid1,
      substrateHumid2Perc, substrateHumid2,
      ambientTemp, ambientHumid,
      lightPerc, lecturaLDR);

    Serial.println(debugMsg);
    if (debugStreamActive) {
      client.publish(PUB_DEBUG_STREAM, debugMsg);
    }
    ledFlash(1, 50, 0); // Quick flash to indicate activity
  }
  

  delay(100); // Small delay to prevent overwhelming the system
}