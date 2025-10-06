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

//MQTT Configuration
const char* mqtt_server = "10.25.12.61";
const int mqtt_port = 1883;
const char* mqtt_user = "yeaberry";
const char* mqtt_pass = "yeaberry";

//MQTT Topics
#define PUB_TEMP "sensor/temperatura"
#define PUB_AMB "sensor/ambienteH"
#define PUB_SUBS "sensor/substrateM"
#define PUB_LIGHT "sensor/light"

#define PUB_ALERT "alerts"

#define SUB_DATA "request/data"
#define PUB_DATA "request/data"


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
const int ldrNoLight = 100;
const int ldrMidLight = 1500;
const int ldrLight = 3000;

// DHT Config
DHT dht(DHTPIN, DHTTYPE);

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

WiFiClient espClient;
PubSubClient client(espClient);
unsigned long lastMsg = 0;

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
  client.setServer(mqtt_server, mqtt_port);
  reconnect();

  Serial.println("Pickafresa Crop Sensor Initialized");
  
}

//MQTT Reconnect
void reconnect() {
  while (!client.connected()) {
    if (client.connect("ESP32Client", mqtt_user, mqtt_pass)) { //mqtt_user, mqtt_pass
      Serial.println("connected!");
      client.subscribe(SUB_DATA);
    } else {
      delay(2000);
    }
    yield();
  }
}

void loop() {
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
    reconnect();
  }
  client.loop();

  unsigned long now = millis();
  if (now - lastMsg >= 30000) {   // every ~7 s
    lastMsg = now;

  delay(1000);
  }

}

void stopwatchReset() {
  // Reset stopwatch timer

}