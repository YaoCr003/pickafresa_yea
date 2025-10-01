#include <Arduino.h>
#include <DHT.h>

#define DHTPIN 12       // Pin for DHT22 
#define DHTTYPE DHT22   // DHT Sensor type

const int substrateHumidSens0Pin = 25; // Pin for substrate humidity sensor 0
const int substrateHumidSens1Pin = 26; // Pin for substrate humidity sensor 1
const int substrateHumidSens2Pin = 27; // Pin for substrate humidity sensor 2
const int ldrPin = 32;                // Pin para el LDR (Light Dependent Resistor)

// ESP32 ADC Config (12-bit ADC)
const float VCC = 3.3;         // ESP-32 VCC
const float ADC_RES = 4095.0;  // ADC resolution

// DHT Config
DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(115200);
  dht.begin();


  analogReadResolution(12); // Ensure 12-bit ADC resolution
}


void loop() {
  // Raw sensor readings
  int substrateHumid0 = analogRead(substrateHumidSens0Pin);
  int substrateHumid1 = analogRead(substrateHumidSens1Pin);
  int substrateHumid2 = analogRead(substrateHumidSens2Pin);

  // DHT Sensor Readings (Temperature and Humidity)
  float ambientHumid = dht.readHumidity();
  float ambientTemp = dht.readTemperature();


  // Raw LDR reading
  int lecturaLDR = analogRead(ldrPin);
  float voltajeLDR = (lecturaLDR * VCC) / ADC_RES;

  // Light intensity calculation (in lux)

  // Light intensity calculation in percetage

  // 