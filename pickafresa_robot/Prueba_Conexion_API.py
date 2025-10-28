from robolink import *  # API RoboDK
from robodk import *
import keyboard
import paho.mqtt.client as mqtt
import numpy as np
import time

RDK = Robolink()
RDK.setRunMode(RUNMODE_SIMULATE)
#RDK.setRunMode(RUNMODE_RUN_ROBOT)

broker="192.168.1.114"
topic="actuador/on_off"
topic_2="actuador/state"

client = mqtt.Client()
client.connect(broker, 1883, 60)

actuador_encendido=False
estado_actuador=""

def actuador_on():
    global actuador_encendido
    actuador_encendido=True
    client.publish(topic, "Gripper encendido")
    print("Actuador encendido")

def actuador_off():
    global actuador_encendido
    actuador_encendido=False
    client.publish(topic, "Gripper apagado")
    print("Actuador apagado")

def on_message(client, userdata, msg):
    global estado_actuador
    mensaje = msg.payload.decode().strip().lower()
    print(f"Mensaje recibido en {msg.topic}: {mensaje}")

    if msg.topic == topic_2:
        if mensaje in ["inflado", "desinflado"]:
            estado_actuador = mensaje

client.on_message = on_message

client.subscribe(topic_2)
client.loop_start()

def esperar_estado(deseado):
    """Bloquea el programa hasta recibir un estado específico."""
    global estado_actuador
    print(f"Esperando estado '{deseado}' del actuador...")
    estado_actuador = ""
    while estado_actuador != deseado:
        if keyboard.is_pressed("esc"):
            print("Programa detenido por el usuario")
            client.loop_stop()
            return False
        time.sleep(0.1)
    print(f"Estado '{deseado}' confirmado.")
    return True

home_target = RDK.Item('Home', ITEM_TYPE_TARGET)# type: ignore
foto_target = RDK.Item('Foto', ITEM_TYPE_TARGET)# type: ignore
T3_target = RDK.Item('Target 3', ITEM_TYPE_TARGET)# type: ignore
T4_target = RDK.Item('Target 4', ITEM_TYPE_TARGET)# type: ignore
T5_target = RDK.Item('Target 5', ITEM_TYPE_TARGET)# type: ignore

item = RDK.ItemUserPick('UR3e', ITEM_TYPE_ROBOT)# type: ignore
item.setSpeed(60, 60) 

#item.Connect()  # Intenta conexión activa (opcional)

while True:
    if keyboard.is_pressed("esc"):
        print("Programa terminado por usuario")
        break
    if item.Valid():
        item.setSpeed(60, 60) 
        print('Conectado correctamente con RoboDK.')
        item.MoveJ(home_target)
        item.MoveJ(foto_target)
        item.MoveJ(T5_target)
        
        actuador_on()
        if not esperar_estado("inflado"):
            break
    
        item.MoveJ(T3_target)
        item.MoveL(T4_target)

        actuador_off()
        if not esperar_estado("desinflado"):
            break

        item.MoveL(T3_target)
        item.MoveJ(home_target)
    else:
        print('No se pudo conectar con RoboDK.')
        break
