from robolink import *  # API RoboDK
from robodk import *
import paho.mqtt.client as mqtt
import numpy as np
import time
import keyboard

#Variables estado inicial
actuador_encendido=False
estado_actuador=""

#MQTT
broker="192.168.1.114"
topic="actuador/on_off"
topic_2="actuador/state"

client = mqtt.Client()
client.connect(broker, 1883, 60)

#Encender o apagar actuador
def actuador_on():
    client.publish(topic, "Gripper encendido")
    global actuador_encendido
    actuador_encendido=True
    print("Actuador encendido")

def actuador_off():
    client.publish(topic, "Gripper apagado")
    global actuador_encendido
    actuador_encendido=False
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



#Estado de robot: simulacion o fisico
RDK = Robolink()
#RDK.setRunMode(RUNMODE_RUN_ROBOT)
RDK.setRunMode(RUNMODE_SIMULATE)

#Definir robot
item = RDK.ItemUserPick('UR3e', ITEM_TYPE_ROBOT)# type: ignore
item.setSpeed(60, 60)

# Intenta conexión activa (opcional)
item.Connect() 


#Posición fresa respecto a camara
fresa_cam=np.array()

#Matriz de transformacion Cámara-Base robot
cam_robot=np.array([
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
])

#Fresa en coordenadas del robot
fresa_robot=cam_robot @ fresa_cam

offset_x=-50

matriz_offset=np.array([
    [1,0,0,offset_x],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1],
])

fresa_robot_offset= fresa_robot @ matriz_offset

#Targets
home_target = RDK.Item('Home', ITEM_TYPE_TARGET)# type: ignore
foto_target = RDK.Item('Foto', ITEM_TYPE_TARGET)# type: ignore
T3_target = RDK.Item('Target 3', ITEM_TYPE_TARGET)# type: ignore
T4_target = RDK.Item('Target 4', ITEM_TYPE_TARGET)# type: ignore
T5_target = RDK.Item('Target 5', ITEM_TYPE_TARGET)# type: ignore
Debajo_fresa=Mat(fresa_robot_offset.tolist())
Fresa=Mat(fresa_robot.tolist())

#Rutina
while True:
    if keyboard.is_pressed("esc") and actuador_encendido==False:
        print("Ciclo Terminado por usuario")
        item.MoveJ(home_target)
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

