from robolink import *  # API RoboDK
from robodk import *
import paho.mqtt.client as mqtt
import numpy as np
import time
import keyboard

#Variables estado inicial
actuador_encendido=False

#MQTT
broker="192.168.9.1"
topic="actuador"

client = mqtt.Client()
client.connect(broker, 1883, 60)

#Encender o apagar actuador
def actuador_on():
    client.publish(topic, "1")
    global actuador_encendido
    actuador_encendido=True
    print("Actuador encendido")

def actuador_off():
    client.publish(topic, "0")
    global actuador_encendido
    actuador_encendido=False
    print("Actuador apagado")



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
        print("Ciclo Terminado")
        actuador_off()
        item.MoveJ(home_target)
        break

    item.MoveJ(home_target)
    item.MoveJ(foto_target)
    item.MoveJ(T5_target)
    item.MoveJ(Debajo_fresa)
    item.MoveL(Fresa)
    time.sleep(2.0)
    actuador_on()
    time.sleep(2.0)
    item.MoveL(Debajo_fresa)
    item.MoveJ(T5_target)
    item.MoveJ(T3_target)
    item.MoveL(T4_target)
    time.sleep(2.0)
    actuador_off()
    time.sleep(2.0)
    item.MoveL(T3_target)
    item.MoveJ(home_target)
    time.sleep(2.0)
