from robolink import *  # API RoboDK
from robodk import *
import paho.mqtt.client as mqtt
import numpy as np
import time


#MQTT
broker="192.168.9.1"
topic="actuador"

client = mqtt.Client()
client.connect(broker, 1883, 60)

#Encender o apagar actuador
def actuador_on():
    client.publish(topic, "1")
    print("Actuador encendido")

def actuador_off():
    client.publish(topic, "0")
    print("Actuador apagado")



#Estado de robot: simulacion o fisico
RDK = Robolink()
#RDK.setRunMode(RUNMODE_RUN_ROBOT)
RDK.setRunMode(RUNMODE_SIMULATE)

#Targets
home_target = RDK.Item('Home', ITEM_TYPE_TARGET)# type: ignore
foto_target = RDK.Item('Foto', ITEM_TYPE_TARGET)# type: ignore
T3_target = RDK.Item('Target 3', ITEM_TYPE_TARGET)# type: ignore
T4_target = RDK.Item('Target 4', ITEM_TYPE_TARGET)# type: ignore
T5_target = RDK.Item('Target 5', ITEM_TYPE_TARGET)# type: ignore

#Definir robot
item = RDK.ItemUserPick('UR3e', ITEM_TYPE_ROBOT)# type: ignore
item.setSpeed(60, 60) 
#item.Connect()  # Intenta conexión activa (opcional)


#Posición fresa respecto a camara
fresa_cam=np.array()

#Matriz de transformacion Cámara-Base robot
cam_robot=np.array([
    [0,0,0,0]
    [0,0,0,0]
    [0,0,0,0]
    [0,0,0,0]
])

#Fresa en coordenadas del robot
fresa_robot=cam_robot*fresa_cam

#Rutina
item.MoveJ(home_target)
item.MoveJ(foto_target)
item.MoveJ(T5_target)
item.MoveJ(T3_target)
item.MoveL(T4_target)
item.MoveL(T3_target)
item.MoveJ(home_target)
