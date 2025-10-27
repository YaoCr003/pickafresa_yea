from robolink import *  # API RoboDK
from robodk import *
import keyboard

RDK = Robolink()
RDK.setRunMode(RUNMODE_SIMULATE)
#RDK.setRunMode(RUNMODE_RUN_ROBOT)

actuador_encendido=False

def actuador_on():
    global actuador_encendido
    actuador_encendido=True
    print("Actuador encendido")

def actuador_off():
    global actuador_encendido
    actuador_encendido=False
    print("Actuador apagado")


home_target = RDK.Item('Home', ITEM_TYPE_TARGET)# type: ignore
foto_target = RDK.Item('Foto', ITEM_TYPE_TARGET)# type: ignore
T3_target = RDK.Item('Target 3', ITEM_TYPE_TARGET)# type: ignore
T4_target = RDK.Item('Target 4', ITEM_TYPE_TARGET)# type: ignore
T5_target = RDK.Item('Target 5', ITEM_TYPE_TARGET)# type: ignore

item = RDK.ItemUserPick('UR3e', ITEM_TYPE_ROBOT)# type: ignore
item.setSpeed(60, 60) 

#item.Connect()  # Intenta conexión activa (opcional)

while True:
    if keyboard.is_pressed("esc") and not actuador_encendido:
        item.setSpeed(80, 80) 
        item.MoveJ(home_target)
        break
    if item.Valid():
        item.setSpeed(80, 80) 
        print('Conectado correctamente con RoboDK.')
        item.MoveJ(home_target)
        item.MoveJ(foto_target)
        item.MoveJ(T5_target)
        time.sleep(2.0)
        actuador_on()
        time.sleep(2.0)
        item.MoveJ(T3_target)
        item.MoveL(T4_target)
        time.sleep(2.0)
        actuador_off()
        time.sleep(2.0)
        item.MoveL(T3_target)
        item.MoveJ(home_target)
    else:
        print('No se pudo conectar con RoboDK.')
