from robolink import *  # API RoboDK
from robodk import *
RDK = Robolink()
RDK.setRunMode(RUNMODE_SIMULATE)
#RDK.setRunMode(RUNMODE_RUN_ROBOT)


home_target = RDK.Item('Home', ITEM_TYPE_TARGET)# type: ignore
foto_target = RDK.Item('Foto', ITEM_TYPE_TARGET)# type: ignore
T3_target = RDK.Item('Target 3', ITEM_TYPE_TARGET)# type: ignore
T4_target = RDK.Item('Target 4', ITEM_TYPE_TARGET)# type: ignore
T5_target = RDK.Item('Target 5', ITEM_TYPE_TARGET)# type: ignore

item = RDK.ItemUserPick('UR3e', ITEM_TYPE_ROBOT)# type: ignore
item.setSpeed(60, 60) 

#item.Connect()  # Intenta conexión activa (opcional)

item.setSpeed(30, 30) 
if item.Valid():
    print('Conectado correctamente con RoboDK.')
    item.MoveJ(home_target, blocking=False)
    item.MoveJ(foto_target)
    item.MoveJ(T5_target)
    item.MoveJ(T3_target)
    item.MoveL(T4_target)
    item.MoveL(T3_target)
    item.MoveJ(home_target)
else:
    print('No se pudo conectar con RoboDK.')
