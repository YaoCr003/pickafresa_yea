from robolink import *  # API RoboDK
from robodk import *
RDK = Robolink()
RDK.setRunMode(RUNMODE_SIMULATE)
RDK.setRunMode(RUNMODE_RUN_ROBOT)


home_target = RDK.Item('Home', ITEM_TYPE_TARGET) # type: ignore

item = RDK.ItemUserPick('UR3e', ITEM_TYPE_ROBOT)# type: ignore

item.Connect()  # Intenta conexión activa (opcional)

item.setSpeed(30, 30) 
if item.Valid():
    print('Conectado correctamente con RoboDK.')
    item.MoveJ(home_target, blocking=False)
else:
    print('No se pudo conectar con RoboDK.')
