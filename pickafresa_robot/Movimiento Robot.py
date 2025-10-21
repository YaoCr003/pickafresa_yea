from robolink import *  # API RoboDK
RDK = Robolink()
RDK.setRunMode(RUNMODE_RUN_ROBOT)


home_target = RDK.Item('Home', ITEM_TYPE_TARGET)
foto_target = RDK.Item('Foto', ITEM_TYPE_TARGET)
T5_target = RDK.Item('Target 5', ITEM_TYPE_TARGET)

item = RDK.ItemUserPick('UR3e', ITEM_TYPE_ROBOT)

item.Connect()  # Intenta conexión activa (opcional)

item.MoveJ(foto_target)
item.MoveJ(T5_target)
item.MoveJ(home_target)
