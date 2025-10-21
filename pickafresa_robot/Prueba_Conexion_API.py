from robolink import *  # API RoboDK
RDK = Robolink()
home_target = RDK.Item('Home', ITEM_TYPE_TARGET)

item = RDK.ItemUserPick('UR3e', ITEM_TYPE_ROBOT)
if item.Valid():
    print('Conectado correctamente con RoboDK.')
    item.MoveJ(home_target)
else:
    print('No se pudo conectar con RoboDK.')
