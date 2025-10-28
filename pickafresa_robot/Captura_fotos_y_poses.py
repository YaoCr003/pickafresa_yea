import os
import json
import cv2
import time
import keyboard
from robolink import *  # API RoboDK
from robodk import *

# ---------- CONFIGURACIÓN ----------
CARPETA_SALIDA = "C:/calibracion_eye_in_hand"  # 📁 cambia la ruta
NOMBRE_ROBOT = "UR3e"  # nombre exacto del robot en RoboDK
CAMERA_INDEX = 0  # índice de cámara (0 por defecto)

# Crear carpeta si no existe
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# Conectar con RoboDK
RDK = Robolink()
RDK.setRunMode(RUNMODE_SIMULATE)  # o RUNMODE_RUN_ROBOT si es físico
robot = RDK.Item(NOMBRE_ROBOT, ITEM_TYPE_ROBOT)

if not robot.Valid():
    raise Exception("No se encontró el robot en RoboDK.")

# Inicializar cámara física
cam = cv2.VideoCapture(CAMERA_INDEX)
if not cam.isOpened():
    raise Exception("No se pudo acceder a la cámara física.")

print("Sistema listo.")
print("Presiona ENTER para capturar una foto y configuración.")
print("Presiona ESC para salir.\n")

contador = 1
while True:
    # Finaliza si se presiona ESC
    if keyboard.is_pressed("esc"):
        print("\n Calibración terminada por el usuario.")
        break

    # Esperar a que se presione Enter
    if keyboard.is_pressed("enter"):
        time.sleep(0.3)  # evitar lecturas múltiples del mismo enter

        # Obtener configuración de articulaciones
        joints = robot.Joints().list()
        joint_data = {
            "id": contador,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "joints_deg": [round(float(j), 4) for j in joints],
        }

        # Guardar JSON
        json_path = os.path.join(CARPETA_SALIDA, f"{contador}.json")
        with open(json_path, "w") as f:
            json.dump(joint_data, f, indent=4)

        # Capturar imagen
        ret, frame = cam.read()
        if not ret:
            print("⚠️ No se pudo capturar imagen.")
            continue

        foto_path = os.path.join(CARPETA_SALIDA, f"{contador}.jpg")
        cv2.imwrite(foto_path, frame)

        print(f"Captura {contador} guardada ({foto_path}, {json_path})")
        contador += 1

# Liberar cámara
cam.release()
print("✅ Programa finalizado. Archivos guardados en:", CARPETA_SALIDA)

