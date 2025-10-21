from robolink import *    # API RoboDK
from robodk import *
import cv2
import numpy as np

RDK = Robolink()
robot = RDK.Item('UR3e')  # cambia el nombre según tu robot en RoboDK
cap = cv2.VideoCapture(0)

poses_robot = []
i = 0

print("Presiona 'c' para capturar imagen y pose, 'ESC' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Camara", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        pose = robot.Pose().Pose()
        poses_robot.append(pose)
        cv2.imwrite(f"captura_{i}.png", frame)
        print(f"Captura {i} guardada.")
        i += 1

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

np.save("poses_robot.npy", poses_robot)
print("Poses guardadas en poses_robot.npy")
