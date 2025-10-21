import cv2
import numpy as np
import glob

# Cargar calibración de cámara
with np.load("calibracion_cam.npz") as X:
    mtx, dist = [X[i] for i in ('mtx','dist')]

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
parameters = cv2.aruco.DetectorParameters_create()

R_target2cam = []
t_target2cam = []

for img_name in sorted(glob.glob("captura_*.png")):
    frame = cv2.imread(img_name)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.03, mtx, dist)
        R, _ = cv2.Rodrigues(rvec[0])
        R_target2cam.append(R)
        t_target2cam.append(tvec[0].reshape(3))
        print(f"Marcador detectado en {img_name}")
    else:
        print(f"⚠️ No se detectó marcador en {img_name}")

np.savez("detecciones_marcador.npz", R_target2cam=R_target2cam, t_target2cam=t_target2cam)
print("Detecciones guardadas en detecciones_marcador.npz")
