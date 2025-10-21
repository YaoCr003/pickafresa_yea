import cv2
import numpy as np
import glob

# Parámetros del patrón de calibración (chessboard)
pattern_size = (9, 6)  # 9x6 esquinas interiores
square_size = 0.025  # 25 mm cada cuadro

# Prepara puntos del mundo real (0,0,0), (1,0,0), ...
objp = np.zeros((np.prod(pattern_size), 3), np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

images = glob.glob("calib_*.png")  # fotos de tablero

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("Matriz intrínseca:\n", mtx)
print("Coeficientes de distorsión:\n", dist)

np.savez("calibracion_cam.npz", mtx=mtx, dist=dist)
print("Calibración guardada como calibracion_cam.npz")
