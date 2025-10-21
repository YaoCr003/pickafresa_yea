import cv2
import numpy as np

poses_robot = np.load("poses_robot.npy", allow_pickle=True)
data = np.load("detecciones_marcador.npz", allow_pickle=True)

R_target2cam = data["R_target2cam"]
t_target2cam = data["t_target2cam"]

R_gripper2base = [pose[:3, :3] for pose in poses_robot]
t_gripper2base = [pose[:3, 3] for pose in poses_robot]

R_tool_cam, t_tool_cam = cv2.calibrateHandEye(
    R_gripper2base, t_gripper2base,
    R_target2cam, t_target2cam,
    method=cv2.CALIB_HAND_EYE_TSAI
)

T_tool_cam = np.eye(4)
T_tool_cam[:3, :3] = R_tool_cam
T_tool_cam[:3, 3] = t_tool_cam.reshape(3)

print("=== MATRIZ T_tool_cam (Cámara respecto a herramienta) ===")
print(T_tool_cam)

np.save("T_tool_cam.npy", T_tool_cam)
print("Guardado en T_tool_cam.npy")
