import numpy as np

data = np.load("camera_intrinsics/camera_calibration.npz")

print(data.files)
print("\nIntrinsic Matrix (K):")
print(data["K"])