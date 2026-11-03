import cv2
import numpy as np
import glob
import os

# ========================
# USER SETTINGS
# ========================
IMAGE_DIR = "calib_images"
COLS = 5        # inner corners per row
ROWS = 7        # inner corners per column
SQUARE_SIZE = 22.0  # mm (or any unit, just be consistent)

# ========================
print("\n=== Camera Calibration ===\n")

images = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
if len(images) == 0:
    print("❌ No images found")
    exit()

print(f"Found {len(images)} images")

# Prepare object points
objp = np.zeros((ROWS * COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

valid_images = 0

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (COLS, ROWS))

    if not ret:
        print(f"⚠ Chessboard not found: {fname}")
        continue

    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), criteria
    )

    objpoints.append(objp)
    imgpoints.append(corners)
    valid_images += 1

    cv2.drawChessboardCorners(img, (COLS, ROWS), corners, ret)
    cv2.imshow("Detected Corners", img)
    cv2.waitKey(200)

cv2.destroyAllWindows()

if valid_images < 5:
    print("❌ Not enough valid images for calibration")
    exit()

print(f"\nUsing {valid_images} valid images")

# Calibrate
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\nIntrinsic Matrix (K):")
print(K)

print("\nDistortion Coefficients:")
print(dist.ravel())

# Reprojection error
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], K, dist
    )
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
    total_error += error / len(imgpoints2)

mean_error = total_error / len(objpoints)
print(f"\nMean reprojection error: {mean_error:.4f} pixels")

# Save calibration
np.savez(
    "camera_calibration.npz",
    K=K,
    dist=dist,
    image_size=gray.shape[::-1]
)

print("\n💾 Calibration saved to camera_calibration.npz")
print("\nDone 🎉")
