# generate_aruco_board.py
import cv2
import numpy as np

# ---- Board parameters (MUST match detector) ----
MARKERS_X = 3
MARKERS_Y = 2
MARKER_LENGTH = 0.064      # meters
MARKER_SEPARATION = 0.02  # meters
DPI = 300

# ---- ArUco dictionary ----
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

board = cv2.aruco.GridBoard_create(
    markersX=MARKERS_X,
    markersY=MARKERS_Y,
    markerLength=MARKER_LENGTH,
    markerSeparation=MARKER_SEPARATION,
    dictionary=aruco_dict
)

# Convert meters → pixels for printing
board_width_m = MARKERS_X * MARKER_LENGTH + (MARKERS_X - 1) * MARKER_SEPARATION
board_height_m = MARKERS_Y * MARKER_LENGTH + (MARKERS_Y - 1) * MARKER_SEPARATION

width_px = int(board_width_m * DPI / 0.0254)
height_px = int(board_height_m * DPI / 0.0254)

img = board.draw((width_px, height_px))

cv2.imwrite("aruco_gridboard.png", img)
print("Saved aruco_gridboard.png")
