import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 1. Setup Pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 2. Start Pipeline
try:
    profile = pipeline.start(config)
except Exception as e:
    print(f"Could not start pipeline: {e}")
    exit()

# 3. Setup Tools
align = rs.align(rs.stream.color)
ply_exporter = rs.save_to_ply("init.ply") # Initial dummy name

# --- CONFIGURE ASCII MODE ---
# 0 = ASCII, 1 = Binary
ply_exporter.set_option(rs.save_to_ply.option_ply_binary, 0) 
# We don't want a mesh (triangles), just the points
ply_exporter.set_option(rs.save_to_ply.option_ply_mesh, 0)   

# 4. Create Folders
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(script_dir, "out1")
for folder in ["rgb", "depth", "ply"]:
    path = os.path.join(base_path, folder)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

print(f"--- RECORDING TO: {base_path} ---")

index = 0
try:
    while index < 20: # Start with a small number to test
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Define file paths
        path_rgb = os.path.join(base_path, "rgb", f"{index}.png")
        path_depth = os.path.join(base_path, "depth", f"{index}.png")
        path_ply = os.path.join(base_path, "ply", f"{index}.ply")

        # 1. Save PNGs
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imwrite(path_rgb, color_image)
        cv2.imwrite(path_depth, np.asanyarray(depth_frame.get_data()))

        # 2. Save ASCII PLY
        # We tell the exporter the specific filename for THIS frame
        # IMPORTANT: RealSense requires a string for the filename option
        # Note: In some versions of SDK, this option is used during the .process() call
        print(f"Exporting: {path_ply}")
        
        # We use the built-in export function to ensure the file is written correctly
        # This is often more reliable than the processing block for specific filenames
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        
        # Unfortunately points.export_to_ply is binary only. 
        # To get ASCII, we MUST use the processing block method:
        # We set the 'out.ply' path and then move it immediately.
        ply_exporter.process(aligned_frames)
        
        # Move the default output to our desired folder
        if os.path.exists("out.ply"):
            os.replace("out.ply", path_ply)
        else:
            # Some SDK versions use the name provided in the constructor
            if os.path.exists("init.ply"):
                os.replace("init.ply", path_ply)

        print(f"Successfully stored frame {index}")
        index += 1

        cv2.imshow("Check Window", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Done.")