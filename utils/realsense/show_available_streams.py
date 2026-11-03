import pyrealsense2 as rs
pipe = rs.pipeline()
profile = pipe.start()
try:
    for i in range(0, 100):
        frames = pipe.wait_for_frames()
        for f in frames:
            print(f.profile)
finally:
    pipe.stop()
# Streams found:

# <pyrealsense2.[video_]stream_profile: Depth(0) 640x480 @ 15fps Z16>
# <pyrealsense2.[video_]stream_profile: Color(0) 640x480 @ 15fps RGB8>
# <pyrealsense2.stream_profile: Accel(0) @ 100fps MOTION_XYZ32F>
# <pyrealsense2.stream_profile: Gyro(0) @ 200fps MOTION_XYZ32F>