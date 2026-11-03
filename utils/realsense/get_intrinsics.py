import pyrealsense2 as rs
import yaml

# Configure to match your target resolution
WIDTH = 640
HEIGHT = 480
FPS = 30

pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
cfg.enable_stream(rs.stream.infrared, 1, WIDTH, HEIGHT, rs.format.y8, FPS)  # Optional: enable infrared stream if needed
cfg.enable_stream(rs.stream.infrared, 2, WIDTH, HEIGHT, rs.format.y8, FPS)

profile = pipeline.start(cfg)
try:
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    ir_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    ir_profile2 = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()

    c = color_profile.get_intrinsics()
    d = depth_profile.get_intrinsics()
    ir = ir_profile.get_intrinsics()
    ir2 = ir_profile2.get_intrinsics()
    # get depth scale
    print('Depth Scale is: ', profile.get_device().first_depth_sensor().get_depth_scale())

    extr = depth_profile.get_extrinsics_to(color_profile)
    extr_ir = ir_profile.get_extrinsics_to(color_profile)
    extr_ir2 = ir_profile2.get_extrinsics_to(color_profile)

    out = {
        'color': {'fx': float(c.fx), 'fy': float(c.fy), 'ppx': float(c.ppx), 'ppy': float(c.ppy)},
        'depth': {'fx': float(d.fx), 'fy': float(d.fy), 'ppx': float(d.ppx), 'ppy': float(d.ppy)},
        'infrared': {'fx': float(ir.fx), 'fy': float(ir.fy), 'ppx': float(ir.ppx), 'ppy': float(ir.ppy)},
        'infrared2': {'fx': float(ir2.fx), 'fy': float(ir2.fy), 'ppx': float(ir2.ppx), 'ppy': float(ir2.ppy)},
        'extrinsics': tuple(list(extr.rotation) + list(extr.translation)),  # 12 floats: 9 rotation, 3 translation,
        'extrinsics_ir': tuple(list(extr_ir.rotation) + list(extr_ir.translation)),  # 12 floats: 9 rotation, 3 translation,
        'extrinsics_ir2': tuple(list(extr_ir2.rotation) + list(extr_ir2.translation))  # 12 floats: 9 rotation, 3 translation,
    }

    with open('realsense_intrinsics.yml', 'w') as f:
        yaml.dump(out, f, default_flow_style=False)
    print('Wrote realsense_intrinsics.yml')
finally:
    pipeline.stop()