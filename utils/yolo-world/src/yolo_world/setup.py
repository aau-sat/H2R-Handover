from setuptools import find_packages, setup

package_name = 'yolo_world'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='caverntwo',
    maintainer_email='caverntwo@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolo_world_node = yolo_world.yolo_world_node:main',
            'yolo_world_roboflow_node = yolo_world.yolo_world_roboflow_node:main',
            'yolo_node = yolo_world.yolo_node:main',
            'boxmot_node = yolo_world.boxmot_node:main',
            'depth_clipped_node = yolo_world.depth_clipped_node:main',
            'double_tracker_yolo_node = yolo_world.double_tracker_yolo_node:main',
            'yoloe_tracker_dual=yolo_world.yoloe_tracker_dual:main',
            'yoloe_tracker_dual_depth=yolo_world.yoloe_tracker_dual_depth:main',
            'yoloe_tracker_single=yolo_world.yoloe_tracker_single:main',
            'yoloe_segmentation_tracker=yolo_world.yoloe_segmentation_tracker:main',
            'rfdetr_tracker_dual=yolo_world.rfdetr_tracker_dual:main',
            'yoloerfdetr_tracker_dual=yolo_world.yoloe_rfdetr_tracker_dual:main',
        ],
    },
)
