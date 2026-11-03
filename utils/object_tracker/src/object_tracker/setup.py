from setuptools import find_packages, setup

package_name = 'object_tracker'

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
    maintainer_email='caverntwo@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # 'object_tracker_node = object_tracker.object_tracker_node:main',
            'depth_clipper_node=object_tracker.depth_clipper_node:main',
            # 'hand_clipper_node=object_tracker.hand_clipper_node:main',
            # 'hand_tracker_node=object_tracker.hand_tracker_node:main',
            # 'hand_tracker_node_depth=object_tracker.hand_tracker_node_depth:main',
            'hand_tracker_node_new=object_tracker.hand_tracker_node_new:main',
            # 'hand_tracker_node_finger=object_tracker.hand_tracker_node_finger:main',
            # 'test_hand_tracker_viewer=object_tracker.test_hand_tracker_viewer:main',
            # 'hand_pose_tracker_node=object_tracker.hand_pose_from_tracker_node:main',
        ],
    },
)
