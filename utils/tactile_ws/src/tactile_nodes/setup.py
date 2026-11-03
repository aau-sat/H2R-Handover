from setuptools import find_packages, setup

package_name = 'tactile_nodes'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', ['launch/tactile_launch.py', 'launch/tactile_sensor_launch.py']),
    ],
    install_requires=['setuptools', 'bleak', 'tactile_detection'],
    zip_safe=True,
    maintainer='nad',
    maintainer_email='nad@example.com',
    description='ROS 2 tactile publisher and contact detector nodes',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'taxel_publisher = tactile_nodes.taxel_publisher:main',
            'taxel_contact_detector = tactile_nodes.taxel_contact_detector:main',
            'tactile_sensor_node = tactile_nodes.tactile_sensor_node_async:main',
            'tactile_sensor_bridge_node = tactile_nodes.tactile_sensor_node_bridge:main',
        ],
    },
)
