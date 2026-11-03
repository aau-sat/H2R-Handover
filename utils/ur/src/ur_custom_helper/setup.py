from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ur_custom_helper'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/obstacles', glob('obstacles/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='caverntwo',
    maintainer_email='caverntwo@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'add_obstacle_node = ur_custom_helper.add_obstacle_node:main',
        ],
    },
)
