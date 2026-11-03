from setuptools import find_packages, setup

package_name = 'visual_properties_detection'

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
            'corsmal_filling_level_predictor = visual_properties_detection.corsmal_filling_level_predictor:main',
            'yolo_filling_level_predictor = visual_properties_detection.yolo_filling_level_predictor:main',
            'gemma_filling_level_predictor = visual_properties_detection.gemma_filling_level_predictor:main'
            
        ],
    },
)
