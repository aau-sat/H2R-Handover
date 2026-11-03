from setuptools import find_packages, setup

package_name = 'handover_detection'

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
			'test_rgb = handover_detection.test_rgb_topic:main',
			'test_yolo_rgb = handover_detection.test_yolo_rgb_topic:main',
			'test_yolo_rgb_with_params = handover_detection.test_yolo_rgb_topic_with_params:main',
			'handover_detector_yolo = handover_detection.handover_detector_yolo:main',
			'handover_detector = handover_detection.handover_detector:main',
		],
	},
)
