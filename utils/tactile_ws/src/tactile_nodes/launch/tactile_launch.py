from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    alfredo_pub = Node(
        package='tactile_nodes',
        executable='taxel_publisher',
        name='taxel_publisher_giovanni',
        output='screen',
        parameters=[{
            'ble_address': 'D9:E9:6D:70:68:D0',
            'device_uid': 'GIOVANNI',
        }],
    )

    alfredo_detector = Node(
        package='tactile_nodes',
        executable='taxel_contact_detector',
        name='taxel_contact_detector_giovanni',
        output='screen',
        parameters=[{
            'device_uid': 'GIOVANNI',
        }],
    )

    giovanni_pub = Node(
        package='tactile_nodes',
        executable='taxel_publisher',
        name='taxel_publisher_alfredo',
        output='screen',
        parameters=[{
            'ble_address': 'FD:7A:24:4C:0B:36',
            'device_uid': 'ALFREDO',
        }],
    )

    giovanni_detector = Node(
        package='tactile_nodes',
        executable='taxel_contact_detector',
        name='taxel_contact_detector_alfredo',
        output='screen',
        parameters=[{
            'device_uid': 'ALFREDO',
        }],
    )

    return LaunchDescription([
        alfredo_pub,
        alfredo_detector,
        TimerAction(
            period=5.0,
            actions=[
                giovanni_pub,
                giovanni_detector,
            ],
        ),
    ])
