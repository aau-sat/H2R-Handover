from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():

    alfredo_detector = Node(
        package='tactile_nodes',
        executable='tactile_sensor_node',
        name='tactile_sensor_node_alfredo',
        output='screen',
        parameters=[{
            'device_uid': 'ALFREDO',
            'ble_address': 'FD:7A:24:4C:0B:36', #'D9:E9:6D:70:68:D0',
            'contact_detection_strategy': 'threshold',
            'adapter': "hci0",
            # 'contact_sensor_count': 3,
            'std_dev_high_multipliers': [27.5, 27.5, 27.5, 27.5, 27.5, 27.5, 27.5],
            'std_dev_low_multipliers': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }],
    )
    # giovanni_detector = Node(
    #     package='tactile_nodes',
    #     executable='tactile_sensor_node',
    #     name='tactile_sensor_node_giovanni',
    #     output='screen',
    #     parameters=[{
    #         'device_uid': 'GIOVANNI',
    #         'ble_address': 'D9:E9:6D:70:68:D0',
    #         #'ble_address': 'FD:7A:24:4C:0B:36',
    #         'contact_detection_strategy': 'threshold',
    #         'adapter': "hci1",
    #         'contact_sensor_count': 1,
    #         'std_dev_high_multipliers': [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
    #         'std_dev_low_multipliers': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    #     }],
    # )
    giovanni_detector = Node(
        package='tactile_nodes',
        executable='tactile_sensor_bridge_node',
        name='tactile_sensor_node_giovanni',
        output='screen',
        parameters=[{
            'device_uid': 'GIOVANNI',
            'ble_address': 'D9:E9:6D:70:68:D0',
            #'ble_address': 'FD:7A:24:4C:0B:36',
            'contact_detection_strategy': 'threshold',
            'adapter': "hci1",
            'contact_sensor_count': 1,
            'std_dev_high_multipliers': [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
            'std_dev_low_multipliers': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            # 'ignore_sensors': [0],
        }],
    )

    return LaunchDescription([
        alfredo_detector,
        #giovanni_detector,
        # TimerAction(
        #    period=1.0,
        #    actions=[
        #        giovanni_detector,
        #    ],
        # ),
    ])
