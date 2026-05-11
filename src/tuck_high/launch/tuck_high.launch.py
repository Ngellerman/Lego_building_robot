from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    ur_type = LaunchConfiguration('ur_type', default='ur7e')

    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ur_moveit_config'),
                'launch',
                'ur_moveit.launch.py'
            )
        ),
        launch_arguments={
            'ur_type': ur_type,
            'launch_rviz': 'false',
        }.items(),
    )

    # Delay tuck node so MoveIt has time to advertise /plan_kinematic_path
    tuck_node = Node(
        package='tuck_high',
        executable='tuck_high',
        name='tuck_high',
        output='screen',
    )

    return LaunchDescription([
        moveit_launch,
        TimerAction(period=10.0, actions=[tuck_node]),
        RegisterEventHandler(
            OnProcessExit(
                target_action=tuck_node,
                on_exit=[Shutdown()],
            )
        ),
    ])
