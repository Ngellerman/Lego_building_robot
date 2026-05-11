from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # -------------------------
    # Declare args
    # -------------------------

    ur_type = LaunchConfiguration("ur_type", default="ur7e")
    launch_rviz = LaunchConfiguration("launch_rviz", default="true") # make false if you don't want rviz to launch when launching moveit

    # -------------------------
    # Includes & Nodes
    # -------------------------
    # RealSense (include rs_launch.py)
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ),
        launch_arguments={
            'pointcloud.enable': 'true',
            'align_depth.enable': 'true',
            'rgb_camera.color_profile': '640x480x30',
        }.items(),
    )

    # Perception node
    perception_node = Node(
        package='perception',
        executable='process_pointcloud',
        name='process_pointcloud',
        output='screen'
    )

    # Planning TF node
    planning_tf_node = Node(
        package='planning',
        executable='tf',
        name='tf_node',
        output='screen'
    )

    # MoveIt include
    moveit_launch_file = os.path.join(
        get_package_share_directory("ur_moveit_config"),
        "launch",
        "ur_moveit.launch.py"
    )
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(moveit_launch_file),
        launch_arguments={
            "ur_type": ur_type,
            "launch_rviz": launch_rviz
        }.items(),
    )

    ik_planner_node = Node(
        package='planning',
        executable='ik',
        name='ik_node',
        output='screen'
    )

    # -------------------------
    # LaunchDescription
    # -------------------------
    return LaunchDescription([
        realsense_launch,
        planning_tf_node,
        moveit_launch,
        ik_planner_node,
        perception_node,
    ])