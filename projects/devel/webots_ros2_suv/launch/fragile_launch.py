#!/usr/bin/env python3
import os
import pathlib
import launch
import yaml
import xacro
from launch.substitutions import Command, LaunchConfiguration
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions.path_join_substitution import PathJoinSubstitution
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from webots_ros2_driver.webots_launcher import WebotsLauncher, Ros2SupervisorLauncher
from webots_ros2_driver.webots_controller import WebotsController
from nav2_common.launch import RewrittenYaml

PACKAGE_NAME = "webots_ros2_suv"
USE_SIM_TIME = True

def generate_launch_description():
    package_dir = get_package_share_directory(PACKAGE_NAME)
    world = LaunchConfiguration("world")

    pointcloud_to_laserscan = Node(
            package="pointcloud_to_laserscan",
            executable="pointcloud_to_laserscan_node",
            remappings=[("cloud_in", ["/lidar"]), ("scan", ["/scan"])],
            parameters=[
                {
                    #"target_frame": "map",
                    "transform_tolerance": 0.01,
                    "min_height": -2.3,
                    "max_height": 0.0,
                    "angle_min": -1.5708,  # -M_PI/2
                    "angle_max": 1.5708,  # M_PI/2
                    "angle_increment": 0.0087,  # M_PI/360.0
                    "scan_time": 0.3333,
                    "range_min": 0.45,
                    "range_max": 15.0,
                    "use_inf": True,
                    "inf_epsilon": 1.0,
                }
            ],
            name="pointcloud_to_laserscan",
        )

    rviz = Node(package="rviz2", executable="rviz2", output="screen")

    
    return LaunchDescription(
        [
           rviz,
           pointcloud_to_laserscan
        ]
    )
