"""World + brain + RViz.

  ros2 launch followme_ros sim.launch.py scenario:=crossing plant:=hardware sensor:=sonar3
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

ARGS = [
    ("scenario", "obstacles", "loop | obstacles | crossing | pillar"),
    ("plant", "hardware", "hardware (kick, delay, duty floors) | ideal"),
    ("profile", "hardware", "brain tuning: hardware | ideal. Match the plant."),
    ("sensor", "sonar3", "sonar3 | lidar | none (camera only)"),
    ("seed", "0", "noise seed"),
    ("auto_arm", "true", "start armed"),
    ("loop", "true", "restart the scenario when it ends"),
    ("use_rviz", "true", "open RViz"),
]


def generate_launch_description():
    c = LaunchConfiguration
    rviz_cfg = PathJoinSubstitution([FindPackageShare("followme_ros"), "rviz", "followme.rviz"])
    return LaunchDescription([
        *[DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in ARGS],
        Node(package="followme_ros", executable="world", name="world", output="screen",
             parameters=[{
                 "scenario": c("scenario"), "plant": c("plant"), "sensor": c("sensor"),
                 "seed": ParameterValue(c("seed"), value_type=int),
                 "loop": ParameterValue(c("loop"), value_type=bool),
             }]),
        Node(package="followme_ros", executable="brain", name="brain", output="screen",
             parameters=[{
                 "profile": c("profile"),
                 "auto_arm": ParameterValue(c("auto_arm"), value_type=bool),
             }]),
        Node(package="rviz2", executable="rviz2", name="rviz2", output="screen",
             arguments=["-d", rviz_cfg], condition=IfCondition(c("use_rviz"))),
    ])
