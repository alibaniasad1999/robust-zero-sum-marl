"""Launch file for the RZSM robust controller node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('rzsm_control')
    default_params = os.path.join(pkg_share, 'config', 'controller_params.yaml')

    return LaunchDescription([
        # ── Arguments ─────────────────────────────────────────────
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Path to controller parameters YAML',
        ),
        DeclareLaunchArgument(
            'namespace',
            default_value='',
            description='Node namespace (for multi-robot)',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock (for Isaac Sim / Gazebo)',
        ),

        # ── Controller node ───────────────────────────────────────
        Node(
            package='rzsm_control',
            executable='robust_controller_node',
            name='robust_controller_node',
            namespace=LaunchConfiguration('namespace'),
            parameters=[
                LaunchConfiguration('params_file'),
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
            ],
            remappings=[
                ('joint_states', 'joint_states'),
                ('imu', 'imu'),
                ('cmd_vel', 'cmd_vel'),
            ],
            output='screen',
            emulate_tty=True,
        ),
    ])
