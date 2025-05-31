from setuptools import find_packages, setup
import os
from glob import glob
import sys

package_name = 'bittle_ros2'

setup(
    name=package_name,
    version='0.0.1',  # Bumped version for update
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.py')),
        (os.path.join('share', package_name), glob('bittle_ros2/*.py')),
    ],
    install_requires=['setuptools'],
    extras_require={
        'test': ['pytest'],
    },
    options={
        'build_scripts': {
            'executable': sys.executable,
        }
    },
    zip_safe=True,
    maintainer='reid',
    maintainer_email='rgraves@andrew.cmu.edu',
    description='Bittle driver for ROS2. This version works with updated messages including an aspect_ratios field in the Yolo message.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'bittle_driver = bittle_ros2.bittle_driver:main',
            'video_subscriber = bittle_ros2.video_subscriber:main',
            'joystick_driver = bittle_ros2.joystick_driver:main',
            'image_save_subscriber = bittle_ros2.image_save_subscriber:main',
            'demo_driver = bittle_ros2.demo_driver:main',
            'serial_sender = bittle_ros2.serial_sender:main',
            'serial_sender2 = bittle_ros2.serial_sender2:main',
            'send_manual_cmd = bittle_ros2.send_manual_cmd:main',
            'webvid_publisher = bittle_ros2.webvid_publisher:main',
            'webvid_subscriber = bittle_ros2.webvid_subscriber:main',
            'yolo_node = bittle_ros2.yolo_node:main',
            'apriltag_node = bittle_ros2.apriltag_node:main',
            'mapping_node = bittle_ros2.mapping_node:main',
            'move_to_point_node = bittle_ros2.move_to_point_node:main',
            'send_manual_goal = bittle_ros2.send_manual_goal:main',
            'robot_to_grid_node = bittle_ros2.robot_to_grid_node:main',
            'occupancy_grid_publisher = bittle_ros2.occupancy_grid_publisher:main',
            'path_planner = bittle_ros2.path_planner:main',
            'LLM_reasoning_node = bittle_ros2.LLM_reasoning_node:main',
            'bittle_command_executor = bittle_ros2.bittle_command_executor:main',
        ],
    },
)

