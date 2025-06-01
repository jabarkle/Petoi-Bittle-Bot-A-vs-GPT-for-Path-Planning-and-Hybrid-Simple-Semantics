# Petoi-Bittle-Bot-A-vs-GPT-for-Path-Planning-and-Hybrid-Simple-Semantics
**Semantic Intelligence: GPT-4 + A* Path Planning for Low-Cost Robotics**  
This repository implements a hybrid navigation system that combines GPT-4’s semantic reasoning with classical A* path planning on a low-cost Petoi Bittle quadruped robot. The system enables context-aware navigation through natural-language instructions without hard-coded finite-state machines.

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Hardware Requirements](#hardware-requirements)
- [Software Dependencies](#software-dependencies)
- [Installation & Setup](#installation--setup)
- [Running Experiments](#running-experiments)
- [Evaluation Scenarios](#evaluation-scenarios)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [File Structure](#file-structure)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Related Work](#related-work)

## Overview
Traditional robot navigation relies on hard-coded state machines and purely geometric planners, limiting adaptability to semantic instructions. This research demonstrates how GPT-4 can interpret high-level commands (e.g., “avoid toxic spills”, “find charging station”) and dynamically guide classical A* pathfinding.

### Key Features
- **Semantic Understanding** — Interprets contextual navigation instructions  
- **Dynamic Safety Buffering** — Adjusts obstacle clearance based on environmental conditions  
- **Sequential Task Reasoning** — Handles multi-stage missions (collect resource → navigate to goal)  
- **Low-Cost Implementation** — Works on affordable hardware (~ $300 robot + webcam)  
- **No Fine-Tuning Required** — Uses GPT-4 API directly with prompt engineering  

### Performance Highlights
- 96 – 100 % success rate on semantic navigation tasks  
- 100 % success on sequential multi-stage missions  
- Enables behaviors impossible with classical planners alone  

## System Architecture
The system consists of three main hardware components communicating over a shared Wi-Fi network via ROS 2.

### Overhead Camera System
- Off-the-shelf webcam with Raspberry Pi Zero 2 W  
- MJPEG streaming server (non-ROS)  
- Provides global view for localization and object detection  

### Laptop (Main Computing Unit)
- All ROS 2 nodes except command executor  
- GPT-4 API integration  
- A* path planning and occupancy-grid generation  
- YOLO object detection and AprilTag tracking  

### Bittle Robot
- Petoi Bittle quadruped with AprilTag marker  
- Raspberry Pi Zero 2 W running Docker container  
- Runs `bittle_command_executor.py` only  
- Receives high-level commands via ROS 2  

All devices communicate using the same `ROS_DOMAIN_ID` over Wi-Fi. The overhead camera streams MJPEG video that the laptop subscribes to for computer-vision processing. The laptop runs all computational nodes and sends high-level movement commands to the Bittle robot.

## Hardware Requirements
### Essential Components
- **Petoi Bittle Robot (~ $300)**  
  - Petoi Bittle quadruped kit  
  - Raspberry Pi Zero 2 W (onboard)  
  - Micro SD card (32 GB +)  
- **Overhead Camera Setup (~ $50)**  
  - USB webcam (resolution ≥ 640 × 480)  
  - Raspberry Pi Zero 2 W  
  - Micro SD card (16 GB +)  
  - Camera mount / tripod  
- **AprilTag**  
  - Printed AprilTag marker (ID = 1) attached to Bittle  
  - Download from *AprilTag Generator*  
- **Computing Device**  
  - Laptop / desktop with Wi-Fi capability  
  - Minimum 8 GB RAM recommended  

### Network Setup
- All devices connected to same Wi-Fi network  
- SSH access to both Raspberry Pi devices  
- ROS 2 domain communication (`ROS_DOMAIN_ID`)  

## Software Dependencies
### Laptop Requirements
    # ROS 2 Humble
    sudo apt update && sudo apt install ros-humble-desktop
    # Python packages
    pip install openai ultralytics opencv-python apriltag cv-bridge
    # Additional ROS 2 packages
    sudo apt install ros-humble-nav-msgs ros-humble-sensor-msgs

### Raspberry Pi Setup (Both Devices)
    # Docker (for Bittle Pi)
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    # ROS 2 Humble (for camera Pi – if needed)
    # Follow standard ROS 2 installation guide

## Installation & Setup
1. **Clone Repository**  
       cd ~/ros2_ws/src
       git clone https://github.com/yourusername/semantic-robot-navigation.git
       cd ~/ros2_ws
       colcon build
       source install/setup.bash

2. **Configure OpenAI API** — edit `LLM_reasoning_node.py` line 61:  
       openai.api_key = "your-openai-api-key-here"

3. **Setup Bittle Robot**  
       ssh pi@<bittle-ip-address>
       docker pull ros:humble
       # copy bittle_command_executor.py to the Pi
       export ROS_DOMAIN_ID=<your-domain-id>
       # run container with executor script ...

4. **Setup Overhead Camera**  
       ssh pi@<camera-ip-address>
       # install & configure MJPEG streaming script

5. **Configure Network Communication**  
       export ROS_DOMAIN_ID=42   # use any 0-101
       ros2 topic list           # verify topics across devices

6. **Update Camera Stream URL** — edit `webvid_publisher.py` line 13:  
       self.stream_url = "http://<camera-pi-ip>:8000/stream.mjpg"

## Running Experiments
### Basic System Startup
**Start Camera Stream (on camera Pi)**  
       python3 mjpeg_server.py   # added to repo

**Launch Core Nodes (on laptop)**  
       ros2 run bittle_ros2 webvid_publisher      # Terminal 1  
       ros2 run bittle_ros2 yolo_node             # Terminal 2  
       ros2 run bittle_ros2 apriltag_node         # Terminal 3  
       ros2 run bittle_ros2 occupancy_grid_publisher  # Terminal 4  
       ros2 run bittle_ros2 path_planner          # Terminal 5  

**Start Robot Control (on Bittle Pi)**  
       ros2 run bittle_ros2 bittle_command_executor

**Launch LLM Reasoning (on laptop)**  
       ros2 run bittle_ros2 LLM_reasoning_node --ros-args -p scenario:=3

### Experiment Scenarios
| Scenario | Command | Description |
|----------|---------|-------------|
| 1 | `ros2 run bittle_ros2 LLM_reasoning_node --ros-args -p scenario:=1` | Pure A* navigation baseline |
| 2 | `ros2 run bittle_ros2 LLM_reasoning_node --ros-args -p scenario:=2` | GPT-4 generates complete waypoint paths |
| 3 | `ros2 run bittle_ros2 LLM_reasoning_node --ros-args -p scenario:=3` | Hybrid — toxic obstacle avoidance |
| 4 | `ros2 run bittle_ros2 LLM_reasoning_node --ros-args -p scenario:=4` | Hybrid — crowded-area avoidance |
| 5 | `ros2 run bittle_ros2 LLM_reasoning_node --ros-args -p scenario:=5` | Hybrid — battery / charging-station |

## Evaluation Scenarios
### Course Complexity
1. **Course 1** — Single obstacle between start and goal  
2. **Course 2** — Two obstacles creating simple maze  
3. **Course 3** — Three obstacles requiring complex navigation  

### Semantic Tasks
- **Toxic Obstacle Avoidance** — Navigate while avoiding hazardous areas  
- **Low Battery Navigation** — Prioritize charging stations over direct goals  
- **Crowded Area Avoidance** — Select less congested paths for safety  

### Sequential Tasks
- Multi-stage missions: collect resource → navigate to final destination  
- Dynamic buffer adjustment based on path complexity  
- Tests multi-step reasoning capabilities  

## Results
**Experimental Results Summary**  
- **Simple Navigation** — A*: 100 % success / 16.4 s average • GPT-4: 60 % / 42.2 s  
- **Semantic Tasks (Hybrid)**  
  - Toxic avoidance: 100 %  
  - Low-battery navigation: 90 %  
  - Crowded-area avoidance: 100 %  
  - Sequential tasks: 100 %  

**Key Findings**  
- Classical A* is superior for pure geometric navigation.  
- GPT-4 hybrid enables semantic behaviors impossible with A* alone.  
- 96.7 % average compliance with semantic instructions.  
- Zero failures on complex sequential reasoning tasks.  

## Troubleshooting
### Common Issues
- **Robot not responding** — verify `ROS_DOMAIN_ID`, network, AprilTag visibility.  
- **GPT-4 API errors** — check key, credits, connectivity, `/home/jesse/Desktop/openai_responses.json`.  
- **Path planning failures** — confirm `/map`, `/yolo/obstacles`, `/yolo/goals` topics.  

### Debug Commands
       ros2 topic list                         # list topics  
       ros2 topic echo /april_tag/bittle_pose  # monitor pose  
       ros2 run rqt_image_view rqt_image_view  # view camera  
       ros2 topic echo /bittlebot/path_json    # check path JSON  

## File Structure  
```
bittle_ros2/
├── apriltag_node.py             # Robot localization via AprilTag
├── bittle_command_executor.py   # Motor control (runs on Bittle Pi)
├── LLM_reasoning_node.py        # GPT-4 integration and decision making
├── occupancy_grid_publisher.py  # Converts detections to occupancy grid
├── path_planner.py              # A* pathfinding with multiple candidates
├── webvid_publisher.py          # MJPEG stream subscriber
├── yolo_node.py                 # Object detection (obstacles/goals)
└── utils/
    └── best_v2.pt               # Custom YOLO model weights
```

## Citation  
```bibtex
@article{barkley2025semantic,
  title = {Semantic Intelligence: Integrating GPT-4 with A* Planning in Low-Cost Robotics},
  author = {Barkley, Jesse and George, Abraham and Farimani, Amir Barati},
  journal = {arXiv preprint arXiv:2505.01931},
  year = {2025}
}
```

## Contributing  
We welcome contributions! Please submit issues, fork the repository, and create pull requests.

## License  
This project is licensed under the **MIT License** — see the `LICENSE` file for details.

## Contact  
- Jesse Barkley — <jabarkle@andrew.cmu.edu>  
  Carnegie Mellon University, Department of Mechanical Engineering


