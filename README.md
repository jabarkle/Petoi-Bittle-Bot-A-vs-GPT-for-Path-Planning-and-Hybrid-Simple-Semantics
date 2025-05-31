# Petoi-Bittle-Bot-A-vs-GPT-for-Path-Planning-and-Hybrid-Simple-Semantics

Semantic Intelligence: GPT-4 + A* Path Planning for Low-Cost Robotics
This repository implements a hybrid navigation system that combines GPT-4's semantic reasoning with classical A* path planning on a low-cost Petoi Bittle quadruped robot. The system enables context-aware navigation through natural language instructions without hardcoded finite state machines.
Table of Contents

Overview
System Architecture
Hardware Requirements
Software Dependencies
Installation & Setup
Running Experiments
Evaluation Scenarios
Results
Citation

Overview
Traditional robot navigation relies on hardcoded state machines and purely geometric planners, limiting adaptability to semantic instructions. This research demonstrates how GPT-4 can interpret high-level commands (e.g., "avoid toxic spills", "find charging station") and dynamically guide classical A* pathfinding.
Key Features

Semantic Understanding: Interprets contextual navigation instructions
Dynamic Safety Buffering: Adjusts obstacle clearance based on environmental conditions
Sequential Task Reasoning: Handles multi-stage missions (collect resource → navigate to goal)
Low-Cost Implementation: Works on affordable hardware ($300 robot + webcam)
No Fine-Tuning Required: Uses GPT-4 API directly with prompt engineering

Performance Highlights

96-100% success rate on semantic navigation tasks
100% success on sequential multi-stage missions
Enables behaviors impossible with classical planners alone

System Architecture
The system consists of three main hardware components communicating over a shared WiFi network via ROS2:
Overhead Camera System:

Off-the-shelf webcam with Raspberry Pi Zero 2W
MJPEG streaming server (non-ROS)
Provides global view for localization and object detection

Laptop (Main Computing Unit):

All ROS2 nodes except command executor
GPT-4 API integration
A* path planning and occupancy grid generation
YOLO object detection and AprilTag tracking

Bittle Robot:

Petoi Bittle quadruped with AprilTag marker
Raspberry Pi Zero 2W running Docker container
Only runs bittle_command_executor.py
Receives high-level commands via ROS2

All devices communicate using the same ROS_DOMAIN_ID over WiFi. The overhead camera streams MJPEG video that the laptop subscribes to for computer vision processing. The laptop runs all computational nodes and sends high-level movement commands to the Bittle robot.
Hardware Requirements
Essential Components

Petoi Bittle Robot (~$300)

Petoi Bittle quadruped kit
Raspberry Pi Zero 2W (onboard)
MicroSD card (32GB+)


Overhead Camera Setup (~$50)

USB webcam (any resolution ≥640x480)
Raspberry Pi Zero 2W
MicroSD card (16GB+)
Camera mount/tripod


AprilTag

Printed AprilTag marker (ID=1) attached to Bittle
Download from: AprilTag Generator


Computing Device

Laptop/desktop with WiFi capability
Minimum 8GB RAM recommended



Network Setup

All devices connected to same WiFi network
SSH access to both Raspberry Pi devices
ROS2 domain communication (same ROS_DOMAIN_ID)

Software Dependencies
Laptop Requirements
bash# ROS2 Humble
sudo apt update && sudo apt install ros-humble-desktop

# Python packages
pip install openai ultralytics opencv-python apriltag cv-bridge

# Additional ROS2 packages  
sudo apt install ros-humble-nav-msgs ros-humble-sensor-msgs
Raspberry Pi Setup (Both Devices)
bash# Docker (for Bittle Pi)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# ROS2 Humble (for camera Pi - if needed)
# Follow standard ROS2 installation guide
Installation & Setup
1. Clone Repository
bashcd ~/ros2_ws/src
git clone https://github.com/yourusername/semantic-robot-navigation.git
cd ~/ros2_ws
colcon build
source install/setup.bash
2. Configure OpenAI API
Edit LLM_reasoning_node.py line 61:
pythonopenai.api_key = "your-openai-api-key-here"
3. Setup Bittle Robot
bash# SSH into Bittle's Pi Zero 2W
ssh pi@<bittle-ip-address>

# Pull ROS2 Docker image (or build custom image with bittle_msgs)
docker pull ros:humble

# Copy bittle_command_executor.py to the Pi
# Run container with bittle_command_executor.py
# Set ROS_DOMAIN_ID to match your network
export ROS_DOMAIN_ID=<your-domain-id>
4. Setup Overhead Camera
bash# SSH into camera Pi Zero 2W  
ssh pi@<camera-ip-address>

# Install MJPEG streaming script (to be added to repo)
# Configure camera parameters and streaming endpoint
5. Configure Network Communication
bash# On all devices, set same domain ID:
export ROS_DOMAIN_ID=42  # Use any number 0-101

# Verify communication
ros2 topic list  # Should show topics across devices
6. Update Camera Stream URL
Edit webvid_publisher.py line 13 to match your camera Pi's IP:
pythonself.stream_url = "http://<camera-pi-ip>:8000/stream.mjpg"
Running Experiments
Basic System Startup

Start Camera Stream (on camera Pi)

bashpython3 mjpeg_server.py  # Will be added to repo

Launch Core Nodes (on laptop)

bash# Terminal 1: Camera feed subscriber
ros2 run bittle_ros2 webvid_publisher

# Terminal 2: Object detection  
ros2 run bittle_ros2 yolo_node

# Terminal 3: Robot localization
ros2 run bittle_ros2 apriltag_node

# Terminal 4: Mapping
ros2 run bittle_ros2 occupancy_grid_publisher

# Terminal 5: Path planning
ros2 run bittle_ros2 path_planner

Start Robot Control (on Bittle Pi)

bash# In Docker container
ros2 run bittle_ros2 bittle_command_executor

Launch LLM Reasoning (on laptop)

bash# Choose scenario: 1=A* only, 2=GPT route generation, 3-5=Hybrid selection
ros2 run bittle_ros2 LLM_reasoning_node --ros-args -p scenario:=3
Experiment Scenarios
Scenario 1: Classical A* Baseline
bashros2 run bittle_ros2 LLM_reasoning_node --ros-args -p scenario:=1

Pure A* navigation without GPT
Baseline for comparison

Scenario 2: GPT Route Generation
bashros2 run bittle_ros2 LLM_reasoning_node --ros-args -p scenario:=2

GPT-4 generates complete waypoint paths
Tests LLM spatial reasoning capabilities

Scenario 3: Hybrid - Toxic Avoidance
bashros2 run bittle_ros2 LLM_reasoning_node --ros-args -p scenario:=3

A* generates candidate paths
GPT selects safest route avoiding "toxic" obstacles

Scenario 4: Hybrid - Crowded Area Avoidance
bashros2 run bittle_ros2 LLM_reasoning_node --ros-args -p scenario:=4

GPT chooses paths avoiding congested regions
Tests understanding of spatial density

Scenario 5: Hybrid - Battery/Charging Station
bashros2 run bittle_ros2 LLM_reasoning_node --ros-args -p scenario:=5

Low battery scenario requiring charging station
Tests goal reinterpretation based on context

Evaluation Scenarios
Course Complexity

Course 1: Single obstacle between start and goal
Course 2: Two obstacles creating simple maze
Course 3: Three obstacles requiring complex navigation

Semantic Tasks

Toxic Obstacle Avoidance: Navigate while avoiding hazardous areas
Low Battery Navigation: Prioritize charging stations over direct goals
Crowded Area Avoidance: Select less congested paths for safety

Sequential Tasks

Multi-stage missions: collect resource → navigate to final destination
Dynamic buffer adjustment based on path complexity
Tests multi-step reasoning capabilities

Results
ScenarioMethodSuccess RateAvg. TimeSimple NavigationA*100%16.4sSimple NavigationGPT-460%42.2sToxic AvoidanceHybrid100%-Low BatteryHybrid90%-Crowded AvoidanceHybrid100%-Sequential TasksHybrid100%-
Key Findings:

Classical A* superior for pure geometric navigation
GPT-4 hybrid enables semantic behaviors impossible with A* alone
96.7% average compliance with semantic instructions
Zero failures on complex sequential reasoning tasks

Troubleshooting
Common Issues
Robot not responding:

Check ROS_DOMAIN_ID matches across all devices
Verify network connectivity: ros2 topic list
Ensure AprilTag is visible and properly oriented

GPT-4 API errors:

Verify OpenAI API key is valid and has credits
Check internet connectivity on laptop
Monitor API call logs in /home/jesse/Desktop/openai_responses.json

Path planning failures:

Ensure occupancy grid is being published: ros2 topic echo /map
Check YOLO detections: ros2 topic echo /yolo/obstacles
Verify goal detection: ros2 topic echo /yolo/goals

Debug Commands
bash# Check all active topics
ros2 topic list

# Monitor robot position
ros2 topic echo /april_tag/bittle_pose

# View camera feed
ros2 run rqt_image_view rqt_image_view

# Check path planning output
ros2 topic echo /bittlebot/path_json
File Structure
bittle_ros2/
├── apriltag_node.py              # Robot localization via AprilTag
├── bittle_command_executor.py    # Motor control (runs on Bittle Pi)
├── LLM_reasoning_node.py         # GPT-4 integration and decision making
├── occupancy_grid_publisher.py   # Converts detections to occupancy grid
├── path_planner.py              # A* pathfinding with multiple candidates
├── webvid_publisher.py          # MJPEG stream subscriber
├── yolo_node.py                 # Object detection (obstacles/goals)
└── utils/
    └── best_v2.pt              # Custom YOLO model weights
Citation
If you use this work in your research, please cite:
bibtex@article{barkley2025semantic,
  title={Semantic Intelligence: Integrating GPT-4 with A* Planning in Low-Cost Robotics},
  author={Barkley, Jesse and George, Abraham and Farimani, Amir Barati},
  journal={arXiv preprint arXiv:2505.01931},
  year={2025}
}
Contributing
We welcome contributions! Please feel free to submit issues, fork the repository, and create pull requests.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Contact

Jesse Barkley - jabarkle@andrew.cmu.edu
Abraham George - aigeorge@andrew.cmu.edu
Amir Barati Farimani - afariman@andrew.cmu.edu

Carnegie Mellon University, Department of Mechanical Engineering
Related Work

SayCan: Grounding Language in Robotic Affordances
LM-Nav: Robotic Navigation with Large Pre-Trained Models
Petoi Bittle Documentation
