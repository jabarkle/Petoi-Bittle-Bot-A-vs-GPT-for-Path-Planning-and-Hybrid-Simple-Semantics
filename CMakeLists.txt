cmake_minimum_required(VERSION 3.8)
project(bittle_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find necessary packages
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)  #Added for Python nodes
find_package(nav_msgs REQUIRED)  #Needed for occupancy grid
find_package(bittle_msgs REQUIRED)  #Uses custom messages
find_package(geometry_msgs REQUIRED)  #Needed for PoseStamped in path planning
find_package(sensor_msgs REQUIRED)

# Install Python modules and nodes
ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY bittle_ros2
  DESTINATION lib/${PROJECT_NAME}
)

# If testing is enabled, configure linting and tests
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

