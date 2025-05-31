#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose
from bittle_msgs.msg import Yolo, AprilTag

class OccupancyGridPublisher(Node):
    def __init__(self):
        super().__init__('occupancy_grid_publisher')
        qos_profile = QoSProfile(depth=10)

        self.sub_yolo = self.create_subscription(
            Yolo,
            '/yolo/obstacles',
            self.yolo_callback,
            qos_profile
        )
        self.sub_apriltag = self.create_subscription(
            AprilTag,
            '/april_tag/bittle_pose',
            self.apriltag_callback,
            qos_profile
        )

        self.pub_grid = self.create_publisher(OccupancyGrid, '/map', qos_profile)
        self.pub_visualization = self.create_publisher(OccupancyGrid, '/visualization', qos_profile)

        # Map dimensions
        self.map_width = 320
        self.map_height = 240
        self.map_resolution = 0.0053  # meters per cell

        self.map_info = MapMetaData()
        self.map_info.resolution = self.map_resolution
        self.map_info.width = self.map_width
        self.map_info.height = self.map_height
        self.map_info.origin = Pose()
        self.map_info.origin.position.x = 0.0
        self.map_info.origin.position.y = 0.0
        self.map_info.origin.position.z = 0.0
        self.map_info.origin.orientation.w = 1.0

        self.bittlebot_position = None
        self.get_logger().info("OccupancyGridPublisher node started.")

    def apriltag_callback(self, apriltag_msg: AprilTag):
        self.bittlebot_position = [apriltag_msg.position[0], apriltag_msg.position[1]]

    def yolo_callback(self, yolo_msg: Yolo):
        # 1) Create an all-free occupancy grid
        occupancy_data = [0 for _ in range(self.map_width * self.map_height)]

        # 2) Convert each YOLO bounding box => Occupied region + buffer
        buffer_cells = 0 # Adjust as needed for your robot’s footprint + extra margin (Do 30 Normally but 0 for testing semantics)
        arr = yolo_msg.xywh
        for i in range(0, len(arr), 4):
            center_x_m = arr[i]
            center_y_m = arr[i + 1]
            width_m    = arr[i + 2]
            height_m   = arr[i + 3]

            center_x_cell = int(center_x_m / self.map_resolution)
            center_y_cell = int(center_y_m / self.map_resolution)
            half_w_cells  = int((width_m / self.map_resolution) / 2.0)
            half_h_cells  = int((height_m / self.map_resolution) / 2.0)

            x_min = center_x_cell - half_w_cells - buffer_cells
            x_max = center_x_cell + half_w_cells + buffer_cells
            y_min = center_y_cell - half_h_cells - buffer_cells
            y_max = center_y_cell + half_h_cells + buffer_cells

            # Clamp to map bounds
            x_min = max(0, x_min)
            x_max = min(self.map_width - 1, x_max)
            y_min = max(0, y_min)
            y_max = min(self.map_height - 1, y_max)

            # Mark as occupied = 100
            for yy in range(y_min, y_max + 1):
                for xx in range(x_min, x_max + 1):
                    idx = yy * self.map_width + xx
                    occupancy_data[idx] = 100

        # 3) Construct OccupancyGrid message
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = "map"
        grid_msg.info = self.map_info
        grid_msg.data = occupancy_data

        # 4) Publish
        self.pub_grid.publish(grid_msg)
        self.pub_visualization.publish(grid_msg)
        self.get_logger().info("Published inflated occupancy grid (obstacles + buffer).")

def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()