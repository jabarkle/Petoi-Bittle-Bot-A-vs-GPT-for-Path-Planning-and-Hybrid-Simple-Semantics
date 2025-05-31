#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import apriltag
import numpy as np
import math
from bittle_msgs.msg import AprilTag

# Update this ID to match the actual AprilTag ID attached to your BittleBot
BITTLEBOT_TAG_ID = 1

class AprilTagNode(Node):
    """
    AprilTagNode
    ------------
    - Subscribes to a camera feed (e.g., /camera/stream).
    - Detects AprilTags in each frame using the apriltag Python library.
    - Publishes the BittleBot's (x, y, theta) pose to /april_tag/bittle_pose 
      whenever a tag with ID=BITTLEBOT_TAG_ID is detected.

    The orientation theta is estimated by a simple vector from the first corner 
    to the second corner of the detected tag, so it’s approximate but typically 
    sufficient for basic heading control.

    Note: 
      - YOLO is used for obstacles and goal detection, but *not* for the BittleBot’s 
        position. The BittleBot’s pose comes *only* from AprilTag for precision 
        and to avoid self-classification as an obstacle.
      - Ensure your map_resolution matches that used by YOLO and the occupancy grid 
        so coordinates align consistently.
    """

    def __init__(self):
        super().__init__('apriltag_node')

        # Subscribe to the camera stream
        self.subscription = self.create_subscription(
            Image,
            '/camera/stream',
            self.listener_callback,
            10
        )

        # Publisher for BittleBot’s position + orientation
        self.detection_publisher = self.create_publisher(AprilTag, '/april_tag/bittle_pose', 10)

        # Publisher for visual debug images with drawn AprilTag detections
        self.image_publisher = self.create_publisher(Image, '/apriltag_detections', 10)

        # Bridge for ROS <-> OpenCV
        self.bridge = CvBridge()

        # Resolution (meters per pixel) consistent with YOLO and occupancy grid
        self.map_resolution = 0.0053  # ~5.3 mm per pixel

        # Initialize the AprilTag detector (can configure as needed)
        self.detector = apriltag.Detector()

        self.get_logger().info("AprilTagNode started and ready to detect BittleBot’s AprilTag!")

    def listener_callback(self, msg: Image):
        """
        Receives an Image from /camera/stream, converts it to OpenCV, detects AprilTags,
        and if it finds the BittleBot’s tag (BITTLEBOT_TAG_ID), publishes its pose.
        """
        # Convert the ROS image to an OpenCV BGR image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags
        results = self.detector.detect(gray_frame)

        # For each detected AprilTag
        for r in results:
            cX, cY = int(r.center[0]), int(r.center[1])

            if r.tag_id == BITTLEBOT_TAG_ID:
                self.get_logger().info(f"BittleBot AprilTag detected at image coords ({cX}, {cY})")

                # Convert image coords -> world coords (meters)
                world_x = cX * self.map_resolution
                world_y = cY * self.map_resolution

                # Estimate orientation from the first two corners
                # (This is approximate but typically sufficient for basic heading)
                if r.corners is not None and len(r.corners) >= 2:
                    corner0 = r.corners[0]
                    corner1 = r.corners[1]
                    theta = math.atan2(corner1[1] - corner0[1], corner1[0] - corner0[0])
                else:
                    theta = 0.0

                self.get_logger().info(
                    f"Updated BittleBot position: Image ({cX}, {cY}) → "
                    f"World ({world_x:.3f}, {world_y:.3f}), Orientation: {theta:.3f} rad"
                )

                # Publish on a custom AprilTag message
                april_tag_message = AprilTag()
                april_tag_message.tag_id = BITTLEBOT_TAG_ID
                # [x_m, y_m, theta_radians]
                april_tag_message.position = [float(world_x), float(world_y), float(theta)]
                self.detection_publisher.publish(april_tag_message)

        # Optionally, we could annotate the frame and publish to /apriltag_detections
        # But for now, we just skip or do minimal debug if needed.
        # (You can draw the corners/center, etc.)
        # example: annotated_frame = frame.copy()
        # ... draw ...
        # annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        # self.image_publisher.publish(annotated_msg)

def main(args=None):
    rclpy.init(args=args)
    node = AprilTagNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
