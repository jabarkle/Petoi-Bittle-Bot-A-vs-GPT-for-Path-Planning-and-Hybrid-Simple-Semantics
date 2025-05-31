#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
from bittle_msgs.msg import Yolo  # Custom message

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Subscribe to camera stream
        self.subscription = self.create_subscription(
            Image,
            '/camera/stream',
            self.listener_callback,
            10
        )

        # Load updated YOLO model with 3 classes:
        #   0: bittlebot, 1: goal, 2: obstacle
        self.model = YOLO('/home/jesse/ros2_ws/src/BittleUpdate/bittle_ros2/bittle_ros2/utils/best_v2.pt')

        # Create ROS2 publishers
        #   /yolo/obstacles => obstacles only (class=2)
        #   /yolo/goals     => goal(s) (class=1)
        # (We ignore bittlebot class 0 for obstacle/goal publishing)
        self.obstacles_pub = self.create_publisher(Yolo, '/yolo/obstacles', 10)
        self.goals_pub = self.create_publisher(Yolo, '/yolo/goals', 10)
        self.annotated_image_pub = self.create_publisher(Image, '/yolo/detections', 10)

        # Define camera and map properties
        self.image_width = 640   # Adjust if needed
        self.image_height = 480
        self.map_resolution = 0.0053  # 5.3 mm per pixel (real-world scaling)
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0

        self.get_logger().info("YOLO Node started with updated best_v2.pt!")

    def listener_callback(self, msg: Image):
        """
        Process YOLO detections and publish obstacles & goals in world coordinates.
        We skip bittlebot (class 0) so it’s never treated as an obstacle.
        """

        # Convert ROS2 Image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Perform YOLO inference
        try:
            results = self.model.predict(frame)
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {str(e)}")
            return

        # Prepare empty messages for obstacles and goals:
        obstacles_msg = Yolo()
        goals_msg = Yolo()

        # New lists for aspect ratios
        obstacle_aspects = []
        goal_aspects = []

        # Run through all detections (if any)
        if not results or len(results[0].boxes) == 0:
            # No detections found; publish empty messages
            self.obstacles_pub.publish(obstacles_msg)  # empty obstacles
            self.goals_pub.publish(goals_msg)          # empty goals

            # Optionally publish unannotated image for debugging
            annotated_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.annotated_image_pub.publish(annotated_msg)
            return

        detections = results[0].boxes
        classes = detections.cls.cpu().tolist()   # Class IDs
        xywh = detections.xywh.cpu().tolist()       # [cx, cy, w, h] in pixels
        xywhn = detections.xywhn.cpu().tolist()     # normalized [cx, cy, w, h]

        for i, raw_class_id in enumerate(classes):
            class_id = int(raw_class_id)  # float -> int
            bbox_xywh = xywh[i]
            bbox_xywhn = xywhn[i]

            # Convert bounding box center from pixel coords => world coords
            px_center = bbox_xywh[0]
            py_center = bbox_xywh[1]
            w_pixels = bbox_xywh[2]
            h_pixels = bbox_xywh[3]

            # Map from pixels to meters
            world_x = px_center * self.map_resolution + self.map_origin_x
            world_y = py_center * self.map_resolution + self.map_origin_y
            world_w = w_pixels * self.map_resolution
            world_h = h_pixels * self.map_resolution

            # Compute aspect ratio (width / height)
            aspect_ratio = w_pixels / h_pixels if h_pixels != 0 else 0.0

            # Based on class ID:
            #   0 => bittlebot (ignore)
            #   1 => goal
            #   2 => obstacle
            if class_id == 1:  # Goal
                goals_msg.class_ids.append(class_id)
                goals_msg.xywh.extend([world_x, world_y, world_w, world_h])
                goals_msg.xywhn.extend(map(float, bbox_xywhn))
                goal_aspects.append(aspect_ratio)
                self.get_logger().info(
                    f"[YOLO] Goal detected at pixel=({px_center:.1f},{py_center:.1f}), "
                    f"world=({world_x:.2f},{world_y:.2f}), aspect_ratio={aspect_ratio:.2f}"
                )
            elif class_id == 2:  # Obstacle
                obstacles_msg.class_ids.append(class_id)
                obstacles_msg.xywh.extend([world_x, world_y, world_w, world_h])
                obstacles_msg.xywhn.extend(map(float, bbox_xywhn))
                obstacle_aspects.append(aspect_ratio)
            else:
                # class_id == 0 => bittlebot => ignore
                self.get_logger().debug(
                    "[YOLO] Detected BittleBot (class=0), ignoring in obstacle/goal topics."
                )

        # Append the aspect ratios to the messages
        goals_msg.aspect_ratios.extend(goal_aspects)
        obstacles_msg.aspect_ratios.extend(obstacle_aspects)

        # Always publish goals (empty or filled)
        self.goals_pub.publish(goals_msg)
        # Always publish obstacles (empty or filled)
        self.obstacles_pub.publish(obstacles_msg)

        # Publish annotated image (for debugging)
        annotated_frame = results[0].plot()
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        self.annotated_image_pub.publish(annotated_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
