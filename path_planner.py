#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Pose
from bittle_msgs.msg import AprilTag, Yolo, BittlePath, BittlePathJSON
import numpy as np
import json
import heapq
import math

class PathPlanner(Node):
    def __init__(self):
        super().__init__('path_planner')
        qos_profile = QoSProfile(depth=10)
        # Subscriptions
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos_profile)
        self.create_subscription(AprilTag, '/april_tag/bittle_pose', self.bittlebot_callback, qos_profile)
        self.create_subscription(Yolo, '/yolo/goals', self.goal_callback, qos_profile)
        # Publishers
        self.path_pub = self.create_publisher(BittlePath, '/bittlebot/path', qos_profile)
        self.json_pub = self.create_publisher(BittlePathJSON, '/bittlebot/path_json', qos_profile)
        self.path_vis_pub = self.create_publisher(Path, '/bittlebot/path_visualization', qos_profile)

        # Map properties (defaults; will be updated)
        self.map_resolution = 0.0053   # meters per cell
        self.map_width = 320
        self.map_height = 240

        # Internal state
        self.occupancy_grid = None           # 2D numpy array
        self.bittlebot_position = None       # in grid coordinates (x, y)
        self.candidate_goals = []            # Each candidate: { "world": [x, y], "grid": (gx, gy) }
        self.candidate_paths = []            # Each candidate path: 
                                             # { "goal": [world_x, world_y],
                                             #   "grid_goal": (gx, gy),
                                             #   "path": [ [world_x, world_y], ... ],
                                             #   "metrics": { "path_length": float, "obstacle_count": int,
                                             #                "min_clearance": float, "avg_clearance": float }
                                             # }

        self.get_logger().info("PathPlanner node started for multiple candidate goals.")

        # Timer to periodically check if the current paths remain valid
        self.path_check_timer = self.create_timer(2.0, self.check_current_path_validity)

    def map_callback(self, msg: OccupancyGrid):
        self.occupancy_grid = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.map_resolution = msg.info.resolution
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.get_logger().info("Occupancy grid updated.")

    def bittlebot_callback(self, msg: AprilTag):
        grid_x = int(msg.position[0] / self.map_resolution)
        grid_y = int(msg.position[1] / self.map_resolution)
        grid_x = max(0, min(self.map_width - 1, grid_x))
        grid_y = max(0, min(self.map_height - 1, grid_y))
        self.bittlebot_position = (grid_x, grid_y)
        self.get_logger().info(f"BittleBot position (grid): {self.bittlebot_position}")

    def goal_callback(self, msg: Yolo):
        """
        Update candidate goals from the Yolo message.
        Each goal is defined by 4 floats in msg.xywh: [cx, cy, w, h].
        """
        self.candidate_goals = []  # Clear previous candidates
        num_goals = len(msg.xywh) // 4
        if num_goals == 0:
            self.get_logger().warn("No valid goal detections received.")
            return

        for i in range(num_goals):
            idx = i * 4
            world_x = msg.xywh[idx]
            world_y = msg.xywh[idx+1]
            goal_x_grid = int(world_x / self.map_resolution)
            goal_y_grid = int(world_y / self.map_resolution)

            candidate = {
                "world": [world_x, world_y],
                "grid": (goal_x_grid, goal_y_grid)
            }
            self.candidate_goals.append(candidate)
            self.get_logger().info(
                f"Candidate goal {i}: world=({world_x:.2f}, {world_y:.2f}), "
                f"grid=({goal_x_grid}, {goal_y_grid})"
            )

        # After updating candidate goals, compute candidate paths
        self.plan_paths_for_candidates()

    def plan_paths_for_candidates(self):
        """
        Runs A* from the BittleBot position to each candidate goal
        and smooths each path. Publishes the result as JSON.
        """
        if (self.occupancy_grid is None or
            self.bittlebot_position is None or
            len(self.candidate_goals) == 0):
            self.get_logger().warn("Missing data for path planning (occupancy grid, robot position, or goals).")
            return

        self.candidate_paths = []  # Clear previous candidate paths
        start = self.bittlebot_position

        for i, candidate in enumerate(self.candidate_goals):
            goal = candidate["grid"]

            # Check map bounds
            if not (0 <= goal[0] < self.map_width and 0 <= goal[1] < self.map_height):
                self.get_logger().error(f"Candidate goal {i} {goal} is out of bounds!")
                continue

            # Ensure the goal cell is free
            if self.occupancy_grid[goal[1], goal[0]] != 0:
                self.get_logger().error(f"Candidate goal {i} {goal} is blocked by an obstacle!")
                continue

            # Plan raw path with A*
            raw_path = self.a_star(start, goal)
            if raw_path:
                # Smooth the path before converting to world coordinates
                smoothed_path = self.smooth_path(raw_path)

                metrics = self.compute_path_metrics(smoothed_path)
                candidate_path = {
                    "goal": candidate["world"],
                    "grid_goal": goal,
                    "path": self.convert_path_to_world(smoothed_path),
                    "metrics": metrics
                }

                self.candidate_paths.append(candidate_path)
                self.get_logger().info(
                    f"Candidate path {i} computed: length={metrics['path_length']:.2f}, "
                    f"obstacle_count={metrics['obstacle_count']}, "
                    f"min_clearance={metrics['min_clearance']:.3f}, avg_clearance={metrics['avg_clearance']:.3f}."
                )
            else:
                self.get_logger().warn(f"A* failed to find a valid path for candidate goal {i}.")

        # Publish all candidate paths as JSON
        self.publish_candidate_paths()

    def a_star(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor, step_cost in self.get_neighbors_8(current):
                tentative_g = g_score[current] + step_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def get_neighbors_8(self, node):
        (x, y) = node
        neighbors_8 = [
            (x+1, y), (x-1, y), (x, y+1), (x, y-1),
            (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)
        ]
        valid_neighbors = []
        for nx, ny in neighbors_8:
            if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                if self.occupancy_grid[ny, nx] == 0:
                    # Diagonal steps have higher cost
                    cost = 1.0 if (nx == x or ny == y) else 1.4142
                    valid_neighbors.append(((nx, ny), cost))
        return valid_neighbors

    def heuristic(self, node, goal):
        (x1, y1) = node
        (x2, y2) = goal
        return math.hypot(x2 - x1, y2 - y1)

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def compute_clearance_for_point(self, x, y):
        """
        Compute the Euclidean distance (in meters) from the given grid cell (x,y)
        to the nearest obstacle cell in the occupancy grid.
        """
        min_dist_cells = float('inf')
        for i in range(self.map_height):
            for j in range(self.map_width):
                if self.occupancy_grid[i, j] != 0:  # obstacle cell
                    dist = math.hypot(j - x, i - y)
                    if dist < min_dist_cells:
                        min_dist_cells = dist
        return min_dist_cells * self.map_resolution if min_dist_cells != float('inf') else float('inf')

    def compute_path_metrics(self, grid_path):
        """
        Measures total path length (in meters), counts how many grid cells
        along the path are near obstacles, and computes clearance metrics.
        """
        path_length = 0.0
        obstacle_count = 0
        clearances = []
        for i in range(1, len(grid_path)):
            x1, y1 = grid_path[i - 1]
            x2, y2 = grid_path[i]
            segment_len_cells = math.hypot(x2 - x1, y2 - y1)
            path_length += segment_len_cells * self.map_resolution
            if not self.is_valid_cell(x2, y2):
                obstacle_count += 1
            # Compute clearance for the current cell
            clearance = self.compute_clearance_for_point(x2, y2)
            clearances.append(clearance)

        min_clearance = min(clearances) if clearances else float('inf')
        avg_clearance = sum(clearances)/len(clearances) if clearances else float('inf')
        return {
            "path_length": path_length,
            "obstacle_count": obstacle_count,
            "min_clearance": min_clearance,
            "avg_clearance": avg_clearance
        }

    def convert_path_to_world(self, grid_path):
        """
        Convert a path in grid coordinates to world coordinates (meters).
        """
        world_path = []
        for (x, y) in grid_path:
            world_x = x * self.map_resolution
            world_y = y * self.map_resolution
            world_path.append([world_x, world_y])
        return world_path

    def smooth_path(self, path):
        """
        Minimal line-of-sight smoothing: tries to 'jump' from one waypoint
        to another if there's no obstacle in between, skipping intermediate points.
        """
        if len(path) < 3:
            return path  # Nothing to smooth if only 0-2 points
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            # Attempt to jump directly from path[i] to path[j].
            while j > i + 1:
                if self.check_line_of_sight(path[i], path[j]):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j
        return smoothed

    def check_line_of_sight(self, point1, point2):
        """
        Bresenham-based check: ensure every cell between point1 and point2 is valid.
        """
        x1, y1 = point1
        x2, y2 = point2
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        n = 1 + dx + dy
        x_inc = 1 if x2 > x1 else -1
        y_inc = 1 if y2 > y1 else -1
        error = dx - dy
        dx *= 2
        dy *= 2
        for _ in range(n):
            if not self.is_valid_cell(x, y):
                return False
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return True

    def is_valid_cell(self, x, y):
        if 0 <= x < self.map_width and 0 <= y < self.map_height:
            return (self.occupancy_grid[y, x] == 0)
        return False

    def publish_candidate_paths(self):
        """
        Publish all candidate paths as JSON for the LLM node, 
        and visualize the *shortest* of them in RViz.
        """
        data = {"candidate_paths": self.candidate_paths}
        json_msg = BittlePathJSON()
        json_msg.json_data = json.dumps(data)
        self.json_pub.publish(json_msg)
        self.get_logger().info("Published candidate paths.")

        if self.candidate_paths:
            # As an example, visualize the path with the minimum path length
            chosen = min(self.candidate_paths, key=lambda cp: cp["metrics"]["path_length"])
            self.publish_visualization(chosen["path"])

    def publish_visualization(self, world_path):
        nav_path = Path()
        nav_path.header.stamp = self.get_clock().now().to_msg()
        nav_path.header.frame_id = "map"
        for coord in world_path:
            pose = PoseStamped()
            pose.pose.position.x = coord[0]
            pose.pose.position.y = coord[1]
            pose.pose.position.z = 0.0
            nav_path.poses.append(pose)
        self.path_vis_pub.publish(nav_path)
        self.get_logger().info("Published visualization of chosen candidate path.")

    def check_current_path_validity(self):
        """
        Periodically checks if candidate paths remain valid
        and replans if an obstacle appears along any path.
        """
        if not self.candidate_paths or self.occupancy_grid is None:
            return

        for cp in self.candidate_paths:
            for point in cp["path"]:
                grid_x = int(point[0] / self.map_resolution)
                grid_y = int(point[1] / self.map_resolution)
                if not self.is_valid_cell(grid_x, grid_y):
                    self.get_logger().info("A candidate path is invalidated due to a new obstacle. Replanning.")
                    self.plan_paths_for_candidates()
                    return

def main(args=None):
    rclpy.init(args=args)
    node = PathPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("PathPlanner node shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
