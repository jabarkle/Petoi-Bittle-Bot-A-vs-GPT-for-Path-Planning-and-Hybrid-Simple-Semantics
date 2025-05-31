#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
import json
import math
import os
import openai
import time

from bittle_msgs.msg import AprilTag, Yolo, BittlePathJSON, BittlePathGPTJSON

LOG_FILE = "/home/jesse/Desktop/openai_responses.json"

def append_to_log(data: dict):
    """
    Append a dict to openai_responses.json for debugging GPT interactions.
    """
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []
    else:
        logs = []
    logs.append(data)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)


class LLMReasoning(Node):
    """
    LLMReasoning Node
    -----------------
    - scenario=1 => Baseline “A* only” (no GPT).
    - scenario=2 => GPT-based route (up to 5 bounding-box–aware waypoints).
    - scenario=3..5 => Hybrid approach using A* candidate paths + GPT selection.
    """

    SUPPORTED_COMMANDS = {
        "move_forward",
        "turn_left",
        "turn_right",
        "stop"
    }

    def __init__(self):
        super().__init__('llm_reasoning_node')
        self.declare_parameter("scenario", 1)
        self.scenario = self.get_parameter("scenario").get_parameter_value().integer_value

        # Subscriptions
        self.create_subscription(BittlePathJSON, '/bittlebot/path_json', self.path_json_callback, 10)
        self.create_subscription(BittlePathGPTJSON, '/bittlebot/path_gpt_json', self.gpt_path_callback, 10)
        self.create_subscription(AprilTag, '/april_tag/bittle_pose', self.apriltag_callback, 10)
        self.create_subscription(Yolo, '/yolo/goals', self.goal_callback, 10)
        self.create_subscription(Yolo, '/yolo/obstacles', self.obstacles_callback, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(String, '/bittle_cmd', 10)
        self.gpt_path_pub = self.create_publisher(BittlePathGPTJSON, '/bittlebot/path_gpt_json', 10)

        # Data caches
        self.robot_position = None     # [x, y, theta]
        self.goal_position = None      # [gx, gy]
        self.obstacles = []            # Each obstacle: {"center": [cx, cy], "size": [w, h]}
        self.occupancy_grid = None
        self.map_resolution = 0.0053

        # Candidate paths
        self.candidate_paths = []
        self.selected_candidate_path = None
        self.decision_locked = False

        # Pure pursuit parameters (slightly adjusted)
        self.path_index = 0
        self.LOOKAHEAD_DIST = 0.1      # was 0.3
        self.ANGLE_THRESHOLD = 0.25    # was 0.4
        self.WP_THRESHOLD = 0.15

        # GPT timers
        self.GPT_CALL_PERIOD = 3.0
        self.CONTROL_TIMER_PERIOD = 0.4
        self.gpt_timer = self.create_timer(self.GPT_CALL_PERIOD, self.gpt_timer_callback)
        self.control_timer = self.create_timer(self.CONTROL_TIMER_PERIOD, self.control_timer_callback)

        # Set your OpenAI key here:
        openai.api_key = ""

        # Scenario 2 variables
        self.scenario2_plan_done = False
        self.gpt_called_once = False

        # For retries if GPT path fails in Scenario 2
        self.scenario2_retry_count = 0
        self.scenario2_max_retries = 10

        self.get_logger().info(f"LLMReasoning node started, scenario={self.scenario}.")

    # -------------------- Subscriptions --------------------

    def apriltag_callback(self, msg: AprilTag):
        self.robot_position = [msg.position[0], msg.position[1], msg.position[2]]

    def goal_callback(self, msg: Yolo):
        if len(msg.xywh) >= 4:
            self.goal_position = [msg.xywh[0], msg.xywh[1]]
        else:
            self.goal_position = None

    def obstacles_callback(self, msg: Yolo):
        """
        Enlarge bounding boxes slightly for scenario=2. 
        Note: We don't touch scenarios 1,3,4,5.
        """
        obs_list = []
        arr = msg.xywh
        for i in range(0, len(arr), 4):
            center = self._round_list([arr[i], arr[i+1]])
            w = arr[i+2]
            h = arr[i+3]

            # If scenario=2, inflate bounding boxes a bit
            if self.scenario == 2:
                # Keep it small (just 0.05) to avoid overly constraining the space
                w += 0.0
                h += 0.0

            obs_list.append({
                "center": center,
                "size": self._round_list([w, h])
            })
        self.obstacles = obs_list

    def map_callback(self, msg: OccupancyGrid):
        self.map_resolution = msg.info.resolution
        w = msg.info.width
        h = msg.info.height
        data = msg.data
        grid_2d = []
        idx = 0
        for _ in range(h):
            grid_2d.append(list(data[idx: idx + w]))
            idx += w
        self.occupancy_grid = grid_2d

    def path_json_callback(self, msg: BittlePathJSON):
        try:
            data = json.loads(msg.json_data)
            if "candidate_paths" in data:
                self.candidate_paths = data["candidate_paths"]
        except json.JSONDecodeError as e:
            self.get_logger().error(f"JSON error reading path: {e}")
            return

        # Scenario 1 only
        if self.scenario == 1 and not self.decision_locked and self.candidate_paths:
            best = min(self.candidate_paths, key=lambda cp: cp["metrics"]["path_length"])
            self.selected_candidate_path = best.get("path", [])
            self.decision_locked = True
            self.path_index = 0
            plen = best["metrics"]["path_length"]
            self.get_logger().info(f"[Scenario1] locked path => length={plen:.2f}, wps={len(self.selected_candidate_path)}")

    def gpt_path_callback(self, msg: BittlePathGPTJSON):
        try:
            data = json.loads(msg.json_data)
            if "candidate_paths" in data:
                self.candidate_paths = data["candidate_paths"]
        except json.JSONDecodeError as e:
            self.get_logger().error(f"JSON error reading GPT path: {e}")
            return

        # If we receive a path from GPT in scenario 2 and haven't decided yet
        if self.scenario == 2 and not self.decision_locked and self.candidate_paths:
            best = self.candidate_paths[0]
            self.selected_candidate_path = best.get("path", [])
            self.decision_locked = True
            self.path_index = 0
            m = best.get("metrics", {})
            plen = m.get("path_length", -1)
            self.get_logger().info(f"[Scenario2 GPT] locked GPT path => length={plen}, wps={len(self.selected_candidate_path)}")

    # -------------------- GPT Timer --------------------

    def gpt_timer_callback(self):
        if self.scenario == 1:
            return
        elif self.scenario == 2:
            if not self.scenario2_plan_done:
                self._scenario2_gpt_call()
        else:
            # Scenarios 3..5 => single GPT call for path selection
            if not self.candidate_paths or self.decision_locked or self.gpt_called_once:
                return
            self.gpt_called_once = True
            self._scenario345_gpt_call()

    # -------------------- Control Timer --------------------

    def control_timer_callback(self):
        if self.decision_locked and self.selected_candidate_path:
            cmd = self._compute_pure_pursuit(self.selected_candidate_path)
            self._publish_command(cmd)

    # -------------------- Scenario 2: GPT Route Planning --------------------

    def _scenario2_gpt_call(self):
        """
        Scenario 2 bounding-box approach with a final collision check.
        Retries up to self.scenario2_max_retries if GPT returns an invalid path.
        """
        if not self.robot_position or not self.goal_position:
            self.scenario2_plan_done = True
            return
        if self._is_near_goal(threshold=0.15):
            self.scenario2_plan_done = True
            return

        data_for_gpt = {
            "robot_position": self._round_list(self.robot_position),
            "goal_position": self._round_list(self.goal_position),
            # Already inflated in obstacles_callback if scenario=2
            "obstacles": self.obstacles 
        }

        # Strengthen prompt to emphasize the clearance requirement
        system_prompt = (
            "You are an LLM controlling a BittleBot in a 2D plane. "
            "The robot's position is [x, y, theta], and the goal is [x, y], all in meters. "
            "Each obstacle is described by a bounding box [center_x, center_y, width, height]. "
            "You must ensure the robot does NOT come closer than 0.05 m to any obstacle's perimeter. "
            "Plan a route consisting of up to 6 absolute [x,y] waypoints that are a minimum of 0.1m apart, "
            "while maintaining at least 0.05 m clearance from every obstacle bounding box. Be safe. "
            "If there are multiple obstacles, take the path that ensures 0.05m clearance from them. "
            "The final waypoint must be exactly the goal center. "
            "Output strictly JSON: {\"mode\":\"waypoint_plan\",\"waypoints\":[[x1,y1],[x2,y2],...]} with NO extra text. "
            "If no collision-free route exists, return an empty list for 'waypoints'."
        )

        # We use temperature=0.0 for more deterministic output
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(data_for_gpt)}
                ],
                max_tokens=256,
                temperature=0.0
            )
            raw_reply = response.choices[0].message.content
            self.get_logger().info(f"[Scenario2 GPT] => {raw_reply}")
            append_to_log({
                "prompt": {"system": system_prompt, "user": data_for_gpt},
                "response": raw_reply
            })
            time.sleep(1.0)
        except Exception as e:
            self.get_logger().error(f"[Scenario2 GPT Error] => {e}")
            self._publish_command("stop")
            self.scenario2_plan_done = True
            return

        mode, wpts = self._parse_gpt_waypoints(raw_reply)
        if mode != "waypoint_plan":
            self.get_logger().warn("GPT returned invalid JSON or no 'waypoint_plan' => no path.")
            wpts = []

        # If GPT gave us no waypoints, check if we can retry
        if len(wpts) == 0:
            if self.scenario2_retry_count < self.scenario2_max_retries:
                self.scenario2_retry_count += 1
                self.get_logger().warn(
                    f"GPT returned an empty or invalid path. Retrying... "
                    f"(attempt {self.scenario2_retry_count}/{self.scenario2_max_retries})"
                )
                return  # Let the timer call us again
            else:
                self.get_logger().warn("GPT returned empty path after max retries => stopping.")
                self.scenario2_plan_done = True
                self._publish_command("stop")
                return

        # Force the final waypoint to be the goal
        wpts[-1] = self.goal_position

        # Final collision check with clearance=0.05
        if not self._check_path_collisions(wpts, self.obstacles, clearance=0.05):
            if self.scenario2_retry_count < self.scenario2_max_retries:
                self.scenario2_retry_count += 1
                self.get_logger().warn(
                    f"Collision check failed. Retrying GPT call... "
                    f"(attempt {self.scenario2_retry_count}/{self.scenario2_max_retries})"
                )
                return  # Let the timer cycle do it again
            else:
                self.get_logger().error("Final collision check failed after max retries => stopping.")
                self.scenario2_plan_done = True
                self._publish_command("stop")
                return

        # If we reach here, the path is accepted
        p_length = 0.0
        prev = None
        for pt in wpts:
            if prev is not None:
                p_length += math.hypot(pt[0] - prev[0], pt[1] - prev[1])
            prev = pt

        candidate = {
            "path": wpts,
            "metrics": {
                "path_length": round(p_length, 3),
                "obstacle_count": 0,
                "min_clearance": 0.0,
                "avg_clearance": 0.0
            }
        }
        data_to_publish = {"candidate_paths": [candidate]}
        msg = BittlePathGPTJSON()
        msg.json_data = json.dumps(data_to_publish)
        self.gpt_path_pub.publish(msg)

        self.get_logger().info("[Scenario2] Published GPT route successfully.")
        self.scenario2_plan_done = True

    # -------------------- Scenarios 3–5: Hybrid Path Selection --------------------

    def _scenario345_gpt_call(self):
        path_summaries = []
        for i, cp in enumerate(self.candidate_paths):
            m = cp.get("metrics", {})
            p_len = round(m.get("path_length", 0), 3)
            obs_ct = m.get("obstacle_count", 0)
            min_clr = round(m.get("min_clearance", 0), 3)
            avg_clr = round(m.get("avg_clearance", 0), 3)
            path_summaries.append({
                "index": i,
                "path_length": p_len,
                "obstacle_count": obs_ct,
                "min_clearance": min_clr,
                "avg_clearance": avg_clr
            })

        data_for_gpt = {"candidate_paths": path_summaries}

        if self.scenario == 3:
            system_prompt = (
                "You are an LLM controlling a BittleBot in a 2D grid. "
                "Review the provided candidate_paths with their metrics, paying special attention to clearance metrics. "
                "If a candidate path has a very low minimum clearance combined with a high obstacle count and indications of long, skinny obstacles (which are considered toxic), avoid that goal and select another. "
                'Return strictly JSON: {"mode":"candidate_selection","selected_candidate":<index>}. '
                "Do not include any extra fields."
            )
        elif self.scenario == 4:
            system_prompt = (
                "You are an LLM controlling a BittleBot in a 2D grid. "
                "Some areas are 'crowded'. Evaluate the obstacle_count and clearance metrics for each path, "
                "and pick the path with fewer obstacles and higher clearance. "
                'Return strictly JSON: {"mode":"candidate_selection","selected_candidate":<index>}. '
                "Do not include any extra fields."
            )
        else:
            system_prompt = (
                "You are an LLM controlling a BittleBot in a 2D grid. "
                "The robot is running low on battery and must recharge immediately. "
                "Under normal circumstances, paths with high clearance from obstacles are preferred. However, in this scenario, a very low minimum clearance indicates that an obstacle is extremely close to the goal—this obstacle should be interpreted as a charging station. "
                "Based solely on the provided clearance and obstacle metrics, select the candidate path that best suggests the presence of a charging station. "
                'Return strictly JSON: {"mode":"candidate_selection","selected_candidate":<index>}. '
                "Do not include any extra fields."
            )

        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(data_for_gpt)}
                ],
                max_tokens=64,
                temperature=0.2
            )
            raw_reply = response.choices[0].message.content
            self.get_logger().info(f"[Hybrid GPT] => {raw_reply}")
            append_to_log({
                "prompt": {"system": system_prompt, "user": data_for_gpt},
                "response": raw_reply
            })
        except Exception as e:
            self.get_logger().error(f"[Hybrid GPT Error] => {e}")
            self._publish_command("stop")
            return

        mode, result = self._parse_gpt_reply(raw_reply)
        if mode == "candidate_selection":
            idx = result
            if 0 <= idx < len(self.candidate_paths):
                path = self.candidate_paths[idx].get("path", [])
                self.selected_candidate_path = path
                self.decision_locked = True
                self.path_index = 0
                self.get_logger().info(f"[Hybrid GPT] locked candidate {idx}, path length={len(path)}")
            else:
                self.get_logger().warn("GPT returned out-of-range index => Stopping.")
                self._publish_command("stop")
        else:
            self.get_logger().warn("GPT returned invalid JSON => Stopping.")
            self._publish_command("stop")

    # -------------------- Local Pure Pursuit --------------------

    def _compute_pure_pursuit(self, path):
        if not path or not self.robot_position:
            return "stop"
        rx, ry, rtheta = self.robot_position
        final_wp = path[-1]
        # If we are near the final waypoint, stop
        if math.hypot(final_wp[0] - rx, final_wp[1] - ry) < self.WP_THRESHOLD:
            return "stop"

        cx, cy = path[self.path_index]
        dist_to_wp = math.hypot(cx - rx, cy - ry)
        if dist_to_wp < self.WP_THRESHOLD:
            self.path_index += 1
            if self.path_index >= len(path):
                return "stop"

        target_idx = self.path_index
        best_idx = target_idx
        # Attempt to find a waypoint at least LOOKAHEAD_DIST away
        while target_idx < len(path):
            tx, ty = path[target_idx]
            if math.hypot(tx - rx, ty - ry) >= self.LOOKAHEAD_DIST:
                best_idx = target_idx
                break
            best_idx = target_idx
            target_idx += 1

        tx, ty = path[best_idx]
        dx = tx - rx
        dy = ty - ry
        desired_theta = math.atan2(dy, dx)
        angle_error = self._normalize_angle(desired_theta - rtheta)

        if abs(angle_error) > self.ANGLE_THRESHOLD:
            return "turn_right" if angle_error > 0 else "turn_left"
        else:
            return "move_forward"

    # -------------------- Utilities --------------------

    def _normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _parse_gpt_reply(self, raw):
        try:
            data = json.loads(raw.strip())
            mode = data.get("mode", None)
            if mode == "candidate_selection":
                idx = data.get("selected_candidate", -1)
                return ("candidate_selection", idx)
            elif mode == "full_control":
                cmd = data.get("command", "stop")
                return ("full_control", cmd)
            else:
                self.get_logger().warn("Reply missing recognized mode => stop")
                return ("full_control", "stop")
        except:
            self.get_logger().error("Invalid GPT JSON => stop")
            return ("full_control", "stop")

    def _parse_gpt_waypoints(self, raw_text):
        """
        Expects:
          {"mode":"waypoint_plan","waypoints":[[x1,y1],[x2,y2],...]}
        """
        try:
            data = json.loads(raw_text.strip())
            if data.get("mode", "") == "waypoint_plan":
                wpts = data.get("waypoints", [])
                valid_wpts = []
                for w in wpts:
                    if isinstance(w, list) and len(w) == 2:
                        valid_wpts.append([float(w[0]), float(w[1])])
                return ("waypoint_plan", valid_wpts)
            else:
                return ("invalid", [])
        except Exception as e:
            self.get_logger().error(f"_parse_gpt_waypoints error => {e}")
            return ("invalid", [])

    def _publish_command(self, cmd: str):
        if cmd not in self.SUPPORTED_COMMANDS:
            cmd = "stop"
        msg = String()
        msg.data = json.dumps({"command": cmd})
        self.cmd_pub.publish(msg)
        self.get_logger().info(f"[publish_command] => {cmd}")

    def _round_list(self, lst, digits=3):
        if not lst:
            return lst
        return [round(x, digits) for x in lst]

    def _is_near_goal(self, threshold=0.15):
        if not self.robot_position or not self.goal_position:
            return False
        dx = self.goal_position[0] - self.robot_position[0]
        dy = self.goal_position[1] - self.robot_position[1]
        return math.hypot(dx, dy) < threshold

    # -------------------- Final Collision Check --------------------
    def _check_path_collisions(self, waypoints, obstacles, clearance=0.0):
        """
        Sample each segment between waypoints. For each sample point,
        check distance to each inflated obstacle bounding box.
        If the distance is < clearance, collision => return False.
        """
        num_samples_per_segment = 10
        for i in range(len(waypoints) - 1):
            x1, y1 = waypoints[i]
            x2, y2 = waypoints[i + 1]
            for s in range(num_samples_per_segment + 1):
                t = s / float(num_samples_per_segment)
                px = x1 + t * (x2 - x1)
                py = y1 + t * (y2 - y1)
                # Check each obstacle
                for obs in obstacles:
                    cx, cy = obs["center"]
                    w, h = obs["size"]
                    if self._distance_point_to_rect(px, py, cx, cy, w, h) < clearance:
                        return False
        return True

    def _distance_point_to_rect(self, px, py, cx, cy, w, h):
        """
        Returns the Euclidean distance from point (px, py) to
        the axis-aligned rectangle whose center is (cx, cy)
        and width/height is w, h.
        """
        x_min = cx - w / 2.0
        x_max = cx + w / 2.0
        y_min = cy - h / 2.0
        y_max = cy + h / 2.0

        # Horizontal distance
        if px < x_min:
            dx = x_min - px
        elif px > x_max:
            dx = px - x_max
        else:
            dx = 0.0

        # Vertical distance
        if py < y_min:
            dy = y_min - py
        elif py > y_max:
            dy = py - y_max
        else:
            dy = 0.0

        return math.hypot(dx, dy)


def main(args=None):
    rclpy.init(args=args)
    node = LLMReasoning()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("LLMReasoning interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
