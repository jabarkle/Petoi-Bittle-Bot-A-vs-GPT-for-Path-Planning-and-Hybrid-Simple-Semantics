#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import json

class BittleCommandExecutor(Node):
    def __init__(self, port='/dev/ttyS0'):
        super().__init__('bittle_command_executor')
        # Subscribe to high-level commands from the LLM_Reasoning node
        self.subscription = self.create_subscription(
            String,
            '/bittle_cmd',
            self.cmd_callback,
            10
        )

        # Initialize serial communication to the BittleBot's motor controller.
        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=115200,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=1
            )
            self.get_logger().info("Serial port opened successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to open serial port: {e}")

        # Mapping from high-level commands to motor controller tokens
        self.command_map = {
            "move_forward": "kwkF",   # Walk Forward
            "move_backward": "kbk",   # Walk Backward
            "walk_left": "kwkL",      # Walk Left
            "walk_right": "kwkR",     # Walk Right
            "turn_left": "kvtL",      # Turn Left
            "turn_right": "kvtR",     # Turn Right
            "stop": "kbalance"        # Stop
        }
        self.get_logger().info("BittleCommandExecutor node started.")

    def cmd_callback(self, msg: String):
        self.get_logger().info(f"Received command message: {msg.data}")
        try:
            # Expect the message to be a JSON string like {"command": "move_forward"}
            command_data = json.loads(msg.data)
            command = command_data.get("command", None)
            if command is None:
                self.get_logger().warn("Command field not found in the received message.")
                return
            if command not in self.command_map:
                self.get_logger().warn(f"Received unknown command: {command}")
                return
            token = self.command_map[command]
            self.send_command(token)
        except Exception as e:
            self.get_logger().error(f"Error processing command message: {e}")

    def send_command(self, token: str):
        # Append newline to mark end of command (as expected by the motor controller)
        cmd_str = token + '\n'
        self.get_logger().info(f"Sending command token: {cmd_str.strip()}")
        try:
            self.ser.write(cmd_str.encode())
            # Optionally, you can also wait and log a response:
            # response = self.ser.readline().decode().strip()
            # self.get_logger().info(f"Received response: {response}")
        except Exception as e:
            self.get_logger().error(f"Error sending command: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = BittleCommandExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("BittleCommandExecutor node shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
