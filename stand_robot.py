#!/usr/bin/env python3
"""
Simple script to put the robot in standing/display position
Based on the neutral position from debug_position.py
"""

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
import time

PORT = "/dev/ttyACM0"
SPEED = 150

# Standing position - all motors at neutral
STANDING_POSITION = {
    "shoulder_pan": 2048,
    "shoulder_lift": 2048,
    "elbow_flex": 2048,
    "wrist_flex": 2048,
    "wrist_roll": 2048,
    "gripper": 2048,
}

print("=" * 70)
print("Moving robot to STANDING position")
print("=" * 70)

# Configure motors
config = FeetechMotorsBusConfig(
    port=PORT,
    motors={
        "shoulder_pan": [1, "sts3215"],
        "shoulder_lift": [2, "sts3215"],
        "elbow_flex": [3, "sts3215"],
        "wrist_flex": [4, "sts3215"],
        "wrist_roll": [5, "sts3215"],
        "gripper": [6, "sts3215"],
    }
)

bus = FeetechMotorsBus(config)
bus.connect()

# Configure motors
print("Configuring motors...")
bus.write("Mode", 0)
bus.write("P_Coefficient", 16)
bus.write("I_Coefficient", 0)
bus.write("D_Coefficient", 32)
bus.write("Lock", 0)
bus.write("Maximum_Acceleration", 254)
bus.write("Acceleration", SPEED)
bus.write("Torque_Enable", 1)

# Move to standing position
print("Moving to standing position (all motors at neutral 2048)...")
bus.write("Goal_Position",
          list(STANDING_POSITION.values()),
          list(STANDING_POSITION.keys()))

time.sleep(3)

# Read final positions
print("\nFinal positions:")
positions = bus.read("Present_Position")
motor_names = list(STANDING_POSITION.keys())

for motor_name, pos in zip(motor_names, positions):
    print(f"  {motor_name}: {pos}")

print("\n" + "=" * 70)
print("Robot is now in STANDING position!")
print("=" * 70)

bus.disconnect()
print("Done!")