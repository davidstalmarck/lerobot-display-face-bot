#!/usr/bin/env python3
"""Move all motors to neutral position (2048) to stand straight"""

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
import time

PORT = "/dev/ttyACM0"
SPEED = 150

print("=" * 70)
print("Moving robot to STRAIGHT/NEUTRAL position")
print("=" * 70)

# Configure all motors
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

# Move all motors to 2048 (neutral position)
print("\nMoving all motors to 2048 (neutral/straight)...")
motors_to_move = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

bus.write("Goal_Position", [2048] * 6, motors_to_move)

print("Moving... (waiting 3 seconds)")
time.sleep(3)

# Read and display final positions
print("\n" + "=" * 70)
print("Final positions:")
print("=" * 70)

for motor_name in motors_to_move:
    try:
        pos = bus.read("Present_Position", [motor_name])[0]
        offset = pos - 2048
        if abs(offset) < 10:
            status = "✅ PERFECT"
        elif abs(offset) < 50:
            status = "✅ Good"
        elif abs(offset) < 200:
            status = "⚠️  Close"
        else:
            status = "❌ Failed"

        print(f"{motor_name:15s}: {pos:4d} (offset: {offset:+4d}) {status}")
    except Exception as e:
        print(f"{motor_name:15s}: ERROR - {e}")

print("=" * 70)

bus.disconnect()
print("\nDone!")
