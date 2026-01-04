#!/usr/bin/env python3
"""Test each motor individually to see which ones respond"""

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
import time

PORT = "/dev/ttyACM0"
SPEED = 150

# Configure all motors
config = FeetechMotorsBusConfig(
    port=PORT,
    motors={
        "shoulder_pan": [1, "sts3215"],
        # "shoulder_lift": [2, "sts3215"],  # DISABLED
        "elbow_flex": [3, "sts3215"],
        "wrist_flex": [4, "sts3215"],
        "wrist_roll": [5, "sts3215"],
        "gripper": [6, "sts3215"],
    }
)

bus = FeetechMotorsBus(config)
bus.connect()

# Configure motors
bus.write("Mode", 0)
bus.write("P_Coefficient", 16)
bus.write("I_Coefficient", 0)
bus.write("D_Coefficient", 32)
bus.write("Lock", 0)
bus.write("Maximum_Acceleration", 254)
bus.write("Acceleration", SPEED)
bus.write("Torque_Enable", 1)

print("=" * 70)
print("Testing each motor individually")
print("=" * 70)

# Test each motor one by one
motors_to_test = [
    ("shoulder_pan", 1),
    ("elbow_flex", 3),
    ("wrist_flex", 4),
    ("wrist_roll", 5),
    ("gripper", 6),
]

for motor_name, motor_id in motors_to_test:
    print(f"\n--- Testing Motor {motor_id}: {motor_name} ---")

    # Read current position
    pos_before = bus.read("Present_Position", [motor_name])[0]
    print(f"Position before: {pos_before}")

    # Try moving to 2048
    print(f"Moving to 2048...")
    bus.write("Goal_Position", [2048], [motor_name])
    time.sleep(2)

    # Read new position
    pos_after = bus.read("Present_Position", [motor_name])[0]
    print(f"Position after:  {pos_after}")

    # Check if it moved
    if abs(pos_after - 2048) < 50:
        print(f"✅ SUCCESS - Motor reached target!")
    elif abs(pos_after - pos_before) > 50:
        print(f"⚠️  PARTIAL - Motor moved but didn't reach target")
    else:
        print(f"❌ FAILED - Motor didn't move")

    time.sleep(1)

print("\n" + "=" * 70)
print("Test complete!")
print("=" * 70)

bus.disconnect()
