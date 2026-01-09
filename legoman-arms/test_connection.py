#!/usr/bin/env python3
"""
Test connection to motors - diagnose communication issues
"""

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
import time

print("=" * 70)
print("TESTING MOTOR CONNECTION")
print("=" * 70)

# Test each motor individually
for motor_id in range(1, 7):
    print(f"\n--- Testing Motor {motor_id} ---")

    config = FeetechMotorsBusConfig(
        port="/dev/ttyACM0",
        motors={f"motor_{motor_id}": [motor_id, "sts3215"]}
    )

    try:
        bus = FeetechMotorsBus(config)
        bus.connect()

        # Try to read position
        pos = bus.read("Present_Position", [f"motor_{motor_id}"])[0]
        print(f"  ✅ Motor {motor_id}: Position = {pos}")

        bus.disconnect()

    except Exception as e:
        print(f"  ❌ Motor {motor_id}: FAILED - {e}")

    time.sleep(0.2)

print("\n" + "=" * 70)
print("TESTING ALL MOTORS TOGETHER (1-5)")
print("=" * 70)

config = FeetechMotorsBusConfig(
    port="/dev/ttyACM0",
    motors={
        "shoulder_pan": [1, "sts3215"],
        "shoulder_lift": [2, "sts3215"],
        "elbow_flex": [3, "sts3215"],
        "wrist_flex": [4, "sts3215"],
        "wrist_roll": [5, "sts3215"],
    }
)

try:
    bus = FeetechMotorsBus(config)
    bus.connect()

    positions = bus.read("Present_Position")
    print("\n✅ ALL MOTORS RESPONDING:")
    for i, (name, pos) in enumerate(zip(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"], positions)):
        print(f"  Motor {i+1} ({name}): {pos}")

    bus.disconnect()

except Exception as e:
    print(f"\n❌ FAILED: {e}")
    print("\nTroubleshooting:")
    print("1. Check power to motors")
    print("2. Check USB cable connection")
    print("3. Verify /dev/ttyACM0 is correct port")
    print("4. Try unplugging and replugging USB")
    print("5. Check if motors are powered on")

print("\n" + "=" * 70)
