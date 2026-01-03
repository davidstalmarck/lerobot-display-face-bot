#!/usr/bin/env python3
"""
Diagnose Motor 6 - Disable torque to check for mechanical issues
"""

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
import time

PORT = "/dev/ttyACM0"

print("=" * 70)
print("Motor 6 Diagnostic - Torque Disable Mode")
print("=" * 70)
print("\nThis will disable motor 6 torque so you can manually check for")
print("mechanical binding or obstructions.")
print("\nPress Ctrl+C when done checking")
print("=" * 70)

config = FeetechMotorsBusConfig(
    port=PORT,
    motors={
        "gripper": [6, "sts3215"],
    }
)

bus = FeetechMotorsBus(config)
bus.connect()

try:
    print("\nDisabling torque on motor 6...")
    bus.write("Torque_Enable", 0, ["gripper"])
    print("✓ Torque disabled!")
    print("\nYou can now manually rotate motor 6.")
    print("It should feel smooth with no binding or resistance.")
    print("If it feels stuck, there's a mechanical problem.\n")

    # Keep reading position while torque is off
    while True:
        try:
            pos = bus.read("Present_Position", ["gripper"])[0]
            print(f"Current position: {pos}   ", end="\r")
            time.sleep(0.5)
        except Exception as e:
            print(f"\nCommunication error: {e}")
            time.sleep(1)

except KeyboardInterrupt:
    print("\n\nRe-enabling torque...")
    try:
        bus.write("Torque_Enable", 1, ["gripper"])
        print("✓ Torque re-enabled")
    except:
        print("⚠ Could not re-enable torque - motor may still be off")

finally:
    bus.disconnect()
    print("Disconnected")