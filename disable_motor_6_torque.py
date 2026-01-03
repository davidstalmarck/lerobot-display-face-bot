#!/usr/bin/env python3
"""
Disable Motor 6 Torque for Manual Inspection
This allows you to manually rotate the motor to find the blockage
"""

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
import time

PORT = "/dev/ttyACM0"

print("=" * 70)
print("MOTOR 6 - TORQUE DISABLE FOR MANUAL INSPECTION")
print("=" * 70)
print("\nThis will disable motor 6 torque so you can manually check")
print("for the mechanical blockage preventing upward rotation.")
print("\nPROBLEM: Motor cannot rotate above position ~1600")
print("=" * 70)

config = FeetechMotorsBusConfig(
    port=PORT,
    motors={"gripper": [6, "sts3215"]}
)

bus = FeetechMotorsBus(config)
bus.connect()

try:
    print("\nCurrent status:")
    pos = bus.read("Present_Position", ["gripper"])[0]
    load = bus.read("Present_Load", ["gripper"])[0]
    print(f"  Position: {pos}")
    print(f"  Load: {load}")

    print("\nDisabling torque on motor 6...")
    bus.write("Torque_Enable", 0, ["gripper"])
    print("✓ Torque DISABLED")

    print("\n" + "=" * 70)
    print("MANUAL INSPECTION INSTRUCTIONS:")
    print("=" * 70)
    print("""
Motor 6 torque is now OFF. You can manually rotate it.

WHAT TO DO:
1. Gently try to rotate motor 6 clockwise (upward/higher values)
2. Feel for what's blocking it around position 1600
3. Check for:
   - Display cables wrapping around the gripper shaft
   - Gripper fingers hitting the display mount
   - Parts colliding or rubbing
   - Screws or brackets in the way

LOOK FOR:
- Wires/cables catching on rotation
- Mechanical interference between parts
- Misaligned mounting brackets
- Gripper assembly hitting the head/display

This script will run for 60 seconds, monitoring position.
The torque will automatically re-enable after 60 seconds.

Press Ctrl+C to stop early and re-enable torque.
    """)

    # Monitor for 60 seconds
    for i in range(60):
        try:
            pos = bus.read("Present_Position", ["gripper"])[0]
            print(f"Position: {pos}   (time: {i+1}s/60s)", end="\r")
            time.sleep(1)
        except:
            print(f"\nCommunication error at {i+1}s")
            time.sleep(1)

    print("\n\n60 seconds elapsed. Re-enabling torque...")

except KeyboardInterrupt:
    print("\n\nStopped by user. Re-enabling torque...")

except Exception as e:
    print(f"\nError: {e}")

finally:
    try:
        bus.write("Torque_Enable", 1, ["gripper"])
        print("✓ Torque re-enabled")
    except:
        print("⚠ Could not re-enable torque - motor may still be disabled")

    bus.disconnect()
    print("\nDisconnected")
    print("=" * 70)