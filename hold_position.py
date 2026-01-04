#!/usr/bin/env python3
"""Hold robot in straight position - motors stay powered and won't fall"""

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
import time
import sys

PORT = "/dev/ttyACM0"
SPEED = 100

print("=" * 70)
print("HOLD POSITION - Robot will stand straight and stay powered")
print("Press Ctrl+C to stop")
print("=" * 70)

# Configure only working motors (exclude gripper motor 6)
config = FeetechMotorsBusConfig(
    port=PORT,
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

    print("\nConfiguring motors...")
    bus.write("Mode", 0)
    bus.write("P_Coefficient", 16)
    bus.write("I_Coefficient", 0)
    bus.write("D_Coefficient", 32)
    bus.write("Lock", 0)
    bus.write("Maximum_Acceleration", 254)
    bus.write("Acceleration", SPEED)
    bus.write("Torque_Enable", 1)

    print("Moving to straight position (2048)...")
    motors_list = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    bus.write("Goal_Position", [2048, 2048, 2048, 2048, 2048], motors_list)

    time.sleep(3)

    # Show positions
    print("\nCurrent positions:")
    for motor in motors_list:
        pos = bus.read("Present_Position", [motor])[0]
        print(f"  {motor:15s}: {pos}")

    print("\n" + "=" * 70)
    print("✅ HOLDING POSITION")
    print("=" * 70)
    print("Motors are now locked and holding position.")
    print("The robot should NOT fall down.")
    print("\nPress Ctrl+C to release motors and exit.")
    print("=" * 70 + "\n")

    # Keep running and periodically refresh position
    counter = 0
    while True:
        time.sleep(10)
        counter += 1

        # Reaffirm position every 10 seconds
        bus.write("Goal_Position", [2048, 2048, 2048, 2048, 2048], motors_list)

        if counter % 6 == 0:  # Every minute
            print(f"Still holding... ({counter * 10}s elapsed)")

except KeyboardInterrupt:
    print("\n\n" + "=" * 70)
    print("Stopping - Releasing motors...")
    print("=" * 70)
    try:
        bus.write("Torque_Enable", 0)
    except:
        pass
    bus.disconnect()
    print("Motors released. Robot may fall now.")
    sys.exit(0)

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    try:
        bus.disconnect()
    except:
        pass
    sys.exit(1)
