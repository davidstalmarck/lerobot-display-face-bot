#!/usr/bin/env python3
"""Move to straight position and HOLD it - keeps torque enabled"""

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
import time

PORT = "/dev/ttyACM0"
SPEED = 100

print("=" * 70)
print("HOLD STRAIGHT POSITION - Press Ctrl+C to stop")
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

print("Configuring motors...")
bus.write("Mode", 0)  # Position control mode
bus.write("P_Coefficient", 16)
bus.write("I_Coefficient", 0)
bus.write("D_Coefficient", 32)
bus.write("Lock", 0)
bus.write("Maximum_Acceleration", 254)
bus.write("Acceleration", SPEED)

# Enable torque
print("Enabling torque...")
bus.write("Torque_Enable", 1)
time.sleep(0.5)

# Move to neutral position
print("\nMoving to straight position (2048)...")
motors_list = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

try:
    # Send all motors to 2048
    bus.write("Goal_Position", [2048] * 5, motors_list)

    print("Moving... (waiting 3 seconds)")
    time.sleep(3)

    # Check positions
    print("\nFinal positions:")
    for motor in motors_list:
        try:
            pos = bus.read("Present_Position", [motor])[0]
            offset = pos - 2048
            print(f"  {motor:15s}: {pos:4d} (offset: {offset:+4d})")
        except:
            pass

    print("\n" + "=" * 70)
    print("HOLDING POSITION - Robot will stay straight")
    print("Press Ctrl+C to stop and disable motors")
    print("=" * 70)

    # Hold position indefinitely
    counter = 0
    while True:
        time.sleep(5)
        counter += 1

        # Re-send position commands periodically to ensure holding
        try:
            bus.write("Goal_Position", [2048] * 5, motors_list)
        except:
            pass

        # Status update every 30 seconds
        if counter % 6 == 0:
            print(f"Still holding... ({counter * 5} seconds elapsed)")

except KeyboardInterrupt:
    print("\n\nStopping...")
    print("Disabling torque...")
    bus.write("Torque_Enable", 0)
    bus.disconnect()
    print("Motors disabled. Robot will now be loose.")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    bus.disconnect()
