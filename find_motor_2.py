#!/usr/bin/env python3
"""Try to find and communicate with motor 2"""

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
import time

PORT = "/dev/ttyACM0"

print("=" * 70)
print("Attempting to find Motor 2 (shoulder_lift)")
print("=" * 70)

# Try with just motor 2
print("\nTest 1: Trying to connect to motor 2 only...")
try:
    config = FeetechMotorsBusConfig(
        port=PORT,
        motors={
            "shoulder_lift": [2, "sts3215"],
        }
    )

    bus = FeetechMotorsBus(config)
    bus.connect()

    print("✅ Connected to motor 2!")

    # Try to read position
    print("\nReading motor 2 position...")
    pos = bus.read("Present_Position", ["shoulder_lift"])
    print(f"Position: {pos}")

    # Try to read voltage
    voltage = bus.read("Present_Voltage", ["shoulder_lift"])
    print(f"Voltage: {voltage}V")

    # Try to read temperature
    temp = bus.read("Present_Temperature", ["shoulder_lift"])
    print(f"Temperature: {temp}°C")

    # Try to enable torque and move it
    print("\nEnabling torque...")
    bus.write("Torque_Enable", 1, ["shoulder_lift"])

    print("Attempting to move to 2048...")
    bus.write("Goal_Position", [2048], ["shoulder_lift"])
    time.sleep(2)

    pos_after = bus.read("Present_Position", ["shoulder_lift"])
    print(f"Position after move: {pos_after}")

    bus.disconnect()
    print("\n✅ Motor 2 is working!")

except Exception as e:
    print(f"\n❌ Error connecting to motor 2: {e}")
    import traceback
    traceback.print_exc()

# Try scanning all possible motor IDs
print("\n" + "=" * 70)
print("Test 2: Scanning for all motors on the bus...")
print("=" * 70)

for motor_id in range(1, 7):
    try:
        config = FeetechMotorsBusConfig(
            port=PORT,
            motors={
                f"motor_{motor_id}": [motor_id, "sts3215"],
            }
        )

        bus = FeetechMotorsBus(config)
        bus.connect()

        pos = bus.read("Present_Position", [f"motor_{motor_id}"])
        print(f"Motor {motor_id}: Found! Position = {pos}")

        bus.disconnect()
    except Exception as e:
        print(f"Motor {motor_id}: Not found or error - {e}")

print("\n" + "=" * 70)
print("Scan complete!")
print("=" * 70)
