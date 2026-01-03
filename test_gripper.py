#!/usr/bin/env python3
"""
Safe Gripper (Motor 6) Rotation Test
IMPORTANT: Motor 6 has been having issues - this uses gentle movements
"""

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
import time

PORT = "/dev/ttyACM0"

print("=" * 70)
print("SAFE GRIPPER (Motor 6) Rotation Test")
print("=" * 70)
print("\nWARNING: Motor 6 has been failing - using careful diagnostics")
print()

config = FeetechMotorsBusConfig(
    port=PORT,
    motors={
        "gripper": [6, "sts3215"],
    }
)

bus = FeetechMotorsBus(config)
bus.connect()

print("Configuring gripper with SAFE settings...")
bus.write("Mode", 0, ["gripper"])
bus.write("P_Coefficient", 16, ["gripper"])
bus.write("I_Coefficient", 0, ["gripper"])
bus.write("D_Coefficient", 32, ["gripper"])
bus.write("Lock", 0, ["gripper"])
bus.write("Maximum_Acceleration", 100, ["gripper"])  # Reduced from 254
bus.write("Acceleration", 30, ["gripper"])  # Very slow: 30 instead of 100
bus.write("Torque_Enable", 1, ["gripper"])

time.sleep(0.5)

try:
    # Read current position and diagnostics
    print("\n" + "=" * 70)
    print("INITIAL DIAGNOSTICS")
    print("=" * 70)

    pos_start = bus.read("Present_Position", ["gripper"])[0]
    print(f"Current position: {pos_start}")

    try:
        voltage = bus.read("Present_Voltage", ["gripper"])[0]
        print(f"Voltage: {voltage / 10.0:.1f}V")
    except Exception as e:
        print(f"Could not read voltage: {e}")

    try:
        temp = bus.read("Present_Temperature", ["gripper"])[0]
        print(f"Temperature: {temp}°C")
    except Exception as e:
        print(f"Could not read temperature: {e}")

    try:
        load = bus.read("Present_Load", ["gripper"])[0]
        print(f"Load: {load} (negative=CCW, positive=CW)")
    except Exception as e:
        print(f"Could not read load: {e}")

    # SAFE incremental test - very small movements from current position
    print("\n" + "=" * 70)
    print("INCREMENTAL MOVEMENT TEST (small steps from current position)")
    print("=" * 70)

    test_offsets = [
        ("Tiny right (+100)", pos_start + 100),
        ("Back to start", pos_start),
        ("Tiny left (-100)", pos_start - 100),
        ("Back to start", pos_start),
        ("Small right (+200)", pos_start + 200),
        ("Back to start", pos_start),
    ]

    for name, target in test_offsets:
        print(f"\n--- {name} (target: {target}) ---")

        # Safety clamp
        if target < 1000 or target > 3000:
            print(f"  SKIPPED - Target {target} outside safe range (1000-3000)")
            continue

        bus.write("Goal_Position", [target], ["gripper"])

        # Monitor movement carefully
        for i in range(30):  # 3 seconds total, 0.1s intervals
            time.sleep(0.1)
            try:
                pos = bus.read("Present_Position", ["gripper"])[0]
                load = bus.read("Present_Load", ["gripper"])[0]
                diff = target - pos

                # Print every 5 iterations (0.5s)
                if i % 5 == 0:
                    print(f"  {i*0.1:.1f}s: pos={pos}, load={load:+4d}, diff={diff:+4d}")

                # Check if reached
                if abs(diff) < 20:
                    print(f"  SUCCESS - Reached {pos} in {i*0.1:.1f}s")
                    break

                # Check for stuck/high load
                if abs(load) > 500:
                    print(f"  WARNING - High load detected: {load}")

            except Exception as e:
                print(f"  ERROR at {i*0.1:.1f}s: {e}")
                print("  Motor may have failed - stopping test")
                break

        time.sleep(1)

except KeyboardInterrupt:
    print("\n\nStopped by user")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("\nDisconnecting...")
    bus.disconnect()
    print("Done!")
