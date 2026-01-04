#!/usr/bin/env python3
"""Move all motors to neutral position (2048) with proper torque and more time"""

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
import time

PORT = "/dev/ttyACM0"
SPEED = 100  # Slower speed for better control

print("=" * 70)
print("Moving robot to STRAIGHT/NEUTRAL position (SLOW & STEADY)")
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

# Configure motors with proper settings
print("Configuring motors...")
bus.write("Mode", 0)  # Position control mode
bus.write("P_Coefficient", 16)
bus.write("I_Coefficient", 0)
bus.write("D_Coefficient", 32)
bus.write("Lock", 0)
bus.write("Maximum_Acceleration", 254)
bus.write("Acceleration", SPEED)

# IMPORTANT: Enable torque AFTER setting all other parameters
print("Enabling torque...")
bus.write("Torque_Enable", 1)

# Give it a moment to engage
time.sleep(0.5)

# Move each motor individually with progress tracking
motors_to_move = [
    ("shoulder_pan", 1),
    ("shoulder_lift", 2),
    ("elbow_flex", 3),
    ("wrist_flex", 4),
    ("wrist_roll", 5),
    ("gripper", 6),
]

print("\nMoving motors to 2048 one by one...")
print("-" * 70)

results = {}

for motor_name, motor_id in motors_to_move:
    print(f"\nMotor {motor_id} ({motor_name}):")

    try:
        # Read current position
        pos_start = bus.read("Present_Position", [motor_name])[0]
        print(f"  Start position: {pos_start}")

        # Send goal position
        bus.write("Goal_Position", [2048], [motor_name])
        print(f"  Moving to 2048...")

        # Wait and monitor
        for i in range(6):  # Monitor for 6 seconds
            time.sleep(1)
            try:
                pos_current = bus.read("Present_Position", [motor_name])[0]
                diff = abs(pos_current - 2048)
                print(f"    {i+1}s: pos={pos_current}, diff={diff}")

                # If close enough, move on
                if diff < 20:
                    break
            except Exception as e:
                print(f"    {i+1}s: Communication error - {e}")
                break

        # Final position
        pos_final = bus.read("Present_Position", [motor_name])[0]
        offset = pos_final - 2048
        results[motor_name] = pos_final

        if abs(offset) < 20:
            status = "✅ SUCCESS"
        elif abs(offset) < 100:
            status = "⚠️  Close"
        else:
            status = "❌ Failed"

        print(f"  Final: {pos_final} (offset: {offset:+d}) {status}")

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        results[motor_name] = "ERROR"

# Summary
print("\n" + "=" * 70)
print("FINAL RESULTS:")
print("=" * 70)

for motor_name, pos in results.items():
    if pos == "ERROR":
        print(f"{motor_name:15s}: ERROR - Communication failed")
    else:
        offset = pos - 2048
        if abs(offset) < 20:
            status = "✅ PERFECT"
        elif abs(offset) < 50:
            status = "✅ Good"
        elif abs(offset) < 100:
            status = "⚠️  Close"
        else:
            status = "❌ Failed"

        print(f"{motor_name:15s}: {pos:4d} (offset: {offset:+4d}) {status}")

print("=" * 70)

bus.disconnect()
print("\nDone!")
