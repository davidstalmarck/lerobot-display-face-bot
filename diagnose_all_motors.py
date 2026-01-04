#!/usr/bin/env python3
"""Comprehensive diagnostics for ALL motors (1-6)"""

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
import time

PORT = "/dev/ttyACM0"

print("=" * 70)
print("COMPREHENSIVE MOTOR DIAGNOSTICS - ALL MOTORS (1-6)")
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

# Enable torque and set basic config for all motors
print("\nConfiguring all motors for testing...")
bus.write("Mode", 0)  # Position control mode
bus.write("P_Coefficient", 16)
bus.write("I_Coefficient", 0)
bus.write("D_Coefficient", 32)
bus.write("Lock", 0)
bus.write("Maximum_Acceleration", 254)
bus.write("Acceleration", 150)  # Medium speed for testing
bus.write("Torque_Enable", 1)  # ENABLE TORQUE
print("✅ All motors configured and torque ENABLED")

motors_info = [
    ("shoulder_pan", 1),
    ("shoulder_lift", 2),
    ("elbow_flex", 3),
    ("wrist_flex", 4),
    ("wrist_roll", 5),
    ("gripper", 6),
]

results = {}

for motor_name, motor_id in motors_info:
    print(f"\n{'=' * 70}")
    print(f"MOTOR {motor_id}: {motor_name}")
    print("=" * 70)

    try:
        # Read current status
        print("\n1. CURRENT STATUS:")
        pos = bus.read("Present_Position", [motor_name])[0]
        print(f"   Position: {pos}")

        voltage = bus.read("Present_Voltage", [motor_name])[0]
        print(f"   Voltage: {voltage/10:.1f}V")

        temp = bus.read("Present_Temperature", [motor_name])[0]
        print(f"   Temperature: {temp}°C")

        load = bus.read("Present_Load", [motor_name])[0]
        print(f"   Load: {load}")

        current = bus.read("Present_Current", [motor_name])[0]
        print(f"   Current: {current}mA")

        # Read limits
        print("\n2. LIMITS & SAFETY:")
        min_angle = bus.read("Min_Angle_Limit", [motor_name])[0]
        print(f"   Min Angle Limit: {min_angle}")

        max_angle = bus.read("Max_Angle_Limit", [motor_name])[0]
        print(f"   Max Angle Limit: {max_angle}")

        max_torque = bus.read("Max_Torque_Limit", [motor_name])[0]
        print(f"   Max Torque Limit: {max_torque}")

        torque_limit = bus.read("Torque_Limit", [motor_name])[0]
        print(f"   Current Torque Limit: {torque_limit}")

        # Check configuration
        print("\n3. CONFIGURATION:")
        mode = bus.read("Mode", [motor_name])[0]
        print(f"   Mode: {mode} (0=Position, 1=Speed, 3=Step)")

        torque_enable = bus.read("Torque_Enable", [motor_name])[0]
        print(f"   Torque Enabled: {torque_enable}")

        # Movement test
        print("\n4. MOVEMENT TEST:")
        print(f"   Starting position: {pos}")

        # Test 1: Move DOWN (to lower value)
        test_pos_down = max(0, pos - 300)
        print(f"   → Testing DOWN movement to {test_pos_down}...")
        bus.write("Goal_Position", [test_pos_down], [motor_name])
        time.sleep(2)
        pos_after_down = bus.read("Present_Position", [motor_name])[0]
        moved_down = pos_after_down - pos
        print(f"     After: {pos_after_down} (moved: {moved_down:+d})")

        # Test 2: Move UP (to higher value)
        test_pos_up = min(4095, pos + 600)
        print(f"   → Testing UP movement to {test_pos_up}...")
        bus.write("Goal_Position", [test_pos_up], [motor_name])
        time.sleep(2)
        pos_after_up = bus.read("Present_Position", [motor_name])[0]
        moved_up = pos_after_up - pos_after_down
        print(f"     After: {pos_after_up} (moved: {moved_up:+d})")

        # Test 3: Return to center
        print(f"   → Returning to center (2048)...")
        bus.write("Goal_Position", [2048], [motor_name])
        time.sleep(2)
        pos_final = bus.read("Present_Position", [motor_name])[0]
        print(f"     Final: {pos_final}")

        # Analysis
        print("\n5. DIAGNOSIS:")
        can_move_down = abs(moved_down) > 50
        can_move_up = abs(moved_up) > 50

        status = "UNKNOWN"
        if can_move_down and can_move_up:
            print("   ✅ Motor can move in BOTH directions - WORKING PROPERLY")
            status = "OK"
        elif can_move_down and not can_move_up:
            print("   ❌ Motor can move DOWN but NOT UP - MECHANICAL BLOCK or LIMIT")
            status = "DOWN_ONLY"
        elif not can_move_down and can_move_up:
            print("   ❌ Motor can move UP but NOT DOWN - MECHANICAL BLOCK or LIMIT")
            status = "UP_ONLY"
        else:
            print("   ❌ Motor CANNOT move in either direction - SERIOUS ISSUE")
            status = "STUCK"

        # Additional warnings
        if pos < 500 or pos_final < 500:
            print("   ⚠️  WARNING: Motor at very low position - may be hitting lower limit")
        if pos > 3595 or pos_final > 3595:
            print("   ⚠️  WARNING: Motor at very high position - may be hitting upper limit")
        if temp > 50:
            print("   ⚠️  WARNING: Temperature elevated")
        if voltage < 60:  # voltage is in 0.1V units
            print("   ⚠️  WARNING: Voltage low")

        # Store results
        results[motor_name] = {
            'id': motor_id,
            'status': status,
            'position': pos_final,
            'can_move_down': can_move_down,
            'can_move_up': can_move_up,
            'temperature': temp,
            'voltage': voltage/10,
        }

    except Exception as e:
        print(f"\n   ❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        results[motor_name] = {
            'id': motor_id,
            'status': 'ERROR',
            'error': str(e)
        }

    time.sleep(0.5)

# Summary
print("\n" + "=" * 70)
print("SUMMARY - ALL MOTORS")
print("=" * 70)

print("\nMotor Status:")
for motor_name, result in results.items():
    motor_id = result['id']
    status = result['status']

    if status == "OK":
        icon = "✅"
    elif status == "ERROR":
        icon = "💥"
    else:
        icon = "❌"

    if status == "ERROR":
        print(f"  {icon} Motor {motor_id} ({motor_name}): {status} - {result.get('error', 'Unknown error')}")
    else:
        pos = result.get('position', 'N/A')
        temp = result.get('temperature', 'N/A')
        voltage = result.get('voltage', 'N/A')
        print(f"  {icon} Motor {motor_id} ({motor_name}): {status} | Pos: {pos} | Temp: {temp}°C | V: {voltage}V")

print("\n" + "=" * 70)
print("RECOMMENDATIONS:")
print("=" * 70)

# Count issues
ok_count = sum(1 for r in results.values() if r['status'] == 'OK')
issue_count = len(results) - ok_count

if ok_count == len(results):
    print("\n✅ All motors are working properly!")
else:
    print(f"\n⚠️  {issue_count} motor(s) have issues:")

    for motor_name, result in results.items():
        if result['status'] != 'OK':
            print(f"\n{motor_name} (Motor {result['id']}):")

            if result['status'] == 'UP_ONLY':
                print("  - Can only move UP, not DOWN")
                print("  - Check for physical obstruction blocking downward movement")
                print("  - Check if at minimum angle limit")
                print("  - Manually move motor (power off) to verify it's not jammed")

            elif result['status'] == 'DOWN_ONLY':
                print("  - Can only move DOWN, not UP")
                print("  - Check for physical obstruction blocking upward movement")
                print("  - Check if at maximum angle limit")
                print("  - Manually move motor (power off) to verify it's not jammed")

            elif result['status'] == 'STUCK':
                print("  - Motor cannot move at all")
                print("  - Check power connections")
                print("  - Check for mechanical jam")
                print("  - Try manually moving motor with power off")

            elif result['status'] == 'ERROR':
                print(f"  - Communication error: {result.get('error', 'Unknown')}")
                print("  - Check motor connections")
                print("  - Verify motor ID is correct")

print("\n" + "=" * 70)

bus.disconnect()
print("\nDone!")