#!/usr/bin/env python3
"""Comprehensive motor diagnostics for motors 2 and 6"""

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
import time

PORT = "/dev/ttyACM0"

print("=" * 70)
print("MOTOR DIAGNOSTICS - Checking motors 2 and 6")
print("=" * 70)

config = FeetechMotorsBusConfig(
    port=PORT,
    motors={
        "shoulder_lift": [2, "sts3215"],
        "gripper": [6, "sts3215"],
    }
)

bus = FeetechMotorsBus(config)
bus.connect()

motors_to_test = ["shoulder_lift", "gripper"]

for motor_name in motors_to_test:
    motor_id = 2 if motor_name == "shoulder_lift" else 6
    print(f"\n{'=' * 70}")
    print(f"MOTOR {motor_id}: {motor_name}")
    print("=" * 70)

    try:
        # Read all important parameters
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

        print("\n2. LIMITS & SAFETY:")
        try:
            min_angle = bus.read("Min_Angle_Limit", [motor_name])[0]
            print(f"   Min Angle Limit: {min_angle}")
        except:
            print(f"   Min Angle Limit: Could not read")

        try:
            max_angle = bus.read("Max_Angle_Limit", [motor_name])[0]
            print(f"   Max Angle Limit: {max_angle}")
        except:
            print(f"   Max Angle Limit: Could not read")

        try:
            max_torque = bus.read("Max_Torque_Limit", [motor_name])[0]
            print(f"   Max Torque Limit: {max_torque}")
        except:
            print(f"   Max Torque Limit: Could not read")

        try:
            torque_limit = bus.read("Torque_Limit", [motor_name])[0]
            print(f"   Current Torque Limit: {torque_limit}")
        except:
            print(f"   Current Torque Limit: Could not read")

        print("\n3. CONFIGURATION:")
        mode = bus.read("Mode", [motor_name])[0]
        print(f"   Mode: {mode} (0=Position, 1=Speed, 3=Step)")

        torque_enable = bus.read("Torque_Enable", [motor_name])[0]
        print(f"   Torque Enabled: {torque_enable}")

        try:
            p_coef = bus.read("P_Coefficient", [motor_name])[0]
            print(f"   P Coefficient: {p_coef}")
        except:
            print(f"   P Coefficient: Could not read")

        try:
            d_coef = bus.read("D_Coefficient", [motor_name])[0]
            print(f"   D Coefficient: {d_coef}")
        except:
            print(f"   D Coefficient: Could not read")

        print("\n4. MOVEMENT TEST:")
        print(f"   Current position: {pos}")

        # Try moving DOWN first (to lower value)
        test_pos_down = max(0, pos - 200)
        print(f"   Attempting to move DOWN to {test_pos_down}...")
        bus.write("Goal_Position", [test_pos_down], [motor_name])
        time.sleep(2)
        pos_after_down = bus.read("Present_Position", [motor_name])[0]
        print(f"   Position after DOWN: {pos_after_down} (moved: {pos_after_down - pos:+d})")

        # Try moving UP (to higher value)
        test_pos_up = min(4095, pos + 400)
        print(f"   Attempting to move UP to {test_pos_up}...")
        bus.write("Goal_Position", [test_pos_up], [motor_name])
        time.sleep(2)
        pos_after_up = bus.read("Present_Position", [motor_name])[0]
        print(f"   Position after UP: {pos_after_up} (moved: {pos_after_up - pos_after_down:+d})")

        # Analysis
        print("\n5. DIAGNOSIS:")
        can_move_down = abs(pos_after_down - pos) > 50
        can_move_up = abs(pos_after_up - pos_after_down) > 50

        if can_move_down and can_move_up:
            print("   ✅ Motor can move in BOTH directions")
        elif can_move_down and not can_move_up:
            print("   ❌ Motor can move DOWN but NOT UP - likely MECHANICAL BLOCK or LIMIT")
        elif not can_move_down and can_move_up:
            print("   ❌ Motor can move UP but NOT DOWN - likely MECHANICAL BLOCK or LIMIT")
        else:
            print("   ❌ Motor CANNOT move in either direction - SERIOUS ISSUE")

        if pos_after_up < 1000:
            print("   ⚠️  WARNING: Motor stuck at very low position - check for:")
            print("      - Physical obstruction preventing upward movement")
            print("      - Min/Max angle limits too restrictive")
            print("      - Motor gear damage")

    except Exception as e:
        print(f"\n   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("RECOMMENDATIONS:")
print("=" * 70)
print("""
Based on the diagnosis:

1. If motors can't move UP:
   - Check for physical obstructions (cables, parts colliding)
   - Check Min/Max_Angle_Limit settings
   - Manually move the motor (power off) to verify it's not jammed

2. If voltage is low (< 6V):
   - Check power supply
   - Check power cable connections

3. If current is very high:
   - Motor is straining against something
   - Reduce load or remove obstruction

4. If temperature is high (> 60°C):
   - Motor is overworking
   - Let it cool down
   - Check for mechanical binding
""")

bus.disconnect()
print("Done!")
