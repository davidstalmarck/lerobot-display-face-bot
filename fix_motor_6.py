#!/usr/bin/env python3
"""
Motor 6 Fix Script - Attempts to resolve mechanical blockage
Tries multiple strategies to get the gripper rotating properly
"""

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
import time
import sys

PORT = "/dev/ttyACM0"

def print_status(bus, motor_name="gripper"):
    """Print current motor status"""
    try:
        pos = bus.read("Present_Position", [motor_name])[0]
        load = bus.read("Present_Load", [motor_name])[0]
        temp = bus.read("Present_Temperature", [motor_name])[0]
        voltage = bus.read("Present_Voltage", [motor_name])[0]
        print(f"  Pos: {pos:4d} | Load: {load:+5d} | Temp: {temp:2d}°C | Voltage: {voltage/10:.1f}V")
        return pos, load, temp
    except Exception as e:
        print(f"  ERROR reading status: {e}")
        return None, None, None

def slow_move_to(bus, target, motor_name="gripper", max_load=800):
    """
    Slowly move to target position, stopping if load gets too high
    Returns: (success, final_position, reason)
    """
    print(f"\nSlowly moving to {target}...")

    current_pos, current_load, _ = print_status(bus, motor_name)
    if current_pos is None:
        return False, None, "Can't read current position"

    # Command the position
    bus.write("Goal_Position", [target], [motor_name])

    # Monitor for 5 seconds
    for i in range(50):
        time.sleep(0.1)
        pos, load, temp = print_status(bus, motor_name)

        if pos is None:
            return False, current_pos, "Communication error"

        # Check if we reached target
        if abs(pos - target) < 20:
            print(f"✓ Reached target {pos}!")
            return True, pos, "Success"

        # Check for excessive load (mechanical resistance)
        if abs(load) > max_load:
            print(f"\n⚠ HIGH LOAD ({load}) - Stopping to prevent damage!")
            return False, pos, f"High load: {load}"

        # Check for overheating
        if temp > 60:
            print(f"\n⚠ OVERHEATING ({temp}°C) - Stopping!")
            return False, pos, f"Temperature: {temp}°C"

        # Check if stuck (not moving)
        if i > 10 and abs(pos - current_pos) < 5:
            print(f"\n⚠ Motor appears stuck at {pos}")
            return False, pos, "Not moving"

        current_pos = pos

    print(f"\nTimeout - only reached {current_pos}")
    return False, current_pos, "Timeout"

def main():
    print("=" * 70)
    print("MOTOR 6 FIX PROCEDURE")
    print("=" * 70)
    print("\nThis script will attempt to fix motor 6 mechanical blockage")
    print("Current problem: Motor stuck at low position (~1509)")
    print("=" * 70)

    # Initialize
    config = FeetechMotorsBusConfig(
        port=PORT,
        motors={"gripper": [6, "sts3215"]}
    )

    print("\nConnecting...")
    bus = FeetechMotorsBus(config)
    bus.connect()

    try:
        # PHASE 1: Initial assessment
        print("\n" + "=" * 70)
        print("PHASE 1: INITIAL ASSESSMENT")
        print("=" * 70)

        current_pos, load, temp = print_status(bus)
        if current_pos is None:
            print("✗ Cannot communicate with motor 6!")
            return

        print(f"\nStarting position: {current_pos}")

        # PHASE 2: Configure for safe movement
        print("\n" + "=" * 70)
        print("PHASE 2: CONFIGURE FOR GENTLE MOVEMENT")
        print("=" * 70)

        print("Setting ultra-safe parameters...")
        bus.write("Mode", 0, ["gripper"])
        bus.write("P_Coefficient", 12, ["gripper"])  # Even softer than before
        bus.write("I_Coefficient", 0, ["gripper"])
        bus.write("D_Coefficient", 24, ["gripper"])
        bus.write("Lock", 0, ["gripper"])
        bus.write("Maximum_Acceleration", 50, ["gripper"])  # Very gentle
        bus.write("Acceleration", 20, ["gripper"])  # Super slow
        bus.write("Torque_Enable", 1, ["gripper"])
        print("✓ Configured for gentle movement")

        time.sleep(0.5)

        # PHASE 3: Try to find free range
        print("\n" + "=" * 70)
        print("PHASE 3: FIND FREE MOVEMENT RANGE")
        print("=" * 70)

        # Test small increments in both directions
        test_sequence = [
            ("Small UP +50", current_pos + 50),
            ("Small UP +100", current_pos + 100),
            ("Small UP +150", current_pos + 150),
            ("Back to start", current_pos),
            ("Small DOWN -50", current_pos - 50),
            ("Small DOWN -100", current_pos - 100),
            ("Back to start", current_pos),
        ]

        successful_positions = []
        blocked_positions = []

        for name, target_pos in test_sequence:
            # Safety clamp
            if target_pos < 1000 or target_pos > 3000:
                print(f"\n{name}: SKIPPED (out of safe range)")
                continue

            print(f"\n--- {name} (target: {target_pos}) ---")
            success, final_pos, reason = slow_move_to(bus, target_pos, max_load=800)

            if success:
                successful_positions.append(target_pos)
                print(f"✓ {reason}")
            else:
                blocked_positions.append((target_pos, reason))
                print(f"✗ {reason}")

            time.sleep(0.5)

        # PHASE 4: Analysis
        print("\n" + "=" * 70)
        print("PHASE 4: ANALYSIS")
        print("=" * 70)

        print(f"\n✓ Successful positions: {successful_positions}")
        print(f"✗ Blocked positions:")
        for pos, reason in blocked_positions:
            print(f"    {pos}: {reason}")

        if not successful_positions:
            print("\n⚠ WARNING: No successful movements!")
            print("Motor appears to be severely mechanically blocked.")

        # PHASE 5: Try to reach center (2048)
        print("\n" + "=" * 70)
        print("PHASE 5: ATTEMPT TO REACH CENTER (2048)")
        print("=" * 70)

        current_pos, _, _ = print_status(bus)

        if current_pos < 2048:
            print(f"\nMotor is below center ({current_pos} < 2048)")
            print("Attempting to move UP to center...")

            # Try incremental steps to 2048
            steps = [
                current_pos + 100,
                current_pos + 200,
                current_pos + 300,
                current_pos + 400,
                current_pos + 500,
                2048
            ]

            for step_target in steps:
                if step_target > 3000:
                    break

                success, final_pos, reason = slow_move_to(bus, step_target, max_load=1000)
                if not success:
                    print(f"\nStopped at {final_pos} due to: {reason}")
                    break

                if abs(final_pos - 2048) < 50:
                    print(f"\n✓✓✓ SUCCESS! Reached center: {final_pos}")
                    break

                time.sleep(0.3)

        # PHASE 6: Manual inspection mode
        print("\n" + "=" * 70)
        print("PHASE 6: MANUAL INSPECTION MODE")
        print("=" * 70)

        response = input("\nDisable torque for manual inspection? (y/n): ").strip().lower()

        if response == 'y':
            print("\nDisabling torque...")
            bus.write("Torque_Enable", 0, ["gripper"])
            print("✓ Torque DISABLED")
            print("\n" + "=" * 70)
            print("MANUAL INSPECTION INSTRUCTIONS:")
            print("=" * 70)
            print("""
1. Gently try to rotate motor 6 by hand
2. Feel for:
   - Cables wrapping/catching
   - Parts colliding
   - Grinding or rough spots
   - Complete blockages

3. Check:
   - Gripper fingers hitting display mount
   - Wires to the screen catching on rotation
   - Loose screws or shifted parts

Press Enter when done inspecting...
            """)
            input()

            print("\nRe-enabling torque...")
            bus.write("Torque_Enable", 1, ["gripper"])
            print("✓ Torque re-enabled")

        # PHASE 7: Final status and recommendations
        print("\n" + "=" * 70)
        print("PHASE 7: FINAL STATUS")
        print("=" * 70)

        final_pos, final_load, final_temp = print_status(bus)

        print(f"\nFinal position: {final_pos}")
        print(f"Starting position was: {current_pos}")
        print(f"Change: {final_pos - current_pos:+d} steps")

        print("\n" + "=" * 70)
        print("RECOMMENDATIONS:")
        print("=" * 70)

        if abs(final_pos - 2048) < 100:
            print("\n✓ Motor is near center - this is a good position")
        elif final_pos < 1800:
            print("\n⚠ Motor is still at low position")
            print("  - Check for physical obstruction preventing upward rotation")
            print("  - Cable management may be needed")

        if blocked_positions:
            print("\n⚠ Movement is still restricted")
            print("  - Motor has mechanical blockage")
            print("  - Manual intervention likely needed:")
            print("    1. Power off the robot")
            print("    2. Physically inspect motor 6 and gripper assembly")
            print("    3. Look for cables, misaligned parts, or collisions")
            print("    4. Adjust/fix the mechanical issue")
            print("    5. Power back on and test")

        if final_temp > 50:
            print(f"\n⚠ Temperature elevated ({final_temp}°C)")
            print("  - Motor has been working hard against resistance")
            print("  - Let it cool before further testing")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "=" * 70)
        print("Disconnecting...")
        bus.disconnect()
        print("Done!")
        print("=" * 70)

if __name__ == "__main__":
    main()