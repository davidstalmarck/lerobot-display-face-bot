#!/usr/bin/env python3
"""
Interactive Position Finder for SO-101 Display Character

This script helps you find the perfect starting position for your character's display/head.
You can either:
1. Edit TARGET_POSITION and run to test
2. Use interactive mode to adjust positions in real-time

Motor Reference:
- shoulder_pan (1): Base rotation (keep at 2048 for neutral)
- shoulder_lift (2): Arm up/down (keep at 2048 for neutral)
- elbow_flex (3): Elbow bend (keep at 2048 for neutral)
- wrist_flex (4): HEAD PITCH - look up/down (main display angle)
- wrist_roll (5): HEAD TILT - tilt left/right (main display tilt)
- gripper (6): NECK - rotate/turn head (main display rotation)

Position values:
- Center: 2048
- Range: ~1000-3000 (be conservative, test incrementally)
- +500 steps ≈ +44 degrees
- -500 steps ≈ -44 degrees
"""

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
import time
import sys

# ============================================================================
# EDIT THIS SECTION TO TEST DIFFERENT POSITIONS
# ============================================================================

TARGET_POSITION = {
    # Keep arm stationary (not used for display)
    "shoulder_pan": 2048, # motor 1
    "shoulder_lift": 1348, # motor 2
    "elbow_flex": 2248, # motor 3

    # HEAD/DISPLAY POSITION - Edit these to find your ideal starting pose!
    "wrist_flex": 2348,    # Head pitch: 2048=level, >2048=look up, <2048=look down
    "wrist_roll": 2048,    # Head tilt: 2048=level, >2048=tilt left, <2048=tilt right
    "gripper": 2048,       # Neck turn: 2048=center, >2048=turn left, <2048=turn right
}

# Movement speed (1-254, higher = faster)
SPEED = 150  # Smooth speed for testing

# Motor port
PORT = "/dev/ttyACM0"

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def print_position_info(positions_dict, label="Target"):
    """Helper to print position information"""
    print(f"\n{label} position:")
    print(f"  HEAD PITCH (wrist_flex): {positions_dict['wrist_flex']} ", end="")
    pitch_offset = positions_dict['wrist_flex'] - 2048
    if pitch_offset > 0:
        print(f"(+{pitch_offset} steps, looking UP)")
    elif pitch_offset < 0:
        print(f"({pitch_offset} steps, looking DOWN)")
    else:
        print("(level)")

    print(f"  HEAD TILT (wrist_roll):  {positions_dict['wrist_roll']} ", end="")
    tilt_offset = positions_dict['wrist_roll'] - 2048
    if tilt_offset > 0:
        print(f"(+{tilt_offset} steps, tilted LEFT)")
    elif tilt_offset < 0:
        print(f"({tilt_offset} steps, tilted RIGHT)")
    else:
        print("(level)")

    print(f"  NECK TURN (gripper):     {positions_dict['gripper']} ", end="")
    neck_offset = positions_dict['gripper'] - 2048
    if neck_offset > 0:
        print(f"(+{neck_offset} steps, turned LEFT)")
    elif neck_offset < 0:
        print(f"({neck_offset} steps, turned RIGHT)")
    else:
        print("(centered)")


def interactive_mode(bus):
    """Interactive mode to adjust positions in real-time"""
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("\nCommands:")
    print("  pitch <value>  - Set head pitch (wrist_flex)")
    print("  tilt <value>   - Set head tilt (wrist_roll)")
    print("  neck <value>   - Set neck turn (gripper)")
    print("  status         - Show current positions")
    print("  preset         - Try some preset positions")
    print("  save           - Print current position as code")
    print("  quit           - Exit interactive mode")
    print("\nPosition shortcuts:")
    print("  up/down/left/right/center - Quick adjustments")
    print("\nExamples:")
    print("  > pitch 2200     (look up slightly)")
    print("  > tilt 2300      (tilt left)")
    print("  > up             (look up by 200 steps)")
    print("  > center         (return to 2048 for all)")
    print()

    current_pos = {
        "shoulder_pan": 2048,
        "shoulder_lift": 2048,
        "elbow_flex": 2048,
        "wrist_flex": 2048,
        "wrist_roll": 2048,
        "gripper": 2048,
    }

    while True:
        try:
            cmd = input("\n> ").strip().lower().split()
            if not cmd:
                continue

            if cmd[0] == "quit" or cmd[0] == "exit" or cmd[0] == "q":
                break

            elif cmd[0] == "status" or cmd[0] == "s":
                positions = bus.read("Present_Position")
                motor_names = list(current_pos.keys())
                actual = dict(zip(motor_names, positions))
                print_position_info(actual, "Current")

            elif cmd[0] == "pitch" and len(cmd) > 1:
                value = int(cmd[1])
                current_pos["wrist_flex"] = value
                bus.write("Goal_Position", [value], ["wrist_flex"])
                print(f"Moving head pitch to {value}")

            elif cmd[0] == "tilt" and len(cmd) > 1:
                value = int(cmd[1])
                current_pos["wrist_roll"] = value
                bus.write("Goal_Position", [value], ["wrist_roll"])
                print(f"Moving head tilt to {value}")

            elif cmd[0] == "neck" and len(cmd) > 1:
                value = int(cmd[1])
                current_pos["gripper"] = value
                bus.write("Goal_Position", [value], ["gripper"])
                print(f"Moving neck turn to {value}")

            elif cmd[0] == "up":
                current_pos["wrist_flex"] += 200
                bus.write("Goal_Position", [current_pos["wrist_flex"]], ["wrist_flex"])
                print(f"Looking up: {current_pos['wrist_flex']}")

            elif cmd[0] == "down":
                current_pos["wrist_flex"] -= 200
                bus.write("Goal_Position", [current_pos["wrist_flex"]], ["wrist_flex"])
                print(f"Looking down: {current_pos['wrist_flex']}")

            elif cmd[0] == "left":
                current_pos["wrist_roll"] += 200
                bus.write("Goal_Position", [current_pos["wrist_roll"]], ["wrist_roll"])
                print(f"Tilting left: {current_pos['wrist_roll']}")

            elif cmd[0] == "right":
                current_pos["wrist_roll"] -= 200
                bus.write("Goal_Position", [current_pos["wrist_roll"]], ["wrist_roll"])
                print(f"Tilting right: {current_pos['wrist_roll']}")

            elif cmd[0] == "center":
                current_pos["wrist_flex"] = 2048
                current_pos["wrist_roll"] = 2048
                current_pos["gripper"] = 2048
                bus.write("Goal_Position",
                         [2048, 2048, 2048],
                         ["wrist_flex", "wrist_roll", "gripper"])
                print("Centered all head motors to 2048")

            elif cmd[0] == "preset":
                print("\nTrying preset positions:")
                presets = [
                    ("Neutral", {"wrist_flex": 2048, "wrist_roll": 2048, "gripper": 2048}),
                    ("Look Up", {"wrist_flex": 2300, "wrist_roll": 2048, "gripper": 2048}),
                    ("Look Down", {"wrist_flex": 1800, "wrist_roll": 2048, "gripper": 2048}),
                    ("Tilt Left", {"wrist_flex": 2048, "wrist_roll": 2300, "gripper": 2048}),
                    ("Tilt Right", {"wrist_flex": 2048, "wrist_roll": 1800, "gripper": 2048}),
                ]
                for name, pos in presets:
                    print(f"\n{name}...")
                    current_pos.update(pos)
                    bus.write("Goal_Position",
                             list(pos.values()),
                             list(pos.keys()))
                    time.sleep(2)

            elif cmd[0] == "save":
                positions = bus.read("Present_Position")
                motor_names = list(current_pos.keys())
                actual = dict(zip(motor_names, positions))
                print("\n" + "=" * 70)
                print("Copy this to your code:")
                print("=" * 70)
                print("TARGET_POSITION = {")
                for motor, pos in actual.items():
                    print(f'    "{motor}": {pos},')
                print("}")
                print("=" * 70)

            else:
                print("Unknown command. Type 'quit' to exit or see commands above.")

        except ValueError:
            print("Invalid value. Use integer positions (e.g., 'pitch 2048')")
        except Exception as e:
            print(f"Error: {e}")


def main():
    print("=" * 70)
    print("SO-101 Display Position Finder")
    print("=" * 70)

    # Check for interactive mode flag
    interactive = "--interactive" in sys.argv or "-i" in sys.argv

    if not interactive:
        print_position_info(TARGET_POSITION, "Target")
        print()
        print("-" * 70)

    # Initialize motor bus
    print("\nInitializing motors...")
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

    # Apply SO-101 preset configuration
    print("Configuring motors...")
    bus.write("Mode", 0)  # Position control mode
    bus.write("P_Coefficient", 16)  # Reduce shakiness
    bus.write("I_Coefficient", 0)
    bus.write("D_Coefficient", 32)
    bus.write("Lock", 0)  # Unlock EPROM
    bus.write("Maximum_Acceleration", 254)
    bus.write("Acceleration", SPEED)
    bus.write("Torque_Enable", 1)

    if interactive:
        # Interactive mode
        interactive_mode(bus)
    else:
        # One-shot mode
        print(f"Moving to target position at speed {SPEED}...")
        bus.write("Goal_Position",
                  list(TARGET_POSITION.values()),
                  list(TARGET_POSITION.keys()))

        # Wait for movement to complete
        print("Moving... (waiting 2 seconds)")
        time.sleep(2)

        # Read actual positions
        positions = bus.read("Present_Position")
        motor_names = list(TARGET_POSITION.keys())
        actual_positions = dict(zip(motor_names, positions))

        print()
        print("=" * 70)
        print("Movement complete! Current position:")
        print("=" * 70)
        print()
        print("HEAD/DISPLAY motors:")
        print(f"  wrist_flex (pitch): {actual_positions['wrist_flex']}")
        print(f"  wrist_roll (tilt):  {actual_positions['wrist_roll']}")
        print(f"  gripper (neck):     {actual_positions['gripper']}")
        print()
        print("ARM motors (should be at 2048):")
        print(f"  shoulder_pan:  {actual_positions['shoulder_pan']}")
        print(f"  shoulder_lift: {actual_positions['shoulder_lift']}")
        print(f"  elbow_flex:    {actual_positions['elbow_flex']}")
        print()
        print("=" * 70)
        print("Tips:")
        print("  - Edit TARGET_POSITION in this file and run again")
        print("  - OR run with --interactive for live adjustments:")
        print("    python debug_position.py --interactive")
        print("  - Typical adjustments: ±200-500 steps")
        print("  - Safe range: 1500-2600 (start conservative!)")
        print("=" * 70)

    # Disconnect
    print("\nDisconnecting motors...")
    bus.disconnect()
    print("Done!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()