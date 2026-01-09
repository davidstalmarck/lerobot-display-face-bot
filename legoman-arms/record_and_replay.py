#!/usr/bin/env python3
"""
Record and Replay Movements - SO-101 Robot

Records motor positions over time and replays them.

Usage:
    python record_and_replay.py record <filename> [--leader|--follower]  # Record movements
    python record_and_replay.py replay <filename> [--leader|--follower]  # Replay movements
    python record_and_replay.py list                                      # List saved recordings

Examples:
    # Record on leader arm (default - /dev/ttyACM0)
    python record_and_replay.py record wave_hello --leader

    # Record on follower arm (/dev/ttyACM1)
    python record_and_replay.py record wave_hello --follower

    # Replay on leader
    python record_and_replay.py replay wave_hello --leader

    # Replay on follower
    python record_and_replay.py replay wave_hello --follower
"""

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
import time
import json
import sys
import os
from pathlib import Path

PORT_LEADER = "/dev/ttyACM0"
PORT_FOLLOWER = "/dev/ttyACM1"
RECORDINGS_DIR = Path("recordings")

# Motor configuration (5 motors)
MOTOR_CONFIG = {
    "shoulder_pan": [1, "sts3215"],
    "shoulder_lift": [2, "sts3215"],
    "elbow_flex": [3, "sts3215"],
    "wrist_flex": [4, "sts3215"],
    "wrist_roll": [5, "sts3215"],
}

MOTOR_NAMES = list(MOTOR_CONFIG.keys())


def initialize_bus(arm="leader"):
    """Initialize motor bus

    Args:
        arm: "leader" or "follower"
    """
    port = PORT_LEADER if arm == "leader" else PORT_FOLLOWER

    config = FeetechMotorsBusConfig(
        port=port,
        motors=MOTOR_CONFIG
    )
    bus = FeetechMotorsBus(config)
    bus.connect()
    return bus


def configure_motors(bus, mode="read"):
    """Configure motors for recording or playback"""
    if mode == "read":
        # For recording - disable torque so user can move freely
        print("Disabling torque for manual movement...")
        bus.write("Torque_Enable", 0)
    elif mode == "write":
        # For playback - enable torque with smooth settings
        print("Enabling torque for playback...")
        bus.write("Mode", 0)  # Position control
        bus.write("P_Coefficient", 16)
        bus.write("I_Coefficient", 0)
        bus.write("D_Coefficient", 32)
        bus.write("Lock", 0)
        bus.write("Maximum_Acceleration", 254)
        bus.write("Acceleration", 150)  # Smooth speed
        bus.write("Torque_Enable", 1)


def read_positions(bus):
    """Read current positions of all motors"""
    positions = bus.read("Present_Position", MOTOR_NAMES)
    # Convert numpy int32 to regular Python int for JSON serialization
    return {name: int(pos) for name, pos in zip(MOTOR_NAMES, positions)}


def write_positions(bus, positions):
    """Write positions to all motors"""
    bus.write("Goal_Position",
              [positions[name] for name in MOTOR_NAMES],
              MOTOR_NAMES)


def record_movement(filename, duration=10, frequency=20, arm="leader"):
    """
    Record movement by reading motor positions over time

    Args:
        filename: Name of recording (without extension)
        duration: How long to record in seconds
        frequency: Samples per second (Hz)
        arm: "leader" or "follower"
    """
    RECORDINGS_DIR.mkdir(exist_ok=True)
    filepath = RECORDINGS_DIR / f"{filename}.json"

    port = PORT_LEADER if arm == "leader" else PORT_FOLLOWER

    print("=" * 70)
    print("RECORDING MOVEMENT")
    print("=" * 70)
    print(f"Arm: {arm.upper()}")
    print(f"Port: {port}")
    print(f"File: {filepath}")
    print(f"Duration: {duration} seconds")
    print(f"Frequency: {frequency} Hz")
    print("=" * 70)

    bus = initialize_bus(arm)

    try:
        configure_motors(bus, mode="read")

        print("\n🎬 Recording will start in 3 seconds...")
        print("   Move the robot arm to create your gesture!")
        time.sleep(3)

        print("\n🔴 RECORDING NOW!\n")

        recording = {
            "arm": arm,
            "frequency": frequency,
            "duration": duration,
            "motor_names": MOTOR_NAMES,
            "frames": []
        }

        interval = 1.0 / frequency
        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < duration:
            frame_start = time.time()

            # Read current positions
            positions = read_positions(bus)
            timestamp = time.time() - start_time

            recording["frames"].append({
                "time": timestamp,
                "positions": positions
            })

            frame_count += 1

            # Progress indicator
            if frame_count % frequency == 0:
                elapsed = int(timestamp)
                remaining = duration - elapsed
                print(f"  [{elapsed}s / {duration}s] - {remaining}s remaining")

            # Sleep to maintain frequency
            elapsed = time.time() - frame_start
            sleep_time = max(0, interval - elapsed)
            time.sleep(sleep_time)

        print(f"\n✅ Recording complete! Captured {frame_count} frames")

        # Save to file
        with open(filepath, 'w') as f:
            json.dump(recording, f, indent=2)

        print(f"💾 Saved to: {filepath}")

        # Re-enable torque
        print("\nRe-enabling torque...")
        configure_motors(bus, mode="write")

        print("\n" + "=" * 70)
        print("To replay: python record_and_replay.py replay " + filename)
        print("=" * 70)

    finally:
        bus.disconnect()


def replay_movement(filename, loop=1, speed_factor=1.0, arm=None):
    """
    Replay recorded movement

    Args:
        filename: Name of recording (without extension)
        loop: Number of times to loop (1 = play once)
        speed_factor: Speed multiplier (1.0 = normal, 2.0 = double speed, 0.5 = half speed)
        arm: "leader" or "follower" (if None, uses arm from recording)
    """
    filepath = RECORDINGS_DIR / f"{filename}.json"

    if not filepath.exists():
        print(f"❌ Recording not found: {filepath}")
        print(f"\nAvailable recordings:")
        list_recordings()
        return

    # Load recording
    with open(filepath, 'r') as f:
        recording = json.load(f)

    frames = recording["frames"]
    duration = recording["duration"]
    recorded_arm = recording.get("arm", "leader")  # Default to leader if not specified

    # Use specified arm or the one from recording
    if arm is None:
        arm = recorded_arm
        print(f"ℹ️  Using arm from recording: {arm}")
    elif arm != recorded_arm:
        print(f"⚠️  Warning: Recording was made on {recorded_arm}, but playing on {arm}")

    port = PORT_LEADER if arm == "leader" else PORT_FOLLOWER

    print("=" * 70)
    print("REPLAYING MOVEMENT")
    print("=" * 70)
    print(f"File: {filepath}")
    print(f"Arm: {arm.upper()}")
    print(f"Port: {port}")
    print(f"Loops: {loop}")
    print(f"Speed: {speed_factor}x")
    print("=" * 70)

    print(f"\nLoaded {len(frames)} frames ({duration:.1f}s)")

    bus = initialize_bus(arm)

    try:
        configure_motors(bus, mode="write")

        print("\n▶️  Starting playback in 2 seconds...")
        time.sleep(2)

        for iteration in range(loop):
            if loop > 1:
                print(f"\n🔁 Loop {iteration + 1}/{loop}")

            print("🎬 Playing...")

            start_time = time.time()

            for i, frame in enumerate(frames):
                target_time = frame["time"] / speed_factor
                positions = frame["positions"]

                # Write positions
                write_positions(bus, positions)

                # Progress indicator
                if i % 20 == 0:
                    progress = (i / len(frames)) * 100
                    print(f"  Progress: {progress:.0f}%")

                # Wait until it's time for next frame
                elapsed = time.time() - start_time
                sleep_time = max(0, target_time - elapsed)
                time.sleep(sleep_time)

            print("  Progress: 100%")

            if iteration < loop - 1:
                time.sleep(0.5)  # Pause between loops

        print("\n✅ Playback complete!")

    finally:
        bus.disconnect()


def list_recordings():
    """List all saved recordings"""
    if not RECORDINGS_DIR.exists():
        print("No recordings found.")
        return

    recordings = list(RECORDINGS_DIR.glob("*.json"))

    if not recordings:
        print("No recordings found.")
        return

    print("\n📁 Available Recordings:")
    print("-" * 70)

    for rec in sorted(recordings):
        try:
            with open(rec, 'r') as f:
                data = json.load(f)

            name = rec.stem
            duration = data.get("duration", "?")
            frames = len(data.get("frames", []))
            frequency = data.get("frequency", "?")
            arm = data.get("arm", "unknown")

            print(f"  📹 {name}")
            print(f"      Arm: {arm} | Duration: {duration}s | Frames: {frames} | Frequency: {frequency}Hz")
            print(f"      File: {rec}")
            print()
        except Exception as e:
            print(f"  ⚠️  {rec.name} (error reading: {e})")

    print("-" * 70)


def parse_arm_arg(args):
    """Parse --leader or --follower from arguments"""
    if "--leader" in args:
        args.remove("--leader")
        return "leader"
    elif "--follower" in args:
        args.remove("--follower")
        return "follower"
    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Make a mutable copy of argv for parsing
    args = sys.argv[1:]
    command = args[0].lower()

    # Parse arm specification
    arm = parse_arm_arg(args)

    if command == "record":
        if len(args) < 2:
            print("Usage: python record_and_replay.py record <filename> [duration] [frequency] [--leader|--follower]")
            print("Example: python record_and_replay.py record wave_hello 10 20 --leader")
            sys.exit(1)

        filename = args[1]
        duration = float(args[2]) if len(args) > 2 else 10
        frequency = int(args[3]) if len(args) > 3 else 20
        arm = arm or "leader"  # Default to leader

        record_movement(filename, duration, frequency, arm)

    elif command == "replay":
        if len(args) < 2:
            print("Usage: python record_and_replay.py replay <filename> [loop] [speed] [--leader|--follower]")
            print("Example: python record_and_replay.py replay wave_hello 3 1.5 --follower")
            sys.exit(1)

        filename = args[1]
        loop = int(args[2]) if len(args) > 2 else 1
        speed = float(args[3]) if len(args) > 3 else 1.0
        # arm can be None here - will use recording's arm by default

        replay_movement(filename, loop, speed, arm)

    elif command == "list":
        list_recordings()

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()