#!/usr/bin/env python3
"""
Simple teleoperation script for SO101 robot.
Makes follower arm (ACM1) mirror leader arm (ACM0) movements.
Press Ctrl+C to stop.
"""

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
import time
import signal
import sys

# Flag for clean shutdown
running = True

def signal_handler(sig, frame):
    global running
    print('\n\nStopping teleoperation...')
    running = False

signal.signal(signal.SIGINT, signal_handler)

# Leader arm configuration (ACM0)
leader_config = FeetechMotorsBusConfig(
    port='/dev/ttyACM0',
    motors={
        'shoulder_pan': [1, 'sts3215'],
        'shoulder_lift': [2, 'sts3215'],
        'elbow_flex': [3, 'sts3215'],
        'wrist_flex': [4, 'sts3215'],
        'wrist_roll': [5, 'sts3215'],
    }
)

# Follower arm configuration (ACM1)
follower_config = FeetechMotorsBusConfig(
    port='/dev/ttyACM1',
    motors={
        'shoulder_pan': [1, 'sts3215'],
        'shoulder_lift': [2, 'sts3215'],
        'elbow_flex': [3, 'sts3215'],
        'wrist_flex': [4, 'sts3215'],
        'wrist_roll': [5, 'sts3215'],
    }
)

print("Connecting to leader arm (ACM0)...")
leader = FeetechMotorsBus(config=leader_config)
leader.connect()
print("✓ Leader connected")

print("Connecting to follower arm (ACM1)...")
follower = FeetechMotorsBus(config=follower_config)
follower.connect()
print("✓ Follower connected")

print("\n" + "="*60)
print("TELEOPERATION STARTED")
print("Move the leader arm (ACM0) and the follower arm (ACM1) will follow")
print("Press Ctrl+C to stop")
print("="*60 + "\n")

loop_count = 0
start_time = time.time()

try:
    while running:
        try:
            # Read leader arm position
            leader_pos = leader.read('Present_Position')

            # Write to follower arm
            follower.write('Goal_Position', leader_pos)

            # Read follower position for monitoring
            follower_pos = follower.read('Present_Position')

            # Print status every 30 loops (~1 second at 30Hz)
            if loop_count % 30 == 0:
                elapsed = time.time() - start_time
                hz = loop_count / elapsed if elapsed > 0 else 0
                print(f"Loop {loop_count:5d} | {hz:5.1f} Hz | Leader: {leader_pos[0]:4d} | Follower: {follower_pos[0]:4d}")

            loop_count += 1

            # Small delay to avoid overwhelming the bus
            time.sleep(0.01)  # ~100Hz max

        except Exception as e:
            print(f"ERROR in control loop: {e}")
            print("Retrying in 1 second...")
            time.sleep(1)

except KeyboardInterrupt:
    print("\n\nKeyboard interrupt received")

finally:
    print("\nDisconnecting arms...")
    try:
        leader.disconnect()
        print("✓ Leader disconnected")
    except:
        pass

    try:
        follower.disconnect()
        print("✓ Follower disconnected")
    except:
        pass

    print("\nTeleoperation stopped. Goodbye!")
