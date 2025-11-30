#!/usr/bin/env python3
"""
SO-101 Follower Arm Motor Control and Customization Script

This script demonstrates ALL available customization options and motor control
capabilities for the LeRobot SO ARM 101 robot.

MOTOR CONTROL TABLE (STS3215):
- ID: Motor identification (1-252)
- Baud_Rate: Serial communication speed
- Min/Max_Angle_Limit: Position limits
- Max_Temperature_Limit: Thermal protection
- Max/Min_Voltage_Limit: Voltage protection
- Max_Torque_Limit: Maximum torque output
- P_Coefficient: Proportional gain (default: 32)
- D_Coefficient: Derivative gain (default: 32)
- I_Coefficient: Integral gain (default: 0)
- Mode: Operating mode (0=Position, 1=Speed, 3=Step)
- Torque_Enable: Enable/disable motor (0/1)
- Acceleration: Acceleration setting (0-254)
- Goal_Position: Target position (0-4095 steps)
- Goal_Speed: Target speed
- Torque_Limit: Current torque limit
- Lock: EPROM write lock (0=unlocked, 1=locked)
- Present_Position: Current position (read-only)
- Present_Speed: Current speed (read-only)
- Present_Load: Current load (read-only)
- Present_Temperature: Current temperature (read-only)
- Present_Voltage: Current voltage (read-only)
- Present_Current: Current consumption (read-only)
- Maximum_Acceleration: Max acceleration limit

POSITION CONVERSION:
- STS3215 Resolution: 4096 steps = 360 degrees
- Center position: 2048 steps = 0 degrees
- Formula: degrees = (steps - 2048) * 360 / 4096
"""

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
import time

# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

print("=" * 80)
print("SO ARM 101 - COMPREHENSIVE MOTOR CONTROL SCRIPT")
print("=" * 80)

# Motor bus configuration
print("\n[1/10] Creating motor bus configuration...")
config = FeetechMotorsBusConfig(
    port="/dev/ttyACM0",  # Change to your port: /dev/ttyACM1, /dev/tty.usbmodem..., etc.
    motors={
        # Format: "motor_name": [motor_id, "model"]
        # Motor IDs must be unique (1-252)
        "shoulder_pan": [1, "sts3215"],
        "shoulder_lift": [2, "sts3215"],
        "elbow_flex": [3, "sts3215"],
        "wrist_flex": [4, "sts3215"],
        "wrist_roll": [5, "sts3215"],
        "gripper": [6, "sts3215"],
    }
)

# Connect to the motor bus
print("[2/10] Creating motor bus...")
bus = FeetechMotorsBus(config)

print("[3/10] Connecting to motors...")
bus.connect()

# ============================================================================
# SECTION 2: INITIAL SETUP & PRESET CONFIGURATION
# ============================================================================

print("\n[4/10] Applying SO-101 preset configuration...")

# Set position control mode for all motors
bus.write("Mode", 0)  # 0 = Position control mode
print("  - Set Mode to Position Control (0)")

# PID Tuning to avoid motor shakiness (SO-101 preset)
bus.write("P_Coefficient", 16)  # Reduced from default 32 to reduce shakiness
bus.write("I_Coefficient", 0)   # Keep at 0
bus.write("D_Coefficient", 32)  # Keep at default
print("  - PID Coefficients: P=16, I=0, D=32")

# Unlock EPROM for writing parameters
bus.write("Lock", 0)
print("  - Unlocked EPROM for parameter writing")

# Set maximum performance
bus.write("Maximum_Acceleration", 254)  # Max = 254
bus.write("Acceleration", 254)
print("  - Acceleration: 254 (maximum)")

# Enable torque on all motors
bus.write("Torque_Enable", 1)
print("  - Torque enabled on all motors")

# ============================================================================
# SECTION 3: READ CURRENT STATUS
# ============================================================================

print("\n[5/10] Reading current motor status...")

# Read positions
positions = bus.read("Present_Position")
print(f"  - Current Positions: {positions}")

# Read speeds
speeds = bus.read("Present_Speed")
print(f"  - Current Speeds: {speeds}")

# Read load/torque
loads = bus.read("Present_Load")
print(f"  - Current Loads: {loads}")

# Read temperatures
temps = bus.read("Present_Temperature")
print(f"  - Current Temperatures: {temps}°C")

# Read voltages
voltages = bus.read("Present_Voltage")
print(f"  - Current Voltages: {voltages}V")

# Read currents
currents = bus.read("Present_Current")
print(f"  - Current Consumption: {currents}mA")

# ============================================================================
# SECTION 4: INDIVIDUAL MOTOR CONTROL EXAMPLES
# ============================================================================

print("\n[6/10] Testing individual motor movements...")

# Example 1: Move single motor
print("\n  Example 1: Moving shoulder_pan to center (2048 steps = 0°)...")
bus.write("Goal_Position", [2048], ["shoulder_pan"])
time.sleep(2)
pos = bus.read("Present_Position", ["shoulder_pan"])
print(f"    shoulder_pan position: {pos}")

# Example 2: Move with custom speed
print("\n  Example 2: Moving shoulder_pan with custom speed...")
bus.write("Goal_Speed", [500], ["shoulder_pan"])  # Set speed
bus.write("Goal_Position", [2500], ["shoulder_pan"])  # Move
time.sleep(2)
pos = bus.read("Present_Position", ["shoulder_pan"])
print(f"    shoulder_pan position: {pos}")

# Example 3: Control gripper
print("\n  Example 3: Gripper control...")
print("    Opening gripper...")
bus.write("Goal_Position", [2048], ["gripper"])
time.sleep(2)
print("    Closing gripper...")
bus.write("Goal_Position", [3000], ["gripper"])
time.sleep(2)

# Example 4: Move multiple motors simultaneously
print("\n  Example 4: Moving multiple motors simultaneously...")
bus.write("Goal_Position",
          [2048, 2048, 2048],  # Target positions
          ["shoulder_pan", "elbow_flex", "wrist_roll"])  # Motor names
time.sleep(2)

# ============================================================================
# SECTION 5: ADVANCED CUSTOMIZATION EXAMPLES
# ============================================================================

print("\n[7/10] Advanced customization examples...")

# Example 5: Adjust PID for specific motor (fine-tuning)
print("\n  Example 5: Custom PID tuning for elbow_flex...")
bus.write("P_Coefficient", [20], ["elbow_flex"])  # Custom P value
bus.write("D_Coefficient", [40], ["elbow_flex"])  # Custom D value
print("    elbow_flex PID: P=20, D=40")

# Example 6: Set torque limit for specific motor
print("\n  Example 6: Setting torque limit for gripper...")
bus.write("Torque_Limit", [800], ["gripper"])  # Reduce max torque
print("    gripper torque limit: 800")

# Example 7: Set acceleration for smooth movement
print("\n  Example 7: Smooth movement with custom acceleration...")
bus.write("Acceleration", [100], ["shoulder_lift"])  # Slower acceleration
bus.write("Goal_Position", [2500], ["shoulder_lift"])
time.sleep(3)
bus.write("Acceleration", [254], ["shoulder_lift"])  # Reset to max
print("    shoulder_lift moved with acceleration=100")

# Example 8: Monitor motor during movement
print("\n  Example 8: Real-time monitoring during movement...")
bus.write("Goal_Position", [3000], ["wrist_flex"])
for i in range(5):
    time.sleep(0.5)
    pos = bus.read("Present_Position", ["wrist_flex"])
    speed = bus.read("Present_Speed", ["wrist_flex"])
    load = bus.read("Present_Load", ["wrist_flex"])
    print(f"    t={i*0.5}s: pos={pos}, speed={speed}, load={load}")

# ============================================================================
# SECTION 6: SAFETY FEATURES
# ============================================================================

print("\n[8/10] Testing safety features...")

# Example 9: Temperature monitoring
print("\n  Example 9: Checking for overheating...")
temps = bus.read("Present_Temperature")
max_safe_temp = 70  # degrees Celsius
for motor_name, temp in zip(config.motors.keys(), temps):
    if temp > max_safe_temp:
        print(f"    WARNING: {motor_name} temperature too high: {temp}°C")
    else:
        print(f"    {motor_name}: {temp}°C (OK)")

# Example 10: Voltage monitoring
print("\n  Example 10: Voltage check...")
voltages = bus.read("Present_Voltage")
min_voltage = 6.0  # Minimum safe voltage
max_voltage = 8.4  # Maximum safe voltage
for motor_name, voltage in zip(config.motors.keys(), voltages):
    if voltage < min_voltage or voltage > max_voltage:
        print(f"    WARNING: {motor_name} voltage out of range: {voltage}V")
    else:
        print(f"    {motor_name}: {voltage}V (OK)")

# ============================================================================
# SECTION 7: POSITION CONVERSION UTILITIES
# ============================================================================

print("\n[9/10] Position conversion examples...")

def steps_to_degrees(steps, center=2048):
    """Convert motor steps to degrees (0 degrees = center position)"""
    return (steps - center) * 360 / 4096

def degrees_to_steps(degrees, center=2048):
    """Convert degrees to motor steps (0 degrees = center position)"""
    return int(center + degrees * 4096 / 360)

# Example 11: Move using degrees
print("\n  Example 11: Moving motors using degree values...")
target_degrees = 45  # 45 degrees from center
target_steps = degrees_to_steps(target_degrees)
print(f"    Target: {target_degrees}° = {target_steps} steps")
bus.write("Goal_Position", [target_steps], ["shoulder_pan"])
time.sleep(2)

current_steps = bus.read("Present_Position", ["shoulder_pan"])[0]
current_degrees = steps_to_degrees(current_steps)
print(f"    Current: {current_steps} steps = {current_degrees:.1f}°")

# ============================================================================
# SECTION 8: PRESET POSITIONS
# ============================================================================

print("\n[10/10] Moving to preset positions...")

# Define useful preset positions
PRESET_POSITIONS = {
    "home": {
        "shoulder_pan": 2048,
        "shoulder_lift": 2048,
        "elbow_flex": 2048,
        "wrist_flex": 2048,
        "wrist_roll": 2048,
        "gripper": 2048,
    },
    "rest": {
        "shoulder_pan": 2048,
        "shoulder_lift": 3000,
        "elbow_flex": 2500,
        "wrist_flex": 2048,
        "wrist_roll": 2048,
        "gripper": 2048,
    },
    "reach_forward": {
        "shoulder_pan": 2048,
        "shoulder_lift": 2300,
        "elbow_flex": 1800,
        "wrist_flex": 2048,
        "wrist_roll": 2048,
        "gripper": 2048,
    }
}

# Move to home position
print("\n  Moving to HOME position...")
home_pos = PRESET_POSITIONS["home"]
bus.write("Goal_Position",
          list(home_pos.values()),
          list(home_pos.keys()))
time.sleep(3)

# Move to rest position
print("  Moving to REST position...")
rest_pos = PRESET_POSITIONS["rest"]
bus.write("Goal_Position",
          list(rest_pos.values()),
          list(rest_pos.keys()))
time.sleep(3)

# Return to home
print("  Returning to HOME position...")
bus.write("Goal_Position",
          list(home_pos.values()),
          list(home_pos.keys()))
time.sleep(3)

# ============================================================================
# SECTION 9: CLEANUP
# ============================================================================

print("\n" + "=" * 80)
print("CUSTOMIZATION OPTIONS SUMMARY:")
print("=" * 80)
print("""
1. Motor Configuration:
   - Change port: config.port = "/dev/ttyACM1"
   - Change motor IDs: motors = {"name": [id, "sts3215"]}

2. PID Tuning:
   - Adjust P_Coefficient (reduce for less shakiness)
   - Adjust D_Coefficient (increase for damping)
   - Adjust I_Coefficient (usually keep at 0)

3. Performance:
   - Acceleration: 0-254 (higher = faster)
   - Maximum_Acceleration: 0-254
   - Goal_Speed: Set target velocity

4. Safety Limits:
   - Torque_Limit: Prevent overload
   - Max_Temperature_Limit: Thermal protection
   - Min/Max_Angle_Limit: Position boundaries

5. Control Modes:
   - Mode 0: Position control
   - Mode 1: Speed control
   - Mode 3: Step control

6. Monitoring:
   - Present_Position, Present_Speed, Present_Load
   - Present_Temperature, Present_Voltage, Present_Current

7. Preset Positions:
   - Define custom position dictionaries
   - Move all motors simultaneously

For full documentation, see:
- /home/david/lerobot/lerobot/common/robot_devices/robots/configs.py (lines 434-494)
- /home/david/lerobot/lerobot/common/robot_devices/motors/feetech.py (lines 55-103)
- /home/david/lerobot/examples/12_use_so101.md
""")

print("\nDisconnecting from motors...")
bus.disconnect()
print("Done!")
print("=" * 80)
