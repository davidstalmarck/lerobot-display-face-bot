# Motor 6 (Gripper) Issue - SO-101 Robot

## Executive Summary

**Problem:** Motor 6 (gripper) on the SO-101 robot experienced mechanical blockage preventing proper rotation, making it unreliable for robotic control.

**Solution:** Disabled Motor 6 entirely and reconfigured the robot to operate successfully with only 5 motors (motors 1-5), utilizing the remaining motors for creative head/display movement gestures.

**Status:** ✅ **RESOLVED** - Robot fully operational with 5-motor configuration as of commit `a698714` (Jan 3, 2026)

---

## Table of Contents

1. [Background](#background)
2. [The Problem](#the-problem)
3. [Diagnosis Process](#diagnosis-process)
4. [Attempted Fixes](#attempted-fixes)
5. [Final Solution](#final-solution)
6. [Implementation Details](#implementation-details)
7. [Impact on Functionality](#impact-on-functionality)
8. [Lessons Learned](#lessons-learned)

---

## Background

### SO-101 Robot Overview

The SO-101 is a low-cost (~€114 per arm), 6-DOF robotic arm from the LeRobot project using:
- **6× Feetech STS3215 servo motors** (originally)
- **Serial communication** via `/dev/ttyACM0` and `/dev/ttyACM1`
- **Leader-follower teleoperation** for imitation learning

### Original Motor Configuration

| Motor ID | Name | Function | Range |
|----------|------|----------|-------|
| 1 | shoulder_pan | Base rotation (left/right) | 0-4095 |
| 2 | shoulder_lift | Shoulder elevation | 0-4095 |
| 3 | elbow_flex | Elbow joint | 0-4095 |
| 4 | wrist_flex | Wrist pitch | 0-4095 |
| 5 | wrist_roll | Wrist roll | 0-4095 |
| 6 | gripper | Gripper open/close | 0-4095 |

### Custom Features Added (Before Issue)

This fork added interactive capabilities:
- **Gesture Server** (`gesture_server.py`) - HTTP API for cute gestures (hello, forward lean, neutral)
- **Face Tracking** - Camera-based head tracking to follow viewers
- **Robot Receptionist** - Web UI with animated face + voice agent
- **Teleoperation** - Leader-follower control for data collection

---

## The Problem

### Initial Symptoms

Motor 6 (gripper) exhibited severe mechanical issues:

1. **Stuck Position**: Motor consistently stuck around position `1509` (well below center `2048`)
2. **Movement Restriction**:
   - Could move DOWN slightly (to lower values)
   - **Could NOT move UP** (to higher values like center `2048`)
3. **High Load Readings**: Excessive load values (`>800`) when attempting upward movement
4. **Inconsistent Communication**: Occasional communication failures with the motor

### Error Manifestation

```bash
# Attempting to move motor 6 to center (2048)
$ python stand_straight.py

shoulder_pan     : 2048 (offset:   +0) ✅ PERFECT
shoulder_lift    : 2050 (offset:   +2) ✅ PERFECT
elbow_flex       : 2045 (offset:   -3) ✅ PERFECT
wrist_flex       : 2051 (offset:   +3) ✅ PERFECT
wrist_roll       : 2049 (offset:   +1) ✅ PERFECT
gripper          : 1509 (offset: -539) ❌ Failed  # STUCK!
```

### Impact

- **Teleoperation unreliable**: Gripper motor wouldn't respond to position commands
- **Gesture system broken**: Couldn't use motor 6 for head/neck movements as designed
- **Safety concerns**: High motor loads risked damaging the servo or robot structure

---

## Diagnosis Process

### 1. Communication Testing

**Test:** Can we communicate with motor 6 at all?

```python
# find_motor_2.py and diagnose_motor_6.py
bus.read("Present_Position", ["gripper"])
```

**Result:** ✅ Communication successful - motor responds to queries

**Conclusion:** Not a communication or ID configuration issue

---

### 2. Movement Range Testing

**Test:** Comprehensive movement diagnostics (`diagnose_all_motors.py`)

```
MOTOR 6: gripper
1. CURRENT STATUS:
   Position: 1509
   Voltage: 6.5V
   Temperature: 31°C
   Load: 156

4. MOVEMENT TEST:
   Starting position: 1509
   → Testing DOWN movement to 1209...
     After: 1480 (moved: -29)          ✅ Small movement
   → Testing UP movement to 2109...
     After: 1512 (moved: +32)          ❌ Barely moved (target was +600!)
   → Returning to center (2048)...
     After: 1509                        ❌ FAILED - stuck at low position

5. DIAGNOSIS:
   ❌ Motor can move DOWN but NOT UP - MECHANICAL BLOCK or LIMIT
```

**Result:** Clear directional restriction

**Conclusion:** Physical obstruction preventing upward rotation

---

### 3. Configuration Parameter Testing

**Test:** Adjusting PID coefficients and acceleration for gentler movement (`fix_motor_6.py`)

```python
# Ultra-gentle settings
bus.write("P_Coefficient", 12, ["gripper"])      # Softer response
bus.write("I_Coefficient", 0, ["gripper"])
bus.write("D_Coefficient", 24, ["gripper"])
bus.write("Maximum_Acceleration", 50, ["gripper"])  # Very slow
bus.write("Acceleration", 20, ["gripper"])          # Super gentle
```

**Result:** ❌ No improvement - still couldn't move up

**Conclusion:** Not a motor tuning or control parameter issue

---

### 4. Physical Inspection

**Test:** Manual inspection with torque disabled (`fix_motor_6.py` Phase 6)

```python
bus.write("Torque_Enable", 0, ["gripper"])  # Allow manual rotation
```

**Findings:**
- ✅ Motor shaft could be rotated manually (with resistance)
- ⚠️ Possible cable wrapping or interference with display mount
- ⚠️ Gripper fingers potentially colliding with other components
- ⚠️ Wires to the mounted screen catching during rotation

**Result:** Identified **mechanical obstruction** as root cause

**Conclusion:** Physical assembly issue, not electrical or firmware

---

### 5. Incremental Movement Testing

**Test:** Small incremental steps to find free range (`fix_motor_6.py` Phase 3)

```python
# Test sequence
test_sequence = [
    ("Small UP +50", current_pos + 50),
    ("Small UP +100", current_pos + 100),
    ("Small UP +150", current_pos + 150),
    ("Small DOWN -50", current_pos - 50),
    ...
]
```

**Result:**
- ✅ Successful: Small downward movements
- ❌ Blocked: All upward movements stopped by high load

**Conclusion:** Consistent mechanical blockage in upward direction

---

## Attempted Fixes

### Fix Attempt #1: Motor Parameter Tuning

**Approach:** Reduce motor stiffness and speed to overcome resistance gently

```python
# Softer PID settings
P_Coefficient: 12 (down from 16)
D_Coefficient: 24 (down from 32)
Maximum_Acceleration: 50 (down from 254)
```

**Result:** ❌ Failed - couldn't overcome mechanical blockage

---

### Fix Attempt #2: Force-Through Movement

**Approach:** Temporarily increase torque limit to power through obstruction

**Status:** ⚠️ **NOT ATTEMPTED** - Too risky

**Reasoning:** Could damage:
- Motor gears (stripped teeth)
- Robot structure (bent parts)
- Cables (severed wires)
- Display mount (broken connection)

---

### Fix Attempt #3: Cable Management Investigation

**Approach:** Check for cable wrapping or catching during rotation

**Findings:**
- Display power/data cables potentially interfering
- Gripper assembly cables may wrap during rotation
- Limited space around motor 6 mounting area

**Result:** ⚠️ Partial - improved cable routing but didn't fully resolve blockage

---

### Fix Attempt #4: Physical Disassembly and Repair

**Approach:** Fully disassemble gripper area to identify and fix mechanical issue

**Status:** ❌ **NOT PURSUED**

**Reasoning:**
1. **Time-intensive** - Would require complete robot disassembly
2. **Risk of damage** - Could break delicate components
3. **Uncertain outcome** - Root cause unclear without full teardown
4. **Alternative available** - Robot can function without motor 6

---

## Final Solution

### Decision: 5-Motor Configuration

After exhausting diagnostic options, the pragmatic solution was:

**Disable Motor 6 entirely and reconfigure robot to use only motors 1-5**

### Why This Works

1. **Core functionality preserved**: Motors 1-5 provide sufficient DOF for:
   - Head pan/tilt movements (repurposed motors 1, 4, 5)
   - Arm positioning (motors 2, 3)
   - Gesture control
   - Face tracking

2. **No gripper requirement**: This robot variant uses:
   - **Display mount** instead of gripper (for animated face UI)
   - **Head movements** for expression
   - **Gestures** for interaction

3. **Improved reliability**: Removing problematic motor eliminates:
   - Unpredictable movement failures
   - High load warnings
   - Communication errors

4. **Easier maintenance**: Simpler configuration with fewer points of failure

---

## Implementation Details

### Code Changes

#### 1. Robot Configuration (`lerobot/common/robot_devices/robots/configs.py:454-471`)

**Leader Arm:**
```python
leader_arms: dict[str, MotorsBusConfig] = field(
    default_factory=lambda: {
        "main": FeetechMotorsBusConfig(
            port="/dev/ttyACM0",
            motors={
                "shoulder_pan": [1, "sts3215"],
                "shoulder_lift": [2, "sts3215"],
                "elbow_flex": [3, "sts3215"],
                "wrist_flex": [4, "sts3215"],
                "wrist_roll": [5, "sts3215"],
                # "gripper": [6, "sts3215"],  # DISABLED - mechanical blockage
            },
        ),
    }
)
```

**Follower Arm:**
```python
follower_arms: dict[str, MotorsBusConfig] = field(
    default_factory=lambda: {
        "main": FeetechMotorsBusConfig(
            port="/dev/ttyACM1",
            motors={
                "shoulder_pan": [1, "sts3215"],
                "shoulder_lift": [2, "sts3215"],
                "elbow_flex": [3, "sts3215"],
                "wrist_flex": [4, "sts3215"],
                "wrist_roll": [5, "sts3215"],
                # "gripper": [6, "sts3215"],  # DISABLED - mechanical blockage
            },
        ),
    }
)
```

#### 2. Gesture Server Updates (`gesture_server.py:44-52`)

Updated neutral position to exclude motor 6:

```python
NEUTRAL_POSITION = {
    "shoulder_pan": 2048,   # motor 1 - base rotation
    "shoulder_lift": 1348,  # motor 2 - arm elevation
    "elbow_flex": 2248,     # motor 3 - elbow joint
    "wrist_flex": 2348,     # motor 4 - head pitch (repurposed)
    "wrist_roll": 3072,     # motor 5 - head tilt (repurposed)
    # NO gripper (motor 6 disabled)
}
```

#### 3. Calibration Files

Created new 5-motor calibration:
- `.cache/calibration/so101/main_leader.json`
- `.cache/calibration/so101/main_follower.json`

(Contains calibration data for motors 1-5 only)

---

### Git Commit

```bash
commit a698714e6354b43a88fddae8f4e8979c3fe503d2
Author: David Stålmarck
Date:   Sat Jan 3 20:56:12 2026 +0100

    trouble with the one of the servos, a solution using only 5 motors

    Modified files:
    - lerobot/common/robot_devices/robots/configs.py (disabled motor 6)
    - .cache/calibration/so101/*.json (5-motor calibration)
    - Added diagnostic scripts for troubleshooting
```

---

## Impact on Functionality

### What Still Works ✅

1. **Gesture Server** - All gestures adapted to 5 motors:
   - `/hello` - Head shake and pan movements (motors 1, 4, 5)
   - `/forward` - Forward lean gesture
   - `/neutral` - Return to center position

2. **Face Tracking** - Uses motors 1 (horizontal) and 4 (vertical):
   ```python
   # gesture_server.py:122-142
   # Motor 1: shoulder_pan (left/right tracking)
   # Motor 4: wrist_flex (up/down tracking)
   ```

3. **Teleoperation** - 5-DOF leader-follower control:
   - Full arm positioning capability
   - Smooth movement replication
   - Data collection for imitation learning

4. **Robot Receptionist** - Web interface + voice agent:
   - Animated face display (no gripper needed)
   - Voice interaction
   - Gesture-based responses

### What Changed ⚠️

1. **No gripper control** - Can't open/close gripper (not needed for display mount variant)
2. **One less DOF** - 5 degrees of freedom instead of 6
3. **Motor naming** - Motors 4-5 repurposed for head/neck control instead of wrist movements

### What Doesn't Work ❌

1. **Traditional gripper tasks** - Can't pick/place objects (by design - display variant)
2. **6-motor gestures** - Any choreography requiring motor 6
3. **Full wrist articulation** - Limited wrist DOF due to repurposed motors

---

## Lessons Learned

### Technical Insights

1. **Hardware debugging workflow**:
   ```
   Communication Test → Range Testing → Parameter Tuning →
   Physical Inspection → Incremental Testing → Root Cause
   ```

2. **When to pivot**: After 10+ diagnostic scripts and multiple failed fixes, **removing the problematic component** was more pragmatic than endless debugging

3. **Mechanical > Electrical**: In robotics, mechanical issues (cables, collisions, misalignment) are often harder to debug than electrical/firmware problems

4. **Safety first**: Avoided force-through solutions that could cause permanent damage

### Design Decisions

1. **Graceful degradation**: Robot remains fully functional with reduced DOF
2. **Purpose-driven**: Since this variant uses a display (not gripper), motor 6 wasn't essential
3. **Documentation matters**: Extensive diagnostic scripts preserved troubleshooting history

### Future Prevention

1. **Cable management**: Better routing to prevent wrapping around rotating joints
2. **Mechanical clearance**: Verify component spacing during assembly
3. **Modular design**: Make problematic components easier to service/replace
4. **Testing protocol**: Test each motor individually during initial assembly

---

## Diagnostic Scripts Reference

The following scripts were created during troubleshooting (now candidates for cleanup):

### Core Diagnostics
- `diagnose_all_motors.py` - Comprehensive 6-motor testing
- `diagnose_motors.py` - Focused testing of motors 2 & 6
- `diagnose_motor_6.py` - Motor 6 specific diagnostics

### Fix Attempts
- `fix_motor_6.py` - Multi-phase repair procedure (main troubleshooting script)
- `disable_motor_6_torque.py` - Manual inspection mode

### Motor Testing
- `test_move_motors.py` - General motor movement demo
- `test_individual_motors.py` - Individual motor testing
- `test_gripper.py` - Motor 6 gripper-specific test
- `find_motor_2.py` - Motor 2 communication test

### Positioning Scripts
- `stand_straight.py` - Move to neutral (all motors including 6)
- `stand_straight_slow.py` - Slow movement to neutral
- `stand_robot.py` - Display position
- `hold_straight.py` - Hold with torque enabled
- `hold_position.py` - Keep powered in position

### Development Tools (Still Useful)
- `debug_position.py` - Interactive position finder
- `debug_tracking.py` - Face tracking debug interface

---

## Current Status

**Robot Configuration:** 5 motors (1-5 active, motor 6 disabled)

**Functionality:** ✅ Fully operational
- Gesture server running
- Face tracking active
- Teleoperation working
- Receptionist UI functional

**Hardware Status:**
- ✅ Motors 1-5: Working perfectly
- ❌ Motor 6: Disabled due to mechanical blockage
- ℹ️ Physical repair possible but not required

**Next Steps:**
1. ✅ Document issue (this README)
2. 🔄 Clean up diagnostic scripts (in progress)
3. ⏭️ Optional: Attempt physical repair when time permits
4. ⏭️ Consider motor 6 replacement if gripper functionality needed

---

## Conclusion

While motor 6's mechanical failure was initially concerning, the 5-motor configuration proved to be a **successful and pragmatic solution**. The robot maintains all core interactive features (gestures, face tracking, teleoperation) while eliminating a source of unreliability.

This demonstrates an important principle in robotics: **functional redundancy and creative repurposing can overcome hardware limitations** without requiring extensive repairs.

The extensive diagnostic process, though ultimately leading to disabling the motor, provided valuable insights into the Feetech STS3215 servo behavior and mechanical debugging workflows for future reference.

---

**Document Version:** 1.0
**Last Updated:** January 3, 2026
**Author:** David Stålmarck (with debugging assistance from Claude Code)
**Robot:** SO-101 with Display Mount (LeRobot Fork)