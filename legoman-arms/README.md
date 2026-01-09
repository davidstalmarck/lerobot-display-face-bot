# Legoman Arms - SO-101 Robot Control Scripts

Interactive control scripts for the SO-101 5-motor robotic arm with display mount.

## Overview

This folder contains scripts for controlling and programming the SO-101 robot, which uses 5 Feetech STS3215 servo motors (motors 1-5). Motor 6 was disabled due to mechanical issues - see [MOTOR_6_ISSUE.md](MOTOR_6_ISSUE.md) for details.

## Hardware Configuration

- **Robot:** SO-101 (LeRobot fork with display mount)
- **Motors:** 5× Feetech STS3215 servos
- **Ports:**
  - Leader arm: `/dev/ttyACM0`
  - Follower arm: `/dev/ttyACM1`

### Motor Mapping (5 motors)

| Motor ID | Name | Function |
|----------|------|----------|
| 1 | shoulder_pan | Base rotation (left/right) |
| 2 | shoulder_lift | Shoulder elevation |
| 3 | elbow_flex | Elbow joint |
| 4 | wrist_flex | Head pitch (repurposed) |
| 5 | wrist_roll | Head tilt (repurposed) |
| ~~6~~ | ~~gripper~~ | **DISABLED** (mechanical blockage) |

## Scripts

### Core Features

#### `gesture_server.py` - Gesture HTTP Server
Interactive gesture control via HTTP API with face tracking.

**Features:**
- Cute gestures (hello, forward, neutral)
- Face tracking with camera
- Real-time video feed with face detection

**Usage:**
```bash
python gesture_server.py
```

**Endpoints:**
- `GET /hello` - Cute hello gesture with head movements
- `GET /forward` - Lean forward gesture
- `GET /neutral` - Return to neutral position
- `GET /track/start` - Start face tracking
- `GET /track/stop` - Stop face tracking
- `GET /video_feed` - Camera feed with face detection overlay
- `GET /status` - Current motor positions

**Example:**
```bash
# Start server
python gesture_server.py

# In another terminal or browser:
curl http://localhost:8080/hello
curl http://localhost:8080/track/start
```

---

#### `robot_receptionist.py` - Robot Receptionist
Web-based receptionist with animated face UI and voice agent.

**Features:**
- Animated face display (served on web interface)
- Voice interaction capabilities
- Gesture integration

**Usage:**
```bash
python robot_receptionist.py
```

Then open the web interface in your browser.

---

#### `record_and_replay.py` - Movement Recording & Playback
Record movements by manually moving the robot, then replay them.

**Usage:**

**Record movement (leader arm):**
```bash
python record_and_replay.py record wave_hello --leader
# Robot torque disables, move it manually for 10 seconds
```

**Record movement (follower arm):**
```bash
python record_and_replay.py record wave_hello --follower
```

**Replay movement:**
```bash
python record_and_replay.py replay wave_hello
# Automatically uses the arm from recording
```

**Replay on different arm:**
```bash
python record_and_replay.py replay wave_hello --follower
```

**Advanced options:**
```bash
# Record for 15 seconds at 30 Hz
python record_and_replay.py record my_gesture 15 30 --leader

# Replay 3 times at 1.5x speed
python record_and_replay.py replay my_gesture 3 1.5 --leader
```

**List recordings:**
```bash
python record_and_replay.py list
```

**Recordings are saved to:** `recordings/<filename>.json`

---

### Development Tools

#### `debug_position.py` - Interactive Position Finder
Find and test motor positions interactively.

**Usage:**
```bash
python debug_position.py
```

Useful for:
- Finding exact positions for gestures
- Testing motor ranges
- Calibrating movements

---

#### `debug_tracking.py` - Face Tracking Debug
Debug and tune face tracking parameters.

**Usage:**
```bash
python debug_tracking.py
```

Features:
- Real-time face detection visualization
- Tracking parameter tuning
- Camera feed debugging

---

### UI Files

#### `index.html` - Animated Face Interface
Web-based animated face display for the robot receptionist.

Used by `robot_receptionist.py` to display an expressive face on the mounted screen.

---

## Installation

### Dependencies

```bash
# Install LeRobot (if not already installed)
pip install lerobot

# Additional dependencies for gesture server
pip install flask opencv-python numpy

# For face detection (if not included)
pip install opencv-contrib-python
```

### Hardware Setup

1. Connect leader arm to `/dev/ttyACM0`
2. Connect follower arm to `/dev/ttyACM1`
3. Ensure motors 1-5 are properly calibrated
4. Mount display/camera if using receptionist or tracking features

---

## Quick Start

### 1. Test Basic Gestures
```bash
python gesture_server.py
# In browser: http://localhost:8080/hello
```

### 2. Record a Custom Gesture
```bash
# Record on leader arm
python record_and_replay.py record my_wave 10 --leader

# Manually move the robot to create the gesture

# Replay it
python record_and_replay.py replay my_wave
```

### 3. Face Tracking Demo
```bash
python gesture_server.py
# In browser: http://localhost:8080/track/start
# View feed: http://localhost:8080/video_feed
```

---

## Troubleshooting

### Motor Connection Issues

**Check ports:**
```bash
ls -l /dev/ttyACM*
```

**Test individual motors:**
```python
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig

config = FeetechMotorsBusConfig(
    port="/dev/ttyACM0",
    motors={"test": [1, "sts3215"]}
)
bus = FeetechMotorsBus(config)
bus.connect()
pos = bus.read("Present_Position", ["test"])
print(f"Motor 1 position: {pos}")
```

### Motor 6 Issues

Motor 6 (gripper) is disabled due to mechanical blockage. See [MOTOR_6_ISSUE.md](MOTOR_6_ISSUE.md) for full details.

**Do not attempt to use motor 6** - it may cause:
- High load warnings
- Overheating
- Communication errors

### Camera Not Found

If face tracking doesn't work:

```bash
# List available cameras
ls -l /dev/video*

# Update camera index in gesture_server.py
# Line 463: camera = cv2.VideoCapture(4)  # Change number
```

### Permission Denied

If you get permission errors accessing `/dev/ttyACM*`:

```bash
# Add user to dialout group
sudo usermod -a -G dialout $USER

# Log out and back in for changes to take effect
```

---

## File Structure

```
legoman-arms/
├── README.md                  # This file
├── MOTOR_6_ISSUE.md          # Motor 6 troubleshooting documentation
│
├── gesture_server.py          # HTTP gesture server with face tracking
├── robot_receptionist.py      # Web receptionist with voice agent
├── record_and_replay.py       # Record and replay movements
│
├── debug_position.py          # Interactive position finder
├── debug_tracking.py          # Face tracking debugger
│
├── index.html                 # Animated face UI
│
└── recordings/                # Saved movement recordings (created on first use)
    └── *.json
```

---

## Configuration

### Neutral Position (defined in `gesture_server.py`)

```python
NEUTRAL_POSITION = {
    "shoulder_pan": 2048,   # motor 1 - centered
    "shoulder_lift": 1348,  # motor 2 - arm position
    "elbow_flex": 2248,     # motor 3 - arm position
    "wrist_flex": 2348,     # motor 4 - head pitch
    "wrist_roll": 3072,     # motor 5 - head tilt
}
```

### Motor Limits

All motors use Feetech STS3215 servos:
- **Range:** 0 - 4095 (12-bit resolution)
- **Center:** 2048
- **Safe operating range:** 500 - 3500 (avoid mechanical limits)

---

## Advanced Usage

### Creating Custom Gestures

Edit `gesture_server.py` to add new gestures:

```python
@app.route('/my_gesture', methods=['GET'])
def my_gesture():
    try:
        # Define positions
        smooth_move({
            "shoulder_pan": 2200,
            "shoulder_lift": 1348,
            "elbow_flex": 2400,
            "wrist_flex": 2500,
            "wrist_roll": 3100,
        }, duration=1.5)

        return jsonify({
            "status": "success",
            "gesture": "my_gesture"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
```

### Teleoperation with LeRobot

Use the main LeRobot teleoperation tools for data collection:

```bash
# From main project directory (not legoman-arms)
python -m lerobot.scripts.control_robot teleoperate \
    --robot-path lerobot/configs/robot/so101.yaml \
    --robot-overrides='~cameras'
```

---

## Safety Notes

⚠️ **Important Safety Guidelines:**

1. **Start slow:** Always test new gestures at low speed first
2. **Check clearance:** Ensure robot has space to move before starting
3. **Emergency stop:** Keep `Ctrl+C` ready to interrupt scripts
4. **Motor 6:** Never try to use motor 6 - it's mechanically blocked
5. **Power off:** When making physical adjustments, power off the robot
6. **Torque limits:** High torque may indicate mechanical blockage - stop immediately

---

## Credits

- **LeRobot Project:** https://github.com/huggingface/lerobot
- **SO-101 Robot:** Low-cost robotic arm design from LeRobot
- **This Fork:** Custom display-mount variant with 5-motor configuration

---

## License

Same as LeRobot project (Apache 2.0)

---

## Related Documentation

- [MOTOR_6_ISSUE.md](MOTOR_6_ISSUE.md) - Detailed motor 6 troubleshooting history
- Main LeRobot docs: https://github.com/huggingface/lerobot/tree/main/docs

---

**Last Updated:** January 4, 2026
