#!/usr/bin/env python3
"""
SO-101 Cute Gesture Server

A simple HTTP server that performs cute, natural gestures with the SO-101 robot.

API Endpoints:
- GET /hello - Tilts head and waves (wrist movements)
- GET /forward - Tilts forward and slightly tilts head to the right
- GET /neutral - Returns to straight, neutral position
- GET /track/start - Start face tracking (follows viewer)
- GET /track/stop - Stop face tracking
- GET /video_feed - View camera feed with face detection
- GET /status - Returns current motor positions and tracking status

Run with: python gesture_server.py
Access at: http://localhost:8080
"""

from flask import Flask, jsonify, Response
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
import time
import threading
import cv2
import numpy as np

app = Flask(__name__)

# Motor port configuration
PORT = "/dev/ttyACM0"
SPEED = 150  # Smooth, natural speed

# Global motor bus (initialized on startup)
bus = None
motor_lock = threading.Lock()

# Face tracking globals
camera = None
face_cascade = None
tracking_active = False
tracking_thread = None

# Neutral starting position - matches your calibrated TARGET_POSITION
NEUTRAL_POSITION = {
    "shoulder_pan": 2048,   # motor 1
    "shoulder_lift": 1348,  # motor 2 - arm position
    "elbow_flex": 2248,     # motor 3 - arm position
    "wrist_flex": 2348,     # Head pitch: looking up slightly from level
    "wrist_roll": 3072,     # Head tilt: level
    "gripper": 2048,        # Neck turn: centered
}


def smooth_move(positions, duration=1.5):
    """Move to positions smoothly over duration"""
    with motor_lock:
        bus.write("Goal_Position",
                  list(positions.values()),
                  list(positions.keys()))
        time.sleep(duration)


def get_current_positions():
    """Read current motor positions"""
    with motor_lock:
        positions = bus.read("Present_Position")
        motor_names = list(NEUTRAL_POSITION.keys())
        return dict(zip(motor_names, positions))


def face_tracking_loop():
    """Main face tracking loop - runs in background thread"""
    global tracking_active, camera

    print("Face tracking started!")

    # Current motor positions (using motor 1 and motor 4 for 2-axis tracking)
    current_shoulder_pan = NEUTRAL_POSITION["shoulder_pan"]  # Motor 1 (left/right)
    current_wrist_flex = NEUTRAL_POSITION["wrist_flex"]      # Motor 4 (up/down)

    # Tracking parameters
    frame_center_x = 320  # Assuming 640x480 camera
    frame_center_y = 240
    dead_zone = 80  # Pixels - don't move if face is within this range of center

    # Movement speed (reduce for smoother tracking)
    move_scale = 0.5  # How much to move per pixel offset
    max_move = 50    # Maximum movement per update

    while tracking_active:
        try:
            ret, frame = camera.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )

            if len(faces) > 0:
                # Get the largest face (closest to camera)
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, w, h = largest_face

                # Calculate face center
                face_center_x = x + w // 2
                face_center_y = y + h // 2

                # Calculate offset from frame center
                offset_x = face_center_x - frame_center_x
                offset_y = face_center_y - frame_center_y

                # Track horizontally (motor 1 - shoulder_pan)
                if abs(offset_x) > dead_zone:
                    # Positive offset_x = face is right, turn right (increase shoulder_pan)
                    # Negative offset_x = face is left, turn left (decrease shoulder_pan)
                    move_x = int(offset_x * move_scale)
                    move_x = max(-max_move, min(max_move, move_x))
                    current_shoulder_pan = max(1500, min(2600, current_shoulder_pan + move_x))

                # Track vertically (motor 4 - wrist_flex)
                if abs(offset_y) > dead_zone:
                    # Positive offset_y = face is down, look down (increase wrist_flex)
                    # Negative offset_y = face is up, look up (decrease wrist_flex)
                    move_y = int(offset_y * move_scale * 0.4)  # Reduce vertical sensitivity
                    move_y = max(-max_move, min(max_move, move_y))
                    current_wrist_flex = max(1800, min(2800, current_wrist_flex + move_y))

                # Apply movements to both motors
                with motor_lock:
                    bus.write("Goal_Position",
                             [current_shoulder_pan, current_wrist_flex],
                             ["shoulder_pan", "wrist_flex"])

            # Small delay to avoid overwhelming the motors
            time.sleep(0.05)  # 20 FPS tracking

        except Exception as e:
            print(f"Error in face tracking: {e}")
            time.sleep(0.1)

    print("Face tracking stopped!")


def generate_video_feed():
    """Generate video frames with face detection overlay"""
    global camera, face_cascade

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Detect faces for visualization
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add tracking status
        status_text = "TRACKING: ON" if tracking_active else "TRACKING: OFF"
        color = (0, 255, 0) if tracking_active else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw center crosshair
        h, w = frame.shape[:2]
        cv2.line(frame, (w//2 - 20, h//2), (w//2 + 20, h//2), (255, 0, 0), 1)
        cv2.line(frame, (w//2, h//2 - 20), (w//2, h//2 + 20), (255, 0, 0), 1)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/hello', methods=['GET'])
def hello_gesture():
    """
    Slow cute hello:
    - MAIN: wrist_roll head shake
    - MORE motor 1: shoulder_pan sway side-to-side
    - subtle elbow_flex wave stays secondary
    """
    try:
        smooth_move(NEUTRAL_POSITION, duration=1.5)

        sp0 = NEUTRAL_POSITION["shoulder_pan"]
        sl0 = NEUTRAL_POSITION["shoulder_lift"]
        ef0 = NEUTRAL_POSITION["elbow_flex"]
        wf0 = NEUTRAL_POSITION["wrist_flex"]
        wr0 = NEUTRAL_POSITION["wrist_roll"]
        g0  = NEUTRAL_POSITION["gripper"]

        # --- Motion sizes ---
        # Wrist roll still the "cute head shake"
        head_left  = wr0 + 140
        head_right = wr0 - 140

        # Motor 1 bigger sway than before (noticeable but not crazy)
        pan_left  = sp0 + 140
        pan_right = sp0 - 140

        # Gentle elbow wave (small)
        elbow_up   = ef0 + 80
        elbow_down = ef0 - 50

        # Slight friendly tilt up
        friendly_up = wf0 + 70

        # --- Timing (slow) ---
        pose_time = 0.9
        beat_time = 0.9

        # 1) Friendly starting pose
        smooth_move({
            "shoulder_pan": sp0,
            "shoulder_lift": sl0,
            "elbow_flex": ef0 + 30,
            "wrist_flex": friendly_up,
            "wrist_roll": wr0,
            "gripper": g0,
        }, duration=pose_time)

        # 2) Bigger side-to-side sway using motor 1
        #    Wrist roll leads each beat, pan follows to exaggerate sway
        for _ in range(2):
            # left beat
            smooth_move({
                "shoulder_pan": pan_left,      # motor 1 strong sway
                "shoulder_lift": sl0,
                "elbow_flex": elbow_up,
                "wrist_flex": friendly_up,
                "wrist_roll": head_left,      # still MAIN shake
                "gripper": g0,
            }, duration=beat_time)

            # right beat
            smooth_move({
                "shoulder_pan": pan_right,     # motor 1 strong sway
                "shoulder_lift": sl0,
                "elbow_flex": elbow_down,
                "wrist_flex": friendly_up,
                "wrist_roll": head_right,     # still MAIN shake
                "gripper": g0,
            }, duration=beat_time)

        # 3) Extra tiny motor-1 wiggle for cuteness (pan only, wrist near center)
        smooth_move({
            "shoulder_pan": sp0 + 80,
            "shoulder_lift": sl0,
            "elbow_flex": ef0,
            "wrist_flex": wf0 + 40,
            "wrist_roll": wr0 + 60,
            "gripper": g0,
        }, duration=0.7)

        smooth_move({
            "shoulder_pan": sp0 - 80,
            "shoulder_lift": sl0,
            "elbow_flex": ef0,
            "wrist_flex": wf0 + 40,
            "wrist_roll": wr0 - 60,
            "gripper": g0,
        }, duration=0.7)

        # 4) Return neutral slowly
        smooth_move(NEUTRAL_POSITION, duration=1.6)

        return jsonify({
            "status": "success",
            "gesture": "hello",
            "message": "Performed slow hello with stronger shoulder-pan sway!"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



@app.route('/forward', methods=['GET'])
def forward_gesture():
    """
    Forward gesture: tilts forward and slightly to the right
    Like looking at something interesting
    """
    try:
        # Start from neutral
        smooth_move(NEUTRAL_POSITION, duration=1.0)

        # Tilt forward (pitch down) and slightly right
        smooth_move({
            "shoulder_pan": 2048,
            "shoulder_lift": 1348,
            "elbow_flex": 2248,
            "wrist_flex": 2050,   # Tilt forward (look down from base 2348)
            "wrist_roll": 1850,   # Slight tilt to the right
            "gripper": 2048,
        }, duration=1.5)

        # Small adjustment - lean in a bit more
        smooth_move({
            "shoulder_pan": 2048,
            "shoulder_lift": 1348,
            "elbow_flex": 2248,
            "wrist_flex": 2000,   # Even more forward
            "wrist_roll": 1850,   # Keep right tilt
            "gripper": 2048,
        }, duration=0.8)

        return jsonify({
            "status": "success",
            "gesture": "forward",
            "message": "Performed forward gesture! (Call /neutral to return)"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/neutral', methods=['GET'])
def neutral_gesture():
    """
    Return to neutral position - head straight and centered
    """
    try:
        smooth_move(NEUTRAL_POSITION, duration=1.5)

        return jsonify({
            "status": "success",
            "gesture": "neutral",
            "message": "Returned to neutral position"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/status', methods=['GET'])
def status():
    """
    Get current motor positions
    """
    try:
        positions = get_current_positions()

        return jsonify({
            "status": "success",
            "positions": positions,
            "neutral": NEUTRAL_POSITION,
            "tracking": tracking_active
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/track/start', methods=['GET'])
def start_tracking():
    """
    Start face tracking
    """
    global tracking_active, tracking_thread

    if tracking_active:
        return jsonify({
            "status": "info",
            "message": "Face tracking is already active"
        })

    try:
        tracking_active = True
        tracking_thread = threading.Thread(target=face_tracking_loop, daemon=True)
        tracking_thread.start()

        return jsonify({
            "status": "success",
            "message": "Face tracking started! View at /video_feed"
        })

    except Exception as e:
        tracking_active = False
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/track/stop', methods=['GET'])
def stop_tracking():
    """
    Stop face tracking
    """
    global tracking_active

    if not tracking_active:
        return jsonify({
            "status": "info",
            "message": "Face tracking is not active"
        })

    try:
        tracking_active = False
        time.sleep(0.2)  # Give thread time to stop

        return jsonify({
            "status": "success",
            "message": "Face tracking stopped"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/video_feed')
def video_feed():
    """
    Video streaming route - displays camera feed with face detection
    """
    return Response(generate_video_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=['GET'])
def index():
    """
    API documentation
    """
    return jsonify({
        "name": "SO-101 Cute Gesture Server",
        "endpoints": {
            "/hello": "Performs a cute hello gesture with head tilts and waves",
            "/forward": "Tilts forward and right, like looking at something interesting",
            "/neutral": "Returns to neutral, straight position",
            "/status": "Returns current motor positions"
        },
        "usage": "Send GET requests to any endpoint above"
    })


def initialize_camera():
    """Initialize camera and face detection"""
    global camera, face_cascade

    print("\nInitializing camera...")
    camera = cv2.VideoCapture(4)  # Use /dev/video4

    if not camera.isOpened():
        print("Warning: Could not open camera! Face tracking will not work.")
        return False

    # Set camera resolution for better performance
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)

    print("Loading face detection model...")
    # Load Haar Cascade for face detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("Warning: Could not load face detection model!")
        return False

    print("Camera and face detection ready!")
    return True


def initialize_motors():
    """Initialize the motor bus"""
    global bus

    print("=" * 70)
    print("SO-101 Gesture Server - Initializing")
    print("=" * 70)

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

    # Configure motors
    print("Configuring motors...")
    bus.write("Mode", 0)  # Position control mode
    bus.write("P_Coefficient", 16)  # Reduce shakiness
    bus.write("I_Coefficient", 0)
    bus.write("D_Coefficient", 32)
    bus.write("Lock", 0)  # Unlock EPROM
    bus.write("Maximum_Acceleration", 254)
    bus.write("Acceleration", SPEED)
    bus.write("Torque_Enable", 1)

    # Move to neutral position
    print("Moving to neutral position...")
    bus.write("Goal_Position",
              list(NEUTRAL_POSITION.values()),
              list(NEUTRAL_POSITION.keys()))
    time.sleep(2)

    # Initialize camera
    initialize_camera()

    print("\n" + "=" * 70)
    print("Server ready!")
    print("=" * 70)
    print("\nAvailable endpoints:")
    print("  Gestures:")
    print("    - http://localhost:8080/hello       - Cute hello gesture")
    print("    - http://localhost:8080/forward     - Forward looking gesture")
    print("    - http://localhost:8080/neutral     - Return to neutral")
    print("  Face Tracking:")
    print("    - http://localhost:8080/track/start - Start face tracking")
    print("    - http://localhost:8080/track/stop  - Stop face tracking")
    print("    - http://localhost:8080/video_feed  - View camera feed")
    print("  Info:")
    print("    - http://localhost:8080/status      - Check motor positions")
    print("\n" + "=" * 70)


def cleanup():
    """Clean shutdown"""
    global bus, camera, tracking_active

    # Stop face tracking if active
    if tracking_active:
        tracking_active = False
        time.sleep(0.3)

    # Release camera
    if camera is not None:
        camera.release()
        print("Camera released")

    # Disconnect motors
    if bus:
        print("\nShutting down...")
        try:
            # Return to neutral before disconnecting
            bus.write("Goal_Position",
                      list(NEUTRAL_POSITION.values()),
                      list(NEUTRAL_POSITION.keys()))
            time.sleep(1)
            bus.disconnect()
            print("Motors disconnected safely")
        except Exception as e:
            print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    try:
        initialize_motors()
        app.run(host='0.0.0.0', port=8080, debug=False)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()