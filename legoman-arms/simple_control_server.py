#!/usr/bin/env env python
"""
Simple control server for robot UI
Provides endpoints for record, play, stop, and relax functionality
"""

import json
import time
import threading
from pathlib import Path
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import cv2
import numpy as np
import base64
import requests

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig

# MediaPipe for hand detection (using new API for v0.10.30+)
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    # Create detectors with new API
    base_options_hands = python.BaseOptions(model_asset_path=None)
    hands_options = vision.HandLandmarkerOptions(
        base_options=base_options_hands,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    hands_detector = vision.HandLandmarker.create_from_options(hands_options)

    base_options_face = python.BaseOptions(model_asset_path=None)
    face_options = vision.FaceDetectorOptions(
        base_options=base_options_face,
        running_mode=vision.RunningMode.VIDEO,
        min_detection_confidence=0.3
    )
    face_detector = vision.FaceDetector.create_from_options(face_options)

    base_options_pose = python.BaseOptions(model_asset_path=None)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=base_options_pose,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    mediapipe_available = True
    print("✓ MediaPipe loaded successfully (v0.10.30+ API)")
except Exception as e:
    print(f"✗ MediaPipe not available: {e}")
    print("Install with: pip install mediapipe")
    mediapipe_available = False
    hands_detector = None
    face_detector = None
    pose_detector = None

app = Flask(__name__)
CORS(app)

# Configuration
PORT = "/dev/ttyACM0"  # Leader arm
RECORDING_DIR = Path(__file__).parent / "recordings"
RECORDING_DIR.mkdir(exist_ok=True)
CAMERA_INDEX = 4  # Use external camera (change to 0 for built-in, 2 or 4 for external)

# Motor configuration (5 motors, motor 6 disabled)
MOTOR_CONFIG = FeetechMotorsBusConfig(
    port=PORT,
    motors={
        "shoulder_pan": [1, "sts3215"],
        "shoulder_lift": [2, "sts3215"],
        "elbow_flex": [3, "sts3215"],
        "wrist_flex": [4, "sts3215"],
        "wrist_roll": [5, "sts3215"],
    }
)

MOTOR_NAMES = list(MOTOR_CONFIG.motors.keys())

# Global state
bus = None
recording_thread = None
playback_thread = None
stop_flag = threading.Event()
current_recording = []
current_recording_name = None

# Camera and detection state
camera = None
latest_detections = []
latest_description = "Starting up..."
detection_lock = threading.Lock()
last_frame_for_llm = None
last_llm_time = 0

# Arm tracking state
arm_tracking_enabled = False
arm_tracking_thread = None
arm_tracking_lock = threading.Lock()

# LLM configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llava"  # Vision-capable model
LLM_INTERVAL = 3.0  # Seconds between LLM calls

# Check if Ollama is available
try:
    response = requests.get("http://localhost:11434/api/tags", timeout=2)
    ollama_available = response.status_code == 200
    if ollama_available:
        print("✓ Ollama is available")
        # Check if llava model is installed
        models = response.json().get('models', [])
        model_names = [m.get('name', '') for m in models]
        if not any('llava' in name for name in model_names):
            print("⚠ llava model not found. Run: ollama pull llava")
            print(f"Available models: {model_names}")
except Exception as e:
    print(f"✗ Ollama not available: {e}")
    print("To use vision AI, install Ollama and run: ollama pull llava")
    ollama_available = False

# Initialize YOLO for object detection (optional, as backup)
try:
    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    use_yolo = True
    print("✓ YOLO loaded successfully")
except Exception as e:
    print(f"✗ YOLO not available (using LLM only): {e}")
    use_yolo = False
    classes = []
    net = None


def initialize_bus():
    """Initialize motor bus connection"""
    global bus
    if bus is None:
        bus = FeetechMotorsBus(MOTOR_CONFIG)
        bus.connect()
    elif not bus.is_connected:
        # Reconnect if bus exists but is disconnected
        print("Reconnecting motor bus...")
        bus.connect()
    return bus


def initialize_camera():
    """Initialize camera"""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(CAMERA_INDEX)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"✓ Camera initialized: /dev/video{CAMERA_INDEX}")
    return camera


def analyze_frame_with_llm(frame):
    """Analyze frame with local LLM"""
    global latest_description, last_llm_time

    if not ollama_available:
        return

    current_time = time.time()
    if current_time - last_llm_time < LLM_INTERVAL:
        return

    last_llm_time = current_time

    # Encode frame as base64
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Call Ollama API
    try:
        prompt = "Describe what you see in this image in one short sentence (10-15 words max). Focus on the main objects and scene."

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False
            },
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            description = result.get('response', '').strip()

            with detection_lock:
                latest_description = description
                print(f"LLM: {description}")
    except Exception as e:
        print(f"LLM error: {e}")


def detect_faces_and_hands(frame):
    """Detect faces and hands using MediaPipe"""
    detections = []

    if not mediapipe_available:
        print("MediaPipe not available!")
        return detections

    # Convert BGR to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]

    # Create MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int(time.time() * 1000)

    # Detect faces
    try:
        face_results = face_detector.detect_for_video(mp_image, timestamp_ms)
        if face_results.detections:
            print(f"✓ Found {len(face_results.detections)} face(s)")
            for detection in face_results.detections:
                bbox = detection.bounding_box
                x = bbox.origin_x
                y = bbox.origin_y
                w = bbox.width
                h = bbox.height

                # Draw face box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)
                cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

                # Get confidence score from first category
                score = detection.categories[0].score if detection.categories else 0.0
                detections.append(f"face ({score:.0%})")
        else:
            # Only log occasionally to avoid spam
            import random
            if random.random() < 0.1:  # 10% of frames
                print("No faces detected in frame")
    except Exception as e:
        print(f"Face detection error: {e}")
        import traceback
        traceback.print_exc()

    # Detect hands
    try:
        hand_results = hands_detector.detect_for_video(mp_image, timestamp_ms)
        if hand_results.hand_landmarks:
            print(f"✓ Found {len(hand_results.hand_landmarks)} hand(s)")
            for idx, hand_landmarks in enumerate(hand_results.hand_landmarks):
                # Get bounding box of hand (from normalized landmarks)
                x_coords = [lm.x for lm in hand_landmarks]
                y_coords = [lm.y for lm in hand_landmarks]

                x_min = int(min(x_coords) * width)
                y_min = int(min(y_coords) * height)
                x_max = int(max(x_coords) * width)
                y_max = int(max(y_coords) * height)

                # Draw hand box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 3)

                # Get hand label (left/right)
                if idx < len(hand_results.handedness):
                    hand_label = hand_results.handedness[idx][0].category_name
                    cv2.putText(frame, f'{hand_label} Hand', (x_min, y_min - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    detections.append(f"{hand_label.lower()} hand")
        else:
            # Only log occasionally to avoid spam
            import random
            if random.random() < 0.1:  # 10% of frames
                print("No hands detected in frame")
    except Exception as e:
        print(f"Hand detection error: {e}")
        import traceback
        traceback.print_exc()

    return detections


def detect_and_track_arm(frame):
    """Detect user's arm and return tracking information"""
    if not mediapipe_available or pose_detector is None:
        return None

    # Convert BGR to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]

    # Create MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int(time.time() * 1000)

    try:
        pose_results = pose_detector.detect_for_video(mp_image, timestamp_ms)

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks[0]  # Get first pose

            # Get right arm landmarks (use left arm if user faces camera)
            # Using LEFT side landmarks because user faces camera (their left = our right)
            # Landmark indices: LEFT_SHOULDER=11, LEFT_ELBOW=13, LEFT_WRIST=15
            shoulder = landmarks[11]
            elbow = landmarks[13]
            wrist = landmarks[15]

            # Draw arm skeleton
            def draw_point(landmark, color):
                cx = int(landmark.x * width)
                cy = int(landmark.y * height)
                cv2.circle(frame, (cx, cy), 8, color, -1)
                return cx, cy

            shoulder_pos = draw_point(shoulder, (255, 0, 0))  # Blue
            elbow_pos = draw_point(elbow, (0, 255, 0))  # Green
            wrist_pos = draw_point(wrist, (0, 0, 255))  # Red

            # Draw arm lines
            cv2.line(frame, shoulder_pos, elbow_pos, (255, 255, 0), 3)
            cv2.line(frame, elbow_pos, wrist_pos, (255, 255, 0), 3)

            # Calculate angles
            import math

            # Shoulder angle (horizontal)
            shoulder_angle_x = math.degrees(math.atan2(elbow.y - shoulder.y, elbow.x - shoulder.x))

            # Elbow angle
            upper_arm_vec = (elbow.x - shoulder.x, elbow.y - shoulder.y)
            forearm_vec = (wrist.x - elbow.x, wrist.y - elbow.y)

            dot_product = upper_arm_vec[0] * forearm_vec[0] + upper_arm_vec[1] * forearm_vec[1]
            upper_arm_len = math.sqrt(upper_arm_vec[0]**2 + upper_arm_vec[1]**2)
            forearm_len = math.sqrt(forearm_vec[0]**2 + forearm_vec[1]**2)

            if upper_arm_len > 0 and forearm_len > 0:
                cos_angle = dot_product / (upper_arm_len * forearm_len)
                cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to [-1, 1]
                elbow_angle = math.degrees(math.acos(cos_angle))
            else:
                elbow_angle = 0

            # Shoulder lift angle (vertical)
            shoulder_angle_y = shoulder.y - elbow.y

            # Wrist angle
            wrist_angle = math.degrees(math.atan2(wrist.y - elbow.y, wrist.x - elbow.x))

            return {
                'shoulder': (shoulder.x, shoulder.y, shoulder.z),
                'elbow': (elbow.x, elbow.y, elbow.z),
                'wrist': (wrist.x, wrist.y, wrist.z),
                'shoulder_angle_x': shoulder_angle_x,
                'shoulder_angle_y': shoulder_angle_y,
                'elbow_angle': elbow_angle,
                'wrist_angle': wrist_angle
            }
    except Exception as e:
        print(f"Arm tracking error: {e}")
        import traceback
        traceback.print_exc()

    return None


def map_arm_to_motors(arm_data):
    """Map arm angles to motor positions"""
    if arm_data is None:
        return None

    # Motor ranges (approximate, may need tuning)
    SHOULDER_PAN_MIN = 1500
    SHOULDER_PAN_MAX = 2600
    SHOULDER_PAN_CENTER = 2048

    SHOULDER_LIFT_MIN = 1000
    SHOULDER_LIFT_MAX = 2000
    SHOULDER_LIFT_CENTER = 1348

    ELBOW_FLEX_MIN = 1500
    ELBOW_FLEX_MAX = 2800
    ELBOW_FLEX_CENTER = 2248

    WRIST_FLEX_MIN = 1800
    WRIST_FLEX_MAX = 2800
    WRIST_FLEX_CENTER = 2348

    # Map shoulder horizontal angle to shoulder_pan
    # arm_data['shoulder_angle_x'] ranges from -180 to 180
    shoulder_pan = SHOULDER_PAN_CENTER + int((arm_data['shoulder_angle_x'] + 90) * (SHOULDER_PAN_MAX - SHOULDER_PAN_MIN) / 180)
    shoulder_pan = max(SHOULDER_PAN_MIN, min(SHOULDER_PAN_MAX, shoulder_pan))

    # Map shoulder vertical position to shoulder_lift
    # arm_data['shoulder_angle_y'] ranges from -1 to 1 (normalized)
    shoulder_lift = SHOULDER_LIFT_CENTER - int(arm_data['shoulder_angle_y'] * 800)
    shoulder_lift = max(SHOULDER_LIFT_MIN, min(SHOULDER_LIFT_MAX, shoulder_lift))

    # Map elbow angle to elbow_flex
    # elbow_angle ranges from 0 (straight) to 180 (bent)
    elbow_flex = ELBOW_FLEX_CENTER + int((180 - arm_data['elbow_angle']) * 2)
    elbow_flex = max(ELBOW_FLEX_MIN, min(ELBOW_FLEX_MAX, elbow_flex))

    # Map wrist angle to wrist_flex
    wrist_flex = WRIST_FLEX_CENTER + int((arm_data['wrist_angle'] + 90) * 2)
    wrist_flex = max(WRIST_FLEX_MIN, min(WRIST_FLEX_MAX, wrist_flex))

    return {
        "shoulder_pan": shoulder_pan,
        "shoulder_lift": shoulder_lift,
        "elbow_flex": elbow_flex,
        "wrist_flex": wrist_flex,
        "wrist_roll": 3072  # Keep constant
    }


def arm_tracking_loop():
    """Arm tracking control loop"""
    global arm_tracking_enabled

    cam = initialize_camera()

    try:
        bus = initialize_bus()
        configure_motors_for_playback()  # Enable torque for controlled movement

        print("Arm tracking started!")

        while arm_tracking_enabled:
            success, frame = cam.read()
            if not success:
                continue

            # Detect arm
            arm_data = detect_and_track_arm(frame)

            if arm_data:
                # Map to motor positions
                motor_positions = map_arm_to_motors(arm_data)

                if motor_positions:
                    try:
                        # Write positions to motors
                        bus.write("Goal_Position",
                                [motor_positions["shoulder_pan"],
                                 motor_positions["shoulder_lift"],
                                 motor_positions["elbow_flex"],
                                 motor_positions["wrist_flex"],
                                 motor_positions["wrist_roll"]],
                                ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"])
                    except Exception as e:
                        print(f"Error writing motor positions: {e}")

            # Small delay to prevent overwhelming the motors
            time.sleep(0.05)  # 20 Hz update rate

    except Exception as e:
        print(f"Arm tracking loop error: {e}")
        import traceback
        traceback.print_exc()


def detect_objects(frame):
    """Detect objects in frame"""
    global latest_detections, last_frame_for_llm

    # Store frame for LLM analysis
    last_frame_for_llm = frame.copy()

    # Detect faces and hands first
    face_hand_detections = detect_faces_and_hands(frame)

    # If arm tracking is enabled, also draw arm tracking
    if arm_tracking_enabled:
        arm_data = detect_and_track_arm(frame)
        if arm_data:
            # Draw "ARM TRACKING" indicator
            cv2.putText(frame, "ARM TRACKING ACTIVE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if use_yolo and net is not None:
        # YOLO detection
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(output_layers)

        detections = []
        height, width = frame.shape[:2]

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    # Get bounding box
                    box = detection[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, w, h) = box.astype("int")

                    x = int(centerX - (w / 2))
                    y = int(centerY - (h / 2))

                    detections.append({
                        "class": classes[class_id],
                        "confidence": float(confidence),
                        "box": [x, y, int(w), int(h)]
                    })

                    # Draw on frame
                    cv2.rectangle(frame, (x, y), (x + int(w), y + int(h)), (0, 255, 0), 2)
                    label = f"{classes[class_id]}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        with detection_lock:
            latest_detections = face_hand_detections + [f"{d['class']} ({d['confidence']:.0%})" for d in detections]
    else:
        # Simple color-based detection as fallback
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        detections = []

        # Detect red objects
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        if cv2.countNonZero(mask_red) > 1000:
            detections.append("red object")

        # Detect blue objects
        lower_blue = np.array([100, 120, 70])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        if cv2.countNonZero(mask_blue) > 1000:
            detections.append("blue object")

        # Detect green objects
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        if cv2.countNonZero(mask_green) > 1000:
            detections.append("green object")

        with detection_lock:
            latest_detections = face_hand_detections + detections

    return frame


def generate_frames():
    """Generate video frames with detection"""
    cam = initialize_camera()

    while True:
        success, frame = cam.read()
        if not success:
            break

        # Detect objects
        frame = detect_objects(frame)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def configure_motors_for_recording():
    """Configure motors for recording mode (torque disabled)"""
    bus = initialize_bus()
    print("Disabling torque for manual movement...")
    bus.write("Torque_Enable", 0)


def configure_motors_for_playback():
    """Configure motors for playback mode (torque enabled with smooth settings)"""
    bus = initialize_bus()
    print("Enabling torque for playback...")
    bus.write("Mode", 0)  # Position control mode
    bus.write("P_Coefficient", 16)
    bus.write("I_Coefficient", 0)
    bus.write("D_Coefficient", 32)
    bus.write("Lock", 0)
    bus.write("Maximum_Acceleration", 254)
    bus.write("Acceleration", 150)  # Smooth speed
    bus.write("Torque_Enable", 1)


def record_loop(name):
    """Recording loop - reads positions at fixed frequency"""
    global current_recording
    current_recording = []

    bus = None
    start_time = time.time()
    frequency = 20  # Hz

    try:
        bus = initialize_bus()
        configure_motors_for_recording()

        interval = 1.0 / frequency

        print(f"Recording '{name}' started...")

        while not stop_flag.is_set():
            loop_start = time.time()

            try:
                # Read current positions
                positions = bus.read("Present_Position", MOTOR_NAMES)

                # Store frame (convert numpy types to Python native types for JSON serialization)
                frame = {
                    "time": time.time() - start_time,
                    "positions": {name: int(pos) for name, pos in zip(MOTOR_NAMES, positions)}
                }
                current_recording.append(frame)
            except Exception as e:
                print(f"Error reading motor positions: {e}")
                # Continue recording even if one frame fails

            # Sleep to maintain frequency
            elapsed = time.time() - loop_start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    except Exception as e:
        print(f"ERROR in record_loop: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Always try to save the recording, even if errors occurred
        try:
            duration = time.time() - start_time
            recording_data = {
                "arm": "leader",
                "frequency": frequency,
                "duration": duration,
                "motor_names": MOTOR_NAMES,
                "frames": current_recording
            }

            recording_file = RECORDING_DIR / f"{name}.json"
            with open(recording_file, 'w') as f:
                json.dump(recording_data, f, indent=2)

            print(f"Recording '{name}' saved: {len(current_recording)} frames, {duration:.2f} seconds")
        except Exception as e:
            print(f"CRITICAL ERROR saving recording '{name}': {e}")
            import traceback
            traceback.print_exc()


def playback_loop(name, speed=1.0):
    """Playback loop - replays recorded movements"""
    recording_file = RECORDING_DIR / f"{name}.json"
    if not recording_file.exists():
        print(f"Recording '{name}' not found")
        return

    try:
        bus = initialize_bus()
        configure_motors_for_playback()

        # Load recording
        with open(recording_file, 'r') as f:
            recording_data = json.load(f)

        frames = recording_data["frames"]
        motor_names = recording_data["motor_names"]

        print(f"Playing back '{name}': {len(frames)} frames at {speed}x speed...")

        start_time = time.time()

        for i, frame in enumerate(frames):
            if stop_flag.is_set():
                break

            # Wait until the correct time (adjusted for speed)
            target_time = frame["time"] / speed
            current_time = time.time() - start_time
            wait_time = target_time - current_time

            if wait_time > 0:
                time.sleep(wait_time)

            try:
                # Write positions
                positions = [frame["positions"][name] for name in motor_names]
                bus.write("Goal_Position", positions, motor_names)
            except Exception as e:
                print(f"Error writing positions at frame {i}: {e}")
                # Continue playback even if one frame fails

        print("Playback complete")

    except Exception as e:
        print(f"ERROR in playback_loop: {e}")
        import traceback
        traceback.print_exc()


@app.route('/start_record', methods=['POST'])
def start_record():
    """Start recording"""
    global recording_thread, stop_flag, current_recording_name

    if recording_thread and recording_thread.is_alive():
        return jsonify({"success": False, "message": "Already recording"})

    data = request.get_json() or {}
    name = data.get('name', 'recording')
    current_recording_name = name

    stop_flag.clear()
    recording_thread = threading.Thread(target=record_loop, args=(name,), daemon=True)
    recording_thread.start()

    return jsonify({"success": True, "message": "Recording started"})


@app.route('/stop_record', methods=['POST'])
def stop_record():
    """Stop recording"""
    global recording_thread, stop_flag

    if not recording_thread or not recording_thread.is_alive():
        return jsonify({"success": False, "message": "Not recording"})

    stop_flag.set()
    recording_thread.join(timeout=10)  # Increased timeout to ensure save completes

    return jsonify({"success": True, "message": f"Recording saved ({len(current_recording)} frames)"})


@app.route('/play', methods=['POST'])
def play():
    """Play recording"""
    global playback_thread, stop_flag

    data = request.get_json() or {}
    name = data.get('name')
    speed = data.get('speed', 1.0)

    if not name:
        return jsonify({"success": False, "message": "No recording name provided"})

    recording_file = RECORDING_DIR / f"{name}.json"
    if not recording_file.exists():
        return jsonify({"success": False, "message": f"Recording '{name}' not found"})

    if playback_thread and playback_thread.is_alive():
        return jsonify({"success": False, "message": "Already playing"})

    # Get duration for UI feedback
    with open(recording_file, 'r') as f:
        recording_data = json.load(f)
    duration = recording_data.get('duration', 10)

    stop_flag.clear()
    playback_thread = threading.Thread(target=playback_loop, args=(name, speed), daemon=True)
    playback_thread.start()

    return jsonify({"success": True, "message": "Playing recording", "duration": duration})


@app.route('/stop', methods=['POST'])
def stop():
    """Stop any current operation"""
    global stop_flag
    stop_flag.set()

    # Wait for threads to stop
    if recording_thread and recording_thread.is_alive():
        recording_thread.join(timeout=2)
    if playback_thread and playback_thread.is_alive():
        playback_thread.join(timeout=2)

    return jsonify({"success": True, "message": "Stopped"})


@app.route('/relax', methods=['POST'])
def relax():
    """Disable torque (make motors soft)"""
    try:
        bus = initialize_bus()
        bus.write("Torque_Enable", 0)
        return jsonify({"success": True, "message": "Motors relaxed"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route('/list_recordings', methods=['GET'])
def list_recordings():
    """List all recordings"""
    recordings = [f.stem for f in RECORDING_DIR.glob("*.json")]
    recordings.sort()
    return jsonify({
        "success": True,
        "recordings": recordings
    })


@app.route('/delete_recording', methods=['POST'])
def delete_recording():
    """Delete a recording"""
    data = request.get_json() or {}
    name = data.get('name')

    if not name:
        return jsonify({"success": False, "message": "No recording name provided"})

    recording_file = RECORDING_DIR / f"{name}.json"
    if not recording_file.exists():
        return jsonify({"success": False, "message": f"Recording '{name}' not found"})

    try:
        recording_file.unlink()
        return jsonify({"success": True, "message": f"Deleted '{name}'"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detections', methods=['GET'])
def get_detections():
    """Get latest detections and description"""
    with detection_lock:
        detections = latest_detections.copy()
        description = latest_description

    return jsonify({
        "success": True,
        "detections": detections,
        "description": description
    })


def llm_analysis_loop():
    """Background thread for LLM analysis"""
    global last_frame_for_llm

    while True:
        try:
            if last_frame_for_llm is not None and ollama_available:
                analyze_frame_with_llm(last_frame_for_llm)
            time.sleep(0.5)  # Check every 500ms
        except Exception as e:
            print(f"LLM loop error: {e}")
            time.sleep(1)


@app.route('/status', methods=['GET'])
def status():
    """Get current status"""
    is_recording = recording_thread and recording_thread.is_alive()
    is_playing = playback_thread and playback_thread.is_alive()
    is_arm_tracking = arm_tracking_thread and arm_tracking_thread.is_alive()

    return jsonify({
        "success": True,
        "recording": is_recording,
        "playing": is_playing,
        "arm_tracking": is_arm_tracking,
        "camera_index": CAMERA_INDEX
    })


@app.route('/switch_camera', methods=['POST'])
def switch_camera():
    """Switch to next camera"""
    global camera, CAMERA_INDEX

    # Release current camera
    if camera is not None:
        camera.release()
        camera = None

    # Try next camera indices
    max_cameras = 6
    original_index = CAMERA_INDEX

    for _ in range(max_cameras):
        CAMERA_INDEX = (CAMERA_INDEX + 1) % max_cameras
        try:
            test_cam = cv2.VideoCapture(CAMERA_INDEX)
            if test_cam.isOpened():
                ret, _ = test_cam.read()
                test_cam.release()
                if ret:
                    # Found a working camera
                    print(f"Switched to camera {CAMERA_INDEX}")
                    return jsonify({
                        "success": True,
                        "camera_index": CAMERA_INDEX,
                        "message": f"Switched to camera {CAMERA_INDEX}"
                    })
        except Exception as e:
            print(f"Camera {CAMERA_INDEX} failed: {e}")
            continue

    # If no camera found, restore original
    CAMERA_INDEX = original_index
    return jsonify({
        "success": False,
        "camera_index": CAMERA_INDEX,
        "message": "No other working cameras found"
    })


@app.route('/start_arm_tracking', methods=['POST'])
def start_arm_tracking():
    """Start arm tracking mode"""
    global arm_tracking_enabled, arm_tracking_thread

    if not mediapipe_available or pose_detector is None:
        return jsonify({"success": False, "message": "MediaPipe not available"})

    if arm_tracking_thread and arm_tracking_thread.is_alive():
        return jsonify({"success": False, "message": "Arm tracking already active"})

    # Stop any recording or playback first
    if (recording_thread and recording_thread.is_alive()) or (playback_thread and playback_thread.is_alive()):
        return jsonify({"success": False, "message": "Stop recording/playback first"})

    arm_tracking_enabled = True
    arm_tracking_thread = threading.Thread(target=arm_tracking_loop, daemon=True)
    arm_tracking_thread.start()

    return jsonify({"success": True, "message": "Arm tracking started"})


@app.route('/stop_arm_tracking', methods=['POST'])
def stop_arm_tracking():
    """Stop arm tracking mode"""
    global arm_tracking_enabled, arm_tracking_thread

    if not arm_tracking_thread or not arm_tracking_thread.is_alive():
        return jsonify({"success": False, "message": "Arm tracking not active"})

    arm_tracking_enabled = False
    arm_tracking_thread.join(timeout=2)

    return jsonify({"success": True, "message": "Arm tracking stopped"})


if __name__ == "__main__":
    print("Starting simple control server on port 5001...")
    print("Open simple_control_ui.html in your browser")

    # Start LLM analysis thread
    if ollama_available:
        llm_thread = threading.Thread(target=llm_analysis_loop, daemon=True)
        llm_thread.start()
        print("✓ LLM analysis thread started")

    try:
        app.run(host='0.0.0.0', port=5001, debug=False)
    finally:
        if bus:
            bus.disconnect()
        if camera:
            camera.release()
