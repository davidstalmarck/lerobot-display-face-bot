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

# MediaPipe for hand detection
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_face_detection = mp.solutions.face_detection
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    face_detector = mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )
    mediapipe_available = True
    print("✓ MediaPipe loaded successfully")
except Exception as e:
    print(f"✗ MediaPipe not available: {e}")
    print("Install with: pip install mediapipe")
    mediapipe_available = False
    hands_detector = None
    face_detector = None

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
        return detections

    # Convert BGR to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]

    # Detect faces
    try:
        face_results = face_detector.process(frame_rgb)
        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)

                # Draw face box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                detections.append(f"face ({detection.score[0]:.0%})")
    except Exception as e:
        print(f"Face detection error: {e}")

    # Detect hands
    try:
        hand_results = hands_detector.process(frame_rgb)
        if hand_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # Get bounding box of hand
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                x_min = int(min(x_coords) * width)
                y_min = int(min(y_coords) * height)
                x_max = int(max(x_coords) * width)
                y_max = int(max(y_coords) * height)

                # Draw hand box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

                # Get hand label (left/right)
                hand_label = hand_results.multi_handedness[idx].classification[0].label
                cv2.putText(frame, f'{hand_label} Hand', (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                detections.append(f"{hand_label.lower()} hand")
    except Exception as e:
        print(f"Hand detection error: {e}")

    return detections


def detect_objects(frame):
    """Detect objects in frame"""
    global latest_detections, last_frame_for_llm

    # Store frame for LLM analysis
    last_frame_for_llm = frame.copy()

    # Detect faces and hands first
    face_hand_detections = detect_faces_and_hands(frame)

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

    return jsonify({
        "success": True,
        "recording": is_recording,
        "playing": is_playing
    })


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
