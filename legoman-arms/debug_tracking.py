#!/usr/bin/env python3
"""
Face Tracking Debug Script for SO-101

This script tracks your face and controls:
- Motor 1 (shoulder_pan) for left/right tracking
- Motor 4 (wrist_flex) for up/down tracking

Features:
- Live camera feed with face detection
- 2-axis real-time motor control
- Easy to adjust tracking parameters

Press 'q' to quit
"""

import cv2
import numpy as np
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
import time

# Configuration
PORT = "/dev/ttyACM0"
CAMERA_ID = 4  # Use /dev/video4 (640x480)

# Starting position
NEUTRAL_POSITION = {
    "shoulder_pan": 2048,
    "shoulder_lift": 1348,
    "elbow_flex": 2248,
    "wrist_flex": 2348,
    "wrist_roll": 3072,  # Corrected for mounting
    "gripper": 2048,
}

# Tracking parameters
DEAD_ZONE = 80  # Pixels from center before we start moving
MOVE_SCALE = 0.5  # How aggressively to track (0.1 = slow, 1.0 = fast)
MAX_MOVE_PER_FRAME = 50  # Maximum position change per frame

def initialize_motors():
    """Initialize the motor bus"""
    print("Initializing motors...")
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
    bus.write("P_Coefficient", 16)
    bus.write("I_Coefficient", 0)
    bus.write("D_Coefficient", 32)
    bus.write("Lock", 0)
    bus.write("Maximum_Acceleration", 254)
    bus.write("Acceleration", 150)  # Smooth speed
    bus.write("Torque_Enable", 1)

    # Move to neutral
    print("Moving to neutral position...")
    bus.write("Goal_Position",
              list(NEUTRAL_POSITION.values()),
              list(NEUTRAL_POSITION.keys()))
    time.sleep(2)

    print("Motors ready!")
    return bus


def detect_face(frame, face_cascade):
    """
    Detect faces in frame using Haar Cascade
    Returns the center and bounding box of the largest face
    """
    # Convert to grayscale
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
        center_x = x + w // 2
        center_y = y + h // 2

        return (center_x, center_y, x, y, w, h)

    return None


def main():
    print("=" * 70)
    print("SO-101 Face Tracking Debug Script")
    print("=" * 70)
    print("\nInstructions:")
    print("  - Move your face in front of the camera")
    print("  - Motor 1 (shoulder_pan) tracks left/right")
    print("  - Motor 4 (wrist_flex) tracks up/down")
    print("  - Green box shows detected face")
    print("  - Red crosshair shows center target")
    print("  - Press 'q' to quit")
    print("=" * 70)

    # Initialize motors
    bus = initialize_motors()

    # Initialize camera
    print("\nOpening camera...")
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print("Error: Could not open camera!")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Camera opened successfully!")

    # Load face detection model
    print("Loading face detection model...")
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("Error: Could not load face detection model!")
        return

    print("Face detection ready!")
    print("\nTracking active. Press 'q' to quit.")

    # Current motor positions
    current_shoulder_pan = NEUTRAL_POSITION["shoulder_pan"]  # Motor 1 (left/right)
    current_wrist_flex = NEUTRAL_POSITION["wrist_flex"]      # Motor 4 (up/down)

    # Get frame dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read initial frame!")
        return

    frame_height, frame_width = frame.shape[:2]
    center_x = frame_width // 2
    center_y = frame_height // 2

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break

        # Detect face
        face_data = detect_face(frame, face_cascade)

        # Draw center crosshair (red)
        cv2.line(frame, (center_x - 30, center_y), (center_x + 30, center_y), (0, 0, 255), 2)
        cv2.line(frame, (center_x, center_y - 30), (center_x, center_y + 30), (0, 0, 255), 2)

        # Draw dead zone rectangle (for both X and Y)
        cv2.rectangle(frame,
                     (center_x - DEAD_ZONE, center_y - DEAD_ZONE),
                     (center_x + DEAD_ZONE, center_y + DEAD_ZONE),
                     (255, 200, 0), 2)

        if face_data is not None:
            face_x, face_y, x, y, w, h = face_data

            # Draw bounding box (green)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "FACE", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw center of face (blue circle)
            cv2.circle(frame, (face_x, face_y), 10, (255, 0, 0), -1)

            # Draw line from center to face
            cv2.line(frame, (center_x, center_y), (face_x, face_y), (255, 255, 0), 2)

            # Calculate offsets from center
            offset_x = face_x - center_x
            offset_y = face_y - center_y

            # Track horizontally (motor 1 - shoulder_pan)
            if abs(offset_x) > DEAD_ZONE:
                # Positive offset_x = face is right, turn right (increase shoulder_pan)
                # Negative offset_x = face is left, turn left (decrease shoulder_pan)
                move_x = int(offset_x * MOVE_SCALE)
                move_x = max(-MAX_MOVE_PER_FRAME, min(MAX_MOVE_PER_FRAME, move_x))
                current_shoulder_pan = max(1500, min(2600, current_shoulder_pan + move_x))

            # Track vertically (motor 4 - wrist_flex)
            if abs(offset_y) > DEAD_ZONE:
                # Positive offset_y = face is down, look down (increase wrist_flex)
                # Negative offset_y = face is up, look up (decrease wrist_flex)
                move_y = int(offset_y * MOVE_SCALE * 0.4)  # Reduce vertical sensitivity
                move_y = max(-MAX_MOVE_PER_FRAME, min(MAX_MOVE_PER_FRAME, move_y))
                current_wrist_flex = max(1800, min(2800, current_wrist_flex + move_y))

            # Apply movements to both motors
            bus.write("Goal_Position",
                     [current_shoulder_pan, current_wrist_flex],
                     ["shoulder_pan", "wrist_flex"])

            # Display tracking info
            in_dead_zone_x = abs(offset_x) <= DEAD_ZONE
            in_dead_zone_y = abs(offset_y) <= DEAD_ZONE

            if in_dead_zone_x and in_dead_zone_y:
                status = "CENTERED"
                color = (0, 255, 255)
            else:
                status = "TRACKING"
                color = (0, 255, 0)

            cv2.putText(frame, f"{status} | X: {offset_x:+4d}px, Y: {offset_y:+4d}px",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Motor 1: {current_shoulder_pan} | Motor 4: {current_wrist_flex}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # No face detected
            cv2.putText(frame, "NO FACE DETECTED",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Motor 1: {current_shoulder_pan} | Motor 4: {current_wrist_flex}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show instructions
        cv2.putText(frame, "Press 'q' to quit",
                   (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the frame
        cv2.imshow("Face Tracking - SO-101", frame)

        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break

    # Cleanup
    print("\nReturning to neutral position...")
    bus.write("Goal_Position",
              list(NEUTRAL_POSITION.values()),
              list(NEUTRAL_POSITION.keys()))
    time.sleep(1)

    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
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