import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize camera and MediaPipe modules
def init_camera():
    return cv2.VideoCapture(0)

def init_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

def get_screen_size():
    return 1920, 1080  # Hardcoded for testing purposes, replace with pyautogui in actual code

# Variables for blink detection thresholds and cooldown
blink_threshold = 3.5  # Distance threshold for blink detection
blink_cooldown = 1  # Cooldown to avoid multiple clicks in quick succession
last_left_blink_time = 0
last_right_blink_time = 0

# Helper function to calculate the distance between two landmarks
def calculate_distance(landmark1, landmark2, frame_w, frame_h):
    x1, y1 = landmark1.x * frame_w, landmark1.y * frame_h
    x2, y2 = landmark2.x * frame_w, landmark2.y * frame_h
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Blink detection logic (no pyautogui, just return a list of actions)
def detect_blink(landmarks, frame_w, frame_h, current_time, blink_threshold, last_left_blink_time, last_right_blink_time):
    actions = []  # Store actions (like left-click or right-click)

    # Left eye (top: 145, bottom: 159)
    left_eye_top = landmarks[145]
    left_eye_bottom = landmarks[159]
    left_eye_distance = calculate_distance(left_eye_top, left_eye_bottom, frame_w, frame_h)

    # Right eye (top: 374, bottom: 386)
    right_eye_top = landmarks[374]
    right_eye_bottom = landmarks[386]
    right_eye_distance = calculate_distance(right_eye_top, right_eye_bottom, frame_w, frame_h)

    # Check if both eyes are closed simultaneously
    both_eyes_closed = (left_eye_distance < blink_threshold) and (right_eye_distance < blink_threshold)

    if not both_eyes_closed and left_eye_distance < blink_threshold and (current_time - last_left_blink_time) > blink_cooldown:
        actions.append("Left blink detected, left click triggered")
        last_left_blink_time = current_time  # Reset the last left blink time

    if not both_eyes_closed and right_eye_distance < blink_threshold and (current_time - last_right_blink_time) > blink_cooldown:
        actions.append("Right blink detected, right click triggered")
        last_right_blink_time = current_time  # Reset the last right blink time

    return actions, last_left_blink_time, last_right_blink_time

# Main loop (mocked for testing purposes)
def process_frame(cam, face_mesh, blink_threshold, blink_cooldown):
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)  # Flip frame horizontally to mirror the webcam
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for facial landmarks
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    actions = []
    if landmark_points:
        landmarks = landmark_points[0].landmark
        current_time = time.time()
        actions, last_left_blink_time, last_right_blink_time = detect_blink(
            landmarks, frame_w, frame_h, current_time, blink_threshold, blink_cooldown, last_left_blink_time, last_right_blink_time
        )

    return frame, actions, last_left_blink_time, last_right_blink_time
