import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize camera and MediaPipe modules
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

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

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)  # Flip frame horizontally to mirror the webcam
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for facial landmarks
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # Blink detection using eye landmarks
        # Left eye (top: 145, bottom: 159)
        left_eye_top = landmarks[145]
        left_eye_bottom = landmarks[159]
        left_eye_distance = calculate_distance(left_eye_top, left_eye_bottom, frame_w, frame_h)

        # Right eye (top: 374, bottom: 386)
        right_eye_top = landmarks[374]
        right_eye_bottom = landmarks[386]
        right_eye_distance = calculate_distance(right_eye_top, right_eye_bottom, frame_w, frame_h)

        # Get current time for cooldown management
        current_time = time.time()

        # Check if both eyes are closed simultaneously
        both_eyes_closed = (left_eye_distance < blink_threshold) and (right_eye_distance < blink_threshold)

        # Check for left eye blink (left-click)
        if not both_eyes_closed and left_eye_distance < blink_threshold and (current_time - last_left_blink_time) > blink_cooldown:
            pyautogui.click(button='left')
            print("Left blink detected, left click triggered")
            last_left_blink_time = current_time  # Reset the last left blink time

        # Check for right eye blink (right-click)
        if not both_eyes_closed and right_eye_distance < blink_threshold and (current_time - last_right_blink_time) > blink_cooldown:
            pyautogui.click(button='right')
            print("Right blink detected, right click triggered")
            last_right_blink_time = current_time  # Reset the last right blink time

        # Visualize the eye landmarks
        # Draw circles around the left eye landmarks
        cv2.circle(frame, (int(left_eye_top.x * frame_w), int(left_eye_top.y * frame_h)), 5, (0, 255, 0), -1)  # Top of left eye
        cv2.circle(frame, (int(left_eye_bottom.x * frame_w), int(left_eye_bottom.y * frame_h)), 5, (0, 255, 0), -1)  # Bottom of left eye

        # Draw circles around the right eye landmarks
        cv2.circle(frame, (int(right_eye_top.x * frame_w), int(right_eye_top.y * frame_h)), 5, (255, 0, 0), -1)  # Top of right eye
        cv2.circle(frame, (int(right_eye_bottom.x * frame_w), int(right_eye_bottom.y * frame_h)), 5, (255, 0, 0), -1)  # Bottom of right eye

    # Show the frame with landmarks
    cv2.imshow('FocusFlow Eye Mouse Module', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close OpenCV windows
cam.release()
cv2.destroyAllWindows()
