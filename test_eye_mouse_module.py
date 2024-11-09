import pytest
import os
from unittest.mock import MagicMock, patch
import time
import numpy as np
import cv2

# Import functions from the actual code
from focusflow import (
    init_camera,
    init_face_mesh,
    get_screen_size,
    calculate_distance,
    detect_blink,
    process_frame
)

# Test the camera initialization function
def test_init_camera():
    cam = init_camera()
    assert isinstance(cam, cv2.VideoCapture), "Camera initialization failed"

# Test the FaceMesh initialization
def test_init_face_mesh():
    face_mesh = init_face_mesh()
    assert isinstance(face_mesh, mp.solutions.face_mesh.FaceMesh), "FaceMesh initialization failed"

# Test the screen size retrieval function (mocking pyautogui)
def test_get_screen_size():
    # Mock pyautogui.size to return a fixed screen size
    pyautogui.size = MagicMock(return_value=(1920, 1080))
    screen_w, screen_h = get_screen_size()
    assert screen_w == 1920, f"Expected screen width to be 1920, but got {screen_w}"
    assert screen_h == 1080, f"Expected screen height to be 1080, but got {screen_h}"

# Test the calculate_distance function
def test_calculate_distance():
    # Mock landmarks with simple x, y values
    class Landmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    landmark1 = Landmark(0.1, 0.2)
    landmark2 = Landmark(0.4, 0.5)
    frame_w, frame_h = 640, 480  # Mock frame size

    distance = calculate_distance(landmark1, landmark2, frame_w, frame_h)
    expected_distance = np.sqrt((0.4 - 0.1) ** 2 + (0.5 - 0.2) ** 2) * frame_w  # Mocked distance formula
    assert np.isclose(distance, expected_distance, atol=1e-2), f"Expected distance {expected_distance}, but got {distance}"

# Test the blink detection function (mocking pyautogui.click)
@patch('pyautogui.click')  # Mock pyautogui.click to prevent actual clicks
def test_detect_blink(mock_click):
    # Create mock landmarks (just mock x, y values for eyes)
    class Landmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    landmarks = [Landmark(0.1, 0.2), Landmark(0.3, 0.4)]  # Mocking eyes' landmarks
    frame_w, frame_h = 640, 480  # Mock frame size
    current_time = time.time()
    blink_threshold = 3.5
    blink_cooldown = 1
    last_left_blink_time = 0
    last_right_blink_time = 0

    # Run the blink detection
    last_left_blink_time, last_right_blink_time = detect_blink(
        landmarks, frame_w, frame_h, current_time, blink_threshold, blink_cooldown, last_left_blink_time, last_right_blink_time
    )

    # Check that the click function was called
    mock_click.assert_called_with(button='left')  # Ensure left click was called
    assert last_left_blink_time != 0, "Last left blink time should have been updated"

# Test the main frame processing logic (mocking GUI and camera)
@patch('pyautogui.click')  # Mock pyautogui.click to prevent actual clicks
@patch.dict('os.environ', {'DISPLAY': ':0'})  # Mock DISPLAY environment variable again
def test_process_frame(mock_click):
    # Mock camera and face_mesh
    cam = MagicMock()
    face_mesh = MagicMock()

    # Mock the frame return value from VideoCapture
    frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Fake a black image
    cam.read = MagicMock(return_value=(True, frame))

    # Mock face mesh output
    class MockFaceMeshOutput:
        def __init__(self):
            self.multi_face_landmarks = [MagicMock()]  # Mocked face landmarks
    face_mesh.process = MagicMock(return_value=MockFaceMeshOutput())

    # Run the frame processing
    result_frame, last_left_blink_time, last_right_blink_time = process_frame(cam, face_mesh, blink_threshold=3.5, blink_cooldown=1)

    # Check that the function completes successfully and that no errors are thrown
    assert result_frame is not None, "Processed frame should not be None"
    assert isinstance(result_frame, np.ndarray), "Processed frame should be a numpy array"
    assert last_left_blink_time == 0, "Left blink time should have been initialized"
    assert last_right_blink_time == 0, "Right blink time should have been initialized"
