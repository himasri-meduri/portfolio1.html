import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Frame and Canvas Dimensions
frame_width, frame_height = 1280, 720
button_width, button_height = 60, 40
button_spacing = 10

# Colors (BGR Format)
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (128, 0, 128), (0, 128, 255),
    (255, 255, 0), (255, 255, 255), (0, 0, 0), (255, 182, 193), (139, 69, 19), (238, 130, 238),
    (255, 0, 255), (128, 128, 0), (0, 255, 128), (128, 128, 128), (128, 0, 0), (85, 107, 47),
    (0, 0, 128), (255, 215, 0), (192, 192, 192), (245, 245, 220), (64, 224, 208), (230, 230, 250),
    (0, 191, 255), (240, 128, 128), (255, 69, 0), (34, 139, 34), (255, 140, 0), (123, 104, 238),
    (75, 0, 130), (250, 128, 114), (46, 139, 87), (220, 20, 60), (144, 238, 144), (255, 160, 122),
    (32, 178, 170), (72, 61, 139), (176, 224, 230), (70, 130, 180), (95, 158, 160), (153, 50, 204),
    (205, 92, 92), (244, 164, 96), (255, 228, 196), (47, 79, 79), (154, 205, 50), (100, 149, 237),
    (255, 250, 205), (0, 128, 128)
]

# Initial Index to the color selected
colorIndex = 0

# Drawing Points Storage
color_points = [[] for _ in range(len(colors))]

# Canvas Setup
paintWindow = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

# Mediapipe Initialization
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Webcam Initialization
cap = cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)

# Scrolling Variables
visible_buttons = 10  # Number of visible color buttons at a time
scroll_index = 0  # Scroll start index

# Button Dimensions
clear_button_width = 60
clear_button_height = 40
clear_button_x1 = frame_width - clear_button_width - 10
clear_button_y1 = 10
clear_button_x2 = clear_button_x1 + clear_button_width
clear_button_y2 = clear_button_y1 + clear_button_height

# Drawing Flag
drawing = False

# Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for mirror effect
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw Clear Button
    cv2.rectangle(frame, (clear_button_x1, clear_button_y1), (clear_button_x2, clear_button_y2), (0, 0, 0), 2)
    cv2.putText(frame, "Clear", (clear_button_x1 + 5, clear_button_y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Buttons for Colors with Scrolling (Left Side)
    for visible_idx in range(visible_buttons):
        actual_idx = scroll_index + visible_idx
        if actual_idx >= len(colors):
            break

        y1 = visible_idx * (button_height + button_spacing)
        y2 = y1 + button_height
        x1 = 10  # Align buttons on the left side of the frame
        x2 = x1 + button_width

        # Draw the color button
        cv2.rectangle(frame, (x1, y1), (x2, y2), colors[actual_idx], -1)
        cv2.putText(frame, str(actual_idx + 1), (x1 + 5, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0) if sum(colors[actual_idx]) > 400 else (255, 255, 255), 1)

        # Highlight the currently selected button
        if actual_idx == colorIndex:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)

    # Hand Detection
    result = hands.process(framergb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
            landmarks = [(int(lm.x * frame_width), int(lm.y * frame_height)) for lm in hand_landmarks.landmark]
            forefinger = landmarks[8]  # Index finger tip
            thumb = landmarks[4]  # Thumb tip

            # Detect if thumb and index finger are close
            if abs(forefinger[1] - thumb[1]) < 30 and abs(forefinger[0] - thumb[0]) < 30:
                drawing = False
            else:
                drawing = True

            # Detect Color Button Click
            if forefinger[0] < x2:  # Within color bar
                button_idx = forefinger[1] // (button_height + button_spacing)
                actual_idx = scroll_index + button_idx
                if 0 <= actual_idx < len(colors):
                    colorIndex = actual_idx

            # Detect Clear Button Click
            if clear_button_x1 < forefinger[0] < clear_button_x2 and clear_button_y1 < forefinger[1] < clear_button_y2:
                paintWindow = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
                color_points = [[] for _ in range(len(colors))]

            # Scroll Up or Down based on hand gestures
            if forefinger[1] < 50:  # Scroll up gesture
                scroll_index = max(0, scroll_index - 1)
            elif forefinger[1] > frame_height - 50:  # Scroll down gesture
                scroll_index = min(len(colors) - visible_buttons, scroll_index + 1)

            # Start Drawing
            if drawing and forefinger[0] > x2:
                if len(color_points[colorIndex]) == 0 or len(color_points[colorIndex][-1]) > 1023:
                    color_points[colorIndex].append(deque(maxlen=1024))
                color_points[colorIndex][-1].appendleft(forefinger)

    # Draw on Canvas
    for i, points in enumerate(color_points):
        for deque_points in points:
            for k in range(1, len(deque_points)):
                if deque_points[k - 1] is None or deque_points[k] is None:
                    continue
                cv2.line(frame, deque_points[k - 1], deque_points[k], colors[i], 2)
                cv2.line(paintWindow, deque_points[k - 1], deque_points[k], colors[i], 2)

    # Display the Output
    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break
    elif key == ord("w"):  # Scroll up using 'W' key
        scroll_index = max(0, scroll_index - 1)
    elif key == ord("s"):  # Scroll down using 'S' key
        scroll_index = min(len(colors) - visible_buttons, scroll_index + 1)

cap.release()
cv2.destroyAllWindows()
