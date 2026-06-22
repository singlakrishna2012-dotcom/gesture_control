import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).with_name(".matplotlib-cache")),
)

import cv2
import pyautogui
import numpy as np
import math
import time
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
)

MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

# ------------------------------------------------------------
# 0. Gesture Creator Table (printed on startup)
# ------------------------------------------------------------
GESTURE_TABLE = """
╔══════════════════════════════════════════════════════════════════╗
║                    GESTURE CONTROLLER TABLE                      ║
╠══════════════════════════════════════════════════════════════════╣
║  GESTURE                          │  MOUSE ACTION                ║
╠══════════════════════════════════════════════════════════════════╣
║  Index finger tip movement        │  Move cursor             q    ║
║  (inside touchpad box)            │                              ║
║ -----------------------------------------------------------------║
║  Short pinch: thumb + index       │  Left click                  ║
║  ( < 0.2 sec)                     │                              ║
║ -----------------------------------------------------------------║
║  Double short pinch               │  Double click                ║
║  thumb + index (within 0.3 sec)   │                              ║
║ -----------------------------------------------------------------║
║  Long pinch: thumb + index        │  Drag & drop (hold & move)   ║
║  ( > 0.2 sec)                     │                              ║
║ -----------------------------------------------------------------║
║  Light touch: thumb + index       │  Left click (tap)            ║
║  (quick tap, < 0.15 sec)          │                              ║
║ -----------------------------------------------------------------║
║  Thumb + middle finger pinch      │  Right click                 ║
║ -----------------------------------------------------------------║
║  Thumb + ring finger pinch        │  Middle click                ║
║ -----------------------------------------------------------------║
║  Index + middle + ring extended   │  Press Enter key             ║
║  (fingers spread, others folded)   │                              ║
║ -----------------------------------------------------------------║
║  Index + middle fingers extended  │  Vertical scroll             ║
║  (ring & pinky folded)            │  (move hand up/down)         ║
║ -----------------------------------------------------------------║
║  Left hand open palm              │  Scroll page vertically      ║
║  Left hand thumb up/down          │  Volume up / down            ║
║  Left hand fist                   │  Play/Pause media            ║
╚══════════════════════════════════════════════════════════════════╝
"""

def get_distance(p1, p2):
    """Euclidean distance between two MediaPipe landmarks."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def is_inside_box(x, y, x_min, x_max, y_min, y_max):
    return x_min <= x <= x_max and y_min <= y <= y_max

def select_active_hand(hand_entries, x_min, x_max, y_min, y_max):
    if not hand_entries:
        return None

    active_candidates = [hand for hand in hand_entries
                         if is_inside_box(hand["index_tip"].x, hand["index_tip"].y,
                                          x_min, x_max, y_min, y_max)]
    if active_candidates:
        right = next((hand for hand in active_candidates if hand["label"] == "Right"), None)
        return right or active_candidates[0]

    return next((hand for hand in hand_entries if hand["label"] == "Right"), hand_entries[0])

def correct_handedness_for_mirror(label):
    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label

def draw_hand_landmarks(image, landmarks, connections):
    h, w, _ = image.shape
    points = []
    for landmark in landmarks:
        x = min(max(int(landmark.x * w), 0), w - 1)
        y = min(max(int(landmark.y * h), 0), h - 1)
        points.append((x, y))

    for connection in connections:
        cv2.line(
            image,
            points[connection.start],
            points[connection.end],
            (0, 255, 120),
            2,
        )

    for point in points:
        cv2.circle(image, point, 3, (255, 255, 255), -1)
        cv2.circle(image, point, 5, (0, 180, 255), 1)

def create_hand_landmarker():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH.name}. Download it into this folder before running gestures.py."
        )

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    return HandLandmarker.create_from_options(options)

def main():
    # ----------------------------------------------------
    # 1. Configuration & Settings
    # ----------------------------------------------------
    print(GESTURE_TABLE)
    pyautogui.PAUSE = 0.001
    pyautogui.FAILSAFE = True

    screen_w, screen_h = pyautogui.size()
    print(f"Screen resolution: {screen_w}x{screen_h}")
    print("Fail-safe active: move mouse to top-left corner to stop.\n")

    # Try opening several camera indices in case 0 isn't correct
    cap = None
    for idx in range(0, 4):
        print(f"Trying camera index {idx}...")
        temp = cv2.VideoCapture(idx)
        if temp is not None and temp.isOpened():
            cap = temp
            print(f"Opened camera index {idx}")
            break
        else:
            if temp is not None:
                temp.release()

    if cap is None or not cap.isOpened():
        print("Error: Could not open any webcam. Check camera connection or index.")
        return

    frame_w, frame_h = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

    # Create named window to ensure it appears on some platforms
    cv2.namedWindow("Gesture Control Virtual Mouse", cv2.WINDOW_NORMAL)

    try:
        hands = create_hand_landmarker()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        cap.release()
        cv2.destroyAllWindows()
        return

    # Active touchpad region (normalized)
    active_x_min, active_x_max = 0.25, 0.75
    active_y_min, active_y_max = 0.25, 0.75

    # Smoothing
    prev_x, prev_y = 0, 0
    smooth_factor_move = 6.0
    smooth_factor_click = 9.0

    # Gesture states
    left_clicked = False        # for drag state
    right_clicked = False



    prev_scroll_y = None
    is_first_frame = True
    last_active_hand_label = None

    # Double‑click / drag detection
    pinch_active = False
    pinch_start_time = 0
    last_release_time = 0
    drag_active = False

    # Enter key gesture state
    enter_pressed = False
    
    # Tap-to-click state (index + thumb light touch)
    tap_active = False
    tap_start_time = 0

    # FPS
    prev_time = 0

    print("Virtual Mouse running. Press 'q' in camera window to exit.\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Mirror and convert
        image = cv2.flip(frame, 1)
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_image)
        timestamp_ms = int(time.time() * 1000)
        results = hands.detect_for_video(mp_image, timestamp_ms)

        # Touchpad box coordinates (pixels)
        bx1, bx2 = int(active_x_min * w), int(active_x_max * w)
        by1, by2 = int(active_y_min * h), int(active_y_max * h)
        color_active = (80, 220, 100)
        cv2.rectangle(image, (bx1, by1), (bx2, by2), color_active, 2)
        cv2.putText(image, "Touchpad Area", (bx1 + 5, by1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_active, 1)

        if results.hand_landmarks:
            hand_entries = []
            for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                raw_label = handedness[0].category_name if handedness else "Unknown"
                label = correct_handedness_for_mirror(raw_label)
                draw_hand_landmarks(
                    image,
                    hand_landmarks,
                    HandLandmarksConnections.HAND_CONNECTIONS,
                )

                lm = hand_landmarks
                wrist = lm[0]
                mcp_middle = lm[9]
                thumb_tip = lm[4]
                index_tip = lm[8]
                index_pip = lm[6]
                middle_tip = lm[12]
                middle_pip = lm[10]
                ring_tip = lm[16]
                ring_pip = lm[14]
                pinky_tip = lm[20]
                pinky_pip = lm[18]

                hand_entries.append({
                    "label": label,
                    "landmarks": lm,
                    "wrist": wrist,
                    "mcp_middle": mcp_middle,
                    "thumb_tip": thumb_tip,
                    "index_tip": index_tip,
                    "index_pip": index_pip,
                    "middle_tip": middle_tip,
                    "middle_pip": middle_pip,
                    "ring_tip": ring_tip,
                    "ring_pip": ring_pip,
                    "pinky_tip": pinky_tip,
                    "pinky_pip": pinky_pip,
                })

                cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                cv2.putText(image, label, (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)

            active_hand = select_active_hand(hand_entries,
                                            active_x_min, active_x_max,
                                            active_y_min, active_y_max)
            active_hand_label = active_hand["label"] if active_hand else None
            secondary_hand = next((hand for hand in hand_entries
                                   if hand["label"] != active_hand_label), None)

            if active_hand_label != last_active_hand_label:
                if left_clicked:
                    pyautogui.mouseUp(button='left')
                    left_clicked = False
                right_clicked = False
                pinch_active = False
                drag_active = False
                prev_scroll_y = None
                is_first_frame = True
                last_active_hand_label = active_hand_label


            if active_hand:
                thumb_tip = active_hand["thumb_tip"]
                index_tip = active_hand["index_tip"]
                index_pip = active_hand["index_pip"]
                middle_tip = active_hand["middle_tip"]
                middle_pip = active_hand["middle_pip"]
                ring_tip = active_hand["ring_tip"]
                ring_pip = active_hand["ring_pip"]
                pinky_tip = active_hand["pinky_tip"]
                pinky_pip = active_hand["pinky_pip"]

                palm_dist = max(get_distance(active_hand["wrist"], active_hand["mcp_middle"]), 0.01)
                left_dist = get_distance(index_tip, thumb_tip) / palm_dist
                right_dist = get_distance(middle_tip, thumb_tip) / palm_dist
                middle_dist = get_distance(ring_tip, thumb_tip) / palm_dist
                scroll_dist = get_distance(index_tip, middle_tip) / palm_dist

                index_ext = index_tip.y < index_pip.y
                middle_ext = middle_tip.y < middle_pip.y
                ring_ext = ring_tip.y < ring_pip.y
                ring_folded = ring_tip.y > ring_pip.y
                pinky_folded = pinky_tip.y > pinky_pip.y

                is_scroll_mode = (index_ext and middle_ext and ring_folded and pinky_folded and scroll_dist < 0.4)

                if is_scroll_mode:
                    if left_clicked:
                        pyautogui.mouseUp(button='left')
                        left_clicked = False
                    drag_active = False
                    pinch_active = False

                    curr_y = index_tip.y
                    if prev_scroll_y is not None:
                        y_diff = prev_scroll_y - curr_y
                        if abs(y_diff) > 0.005:
                            scroll_amt = int(y_diff * 1800)
                            pyautogui.scroll(scroll_amt)
                    prev_scroll_y = curr_y

                    cv2.putText(image, "SCROLL MODE", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 180, 52), 2)
                    cv2.circle(image, (int(index_tip.x * w), int(index_tip.y * h)), 10, (235, 180, 52), -1)
                    cv2.circle(image, (int(middle_tip.x * w), int(middle_tip.y * h)), 10, (235, 180, 52), -1)

                else:
                    prev_scroll_y = None

                    is_enter_gesture = (index_ext and middle_ext and ring_ext and pinky_folded and 
                                       left_dist > 0.35 and right_dist > 0.35)
                    
                    if is_enter_gesture and not enter_pressed:
                        pyautogui.press('enter')
                        enter_pressed = True
                        print("Enter key pressed")
                    elif not is_enter_gesture and enter_pressed:
                        enter_pressed = False

                    if is_enter_gesture:
                        cv2.putText(image, "ENTER GESTURE", (20, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 150, 255), 2)

                    norm_x = (index_tip.x - active_x_min) / (active_x_max - active_x_min)
                    norm_y = (index_tip.y - active_y_min) / (active_y_max - active_y_min)
                    norm_x = max(0.0, min(1.0, norm_x))
                    norm_y = max(0.0, min(1.0, norm_y))
                    target_x = norm_x * screen_w
                    target_y = norm_y * screen_h

                    smooth = smooth_factor_click if pinch_active else smooth_factor_move
                    if is_first_frame:
                        curr_x, curr_y = target_x, target_y
                        is_first_frame = False
                    else:
                        curr_x = prev_x + (target_x - prev_x) / smooth
                        curr_y = prev_y + (target_y - prev_y) / smooth

                    try:
                        pyautogui.moveTo(int(curr_x), int(curr_y))
                    except pyautogui.FailSafeException:
                        print("Fail-safe triggered. Exiting.")
                        break
                    prev_x, prev_y = curr_x, curr_y

                    is_pinching = left_dist < 0.33

                    is_tapping = (left_dist < 0.25) and not is_pinching
                    
                    if is_tapping and not tap_active:
                        tap_active = True
                        tap_start_time = time.time()
                    
                    elif not is_tapping and tap_active:
                        tap_duration = time.time() - tap_start_time
                        if tap_duration < 0.15:
                            now = time.time()
                            if now - last_release_time < 0.3:
                                pyautogui.doubleClick()
                                print("Double click (tap)")
                            else:
                                pyautogui.click()
                                print("Left click (tap)")
                            last_release_time = now
                        tap_active = False

                    if is_tapping:
                        tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                        ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                        cv2.line(image, (tx, ty), (ix, iy), (255, 200, 0), 2)
                        cv2.putText(image, "TAP CLICK", (20, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)

                    if is_pinching and not pinch_active:
                        pinch_active = True
                        pinch_start_time = time.time()
                        drag_active = False

                    elif not is_pinching and pinch_active:
                        pinch_duration = time.time() - pinch_start_time
                        if pinch_duration < 0.2:
                            now = time.time()
                            if now - last_release_time < 0.3:
                                pyautogui.doubleClick()
                                print("Double click")
                            else:
                                pyautogui.click()
                                print("Left click")
                            last_release_time = now
                        else:
                            if left_clicked:
                                pyautogui.mouseUp(button='left')
                                left_clicked = False
                                print("Drag release")
                        pinch_active = False
                        drag_active = False

                    if pinch_active and not drag_active:
                        if time.time() - pinch_start_time > 0.2:
                            drag_active = True
                            if not left_clicked:
                                pyautogui.mouseDown(button='left')
                                left_clicked = True
                                print("Drag start")

                    if is_pinching:
                        tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                        ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                        cv2.line(image, (tx, ty), (ix, iy), (0, 0, 255), 3)
                        if drag_active:
                            cv2.putText(image, "DRAG ACTIVE", (20, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            cv2.putText(image, "LEFT PINCH", (20, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    is_right = (right_dist < 0.33) and not pinch_active
                    if is_right and not right_clicked:
                        pyautogui.rightClick()
                        right_clicked = True
                        print("Right click")
                    elif not is_right:
                        right_clicked = False

                    if is_right:
                        tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                        mx, my = int(middle_tip.x * w), int(middle_tip.y * h)
                        cv2.line(image, (tx, ty), (mx, my), (0, 0, 255), 3)
                        cv2.putText(image, "RIGHT CLICK", (20, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    is_middle_click = (middle_dist < 0.33) and not pinch_active and not is_right
                    if is_middle_click:
                        pyautogui.click(button='middle')
                        print("Middle click")
                        tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                        rx, ry = int(ring_tip.x * w), int(ring_tip.y * h)
                        cv2.line(image, (tx, ty), (rx, ry), (255, 100, 100), 3)
                        cv2.putText(image, "MIDDLE CLICK", (20, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)

                    cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                    cv2.circle(image, (cx, cy), 12, (255, 0, 0), 2)

            if active_hand_label:
                cv2.putText(image, f"ACTIVE HAND: {active_hand_label}", (20, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 255, 180), 2)

        else:
            if left_clicked:
                pyautogui.mouseUp(button='left')
                left_clicked = False
            right_clicked = False
            pinch_active = False
            drag_active = False
            prev_scroll_y = None
            is_first_frame = True
            last_active_hand_label = None
            cv2.putText(image, "NO HAND DETECTED", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- FPS and instructions ---
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(image, f"FPS: {fps}", (w - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(image, "Press 'Q' to Exit", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Gesture Control Virtual Mouse", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    if left_clicked:
        pyautogui.mouseUp(button='left')
    print("\nApplication closed.")

if __name__ == "__main__":
    main()
