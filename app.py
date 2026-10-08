"""
Hand Gesture Controlled Computer Interface
------------------------------------------
Control the mouse and common system functions using hand gestures
captured from a webcam. 15 gestures are supported.

Libraries : OpenCV, MediaPipe Hands, PyAutoGUI, NumPy
Run       : python app.py
Keys      : q = quit, p = pause / resume control

Keep the hand upright with the palm facing the camera.

 #  Gesture                               Action
 1  Index finger up                       Move cursor
 2  Thumb + index pinch                   Left click
 3  Thumb + middle pinch                  Right click
 4  Thumb + ring pinch                    Double click
 5  Index + middle + ring up              Drag and drop (hold and move)
 6  Index + middle up (peace sign)        Scroll (move hand up / down)
 7  Thumb + index + middle up             Zoom in / out (move hand up / down)
 8  Index + pinky up (rock sign)          Volume up / down (move hand up / down)
 9  Thumbs up (hold)                      Mute / unmute
10  Open palm, 5 fingers (hold)           Play / pause media
11  Pinky only (hold)                     Show desktop
12  Thumb + pinky (call-me sign, hold)    Take screenshot
13  Four fingers + swipe left             Browser back (Alt + Left)
14  Four fingers + swipe right            Browser forward (Alt + Right)
15  Closed fist                           Pause (no action)

Safety: move the real mouse to the TOP-LEFT corner of the screen to
trigger the PyAutoGUI fail-safe and stop the program immediately.
"""

import math
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# ----------------------------------------------------------------------
# Configuration (tune these values for your camera and comfort)
# ----------------------------------------------------------------------
CAM_WIDTH, CAM_HEIGHT = 640, 480
PANEL_WIDTH = 340            # width of the gesture guide panel (pixels)
FRAME_MARGIN = 100           # active area margin inside the camera frame
SMOOTHING = 5                # higher = smoother but slower cursor
PINCH_THRESHOLD = 0.30       # pinch distance relative to hand size
THUMB_FACTOR = 1.15          # thumb is "extended" if tip is this much farther
STABLE_FRAMES = 3            # frames a gesture must last before it is accepted
HOLD_TIME = 0.8              # seconds to hold "hold" gestures
CLICK_COOLDOWN = 0.5         # minimum seconds between two clicks
SCROLL_SENSITIVITY = 1500    # higher = faster scrolling
ZOOM_SENSITIVITY = 1000      # higher = faster zooming
SCROLL_DEADZONE = 0.004      # ignore tiny hand movements
VOLUME_STEP = 0.04           # hand movement needed for one volume step
SWIPE_DISTANCE = 0.25        # horizontal hand travel (fraction of frame width)
SWIPE_WINDOW = 0.5           # seconds in which the swipe must happen
SWIPE_COOLDOWN = 1.0         # seconds before another swipe is accepted

pyautogui.FAILSAFE = True    # keep the safety stop enabled
pyautogui.PAUSE = 0          # no delay between PyAutoGUI calls

SCREEN_W, SCREEN_H = pyautogui.size()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

FINGER_TIPS = [8, 12, 16, 20]    # index, middle, ring, pinky
FINGER_PIPS = [6, 10, 14, 18]

# Actions that must be held for HOLD_TIME seconds before they run
HOLD_ACTIONS = {
    "MUTE": (lambda: pyautogui.press("volumemute"), "Mute / Unmute"),
    "PLAY/PAUSE": (lambda: pyautogui.press("playpause"), "Play / Pause"),
    "SHOW DESKTOP": (lambda: pyautogui.hotkey("win", "d"), "Show Desktop"),
    "SCREENSHOT": (lambda: pyautogui.hotkey("win", "printscreen"),
                   "Screenshot saved"),
}

# Gesture guide shown on the side panel: (gesture name, description)
GUIDE = [
    ("MOVE", "Index up: Move cursor"),
    ("LEFT CLICK", "Thumb+Index pinch: Left click"),
    ("RIGHT CLICK", "Thumb+Middle pinch: Right click"),
    ("DOUBLE CLICK", "Thumb+Ring pinch: Double click"),
    ("DRAG", "3 fingers up: Drag and drop"),
    ("SCROLL", "Peace sign: Scroll"),
    ("ZOOM", "Thumb+Index+Middle: Zoom"),
    ("VOLUME", "Index+Pinky: Volume"),
    ("MUTE", "Thumbs up (hold): Mute"),
    ("PLAY/PAUSE", "Open palm (hold): Play/Pause"),
    ("SHOW DESKTOP", "Pinky only (hold): Show desktop"),
    ("SCREENSHOT", "Thumb+Pinky (hold): Screenshot"),
    ("FOUR FINGERS", "4 fingers swipe L/R: Back/Forward"),
    ("PAUSE", "Fist: No action"),
]


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def distance(p1, p2):
    """Euclidean distance between two (x, y) points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def get_points(hand_landmarks):
    """Convert MediaPipe landmarks to a list of normalized (x, y) points."""
    return [(lm.x, lm.y) for lm in hand_landmarks.landmark]


def fingers_up(points):
    """Return [index, middle, ring, pinky] as booleans (True = finger up)."""
    return [points[tip][1] < points[pip][1]
            for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)]


def thumb_extended(points):
    """True if the thumb is stretched out (works for left and right hand)."""
    return (distance(points[4], points[17])
            > distance(points[3], points[17]) * THUMB_FACTOR)


def pinch_ratio(points, a, b):
    """Distance between landmarks a and b divided by the hand size."""
    hand_size = distance(points[0], points[9])  # wrist to middle-finger base
    return distance(points[a], points[b]) / max(hand_size, 1e-6)


def classify_gesture(thumb, fingers, thumb_index, thumb_middle, thumb_ring):
    """Map finger states and pinch distances to a gesture name."""
    index, middle, ring, pinky = fingers

    # Pinch gestures first (the thumb touches another fingertip)
    if middle and ring and pinky and thumb_index < PINCH_THRESHOLD:
        return "LEFT CLICK"
    if index and middle and pinky and thumb_ring < PINCH_THRESHOLD:
        return "DOUBLE CLICK"
    if index and ring and pinky and thumb_middle < PINCH_THRESHOLD:
        return "RIGHT CLICK"

    # Pose gestures
    if not (thumb or index or middle or ring or pinky):
        return "PAUSE"
    if thumb and index and middle and ring and pinky:
        return "PLAY/PAUSE"
    if index and middle and ring and pinky:
        return "FOUR FINGERS"
    if index and middle and ring:
        return "DRAG"
    if index and not middle and not ring and not pinky:
        return "MOVE"
    if index and middle and not ring and not pinky:
        return "ZOOM" if thumb else "SCROLL"
    if index and pinky and not middle and not ring:
        return "VOLUME"
    if pinky and not (index or middle or ring):
        return "SCREENSHOT" if thumb else "SHOW DESKTOP"
    if thumb and not (index or middle or ring or pinky):
        return "MUTE"
    return "NONE"


def put_text(img, text, pos, scale=0.7, color=(255, 255, 255), thickness=2):
    """Draw text with a dark outline so it is readable on any background."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


# ----------------------------------------------------------------------
# Gesture controller: keeps state and performs the computer actions
# ----------------------------------------------------------------------
class GestureController:
    def __init__(self):
        self.cur_x, self.cur_y = SCREEN_W / 2, SCREEN_H / 2
        self.candidate = "NO HAND"
        self.candidate_frames = 0
        self.active = "NO HAND"
        self.active_since = time.time()
        self.fired = False
        self.prev_y = None
        self.anchor_y = None
        self.dragging = False
        self.last_click_time = 0.0
        self.swipe_history = deque()
        self.swipe_block_until = 0.0
        self.message = ""
        self.message_until = 0.0

    def notify(self, text):
        """Show a short message on the screen."""
        self.message = text
        self.message_until = time.time() + 1.5

    def release_all(self):
        """Release a held mouse button (used when leaving drag mode)."""
        if self.dragging:
            pyautogui.mouseUp()
            self.dragging = False

    def hold_progress(self, now):
        """Progress (0 to 1) of a hold gesture, used for the progress bar."""
        if self.active in HOLD_ACTIONS and not self.fired:
            return min(1.0, (now - self.active_since) / HOLD_TIME)
        return 0.0

    def stabilize(self, raw, now):
        """Accept a gesture only after it is seen for a few frames."""
        if raw == self.candidate:
            self.candidate_frames += 1
        else:
            self.candidate = raw
            self.candidate_frames = 1

        if self.candidate_frames >= STABLE_FRAMES and self.candidate != self.active:
            self.release_all()
            self.active = self.candidate
            self.active_since = now
            self.fired = False
            self.prev_y = None
            self.anchor_y = None
            self.swipe_history.clear()
        return self.active

    def move_cursor(self, points, w, h):
        """Move the mouse using the index fingertip. Returns its pixel position."""
        ix, iy = points[8][0] * w, points[8][1] * h
        target_x = np.interp(ix, (FRAME_MARGIN, w - FRAME_MARGIN), (0, SCREEN_W))
        target_y = np.interp(iy, (FRAME_MARGIN, h - FRAME_MARGIN), (0, SCREEN_H))
        self.cur_x += (target_x - self.cur_x) / SMOOTHING
        self.cur_y += (target_y - self.cur_y) / SMOOTHING
        pyautogui.moveTo(int(np.clip(self.cur_x, 5, SCREEN_W - 5)),
                         int(np.clip(self.cur_y, 5, SCREEN_H - 5)))
        return int(ix), int(iy)

    def perform(self, points, w, h, now, frame):
        """Run the action that belongs to the active gesture."""
        g = self.active

        if g == "MOVE":
            pos = self.move_cursor(points, w, h)
            cv2.circle(frame, pos, 12, (0, 255, 0), cv2.FILLED)

        elif g in ("LEFT CLICK", "RIGHT CLICK", "DOUBLE CLICK"):
            if not self.fired and now - self.last_click_time > CLICK_COOLDOWN:
                if g == "LEFT CLICK":
                    pyautogui.click()
                elif g == "RIGHT CLICK":
                    pyautogui.click(button="right")
                else:
                    pyautogui.doubleClick()
                self.last_click_time = now
                self.fired = True
                self.notify(g.title())

        elif g == "DRAG":
            if not self.fired:
                pyautogui.mouseDown()
                self.dragging = True
                self.fired = True
            pos = self.move_cursor(points, w, h)
            cv2.circle(frame, pos, 12, (0, 165, 255), cv2.FILLED)

        elif g in ("SCROLL", "ZOOM"):
            y = points[9][1]
            if self.prev_y is not None:
                dy = self.prev_y - y            # hand up = positive
                if abs(dy) > SCROLL_DEADZONE:
                    if g == "SCROLL":
                        pyautogui.scroll(int(dy * SCROLL_SENSITIVITY))
                    else:
                        pyautogui.keyDown("ctrl")
                        pyautogui.scroll(int(dy * ZOOM_SENSITIVITY))
                        pyautogui.keyUp("ctrl")
            self.prev_y = y

        elif g == "VOLUME":
            y = points[9][1]
            if self.anchor_y is None:
                self.anchor_y = y
            elif self.anchor_y - y > VOLUME_STEP:
                pyautogui.press("volumeup")
                self.anchor_y = y
            elif y - self.anchor_y > VOLUME_STEP:
                pyautogui.press("volumedown")
                self.anchor_y = y

        elif g in HOLD_ACTIONS:
            if not self.fired and now - self.active_since >= HOLD_TIME:
                action, message = HOLD_ACTIONS[g]
                action()
                self.fired = True
                self.notify(message)

        elif g == "FOUR FINGERS":
            x = points[9][0]
            self.swipe_history.append((now, x))
            while self.swipe_history and now - self.swipe_history[0][0] > SWIPE_WINDOW:
                self.swipe_history.popleft()
            if now > self.swipe_block_until and len(self.swipe_history) > 1:
                dx = x - self.swipe_history[0][1]
                if dx < -SWIPE_DISTANCE:
                    pyautogui.hotkey("alt", "left")
                    self.notify("Back")
                elif dx > SWIPE_DISTANCE:
                    pyautogui.hotkey("alt", "right")
                    self.notify("Forward")
                else:
                    return
                self.swipe_block_until = now + SWIPE_COOLDOWN
                self.swipe_history.clear()


# ----------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------
def build_guide_panel(height, active):
    """Create the side panel that lists all gestures."""
    panel = np.full((height, PANEL_WIDTH, 3), 30, dtype=np.uint8)
    put_text(panel, "GESTURE GUIDE", (12, 28), 0.65, (0, 200, 255))
    row_h = 29
    for i, (name, text) in enumerate(GUIDE):
        y = 62 + i * row_h
        is_active = (name == active)
        if is_active:
            cv2.rectangle(panel, (4, y - 18), (PANEL_WIDTH - 4, y + 6),
                          (0, 120, 0), cv2.FILLED)
        color = (255, 255, 255) if is_active else (200, 200, 200)
        cv2.putText(panel, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    color, 1, cv2.LINE_AA)
    return panel


# ----------------------------------------------------------------------
# Main application
# ----------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        print("Error: cannot open the camera.")
        return

    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,              # lighter model, faster on CPU
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    controller = GestureController()
    control_enabled = True
    prev_time = time.time()
    fps_smooth = 0.0

    print("Hand Gesture Controlled Computer Interface started.")
    print("Press 'q' to quit, 'p' to pause or resume control.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)               # mirror view
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            now = time.time()

            # Active area: the cursor covers the whole screen inside this box
            cv2.rectangle(frame, (FRAME_MARGIN, FRAME_MARGIN),
                          (w - FRAME_MARGIN, h - FRAME_MARGIN), (255, 0, 255), 2)

            raw_gesture = "NO HAND"
            points = None
            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks,
                                       mp_hands.HAND_CONNECTIONS)
                points = get_points(hand_landmarks)
                raw_gesture = classify_gesture(
                    thumb_extended(points),
                    fingers_up(points),
                    pinch_ratio(points, 4, 8),
                    pinch_ratio(points, 4, 12),
                    pinch_ratio(points, 4, 16),
                )

            gesture = controller.stabilize(raw_gesture, now)

            if control_enabled and points is not None:
                controller.perform(points, w, h, now, frame)
            elif not control_enabled:
                controller.release_all()

            # ---- FPS ----
            fps = 1 / max(time.time() - prev_time, 1e-6)
            prev_time = time.time()
            fps_smooth = fps if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * fps

            # ---- On-screen information ----
            status = "ON" if control_enabled else "PAUSED"
            status_color = (0, 255, 0) if control_enabled else (0, 0, 255)
            put_text(frame, f"FPS: {fps_smooth:.1f}", (10, 30), 0.8, (0, 0, 255))
            put_text(frame, f"Gesture: {gesture}", (10, 65), 0.8, (255, 200, 0))
            put_text(frame, f"Control: {status}", (10, 100), 0.8, status_color)

            progress = controller.hold_progress(now)
            if progress > 0:
                cv2.rectangle(frame, (10, 115), (210, 135), (255, 255, 255), 2)
                cv2.rectangle(frame, (12, 117),
                              (12 + int(196 * progress), 133),
                              (0, 255, 255), cv2.FILLED)

            if now < controller.message_until:
                put_text(frame, controller.message, (10, h - 45), 0.9,
                         (0, 255, 0))
            put_text(frame, "q: quit   p: pause/resume", (10, h - 15), 0.55)

            panel = build_guide_panel(h, gesture)
            cv2.imshow("Hand Gesture Controlled Computer Interface",
                       np.hstack((frame, panel)))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                control_enabled = not control_enabled
    finally:
        controller.release_all()
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")


if __name__ == "__main__":
    main()