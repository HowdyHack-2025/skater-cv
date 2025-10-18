# opencv + mediapipe + pydirectinput hold/release integration
import time
import cv2 as cv
import mediapipe as mp

try:
    import pydirectinput
except Exception as e:
    raise SystemExit("pydirectinput not available. Install with: pip install pydirectinput") from e

# MediaPipe drawing/helpers
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Gesture detection (unchanged)
def getHandMove(hand_landmarks):
    threshold = 0.1
    landmarks = hand_landmarks.landmark
    if all([(landmarks[i+3].x - landmarks[i].x) > threshold for i in range(5,20,4)]): return "Backspin"
    if all([(landmarks[i+3].x - landmarks[i].x) < -threshold for i in range(5,20,4)]): return "Frontspin"
    if all([landmarks[i].y < landmarks[i+3].y for i in range(5, 20, 4)]): return "STOP"
    if all([landmarks[i].y > landmarks[i+3].y for i in range(5, 20, 4)]): return "GO"
    else: return "Pending Hand Movement"

# Map gestures -> pydirectinput key names
key_mapping = {
    "GO": "w",
    "STOP": "s",
    "Frontspin": "e",
    "Backspin": "q",
    "Pending Hand Movement": None
}

# State for hold logic
current_key_held = None        # key name currently held (e.g., 'w'), or None
last_gesture = None            # last recognized gesture string
gesture_start_time = 0.0       # when current gesture started
DEBOUNCE_SECONDS = 0.08        # ignore very short-lived changes

# Video capture
vid = cv.VideoCapture(0)
if not vid.isOpened():
    raise SystemExit("Could not open camera")

with mp_hands.Hands(model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:

    while True:
        ret, frame = vid.read()
        if not ret or frame is None:
            break

        # process frame
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)

        # draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # mirror for user
        frame_bgr = cv.flip(frame_bgr, 1)

        # Determine gesture and target key
        gameText = ""
        target_key = None
        hls = results.multi_hand_landmarks

        if hls and len(hls) == 1:
            gesture = getHandMove(hls[0])
            gameText = gesture
            target_key = key_mapping.get(gesture)
        else:
            gesture = None
            gameText = "Put 1 hand in the frame at a time!"
            target_key = None

        now = time.time()

        # Gesture start/time logic for debounce
        if gesture != last_gesture:
            # gesture changed; start new timer (debounce)
            last_gesture = gesture
            gesture_start_time = now

        # Only act if gesture has been stable longer than debounce
        if (now - gesture_start_time) >= DEBOUNCE_SECONDS:
            # If new target_key is different from currently held key, update holds
            if target_key != current_key_held:
                # release previously held key (if any)
                if current_key_held is not None:
                    try:
                        pydirectinput.keyUp(current_key_held)
                    except Exception as e:
                        print(f"Warning: keyUp failed for {current_key_held}: {e}")
                    current_key_held = None

                # press new key down if target_key is not None
                if target_key is not None:
                    try:
                        pydirectinput.keyDown(target_key)
                        current_key_held = target_key
                    except Exception as e:
                        print(f"Warning: keyDown failed for {target_key}: {e}")
                        current_key_held = None
        else:
            # still in debounce window -> do nothing (keep previous hold)
            pass

        # show status on frame
        cv.putText(frame_bgr, gameText, (25, 40), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2, cv.LINE_AA)
        cv.imshow('frame', frame_bgr)

        # quit on 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# clean up: release any held key
if current_key_held is not None:
    try:
        pydirectinput.keyUp(current_key_held)
    except Exception:
        pass

vid.release()
cv.destroyAllWindows()