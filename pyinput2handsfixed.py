# opencv + mediapipe + pydirectinput dual-hand hold integration
import cv2 as cv
import mediapipe as mp
import pydirectinput
import time

pydirectinput.PAUSE = 0  # No delay between inputs

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def getHandMove(hand_landmarks):
    threshold = 0.1
    landmarks = hand_landmarks.landmark
    if all([(landmarks[i+3].x - landmarks[i].x) > threshold for i in range(5,20,4)]): return "Backspin"
    if all([(landmarks[i+3].x - landmarks[i].x) < -threshold for i in range(5,20,4)]): return "Frontspin"
    if all([landmarks[i].y < landmarks[i+3].y for i in range(5, 20, 4)]): return "STOP"
    if all([landmarks[i].y > landmarks[i+3].y for i in range(5, 20, 4)]): return "GO"
    else: return "Pending Hand Movement"

def getHandMove2(hand_landmarks):
    threshold = 0.1
    landmarks = hand_landmarks.landmark
    if all([(landmarks[i+3].x - landmarks[i].x) > threshold for i in range(5,20,4)]): return "Backspin2"
    if all([(landmarks[i+3].x - landmarks[i].x) < -threshold for i in range(5,20,4)]): return "Frontspin2"
    if all([landmarks[i].y < landmarks[i+3].y for i in range(5, 20, 4)]): return "STOP2"
    if all([landmarks[i].y > landmarks[i+3].y for i in range(5, 20, 4)]): return "GO2"
    else: return "Pending Hand Movement"

key_mapping = {
    "Left": {
        "GO": "w",
        "STOP": "s",
        "Frontspin": "e",
        "Backspin": "q",
        "Pending Hand Movement": None
    },
    "Right" : {
        "GO2": "i",
        "STOP2": "k",
        "Frontspin2": "l",
        "Backspin2": "j",
        "Pending Hand Movement": None
    }   
}

# Current key held for each hand
current_keys = {"Left": None, "Right": None}
last_gestures = {"Left": None, "Right": None}

# Open camera
vid = cv.VideoCapture(0)

with mp_hands.Hands(model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    max_num_hands=2) as hands:
    
    while True:
        ret, frame = vid.read()
        if not ret or frame is None:
            break

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
        frame_bgr = cv.flip(frame_bgr, 1)

        gameText = ""

        hls = results.multi_hand_landmarks

        if hls and len(hls) == 2:
            # Assume: left hand is hls[0], right hand is hls[1]
            # (You could later improve this by classifying handedness)
            p1_gesture = getHandMove(hls[0])
            p2_gesture = getHandMove2(hls[1])

            key1 = key_mapping["Left"].get(p1_gesture)
            key2 = key_mapping["Right"].get(p2_gesture)

            # Handle Left Hand (Player 1)
            if p1_gesture != last_gestures["Left"]:
                # Release old key
                if current_keys["Left"]:
                    pydirectinput.keyUp(current_keys["Left"])
                # Press new key
                if key1:
                    pydirectinput.keyDown(key1)
                current_keys["Left"] = key1
                last_gestures["Left"] = p1_gesture

            # Handle Right Hand (Player 2)
            if p2_gesture != last_gestures["Right"]:
                if current_keys["Right"]:
                    pydirectinput.keyUp(current_keys["Right"])
                if key2:
                    pydirectinput.keyDown(key2)
                current_keys["Right"] = key2
                last_gestures["Right"] = p2_gesture

            gameText = f'P1: {p1_gesture}, P2: {p2_gesture}'

            # Draw landmarks
            for hand_landmarks in hls:
                mp_drawing.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        else:
            gameText = "Put 2 hands in the frame!"
            # Release keys if hands lost
            for side in ["Left", "Right"]:
                if current_keys[side]:
                    pydirectinput.keyUp(current_keys[side])
                    current_keys[side] = None
                    last_gestures[side] = None

        # Show gesture text
        cv.putText(frame_bgr, gameText, (25, 40), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)

        # Show frame
        cv.imshow('frame', frame_bgr)

        # Quit key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
for side in ["Left", "Right"]:
    if current_keys[side]:
        pydirectinput.keyUp(current_keys[side])

vid.release()
cv.destroyAllWindows()
wiejijwij