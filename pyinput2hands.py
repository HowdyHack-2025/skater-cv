#opencv and mediapipe
import cv2 as cv
import mediapipe as mp
import pydirectinput
pydirectinput.PAUSE = 0

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

vid = cv.VideoCapture(0)

p1_move = None
p2_move = None
gameText = ""

current_keys_down = {"Left": None, "Right": None}

with mp_hands.Hands(model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = vid.read()
        if not ret or frame is None: break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        results = hands.process(frame)
        
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        if results.multi_hand_landmarks: 
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, 
                                          hand_landmarks,
                                          mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())
                
        frame = cv.flip(frame, 1)
            
        
       
        hls = results.multi_hand_landmarks
        if hls and len(hls) == 2:
            p1_move = getHandMove(hls[0])
            p2_move = getHandMove2(hls[1])
            gameText = f'P1: {str(p1_move)}, P2: {str(p2_move)}'

            key1 = key_mapping["Left"].get(p1_move)
            prev_key1 = current_keys_down["Left"]

            if key1 != prev_key1:
                if prev_key1:
                    pydirectinput.keyUp(prev_key1)
                if key1:
                    pydirectinput.keyDown(key1)
                current_keys_down["Left"] = key1


            key2 = key_mapping["Right"].get(p2_move)
            prev_key2 = current_keys_down["Right"]
            if key2 != prev_key2: 
                if prev_key2:
                    pydirectinput.keyUp(prev_key2)
                if key2:
                    pydirectinput.keyDown(key2)
                current_keys_down["Right"] = key2

            # Update the state to remember which key is now being held.
        else:
            gameText = "Put 2 hands in the frame at a time!"       
            for side in ["Left", "Right"]:
                if current_keys_down[side]:
                    pydirectinput.keyUp(current_keys_down[side])
                    current_keys_down[side] = None   
        
        cv.putText(frame, gameText, (25, 40), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2, cv.LINE_AA)
        cv.imshow('frame', frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'): break
for side in ["Left", "Right"]:
    if current_keys_down[side]:
        pydirectinput.keyUp(current_keys_down[side])
vid.release()
cv.destroyAllWindows()    

