#opencv and mediapipe
import cv2 as cv
import mediapipe as mp
import keyboard

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

key_mapping = {
    "GO": "w",
    "STOP": "s",
    "Frontspin": "e", # Assuming Frontspin is a leftward gesture for 'a'
    "Backspin": "q",  # Assuming Backspin is a rightward gesture for 'd'
    "Pending Hand Movement": None
}

current_key_pressed = None

vid = cv.VideoCapture(0)

p1_move = None
gameText = ""

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
        
        target_key = None
       
        
       
        hls = results.multi_hand_landmarks
        if hls and len(hls) == 1:
            p1_move = getHandMove(hls[0])
            gameText = str(p1_move)
            target_key = key_mapping.get(p1_move) 
            # If a key was held down, release it first.
            #if current_key_pressed is not None:
               # keyboard.release(current_key_pressed)
        
                # If the new gesture needs a key, press it dowwn.
            if not target_key is None:
                
                keyboard.send(target_key)
        
            # Update the state to remember which key is now being held.
            current_key_pressed = target_key  
        else:
            gameText = "Put 1 hand in the frame at a time!"
            target_key=None
        
        
        
        
        cv.putText(frame, gameText, (25, 40), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2, cv.LINE_AA)
        cv.imshow('frame', frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'): break
        
if current_key_pressed is not None:
    keyboard.release(current_key_pressed)
        
vid.release()
cv.destroyAllWindows()    

