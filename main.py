#opencv and mediapipe
import cv2 as cv
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def getHandMove(hand_landmarks):
    threshold = 20
    landmarks = hand_landmarks.landmark
    if all([landmarks[i].y < landmarks[i+3].y for i in range(5, 20, 4)]): return "STOP"
    elif all([landmarks[i].y < landmarks[i+3].y for i in range(5, 20, 4)]): return "GO"
    elif all([(landmarks[i+3].x - landmarks[i].x) > threshold for i in range(5,20,4)]): return "frontspin"
    elif all([(landmarks[i+3].x - landmarks[i].x) < -threshold for i in range(5,20,4)]): return "backspin"
    else: return "Pending Hand Movement"
vid = cv.VideoCapture(0)

clock = 0
p1_move = None
gameText = ""
success = True

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
        
        if 0 == clock < 20:
            success = True
            gameText = "Ready?"
        elif clock == 21:
            hls = results.multi_hand_landmarks
            if hls and len(hls) == 1:
                p1_move = getHandMove(hls[0])
            else:
                success = False
        
        
        cv.imshow('frame', frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'): break
        
vid.release()
cv.destroyAllWindows()    

