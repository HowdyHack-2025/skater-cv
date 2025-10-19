# opencv and mediapipe
import cv2 as cv
import mediapipe as mp
import keyboard

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def getHandMove(hand_landmarks):
    threshold = 0.1
    landmarks = hand_landmarks.landmark
    if all([(landmarks[i+3].x - landmarks[i].x) > threshold for i in range(5,20,4)]): 
        return "Backspin"
    if all([(landmarks[i+3].x - landmarks[i].x) < -threshold for i in range(5,20,4)]): 
        return "Frontspin"
    if all([landmarks[i].y < landmarks[i+3].y for i in range(5, 20, 4)]): 
        return "STOP"
    if all([landmarks[i].y > landmarks[i+3].y for i in range(5, 20, 4)]): 
        return "GO"
    else: 
        return "Pending"

# Different key mappings for each hand
left_hand_mapping = {
    "GO": "w",
    "STOP": "s",
    "Frontspin": "d",
    "Backspin": "a",
    "Pending": None
}

right_hand_mapping = {
    "GO": "i",
    "STOP": "k",
    "Frontspin": "l",
    "Backspin": "j",
    "Pending": None
}

vid = cv.VideoCapture(0)
current_key_pressed = None

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,  # detect up to 2 hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while True:
        ret, frame = vid.read()
        if not ret or frame is None: 
            break

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        gameText = ""

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                hand_label = handedness.classification[0].label  # 'Left' or 'Right'
                move = getHandMove(hand_landmarks)
        
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
                # Map left/right to Player 1/Player 2
                if hand_label == "Left":
                    player = "Player 1"
                    target_key = left_hand_mapping.get(move)
                else:
                    player = "Player 2"
                    target_key = right_hand_mapping.get(move)
        
                if target_key:
                    keyboard.send(target_key)
        
                # Format text as: "Player 1 (Left): GO"
                gameText += f"{player} ({hand_label}): {move}  "

        else:
            gameText = "Show both hands!"


        cv.putText(frame, gameText, (25, 40), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 2, cv.LINE_AA)
        cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if current_key_pressed is not None:
    keyboard.release(current_key_pressed)

vid.release()
cv.destroyAllWindows()
