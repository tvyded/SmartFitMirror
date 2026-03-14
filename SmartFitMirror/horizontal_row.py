import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
draw = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    ang = abs(rad * 180 / np.pi)
    return 360 - ang if ang > 180 else ang

# Настройки (подбери под свое расстояние до камеры)
SIDE = 'right'
HAND_CLOSE_THRESH = 0.15 
HAND_FAR_THRESH = 0.35    

cap = cv2.VideoCapture(0)
reps = 0
stage = "WAIT"           
has_error = False        # Если True — реп испорчен и не будет засчитан
status_text = "STRETCH ARMS TO START"
status_color = (255, 255, 255)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        
        if SIDE == 'right':
            shoulder, elbow, wrist, hip = lm[12], lm[14], lm[16], lm[24]
        else:
            shoulder, elbow, wrist, hip = lm[11], lm[13], lm[15], lm[23]

        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        hand_to_hip = abs(wrist.x - hip.x)

        # --- СУПЕР СТРОГАЯ ЛОГИКА ---

        # 1. ТОЧКА ОБНУЛЕНИЯ (Полное выпрямление)
        if hand_to_hip > HAND_FAR_THRESH and elbow_angle > 160:
            if stage != "READY":
                stage = "READY"
                has_error = False # Сброс ошибки только здесь
                status_text = "READY: PULL NOW"
                status_color = (0, 255, 0)

        # 2. ФИКСАЦИЯ ОШИБКИ (Начал тянуть раньше времени)
        if stage == "WAIT" and hand_to_hip < (HAND_FAR_THRESH - 0.05):
            has_error = True
            status_text = "ERROR: STRETCH FIRST"
            status_color = (0, 0, 255)

        # 3. ФАЗА ТЯГИ И ПРОВЕРКА ЗАВЕРШЕНИЯ
        if hand_to_hip < HAND_CLOSE_THRESH:
            if stage == "READY":
                if not has_error:
                    reps += 1
                    status_text = "GOOD REP!"
                    status_color = (0, 255, 0)
                else:
                    status_text = "BAD FORM: NO COUNT"
                    status_color = (0, 0, 255)
                
                # После того как рука у корпуса, всегда уходим в WAIT
                stage = "WAIT" 
            
            elif stage == "WAIT" and not has_error:
                # Если человек просто держит руку у корпуса, не вытянув её
                has_error = True 
                status_text = "ERROR: NOT STRETCHED"
                status_color = (0, 0, 255)

        # --- ИНТЕРФЕЙС ---
        cv2.rectangle(img, (0, 0), (550, 100), (0, 0, 0), -1)
        
        # Визуальная блокировка (Рамка)
        if has_error:
            cv2.rectangle(img, (0,0), (img.shape[1], img.shape[0]), (0,0,255), 20)

        cv2.putText(img, f"REPS: {reps}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, status_text, (20, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        draw.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Extreme Strict Row", img)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()