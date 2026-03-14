import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
draw = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    ang = abs(rad * 180 / np.pi)
    return 360 - ang if ang > 180 else ang

cap = cv2.VideoCapture(0)
reps = 0
stage = "UP" 
fully_extended = False  # Флаг полной растяжки
has_error = False       # Флаг ошибки (неполная амплитуда)
status_text = "EXTEND ARM FULLY"
status_color = (255, 255, 255)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark

        # Правая рука (12, 14, 16)
        shoulder = [lm[12].x, lm[12].y]
        elbow = [lm[14].x, lm[14].y]
        wrist = [lm[16].x, lm[16].y]

        angle = calculate_angle(shoulder, elbow, wrist)

        # --- ЛОГИКА БИЦЕПСА ---

        # 1. НИЖНЯЯ ТОЧКА (Полное разгибание)
        if angle > 165:
            if not fully_extended:
                fully_extended = True
                has_error = False  # Сброс ошибки только при полном разгибании
                status_text = "READY! CURL NOW"
                status_color = (0, 255, 0)
            stage = "DOWN"

        # 2. ФИКСАЦИЯ ОШИБКИ (Начал сгибать, не выпрямив до конца)
        # Если рука в "серой зоне" и мы не зафиксировали полное разгибание
        if stage == "UP" and angle < 140 and not fully_extended:
            has_error = True
            status_text = "NOT FULLY EXTENDED!"
            status_color = (0, 0, 255)

        # 3. ВЕРХНЯЯ ТОЧКА (Полное сгибание)
        if angle < 35:
            if stage == "DOWN":
                if fully_extended and not has_error:
                    reps += 1
                    status_text = "GOOD REP!"
                    status_color = (0, 255, 0)
                else:
                    status_text = "BAD FORM: NO COUNT"
                    status_color = (0, 0, 255)
                
                # Переход в режим ожидания разгибания
                stage = "UP"
                fully_extended = False

        # --- ИНТЕРФЕЙС ---
        # Плашка
        cv2.rectangle(img, (0, 0), (550, 100), (0, 0, 0), -1)
        
        # КРАСНАЯ РАМКА (Блокировка визуально)
        if has_error:
            cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 20)

        cv2.putText(img, f"REPS: {reps}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, status_text, (20, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        draw.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Strict Bicep Control", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()