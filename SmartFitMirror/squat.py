import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
draw = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    ang = abs(rad * 180 / np.pi)
    return 360 - ang if ang > 180 else ang

cap = cv2.VideoCapture(0)
reps = 0
stage = "UP"
has_error = False      # Флаг ошибки (недосел)
reached_depth = False   # Достиг ли правильной глубины

status_text = "SQUAT DOWN"
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

        # Берем точки (таз, колено, голеностоп)
        hip = [lm[24].x, lm[24].y]
        knee = [lm[26].x, lm[26].y]
        ankle = [lm[28].x, lm[28].y]

        k_angle = calculate_angle(hip, knee, ankle)

        # --- СТРОГАЯ ЛОГИКА ---

        # 1. НИЖНЯЯ ТОЧКА (Глубина приседа)
        if 70 <= k_angle <= 105:
            if not reached_depth:
                reached_depth = True
                has_error = False # Сброс ошибки при достижении глубины
                status_text = "GOOD DEPTH! GO UP"
                status_color = (0, 255, 0)
            stage = "DOWN"

        # 2. ФИКСАЦИЯ ОШИБКИ (Начал вставать, не досев)
        if stage == "UP" and 110 < k_angle < 150 and not reached_depth:
            has_error = True
            status_text = "NOT DEEP ENOUGH!"
            status_color = (0, 0, 255)

        # 3. ВЕРХНЯЯ ТОЧКА (Полное выпрямление)
        if k_angle > 165:
            if stage == "DOWN":
                if reached_depth and not has_error:
                    reps += 1
                    status_text = "GOOD REP!"
                    status_color = (0, 255, 0)
                else:
                    status_text = "BAD FORM: NO COUNT"
                    status_color = (0, 0, 255)
                
                # Сброс для следующего раза
                stage = "UP"
                reached_depth = False
            
            # Если стоим прямо и нет ошибок
            if not has_error and not reached_depth:
                status_text = "READY"
                status_color = (255, 255, 255)

        # --- ИНТЕРФЕЙС ---
        cv2.rectangle(img, (0, 0), (550, 100), (0, 0, 0), -1)
        
        # КРАСНАЯ РАМКА
        if has_error:
            cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 20)

        cv2.putText(img, f"REPS: {reps}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, status_text, (20, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Подсказка по углу для отладки
        cv2.putText(img, f"Angle: {int(k_angle)}", (400, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        draw.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Strict Squat Checker", img)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()