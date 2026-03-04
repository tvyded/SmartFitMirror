import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
draw = mp.solutions.drawing_utils

def angle(a, b, c):
    a, b, c = np.array([a.x, a.y]), np.array([b.x, b.y]), np.array([c.x, c.y])
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    ang = abs(rad * 180 / np.pi)
    return 360 - ang if ang > 180 else ang

cap = cv2.VideoCapture(0)
reps = 0
stage = "DOWN"
good_form = True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark

        # Точки для обеих рук
        # Правая рука
        r_shoulder = lm[12]
        r_elbow = lm[14]
        r_wrist = lm[16]
        
        # Левая рука
        l_shoulder = lm[11]
        l_elbow = lm[13]
        l_wrist = lm[15]

        # Высота запястий (y - чем меньше, тем выше)
        r_wrist_y = r_wrist.y
        l_wrist_y = l_wrist.y
        
        # Высота плеч (для проверки "не выше плеч")
        r_shoulder_y = r_shoulder.y
        l_shoulder_y = l_shoulder.y

        # Углы в локтях (должны быть слегка согнуты)
        r_elbow_angle = angle(r_shoulder, r_elbow, r_wrist)
        l_elbow_angle = angle(l_shoulder, l_elbow, l_wrist)

        # Средняя высота запястий (для определения фазы)
        avg_wrist_y = (r_wrist_y + l_wrist_y) / 2

        # --- Логика повторений ---
        if stage == "DOWN" and avg_wrist_y < 0.3:  # руки подняты
            stage = "UP"
            good_form = True
        elif stage == "UP" and avg_wrist_y > 0.4:  # руки опущены
            stage = "DOWN"
            if good_form:
                reps += 1

        # --- Оценка формы ---
        status = "GOOD FORM"
        color = (0, 255, 0)
        
        # Проверяем обе руки
        # 1. Локти не должны быть заблокированы (слишком прямые)
        if r_elbow_angle > 160 or l_elbow_angle > 160:
            status = "LOCKED ELBOWS"
            color = (0, 0, 255)
            good_form = False
        # 2. Локти не должны быть слишком согнуты
        elif r_elbow_angle < 30 or l_elbow_angle < 30:
            status = "ELBOWS TOO BENT"
            color = (0, 165, 255)
            good_form = False
        # 3. Руки не должны подниматься выше плеч
        elif r_wrist_y < r_shoulder_y - 0.05 or l_wrist_y < l_shoulder_y - 0.05:
            status = "TOO HIGH"
            color = (0, 0, 255)
            good_form = False
        # 4. Руки должны подниматься синхронно (разница не больше 0.05)
        elif abs(r_wrist_y - l_wrist_y) > 0.05:
            status = "NOT SYNC"
            color = (0, 165, 255)
            good_form = False

        # Интерфейс
        cv2.rectangle(img, (0, 0), (300, 150), (0, 0, 0), -1)
        cv2.putText(img, f"REPS: {reps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.putText(img, f"FORM: {status}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Информация для отладки
        cv2.putText(img, f"R angle: {r_elbow_angle:.0f} L angle: {l_elbow_angle:.0f}", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(img, f"R wrist Y: {r_wrist_y:.2f} L wrist Y: {l_wrist_y:.2f}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        draw.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Shoulder Press (Lateral Raises)", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()