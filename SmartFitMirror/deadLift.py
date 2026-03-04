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

# ========== НАСТРОЙКИ ==========
SIDE = 'right'           # какой бок к камере: 'right' или 'left'
# Пороги глубины (y бедра: 0 = верх, 1 = низ)
LOWER_THRESH = 0.65      # при опускании ниже этого → начинаем повторение
RAISE_THRESH = 0.55      # при подъёме выше этого → завершаем повторение
# (LOWER должен быть БОЛЬШЕ RAISE, чтобы был зазор)
TOO_DEEP_THRESH = 0.72   # если ниже этого → слишком глубоко
BACK_ANGLE_THRESH = 70  # если угол спины меньше этого → спина круглая
# ===============================

cap = cv2.VideoCapture(0)
reps = 0
stage = "UP"           # UP = стоим, DOWN = опускаемся/поднимаемся
good_form = True       # был ли хоть один косяк во время движения

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

        # Выбираем правую или левую сторону
        if SIDE == 'right':
            shoulder = lm[12]
            hip = lm[24]
            knee = lm[26]
        else:
            shoulder = lm[11]
            hip = lm[23]
            knee = lm[25]

        back_angle = angle(shoulder, hip, knee)
        hip_y = hip.y

        # --- Логика переходов ---
        if stage == "UP" and hip_y > LOWER_THRESH:
            stage = "DOWN"
            good_form = True          # новый цикл – начинаем с чистой формы
        elif stage == "DOWN" and hip_y < RAISE_THRESH:
            stage = "UP"
            if good_form:
                reps += 1

        # --- Оценка формы (только один статус) ---
        if hip_y > TOO_DEEP_THRESH:
            status, color = "TOO DEEP", (0, 0, 255)
            good_form = False
        elif hip_y < RAISE_THRESH:
            status, color = "TOO HIGH", (0, 165, 255)
            # Не сбрасываем good_form, потому что в начале подхода это нормально
        else:
            # Между RAISE и TOO_DEEP – рабочий диапазон, проверяем спину
            if back_angle < BACK_ANGLE_THRESH:
                status, color = "ROUND BACK", (0, 0, 255)
                good_form = False
            else:
                status, color = "GOOD FORM", (0, 255, 0)

        # Интерфейс
        cv2.rectangle(img, (0, 0), (300, 120), (0, 0, 0), -1)
        cv2.putText(img, f"REPS: {reps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.putText(img, f"FORM: {status}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # Для отладки (можно удалить)
        cv2.putText(img, f"hip_y: {hip_y:.2f} ang:{back_angle:.1f}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        draw.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Deadlift checker", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()