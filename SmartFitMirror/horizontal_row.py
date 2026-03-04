import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
draw = mp.solutions.drawing_utils

def angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    ang = abs(rad * 180 / np.pi)
    return 360 - ang if ang > 180 else ang

def distance_x(a, b):
    return abs(a.x - b.x)

# ========== НАСТРОЙКИ ==========
SIDE = 'right'           # 'right' или 'left' - какой стороной к камере
ANGLE_TARGET = 90        # целевой угол в локте (90 градусов)
ANGLE_TOLERANCE = 20     # допуск ±20 градусов (70-110)
HAND_CLOSE_THRESH = 0.15 # расстояние до корпуса, когда рука прижата
HAND_FAR_THRESH = 0.25   # расстояние, когда рука вытянута
# ===============================

cap = cv2.VideoCapture(0)
reps = 0
stage = "START"  # START, PULL, HOLD
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

        # Выбираем сторону
        if SIDE == 'right':
            shoulder = lm[12]
            elbow = lm[14]
            wrist = lm[16]
            hip = lm[24]
        else:
            shoulder = lm[11]
            elbow = lm[13]
            wrist = lm[15]
            hip = lm[23]

        # Угол в локте
        elbow_angle = angle(shoulder, elbow, wrist)
        
        # Расстояние от запястья до бедра (по горизонтали)
        hand_to_hip = distance_x(wrist, hip)

        # --- Логика повторений ---
        if stage == "START":
            # Рука вытянута вперед
            if elbow_angle > 150 and hand_to_hip > HAND_FAR_THRESH:
                stage = "PULL"
                good_form = True
                
        elif stage == "PULL":
            # Тянем руку к корпусу
            if hand_to_hip < HAND_CLOSE_THRESH:
                # Проверяем угол в локте
                if (ANGLE_TARGET - ANGLE_TOLERANCE) <= elbow_angle <= (ANGLE_TARGET + ANGLE_TOLERANCE):
                    status, color = "GOOD FORM", (0, 255, 0)
                    good_form = True
                else:
                    status, color = "BAD ANGLE", (0, 0, 255)
                    good_form = False
                stage = "HOLD"
            else:
                status, color = "PULLING...", (255, 255, 0)
                
        elif stage == "HOLD":
            # Возвращаем руку в исходное положение
            if hand_to_hip > HAND_FAR_THRESH and elbow_angle > 150:
                stage = "START"
                if good_form:
                    reps += 1
                    status, color = "GOOD REP", (0, 255, 0)
                else:
                    status, color = "BAD REP", (0, 0, 255)

        # Определяем статус для отображения
        if stage == "START":
            if elbow_angle > 150:
                status, color = "READY", (255, 255, 0)
            else:
                status, color = "EXTEND ARM", (0, 165, 255)
        elif stage == "PULL":
            status, color = "PULLING...", (255, 255, 0)
        elif stage == "HOLD":
            if (ANGLE_TARGET - ANGLE_TOLERANCE) <= elbow_angle <= (ANGLE_TARGET + ANGLE_TOLERANCE):
                status, color = "HOLD GOOD", (0, 255, 0)
            else:
                status, color = "HOLD BAD", (0, 0, 255)

        # Интерфейс
        cv2.rectangle(img, (0, 0), (350, 150), (0, 0, 0), -1)
        cv2.putText(img, f"REPS: {reps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.putText(img, f"STATUS: {status}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img, f"ANGLE: {elbow_angle:.0f}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(img, f"HAND DIST: {hand_to_hip:.2f}", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        draw.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Horizontal Row Tracker", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()