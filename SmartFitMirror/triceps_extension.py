import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
draw = mp.solutions.drawing_utils


def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    ang = abs(rad * 180 / np.pi)
    return 360 - ang if ang > 180 else ang


cap = cv2.VideoCapture(0)
reps = 0
stage = "UP"  # UP - поднимание, DOWN - опускание
good_flag = False  # флаг что был хороший угол
bad_flag = False   # флаг что был плохой угол

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark

        # Точки для правой руки
        shoulder = [lm[12].x, lm[12].y]
        elbow = [lm[14].x, lm[14].y]
        wrist = [lm[16].x, lm[16].y]

        # Угол в локте
        arm_angle = angle(shoulder, elbow, wrist)

        # Определяем фазы по углу
        if arm_angle > 150:  # Рука прямая - фаза поднимания
            if stage == "DOWN":  # Только что закончили опускание
                if good_flag and not bad_flag:
                    reps += 1
                    result_text = "GOOD REP!"
                    result_color = (0, 255, 0)
                else:
                    result_text = "BAD REP!"
                    result_color = (0, 0, 255)
                print(f"{result_text} Total: {reps}")
            stage = "UP"
            good_flag = False
            bad_flag = False
            
        elif arm_angle < 100:  # Рука согнута - фаза опускания
            stage = "DOWN"
            # Проверяем углы во время опускания
            if 80 <= arm_angle <= 100:
                good_flag = True
            elif arm_angle < 70 or arm_angle > 150:
                bad_flag = True

        # Интерфейс
        cv2.rectangle(img, (0, 0), (300, 180), (0, 0, 0), -1)
        cv2.putText(img, f"REPS: {reps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(img, f"ANGLE: {arm_angle:.0f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"PHASE: {stage}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Флаги
        if good_flag:
            cv2.putText(img, "GOOD FLAG: YES", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(img, "GOOD FLAG: NO", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
        if bad_flag:
            cv2.putText(img, "BAD FLAG: YES", (10, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.putText(img, "BAD FLAG: NO", (10, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        draw.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Triceps Extension", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()