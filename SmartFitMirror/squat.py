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
stage = "UP"
correct_down = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark

        hip = [lm[24].x, lm[24].y]
        knee = [lm[26].x, lm[26].y]
        ankle = [lm[28].x, lm[28].y]

        k_angle = angle(hip, knee, ankle)

        # --- Stage logic ---
        if stage == "UP" and 80 <= k_angle <= 100:
            stage = "DOWN"
            correct_down = True
        elif stage == "DOWN" and k_angle > 160:
            stage = "UP"
            if correct_down:
                reps += 1
            correct_down = False

        
        if 80 <= k_angle <= 100:
            status, color = "GOOD DEPTH", (0, 255, 0)
        elif k_angle < 70:
            status, color = "TOO DEEP", (0, 0, 255)
        else:
            status, color = "BAD FORM", (0, 165, 255)

        cv2.rectangle(img, (0, 0), (260, 90), (0, 0, 0), -1)
        cv2.putText(img, f"REPS: {reps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(img, f"FORM: {status}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        draw.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Squat checker", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()