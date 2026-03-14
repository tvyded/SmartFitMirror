import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
reps = 0
stage = "DOWN"

# Системные флаги
good_top = False      # Достиг ли правильной высоты
bad_form_flag = False # Была ли ошибка (слишком высоко)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark

        # Координаты плеч и запястий (среднее значение для стабильности)
        avg_shoulder_y = (lm[12].y + lm[11].y) / 2
        avg_wrist_y = (lm[16].y + lm[15].y) / 2

        # --- ЛОГИКА ПОВТОРЕНИЙ ---

        # 1. НИЖНЯЯ ТОЧКА (Руки опущены)
        if avg_wrist_y > avg_shoulder_y + 0.3: 
            if stage == "UP":
                # Если дошли до верха и НЕ накосячили по пути — считаем реп
                if good_top and not bad_form_flag:
                    reps += 1
                    print(f"GOOD REP! Total: {reps}")
                else:
                    print("BAD REP! Form issue or not high enough.")
            
            # Сброс всех флагов в нижней точке
            stage = "DOWN"
            good_top = False
            bad_form_flag = False

        # 2. ВЕРХНЯЯ ТОЧКА (Мах до уровня плеч)
        elif avg_wrist_y < avg_shoulder_y + 0.1:
            stage = "UP"
            
            # Если поднял достаточно высоко (до уровня плеч)
            if avg_wrist_y <= avg_shoulder_y + 0.1:
                good_top = True
            
            # ЕСЛИ ПОДНЯЛ СЛИШКОМ ВЫСОКО (Ошибка)
            if avg_wrist_y < avg_shoulder_y - 0.1:
                bad_form_flag = True

        # --- ИНТЕРФЕЙС ---
        # Статус сообщения
        if bad_form_flag:
            status_text = "TOO HIGH! REP RUINED"
            status_color = (0, 0, 255) # Красный
        elif good_top:
            status_text = "GOOD HEIGHT"
            status_color = (0, 255, 0) # Зеленый
        elif stage == "UP":
            status_text = "LIFTING..."
            status_color = (0, 255, 255) # Желтый
        else:
            status_text = "READY / DOWN"
            status_color = (255, 255, 255) # Белый

        # Отрисовка плашки
        cv2.rectangle(img, (0, 0), (450, 100), (0, 0, 0), -1)
        
        # Если ошибка — рисуем жирную красную рамку
        if bad_form_flag:
            cv2.rectangle(img, (0,0), (img.shape[1], img.shape[0]), (0,0,255), 15)

        cv2.putText(img, f"REPS: {reps}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, status_text, (20, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        draw.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Strict Lateral Raises", img)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()