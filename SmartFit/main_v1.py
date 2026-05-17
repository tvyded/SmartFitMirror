import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os

# ═══════════════════════════════════════════════════════════════
# 🎨 ЦВЕТОВАЯ СХЕМА
# ═══════════════════════════════════════════════════════════════
BG = "#ffffff"           # Основной фон (белый)
PANEL = "#f0f2f5"        # Панели (светло-серый)
CARD = "#ffffff"         # Карточки (белый)
CARD_HOVER = "#e6e9ed"   # Ховер карточек
TEXT = "#1a1a1a"         # Основной текст (почти чёрный)
TEXT_SECONDARY = "#4a4a4a"  # Вторичный текст
ACCENT = "#00aa55"       # Акцент (насыщенный зелёный)
ACCENT_DARK = "#008844"  # Акцент при наведении
ERROR = "#b30000"        # Ошибка (тёмно-красный)
WARNING = "#b37700"      # Предупреждение (тёмный янтарь)
SUCCESS = "#009944"      # Успех
MUTED = "#999999"        # Приглушённый
BORDER = "#d0d5dd"       # Границы элементов


# ═══════════════════════════════════════════════════════════════
# 📐 ШРИФТЫ
# ═══════════════════════════════════════════════════════════════
def get_font(size, weight="normal"):
    """Возвращает корректный кортеж шрифта для tkinter"""
    if weight == "normal":
        return ("Segoe UI", size)
    else:
        return ("Segoe UI", size, weight)


# ═══════════════════════════════════════════════════════════════
# 🦴 ОТРИСОВКА СКЕЛЕТА
# ═══════════════════════════════════════════════════════════════
def draw_white_skeleton_on_black(frame_shape, landmarks, connections):
    h, w = frame_shape
    black_frame = np.zeros((h, w, 3), dtype=np.uint8)

    for conn in connections:
        lm_s, lm_e = landmarks[conn[0]], landmarks[conn[1]]
        if lm_s.visibility > 0.4 and lm_e.visibility > 0.4:
            x1, y1 = int(lm_s.x * w), int(lm_s.y * h)
            x2, y2 = int(lm_e.x * w), int(lm_e.y * h)
            cv2.line(black_frame, (x1, y1), (x2, y2), (255, 255, 255), 8)

    for lm in landmarks:
        if lm.visibility > 0.4:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(black_frame, (x, y), 8, (100, 255, 200), -1)
            cv2.circle(black_frame, (x, y), 4, (255, 255, 255), -1)
    return black_frame


def draw_skeleton_on_frame(frame, landmarks, connections):
    h, w = frame.shape[:2]
    for conn in connections:
        lm_s, lm_e = landmarks[conn[0]], landmarks[conn[1]]
        if lm_s.visibility > 0.4 and lm_e.visibility > 0.4:
            x1, y1 = int(lm_s.x * w), int(lm_s.y * h)
            x2, y2 = int(lm_e.x * w), int(lm_e.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 153), 7)
    for lm in landmarks:
        if lm.visibility > 0.4:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 8, (0, 255, 153), -1)
            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
    return frame


def draw_progress_bar(frame, elapsed, total=3):
    h, w = frame.shape[:2]
    bw, bh = 450, 20
    bx, by = w // 2 - bw // 2, h - 60

    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)
    filled = int(bw * min(elapsed / total, 1.0))
    cv2.rectangle(frame, (bx, by), (bx + filled, by + bh), (0, 255, 153), -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)
    return frame


def draw_error_frame(frame, error_flag):
    if error_flag:
        alpha = 0.6 + 0.4 * math.sin(time.time() * 8)
        color = (int(255 * alpha), int(85 * alpha), int(85 * alpha))
        cv2.rectangle(frame, (15, 15), (frame.shape[1] - 15, frame.shape[0] - 15), color, 5)
    return frame


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    ang = abs(rad * 180 / np.pi)
    return 360 - ang if ang > 180 else ang


def landmark_xy(lm):
    return [lm.x, lm.y]


def load_icon(path, size=(70, 70)):
    """Безопасная загрузка иконки с заглушкой"""
    try:
        if os.path.exists(path):
            img = Image.open(path).resize(size, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img)
    except:
        pass
    return None


# ═══════════════════════════════════════════════════════════════
# 🎬 ЗАСТАВКА
# ═══════════════════════════════════════════════════════════════
class SplashScreen(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.overrideredirect(True)
        self.configure(bg=BG)

        screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{screen_w}x{screen_h}+0+0")
        self.attributes('-alpha', 0.0)

        self.setup_ui()
        self.fade_in()
        self.after(3000, self.fade_out_and_close)

    def setup_ui(self):
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True)

        center = tk.Frame(main, bg=BG)
        center.place(relx=0.5, rely=0.5, anchor="center")

        self.canvas = tk.Canvas(center, width=140, height=140, bg=BG, highlightthickness=0)
        self.canvas.pack(pady=(0, 25))
        self.outer = self.canvas.create_oval(15, 15, 125, 125, outline=ACCENT, width=4)
        self.inner = self.canvas.create_oval(45, 45, 95, 95, outline=ACCENT_DARK, width=3)

        tk.Label(center, text="🏋️ SMARTFIT MIRROR", font=get_font(36, "bold"),
                 fg=ACCENT, bg=BG).pack(pady=(0, 8))
        tk.Label(center, text="Умное фитнес-зеркало", font=get_font(16),
                 fg=TEXT_SECONDARY, bg=BG).pack()

        tk.Frame(center, bg=MUTED, height=2, width=400).pack(pady=35)

        logos = tk.Frame(center, bg=BG)
        logos.pack(pady=10)

        # Список логотипов: (путь_к_файлу, название)
        logo_files = [
            ("istok.png", "НПП «Исток» им. Шокина"),
            ("rtu_mirea.png", "РТУ МИРЭА")
        ]

        for logo_path, name in logo_files:
            f = tk.Frame(logos, bg=CARD, relief="flat", bd=0, padx=25, pady=15)
            f.pack(side="left", padx=20)

            try:
                if os.path.exists(logo_path):
                    logo_img = Image.open(logo_path)

                    # 📐 Сохраняем пропорции:
                    # 1. Задаем максимальную ширину
                    max_width = 180

                    # 2. Считаем коэффициент сжатия
                    ratio = max_width / logo_img.width

                    # 3. Считаем новую высоту (чтобы пропорции остались верными)
                    new_height = int(logo_img.height * ratio)

                    # 4. Масштабируем
                    logo_img = logo_img.resize((max_width, new_height), Image.Resampling.LANCZOS)

                    logo_photo = ImageTk.PhotoImage(logo_img)
                    logo_label = tk.Label(f, image=logo_photo, bg=CARD)
                    logo_label.image = logo_photo
                    logo_label.pack(pady=(0, 12))
                else:
                    tk.Label(f, text="📁", font=get_font(30), bg=CARD, fg=MUTED).pack(pady=(0, 10))
            except:
                tk.Label(f, text="⚠️", font=get_font(30), bg=CARD, fg=WARNING).pack(pady=(0, 10))

            tk.Label(f, text=name, font=get_font(12, "bold"),
                     fg=TEXT_SECONDARY, bg=CARD, justify="center").pack()
        # 🔧 Прогресс-бар (исправленная настройка)
        style = ttk.Style()
        try:
            style.layout("Accent.TProgressbar", [
                ('Horizontal.Progressbar.trough', {
                    'sticky': 'nswe',
                    'children': [('Horizontal.Progressbar.pbar', {'sticky': 'nswe'})]
                })
            ])
            style.configure("Accent.TProgressbar",
                            background=ACCENT, troughcolor=CARD, borderwidth=0)
            pb_style = "Accent.TProgressbar"
        except:
            pb_style = "Horizontal.TProgressbar"

        self.progress = ttk.Progressbar(center, length=400, mode='determinate', style=pb_style)
        self.progress.pack(pady=25)

        self.animate_progress()
        self.animate_logo()

    def animate_progress(self):
        val = self.progress['value']
        if val < 100:
            self.progress['value'] = val + 2.5
            self.after(40, self.animate_progress)

    def animate_logo(self):
        s = 125 + 10 * math.sin(time.time() * 4)
        self.canvas.coords(self.outer, 15, 15, s, s)
        self.canvas.coords(self.inner, 45, 45, s - 30, s - 30)
        self.after(40, self.animate_logo)

    def fade_in(self):
        a = self.attributes('-alpha')
        if a < 1.0:
            self.attributes('-alpha', a + 0.06)
            self.after(18, self.fade_in)

    def fade_out_and_close(self):
        a = self.attributes('-alpha')
        if a > 0:
            self.attributes('-alpha', a - 0.06)
            self.after(18, self.fade_out_and_close)
        else:
            self.destroy()
            self.parent.deiconify()
            self.parent.attributes("-fullscreen", True)


# ═══════════════════════════════════════════════════════════════
# 🎥 ВИДЕО-ПОДСКАЗКА
# ═══════════════════════════════════════════════════════════════
class VideoPlayer:
    def __init__(self, parent, video_path, max_w=360, max_h=240):
        self.parent = parent
        self.video_path = video_path
        self.max_w, self.max_h = max_w, max_h
        self.cap = None
        self.playing = False
        self.job = None

        self.frame = tk.Frame(parent, bg=CARD, relief="flat", bd=0)
        self.frame.pack(fill="x", pady=(8, 12), padx=18)

        tk.Label(self.frame, text="📹 ТЕХНИКА ВЫПОЛНЕНИЯ",
                 font=get_font(11, "bold"), fg=TEXT_SECONDARY, bg=CARD).pack(anchor="w", padx=12, pady=(10, 6))

        self.container = tk.Frame(self.frame, bg=CARD)
        self.container.pack(padx=12, pady=(0, 12))
        self.label = tk.Label(self.container, bg=CARD)
        self.label.pack()
        self.load_video()

    def load_video(self):
        if os.path.exists(self.video_path):
            try:
                self.cap = cv2.VideoCapture(self.video_path)
                if self.cap.isOpened():
                    self.playing = True
                    self.play()
                else:
                    self.show_placeholder("❌ Не удалось открыть видео")
            except:
                self.show_placeholder("⚠️ Ошибка видео")
        else:
            self.show_placeholder("🎬 Видео будет доступно позже")

    def show_placeholder(self, text):
        tk.Label(self.container, text=text, font=get_font(10),
                 fg=MUTED, bg=CARD, pady=20).pack()

    def play(self):
        if self.playing and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                h, w = frame.shape[:2]
                scale = min(self.max_w / w, self.max_h / h)
                nw, nh = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (nw, nh))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.label.config(image=photo)
                self.label.image = photo
                self.job = self.label.after(33, self.play)
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.job = self.label.after(33, self.play)

    def stop(self):
        self.playing = False
        if self.job:
            self.label.after_cancel(self.job)
        if self.cap:
            self.cap.release()


# ═══════════════════════════════════════════════════════════════
# 🏋️ БАЗОВЫЙ КЛАСС УПРАЖНЕНИЯ
# ═══════════════════════════════════════════════════════════════
class ExerciseBase:
    name = "УПРАЖНЕНИЕ"
    video_file = None

    def __init__(self):
        self.reset()

    def reset(self):
        self.reps = 0
        self.stage = "IDLE"
        self.status_text = "ГОТОВ"
        self.status_color = TEXT_SECONDARY
        self.error_flag = False

    def draw(self, frame):
        return draw_error_frame(frame, self.error_flag)

    def process(self, frame, lm):
        return self.draw(frame)


# ═══════════════════════════════════════════════════════════════
# 💪 УПРАЖНЕНИЯ
# ═══════════════════════════════════════════════════════════════
class BicepsExercise(ExerciseBase):
    name = "💪 СГИБАНИЕ НА БИЦЕПС"
    video_file = "biceps.mp4"

    def reset(self):
        super().reset()
        self.stage = "UP"
        self.fully_extended = False
        self.status_text = "👉 Выпрямите руку полностью"

    def process(self, frame, lm):
        if lm:
            s, e, w = landmark_xy(lm[12]), landmark_xy(lm[14]), landmark_xy(lm[16])
            angle = calculate_angle(s, e, w)
            if angle > 165:
                if not self.fully_extended:
                    self.fully_extended = True
                    self.error_flag = False
                    self.status_text = "✅ Готово! Сгибайте руку"
                    self.status_color = SUCCESS
                self.stage = "DOWN"
            if self.stage == "UP" and angle < 140 and not self.fully_extended:
                self.error_flag = True
                self.status_text = "⚠️ Выпрямите сильнее!"
                self.status_color = WARNING
            if angle < 35:
                if self.stage == "DOWN":
                    if self.fully_extended and not self.error_flag:
                        self.reps += 1
                        self.status_text = "🎉 Отлично! Повтор засчитан"
                        self.status_color = SUCCESS
                    else:
                        self.status_text = "❌ Начните с выпрямленной руки"
                        self.status_color = ERROR
                    self.stage = "UP"
                    self.fully_extended = False
        return self.draw(frame)


class ShouldersExercise(ExerciseBase):
    name = "🤲 ПОДЪЁМ РУК (ПЛЕЧИ)"
    video_file = "shoulders.mp4"

    def reset(self):
        super().reset()
        self.stage = "DOWN"              # DOWN — руки опущены, UP — поднимаем
        self.reached_top = False         # Была ли достигнута правильная высота
        self.too_high_flag = False       # Флаг: слишком высоко
        self.too_low_flag = False        # Флаг: недостаточно высоко
        self.arms_not_sideways = False   # Флаг: руки не в стороны
        self.elbows_bent = False         # Флаг: согнуты локти
        self.rep_feedback = ""           # Обратная связь после повтора
        self.status_text = "👉 Разведите руки в стороны и поднимите до уровня плеч"
        self.status_color = TEXT_SECONDARY

    def process(self, frame, lm):
        if lm:
            # Ключевые точки
            left_shoulder = lm[11]
            right_shoulder = lm[12]
            left_elbow = lm[13]
            right_elbow = lm[14]
            left_wrist = lm[15]
            right_wrist = lm[16]
            left_hip = lm[23]
            right_hip = lm[24]
            
            # Углы в плечевых суставах (корпус-плечо-локоть)
            left_shoulder_angle = calculate_angle(
                landmark_xy(left_hip),
                landmark_xy(left_shoulder),
                landmark_xy(left_elbow)
            )
            right_shoulder_angle = calculate_angle(
                landmark_xy(right_hip),
                landmark_xy(right_shoulder),
                landmark_xy(right_elbow)
            )
            
            # Углы в локтях (должны быть прямыми)
            left_elbow_angle = calculate_angle(
                landmark_xy(left_shoulder),
                landmark_xy(left_elbow),
                landmark_xy(left_wrist)
            )
            right_elbow_angle = calculate_angle(
                landmark_xy(right_shoulder),
                landmark_xy(right_elbow),
                landmark_xy(right_wrist)
            )
            
            # Средняя высота запястий и плеч
            avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            avg_wrist_y = (left_wrist.y + right_wrist.y) / 2
            
            # ✅ КРИТЕРИИ ПРАВИЛЬНОСТИ (жёсткие)
            
            # 1. Руки в стороны (угол в плече 80-100 градусов)
            arms_sideways = (80 < left_shoulder_angle < 100 and 
                           80 < right_shoulder_angle < 100)
            
            # 2. Локти прямые (>160 градусов)
            elbows_straight = (left_elbow_angle > 160 and right_elbow_angle > 160)
            
            # 3. Запястья СТРОГО на уровне плеч (допуск ±0.04)
            at_correct_height = abs(avg_wrist_y - avg_shoulder_y) < 0.04
            
            # 4. Слишком высоко
            too_high = avg_wrist_y < avg_shoulder_y - 0.08
            
            # 5. Слишком низко (для стадии UP)
            too_low_for_up = avg_wrist_y > avg_shoulder_y + 0.06
            
            # 🔄 ЛОГИКА СТАДИЙ
            
            if self.stage == "DOWN":
                # Сбрасываем все флаги ошибок
                self.too_high_flag = False
                self.too_low_flag = False
                self.arms_not_sideways = False
                self.elbows_bent = False
                
                # Ждём начала подъёма
                if avg_wrist_y < avg_shoulder_y + 0.08:
                    self.stage = "UP"
                    self.reached_top = False
                    self.status_text = "📈 Поднимайте до уровня плеч"
                    self.status_color = TEXT_SECONDARY
            
            elif self.stage == "UP":
                # Обновляем флаги ошибок ТОЛЬКО если ещё не достигли правильной позиции
                if not self.reached_top:
                    if too_high:
                        self.too_high_flag = True
                    if too_low_for_up:
                        self.too_low_flag = True
                    if not arms_sideways:
                        self.arms_not_sideways = True
                    if not elbows_straight:
                        self.elbows_bent = True
                
                # Проверяем, достиг ли идеальной позиции
                if at_correct_height and arms_sideways and elbows_straight:
                    if not self.reached_top:
                        self.reached_top = True
                        # Сбрасываем флаги ошибок, т.к. достиг правильно
                        self.too_high_flag = False
                        self.too_low_flag = False
                        self.arms_not_sideways = False
                        self.elbows_bent = False
                        self.status_text = "✅ Идеально! Опускайте руки"
                        self.status_color = SUCCESS
                
                # Ждём опускания рук
                if avg_wrist_y > avg_shoulder_y + 0.25:
                    # 🔥 ЗАСЧИТЫВАЕМ ТОЛЬКО ЕСЛИ:
                    # - Достиг правильной высоты (reached_top)
                    # - НЕ было ошибок (все флаги False)
                    if (self.reached_top and 
                        not self.too_high_flag and 
                        not self.too_low_flag and 
                        not self.arms_not_sideways and 
                        not self.elbows_bent):
                        
                        self.reps += 1
                        self.status_text = "🎉 Отлично! Повтор засчитан"
                        self.status_color = SUCCESS
                        self.rep_feedback = self.generate_feedback()
                    else:
                        # Формируем объяснение, почему не засчитано
                        errors = []
                        if not self.reached_top:
                            errors.append("поднимайте до уровня плеч")
                        if self.too_high_flag:
                            errors.append("не поднимайте выше плеч")
                        if self.too_low_flag:
                            errors.append("поднимайте выше")
                        if self.arms_not_sideways:
                            errors.append("разводите руки строго в стороны")
                        if self.elbows_bent:
                            errors.append("выпрямляйте локти")
                        
                        self.status_text = "❌ " + ", ".join(errors)
                        self.status_color = ERROR
                    
                    # Возвращаемся в исходное положение
                    self.stage = "DOWN"
                    self.reached_top = False
                    self.too_high_flag = False
                    self.too_low_flag = False
                    self.arms_not_sideways = False
                    self.elbows_bent = False
            
            # Обновляем подсказки в реальном времени
            if self.stage == "UP" and not self.reached_top:
                if self.too_high_flag:
                    self.status_text = "⚠️ Слишком высоко! Опустите до уровня плеч"
                    self.status_color = WARNING
                elif self.too_low_flag:
                    self.status_text = "⚠️ Поднимите выше — до уровня плеч"
                    self.status_color = WARNING
                elif self.arms_not_sideways:
                    self.status_text = "⚠️ Разведите руки строго в стороны"
                    self.status_color = WARNING
                elif self.elbows_bent:
                    self.status_text = "⚠️ Выпрямите локти полностью"
                    self.status_color = WARNING
        
        # Визуализация ошибок (красная рамка)
        self.error_flag = (self.too_high_flag or self.too_low_flag or 
                          self.arms_not_sideways or self.elbows_bent)
        return self.draw(frame)
    
    def generate_feedback(self):
        """Генерирует подсказку после успешного повтора"""
        if self.reps == 1:
            return "Отличное начало! Держите руки на уровне плеч"
        elif self.reps == 5:
            return "Хорошая серия! Следите за дыханием"
        elif self.reps == 10:
            return "Супер! Можете увеличить амплитуду"
        else:
            return "Продолжайте в том же темпе"


class SquatExercise(ExerciseBase):
    name = "🦵 ПРИСЕДАНИЯ"
    video_file = "squat.mp4"

    def reset(self):
        super().reset()
        self.stage = "UP"
        self.reached_depth = False
        self.status_text = "👉 Приседайте вниз"

    def process(self, frame, lm):
        if lm:
            hip, knee, ankle = landmark_xy(lm[24]), landmark_xy(lm[26]), landmark_xy(lm[28])
            k_angle = calculate_angle(hip, knee, ankle)
            if 70 <= k_angle <= 105:
                if not self.reached_depth:
                    self.reached_depth = True
                    self.error_flag = False
                    self.status_text = "✅ Глубина отличная! Поднимайтесь"
                    self.status_color = SUCCESS
                self.stage = "DOWN"
            if self.stage == "UP" and 110 < k_angle < 150 and not self.reached_depth:
                self.error_flag = True
                self.status_text = "⚠️ Приседайте глубже!"
                self.status_color = WARNING
            if k_angle > 165:
                if self.stage == "DOWN":
                    if self.reached_depth and not self.error_flag:
                        self.reps += 1
                        self.status_text = "🎉 Отлично! Повтор засчитан"
                        self.status_color = SUCCESS
                    else:
                        self.status_text = "❌ Приседайте глубже в следующий раз"
                        self.status_color = ERROR
                    self.stage = "UP"
                    self.reached_depth = False
        return self.draw(frame)



# ═══════════════════════════════════════════════════════════════
# 📊 ЛЕВАЯ ПАНЕЛЬ
# ═══════════════════════════════════════════════════════════════
class StatsPanel(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self.current_video = None
        self.setup_ui()

    def setup_ui(self):
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview, bg=PANEL)
        self.scrollable = tk.Frame(self.canvas, bg=BG)
        self.scrollable.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.stats = tk.Frame(self.scrollable, bg=BG)
        self.help = tk.Frame(self.scrollable, bg=BG)
        self.setup_stats_ui()
        self.setup_help_ui()
        self.bind_mousewheel()

    def bind_mousewheel(self):
        def scroll(e):
            self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        self.canvas.bind("<MouseWheel>", scroll)
        self.scrollable.bind("<MouseWheel>", scroll)

    def setup_stats_ui(self):
        tk.Label(self.stats, text="💪", font=get_font(42), bg=BG, fg=ACCENT).pack(pady=(20, 8))
        self.ex_name = tk.Label(self.stats, text="", font=get_font(16, "bold"),
                                fg=TEXT, bg=BG, wraplength=340, justify="center")
        self.ex_name.pack(pady=(0, 12))

        reps_f = tk.Frame(self.stats, bg=BG)
        reps_f.pack(pady=15)
        self.reps = tk.Label(reps_f, text="0", font=get_font(90, "bold"), fg=ACCENT, bg=BG)
        self.reps.pack()
        tk.Label(reps_f, text="ПОВТОРЕНИЙ", font=get_font(13), fg=TEXT_SECONDARY, bg=BG).pack()

        status_f = tk.Frame(self.stats, bg=CARD, relief="flat", bd=0)
        status_f.pack(pady=18, padx=22, fill="x")
        self.status_icon = tk.Label(status_f, text="⏺", font=get_font(16), bg=CARD, fg=TEXT_SECONDARY)
        self.status_icon.pack(side="left", padx=14, pady=10)
        self.status_text = tk.Label(status_f, text="", font=get_font(11),
                                    fg=TEXT_SECONDARY, bg=CARD, wraplength=240, justify="left")
        self.status_text.pack(side="left", padx=10, pady=10, fill="x", expand=True)

        self.video_f = tk.Frame(self.stats, bg=BG)
        self.video_f.pack(fill="x", pady=12)

    def setup_help_ui(self):
        """Режим выбора — инструкции с ВАШИМИ иконками"""
        tk.Label(self.help, text="КАК УПРАВЛЯТЬ", font=get_font(20, "bold"),
                 fg=ACCENT, bg=BG).pack(pady=(25, 18))

        # ─────────────────────────────────────
        # 🖼️ Блок с иконками жестов
        # ─────────────────────────────────────
        icons_frame = tk.Frame(self.help, bg=CARD, relief="flat", bd=0)
        icons_frame.pack(fill="x", pady=8, padx=18)

        tk.Label(icons_frame, text="✋ ЖЕСТЫ ДЛЯ УПРАВЛЕНИЯ", font=get_font(12, "bold"),
                 fg=TEXT, bg=CARD).pack(anchor="w", padx=14, pady=(12, 8))

        gestures_row = tk.Frame(icons_frame, bg=CARD)
        gestures_row.pack(fill="x", padx=14, pady=(0, 12))

        # ─── Левая рука ← ───
        left_col = tk.Frame(gestures_row, bg=CARD)
        left_col.pack(side="left", expand=True, padx=5)

        left_icon = load_icon("left.png", (70, 70))
        if left_icon:
            tk.Label(left_col, image=left_icon, bg=CARD).pack(pady=(0, 8))
            left_col.left_icon = left_icon  # 🔥 Сохраняем ссылку!
        else:
            tk.Label(left_col, text="🖐️", font=get_font(28), bg=CARD, fg=ACCENT).pack(pady=(0, 8))

        tk.Label(left_col, text="Левая рука вверх", font=get_font(9),
                 fg=TEXT_SECONDARY, bg=CARD, justify="center").pack()
        tk.Label(left_col, text="← Предыдущее", font=get_font(10, "bold"),
                 fg=WARNING, bg=CARD).pack(pady=(2, 0))

        # ─── Обе руки 🙌 ───
        up_col = tk.Frame(gestures_row, bg=CARD)
        up_col.pack(side="left", expand=True, padx=5)

        up_icon = load_icon("up.png", (70, 70))
        if up_icon:
            tk.Label(up_col, image=up_icon, bg=CARD).pack(pady=(0, 8))
            up_col.up_icon = up_icon
        else:
            tk.Label(up_col, text="🙌", font=get_font(28), bg=CARD, fg=SUCCESS).pack(pady=(0, 8))

        tk.Label(up_col, text="Обе руки вверх", font=get_font(9),
                 fg=TEXT_SECONDARY, bg=CARD, justify="center").pack()
        tk.Label(up_col, text="3 секунды = старт", font=get_font(10, "bold"),
                 fg=SUCCESS, bg=CARD).pack(pady=(2, 0))

        # ─── Правая рука → ───
        right_col = tk.Frame(gestures_row, bg=CARD)
        right_col.pack(side="left", expand=True, padx=5)

        right_icon = load_icon("right.png", (70, 70))
        if right_icon:
            tk.Label(right_col, image=right_icon, bg=CARD).pack(pady=(0, 8))
            right_col.right_icon = right_icon
        else:
            tk.Label(right_col, text="🖐️", font=get_font(28), bg=CARD, fg=ACCENT).pack(pady=(0, 8))

        tk.Label(right_col, text="Правая рука вверх", font=get_font(9),
                 fg=TEXT_SECONDARY, bg=CARD, justify="center").pack()
        tk.Label(right_col, text="Следующее →", font=get_font(10, "bold"),
                 fg=WARNING, bg=CARD).pack(pady=(2, 0))

        # ─────────────────────────────────────
        # 🚀 Блок начала тренировки
        # ─────────────────────────────────────
        start = tk.Frame(self.help, bg=CARD, relief="flat", bd=0)
        start.pack(fill="x", pady=8, padx=18)

        tk.Label(start, text="▶️ НАЧАТЬ ТРЕНИРОВКУ", font=get_font(12, "bold"),
                 fg=TEXT, bg=CARD).pack(anchor="w", padx=14, pady=(12, 8))

        hint = tk.Frame(start, bg=CARD)
        hint.pack(fill="x", padx=14, pady=(0, 12))

        start_icon = load_icon("up.png", (50, 50))
        if start_icon:
            tk.Label(hint, image=start_icon, bg=CARD).pack(pady=(0, 8))
            hint.start_icon = start_icon
        else:
            tk.Label(hint, text="🙌", font=get_font(28), bg=CARD, fg=SUCCESS).pack(pady=(0, 8))

        tk.Label(hint, text="Поднимите ОБЕ руки", font=get_font(11, "bold"),
                 fg=TEXT, bg=CARD, justify="center").pack()
        tk.Label(hint, text="и удерживайте 3 секунды", font=get_font(10),
                 fg=TEXT_SECONDARY, bg=CARD, justify="center").pack(pady=(2, 10))

        bar = tk.Frame(hint, bg="#3a3a3a", height=10)
        bar.pack(fill="x", padx=30)
        tk.Frame(bar, bg=SUCCESS, width=120).pack(side="left", fill="y")
        tk.Label(hint, text="3 сек", font=get_font(9), fg=MUTED, bg=CARD).pack(pady=(5, 0))

        # ─────────────────────────────────────
        # 💡 Советы
        # ─────────────────────────────────────
        tips = tk.Frame(self.help, bg=CARD, relief="flat", bd=0)
        tips.pack(fill="x", pady=8, padx=18)

        tk.Label(tips, text="💡 СОВЕТЫ", font=get_font(12, "bold"),
                 fg=ACCENT, bg=CARD).pack(anchor="w", padx=14, pady=(12, 8))

        for icon, txt in [
            ("👁️", "Встаньте так, чтобы камера видела вас целиком"),
            ("✋", "Поднимайте руки выше плеч — так лучше распознаётся"),
            ("🧍", "Держите спину прямо для точного счёта"),
            ("⏱️", "Не отпускайте руки раньше 3 секунд"),
            ("🖱️", "Можно кликать по кнопкам мышкой, если удобно")
        ]:
            f = tk.Frame(tips, bg=CARD)
            f.pack(fill="x", padx=14, pady=3)
            tk.Label(f, text=icon, font=get_font(15), bg=CARD, fg=ACCENT).pack(side="left", padx=(0, 10))
            tk.Label(f, text=txt, font=get_font(10), fg=TEXT_SECONDARY,
                     bg=CARD, wraplength=290, justify="left").pack(side="left")

        tk.Frame(self.help, bg=BG, height=20).pack()

    def update_video_hint(self, video_file):
        if self.current_video:
            self.current_video.stop()
            self.current_video.frame.destroy()
        if video_file and os.path.exists(video_file):
            self.current_video = VideoPlayer(self.video_f, video_file, max_w=360, max_h=220)
        else:
            tk.Label(self.video_f, text="🎬 Видео техники будет доступно",
                     font=get_font(10), fg=MUTED, bg=CARD, pady=15).pack(fill="x")

    def update_reps_size(self):
        n = len(str(self.app.current_exercise.reps))
        size = {1: 90, 2: 90, 3: 70, 4: 55}.get(n, 45)
        self.reps.config(font=get_font(size, "bold"))

    def show_workout(self):
        self.stats.pack(fill="both", expand=True)
        self.help.pack_forget()
        self.update_video_hint(getattr(self.app.current_exercise, 'video_file', None))
        self.update_stats()

    def show_help(self):
        if self.current_video:
            self.current_video.stop()
            self.current_video = None
        for w in self.video_f.winfo_children(): w.destroy()
        self.stats.pack_forget()
        self.help.pack(fill="both", expand=True)
        self.canvas.yview_moveto(0)

    def update_stats(self):
        ex = self.app.current_exercise
        self.ex_name.config(text=ex.name)
        self.reps.config(text=str(ex.reps))
        self.update_reps_size()
        self.status_text.config(text=ex.status_text, fg=ex.status_color)
        icons = {SUCCESS: "✅", ERROR: "❌", WARNING: "⚠️"}
        self.status_icon.config(text=icons.get(ex.status_color, "⏺"),
                                fg=ex.status_color if ex.status_color != TEXT_SECONDARY else MUTED)
        self.reps.config(fg=ACCENT)
        self.reps.after(150, lambda: self.reps.config(fg=ACCENT))


# ═══════════════════════════════════════════════════════════════
# 🔘 КНОПКИ УПРАЖНЕНИЙ (только текст)
# ═══════════════════════════════════════════════════════════════
class BigButton(tk.Frame):
    def __init__(self, parent, text, command=None):
        super().__init__(parent, bg=CARD, cursor="hand2")
        self.command = command
        tk.Label(self, text=text, font=get_font(13, "bold"),
                 fg=TEXT, bg=CARD).pack(pady=18, padx=20)
        for w in [self] + list(self.winfo_children()):
            w.bind("<Enter>", self.on_enter)
            w.bind("<Leave>", self.on_leave)
            w.bind("<Button-1>", self.on_click)

    def on_enter(self, e):
        self.config(bg=ACCENT_DARK)
        for w in self.winfo_children():
            w.config(bg=ACCENT_DARK, fg="#000000")

    def on_leave(self, e):
        self.config(bg=CARD)
        for w in self.winfo_children():
            w.config(bg=CARD, fg=TEXT)

    def on_click(self, e):
        if self.command: self.command()


# ═══════════════════════════════════════════════════════════════
# 🖥️ ГЛАВНОЕ ПРИЛОЖЕНИЕ
# ═══════════════════════════════════════════════════════════════
class WorkoutMenuApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SmartFit Mirror")
        self.root.configure(bg=BG)
        self.root.withdraw()

        self.fullscreen = True
        self.video_job = None
        self.current_photo = None

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Не удалось открыть камеру!")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.exercises = {
            "biceps": BicepsExercise(),
            "shoulders": ShouldersExercise(),
            "squat": SquatExercise(),
            
        }
        self.exercise_order = ["biceps", "shoulders", "squat"]
        self.current_key = "shoulders"
        self.current_exercise = self.exercises[self.current_key]

        self.selection_mode = True
        self.both_start = None
        self.last_switch = 0
        self.switch_delay = 0.5

        self.splash = SplashScreen(self.root)
        self.root.after(3100, self.setup_ui)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Escape>", self.handle_escape)
        self.root.bind("<F11>", lambda e: self.toggle_fullscreen())

    def setup_ui(self):
        self.root.deiconify()

        paned = tk.PanedWindow(self.root, bg=BG, sashwidth=4, sashrelief="flat")
        paned.pack(fill="both", expand=True, padx=12, pady=12)

        self.left = StatsPanel(paned, self)
        paned.add(self.left, width=440, minsize=380)

        right = tk.Frame(paned, bg=PANEL)
        paned.add(right, width=850)

        self.v_frame = tk.Frame(right, bg=BG)
        self.v_frame.pack(fill="both", expand=True, padx=12, pady=12)
        self.v_label = tk.Label(self.v_frame, bg=BG)
        self.v_label.pack(expand=True)

        bottom = tk.Frame(right, bg=PANEL, height=85)
        bottom.pack(fill="x", side="bottom")
        bottom.pack_propagate(False)

        btn_f = tk.Frame(bottom, bg=PANEL)
        btn_f.pack(expand=True)

        # 🔤 Кнопки только с текстом (без эмодзи)
        ex_cfg = {
            "biceps": "БИЦЕПС",
            "shoulders": "ПЛЕЧИ",
            "squat": "ПРИСЕД",
            "row": "ТЯГА",
        }

        self.buttons = {}
        for key, name in ex_cfg.items():
            btn = BigButton(btn_f, name, command=lambda k=key: self.select_exercise(k))
            btn.pack(side="left", padx=12, pady=10)
            self.buttons[key] = btn

        self.mode_f = tk.Frame(bottom, bg=ACCENT if not self.selection_mode else CARD, relief="flat", bd=0)
        self.mode_f.pack(side="left", padx=18, pady=10)
        self.mode_lbl = tk.Label(self.mode_f,
                                 text="ТРЕНИРОВКА" if not self.selection_mode else "ВЫБОР",
                                 font=get_font(10, "bold"),
                                 fg='#000000' if not self.selection_mode else TEXT_SECONDARY,
                                 bg=ACCENT if not self.selection_mode else CARD,
                                 padx=14, pady=5)
        self.mode_lbl.pack()

        hint_f = tk.Frame(right, bg=PANEL, height=28)
        hint_f.pack(fill="x", side="bottom")
        hint_f.pack_propagate(False)
        self.hint_lbl = tk.Label(hint_f, text="", font=get_font(9), bg=PANEL)
        self.hint_lbl.pack(expand=True)
        self.update_hint()

        self.update_buttons()
        if self.selection_mode:
            self.left.show_help()
        else:
            self.left.show_workout()
            self.left.update_stats()

        self.root.after(15, self.update_video)

    def update_hint(self):
        txt = "🖐️ Жесты: ←/→ смена  |  🙌 3 сек = старт" if self.selection_mode else \
            "🙌 Обе руки 3 сек = смена упражнения  |  ✨ Только скелет"
        self.hint_lbl.config(text=txt, fg=ACCENT if self.selection_mode else WARNING)

    def update_buttons(self):
        for key, btn in self.buttons.items():
            if key == self.current_key:
                btn.config(bg=ACCENT_DARK)
                for w in btn.winfo_children():
                    w.config(bg=ACCENT_DARK, fg="#ffffff")
            else:
                btn.config(bg=CARD)
                for w in btn.winfo_children():
                    w.config(bg=CARD, fg=TEXT)

    def update_info(self):
        if self.selection_mode:
            self.left.show_help()
            self.mode_f.config(bg=CARD)
            self.mode_lbl.config(text="ВЫБОР", fg=TEXT_SECONDARY, bg=CARD)
        else:
            self.left.show_workout()
            self.left.update_stats()
            self.mode_f.config(bg=ACCENT)
            self.mode_lbl.config(text="ТРЕНИРОВКА", fg='#000000', bg=ACCENT)
        self.update_hint()

    def select_exercise(self, key):
        if key == self.current_key: return
        self.current_key = key
        self.current_exercise = self.exercises[key]
        self.current_exercise.reset()
        self.update_buttons()
        self.update_info()

    def detect_gesture(self, lm):
        if not lm: return None
        lw, rw = lm[15], lm[16]
        ls, rs = lm[11], lm[12]
        l_up = lw.y < ls.y - 0.12
        r_up = rw.y < rs.y - 0.12
        if l_up and r_up: return "both"
        if r_up: return "right"
        if l_up: return "left"
        return None

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def handle_escape(self, e=None):
        if self.fullscreen:
            self.fullscreen = False
            self.root.attributes("-fullscreen", False)
        else:
            self.on_close()

    def resize_frame(self, frame, max_w, max_h):
        h, w = frame.shape[:2]
        if w == 0 or h == 0: return frame
        s = min(max_w / w, max_h / h)
        return cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)

    def update_video(self):
        if not self.cap or not self.cap.isOpened():
            self.root.after(100, self.update_video)
            return

        ok, frame = self.cap.read()
        if not ok:
            self.root.after(100, self.update_video)
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        lm = None
        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            gesture = self.detect_gesture(lm)
            now = time.time()

            if gesture == "both":
                if self.both_start is None:
                    self.both_start = now
                elif now - self.both_start >= 3:
                    self.selection_mode = not self.selection_mode
                    if not self.selection_mode:
                        self.current_exercise.reset()
                    self.update_info()
                    self.both_start = None
            else:
                self.both_start = None

            if self.selection_mode:
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                frame = draw_skeleton_on_frame(frame, lm, self.mp_pose.POSE_CONNECTIONS)
                if now - self.last_switch > self.switch_delay:
                    if gesture == "right":
                        idx = self.exercise_order.index(self.current_key)
                        self.select_exercise(self.exercise_order[(idx + 1) % len(self.exercise_order)])
                        self.last_switch = now
                    elif gesture == "left":
                        idx = self.exercise_order.index(self.current_key)
                        self.select_exercise(self.exercise_order[(idx - 1) % len(self.exercise_order)])
                        self.last_switch = now
                processed = frame
            else:
                h, w = frame.shape[:2]
                black = draw_white_skeleton_on_black((h, w), lm, self.mp_pose.POSE_CONNECTIONS)
                processed = self.current_exercise.process(black, lm)
                self.left.update_stats()

            if gesture == "both" and self.both_start:
                elapsed = now - self.both_start
                if elapsed < 3:
                    processed = draw_progress_bar(processed, elapsed)
        else:
            processed = frame if self.selection_mode else np.zeros((480, 640, 3), dtype=np.uint8)

        dw, dh = self.v_frame.winfo_width(), self.v_frame.winfo_height()
        if dw > 10 and dh > 10:
            processed = self.resize_frame(processed, dw - 24, dh - 24)

        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        self.current_photo = ImageTk.PhotoImage(image=Image.fromarray(processed))
        self.v_label.config(image=self.current_photo)
        self.video_job = self.root.after(15, self.update_video)

    def on_close(self):
        if self.video_job: self.root.after_cancel(self.video_job)
        if self.cap: self.cap.release()
        try:
            self.pose.close()
        except:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app = WorkoutMenuApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()