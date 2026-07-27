"""
Finger AR Overlay — привязка картинки к кончикам пальцев через камеру.

Стек: OpenCV (камера) + MediaPipe Hands (21 landmark) + warpPerspective (AR-оверлей).

Как работает:
    1. Камера захватывает кадр.
    2. MediaPipe находит руку и 21 точку (landmark).
    3. Кончики 4 пальцев (index/middle/ring/pinky → точки 8/12/16/20)
       задают 4 угла перспективного преобразования.
    4. Картинка из images/ накладывается на кадр через warpPerspective.
    5. Двигаешь пальцами → картинка тянется, сжимается, поворачивается.

Управление:
    1-4 — выбор картинки из списка
    SPACE — следующая картинка
    S — сохранить скриншот
    Q / ESC — выход
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import glob
import time
from dataclasses import dataclass
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    max_num_hands: int = 1
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    # Индексы кончиков пальцев в MediaPipe (8=index, 12=middle, 16=ring, 20=pinky)
    fingertip_ids: tuple = (8, 12, 16, 20)
    images_dir: str = "images"
    screenshot_dir: str = "screenshots"


# ──────────────────────────────────────────────────────────────────────────────
# Загрузчик картинок
# ──────────────────────────────────────────────────────────────────────────────

class ImageLoader:
    """Сканирует images/ и держит список доступных картинок."""

    SUPPORTED_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

    def __init__(self, images_dir: str):
        self.images_dir = images_dir
        self.paths: list[str] = []
        self.current_idx: int = 0
        self._scan()

    def _scan(self) -> None:
        if not os.path.isdir(self.images_dir):
            os.makedirs(self.images_dir, exist_ok=True)
            print(f"[ImageLoader] Создана папка {self.images_dir}/ — кидай туда картинки.")
            return
        for ext in self.SUPPORTED_EXT:
            self.paths.extend(glob.glob(os.path.join(self.images_dir, f"*{ext}")))
        self.paths.sort()
        if self.paths:
            print(f"[ImageLoader] Найдено картинок: {len(self.paths)}")
        else:
            print(f"[ImageLoader] Папка {self.images_dir}/ пуста — кинь туда картинки.")

    @property
    def count(self) -> int:
        return len(self.paths)

    def current(self) -> Optional[np.ndarray]:
        if not self.paths:
            return None
        img = cv2.imread(self.paths[self.current_idx], cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        # Конвертируем в BGR или BGRA
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 3:
            pass  # уже BGR
        elif img.shape[2] == 4:
            pass  # BGRA — альфа-канал используем для маски
        return img

    def next(self) -> None:
        if self.paths:
            self.current_idx = (self.current_idx + 1) % len(self.paths)

    def select(self, idx: int) -> None:
        if self.paths and 0 <= idx < len(self.paths):
            self.current_idx = idx

    def current_name(self) -> str:
        if not self.paths:
            return "<нет картинок>"
        return os.path.basename(self.paths[self.current_idx])


# ──────────────────────────────────────────────────────────────────────────────
# Детектор руки
# ──────────────────────────────────────────────────────────────────────────────

class HandDetector:
    """Обёртка над MediaPipe Hands."""

    def __init__(self, cfg: Config):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=cfg.max_num_hands,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, Optional[list]]:
        """
        Возвращает (аннотированный кадр, список кончиков пальцев в пикселях).
        Список = [(x0,y0), (x1,y1), (x2,y2), (x3,y3)] или None если рука не найдена.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        fingertips = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            h, w = frame.shape[:2]
            fingertips = []
            for idx in Config().fingertip_ids:
                lm = hand_landmarks.landmark[idx]
                px, py = int(lm.x * w), int(lm.y * h)
                fingertips.append((px, py))
            # Рисуем скелет
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style(),
            )
            # Подсвечиваем кончики
            for (px, py) in fingertips:
                cv2.circle(frame, (px, py), 10, (0, 255, 255), -1)
                cv2.circle(frame, (px, py), 12, (0, 0, 255), 2)

        return frame, fingertips

    def close(self) -> None:
        self.hands.close()


# ──────────────────────────────────────────────────────────────────────────────
# AR-оверлей: warpPerspective
# ──────────────────────────────────────────────────────────────────────────────

class AROverlay:
    """Накладывает картинку на кадр по 4 точкам через перспективное преобразование."""

    @staticmethod
    def overlay(frame: np.ndarray, image: np.ndarray, pts_dst: list) -> np.ndarray:
        """
        frame  — кадр камеры (H, W, 3)
        image  — картинка (h, w, 3|4)
        pts_dst — 4 точки назначения [(x,y), ...] в пикселях кадра
        """
        h_img, w_img = image.shape[:2]

        # Исходные 4 угла картинки
        pts_src = np.float32([
            [0, 0],
            [w_img, 0],
            [w_img, h_img],
            [0, h_img],
        ])
        pts_dst_np = np.float32(pts_dst)

        # Матрица перспективного преобразования
        M = cv2.getPerspectiveTransform(pts_src, pts_dst_np)

        # Размер выходного холста = размер кадра
        h_frame, w_frame = frame.shape[:2]
        warped = cv2.warpPerspective(
            image, M, (w_frame, h_frame),
            borderMode=cv2.BORDER_TRANSPARENT,
        )

        # Наложение с учётом альфа-канала (если есть)
        if warped.shape[2] == 4:
            # Разделяем цвет и маску
            alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
            rgb = warped[:, :, :3].astype(np.float32)
            frame_float = frame.astype(np.float32)
            blended = (rgb * alpha + frame_float * (1 - alpha)).astype(np.uint8)
            return blended
        else:
            # Без альфа: маска по нечёрным пикселям
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            fg = cv2.bitwise_and(warped, warped, mask=mask)
            return cv2.add(bg, fg)


# ──────────────────────────────────────────────────────────────────────────────
# Главный цикл
# ──────────────────────────────────────────────────────────────────────────────

class FingerARApp:
    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        self.loader = ImageLoader(cfg.images_dir)
        self.detector = HandDetector(cfg)
        self.overlay = AROverlay()
        self.cap: Optional[cv2.VideoCapture] = None
        self.last_fingertips: Optional[list] = None
        self.hand_lost_time: float = 0
        self.HAND_LOST_TIMEOUT: float = 1.0  # секунд держим последнюю позу

    def _init_camera(self) -> bool:
        self.cap = cv2.VideoCapture(self.cfg.camera_index)
        if not self.cap.isOpened():
            print("[Ошибка] Не удалось открыть камеру.")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera_height)
        return True

    def _draw_hud(self, frame: np.ndarray) -> None:
        """Панель с информацией в углу."""
        h, w = frame.shape[:2]
        # Полупрозрачный фон
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (420, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, f"Картинка: {self.loader.current_name()}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Картинок загружено: {self.loader.count}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "1-4:выбор SPACE:след S:скриншот Q:выход", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        status = "Рука найдена" if self.last_fingertips else "Покажи руку!"
        color = (0, 255, 0) if self.last_fingertips else (0, 0, 255)
        cv2.putText(frame, status, (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def _save_screenshot(self, frame: np.ndarray) -> None:
        os.makedirs(self.cfg.screenshot_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.cfg.screenshot_dir, f"screenshot_{ts}.png")
        cv2.imwrite(path, frame)
        print(f"[Скриншот] {path}")

    def run(self) -> None:
        if not self._init_camera():
            return

        print("\n" + "=" * 50)
        print("  Finger AR Overlay — запущен!")
        print("  Покажи руку, разведи пальцы — картинка привяжется")
        print("=" * 50 + "\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Зеркалим для естественности
                frame = cv2.flip(frame, 1)

                # Детекция руки
                frame, fingertips = self.detector.process(frame)

                if fingertips:
                    self.last_fingertips = fingertips
                    self.hand_lost_time = 0
                else:
                    self.hand_lost_time += 1 / 30  # ~30 fps

                # Накладываем картинку
                image = self.loader.current()
                if image is not None and self.last_fingertips is not None:
                    # Если рука потеряна недавно — держим последнюю позу
                    if self.hand_lost_time <= self.HAND_LOST_TIMEOUT:
                        frame = self.overlay.overlay(frame, image, self.last_fingertips)

                self._draw_hud(frame)
                cv2.imshow("Finger AR Overlay", frame)

                # Обработка клавиш
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q или ESC
                    break
                elif key == ord(' '):  # SPACE — следующая картинка
                    self.loader.next()
                elif key == ord('s'):  # S — скриншот
                    self._save_screenshot(frame)
                elif ord('1') <= key <= ord('4'):
                    self.loader.select(key - ord('1'))

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        if self.cap:
            self.cap.release()
        self.detector.close()
        cv2.destroyAllWindows()
        print("[Finger AR] Завершено.")


# ──────────────────────────────────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = FingerARApp()
    app.run()
