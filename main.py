"""
Finger AR Overlay — привязка картинки к кончикам пальцев через камеру.

Стек: OpenCV (камера) + MediaPipe Tasks Vision (HandLandmarker) + warpPerspective (AR-оверлей).

Как работает:
    1. Камера захватывает кадр.
    2. MediaPipe HandLandmarker находит руку и 21 точку (landmark).
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
import urllib.request
import logging
from dataclasses import dataclass
from typing import Optional

# Подавляем libpng warning
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
cv2.setLogLevel(0)
logging.getLogger().setLevel(logging.ERROR)


# ──────────────────────────────────────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    max_num_hands: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # 4 точки: большой палец(4), указательный(8), безымянный(16), мизинец(20)
    # thumb даёт широкий охват — картинка не переворачивается
    fingertip_ids: tuple = (4, 8, 16, 20)
    images_dir: str = "images"
    screenshot_dir: str = "screenshots"
    model_path: str = "hand_landmarker.task"
    model_url: str = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


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
        # Конвертируем в BGR или BGRA (тихий импорт без libpng warning)
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
# Детектор руки (MediaPipe Tasks Vision API)
# ──────────────────────────────────────────────────────────────────────────────

class HandDetector:
    """Обёртка над MediaPipe HandLandmarker (новый API)."""

    # Соединения для рисования скелета (пары индексов)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # большой
        (0, 5), (5, 6), (6, 7), (7, 8),      # указательный
        (0, 9), (9, 10), (10, 11), (11, 12), # средний
        (0, 13), (13, 14), (14, 15), (15, 16), # безымянный
        (0, 17), (17, 18), (18, 19), (19, 20), # мизинец
        (5, 9), (9, 13), (13, 17),           # ладонь
    ]

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._download_model_if_needed(cfg.model_path, cfg.model_url)

        # Правильные импорты для MediaPipe 0.10.x
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode  # НЕ VisionRunningMode!

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=cfg.model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=cfg.max_num_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    @staticmethod
    def _download_model_if_needed(path: str, url: str) -> None:
        if os.path.exists(path):
            return
        print(f"[HandDetector] Скачиваю модель {path} ...")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"[HandDetector] Модель скачана: {path}")
        except Exception as e:
            print(f"[HandDetector] Ошибка скачивания: {e}")
            print(f"Скачай вручную: {url}")
            print(f"Или установи старую версию: pip install mediapipe==0.10.9")
            raise

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, Optional[list]]:
        """
        Возвращает (аннотированный кадр, список кончиков пальцев в пикселях).
        Список = [(x0,y0), (x1,y1), (x2,y2), (x3,y3)] или None если рука не найдена.
        """
        h, w = frame.shape[:2]
        fingertips = None

        # Конвертируем BGR → RGB для MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect(mp_image)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]  # первая рука
            fingertips = []
            for idx in self.cfg.fingertip_ids:
                lm = landmarks[idx]
                px, py = int(lm.x * w), int(lm.y * h)
                fingertips.append((px, py))

            # Рисуем скелет
            self._draw_skeleton(frame, landmarks, w, h)
            # Подсвечиваем кончики
            for (px, py) in fingertips:
                cv2.circle(frame, (px, py), 10, (0, 255, 255), -1)
                cv2.circle(frame, (px, py), 12, (0, 0, 255), 2)

        return frame, fingertips

    def _draw_skeleton(self, frame: np.ndarray, landmarks, w: int, h: int) -> None:
        """Рисует соединения скелета вручную."""
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for (i, j) in self.HAND_CONNECTIONS:
            cv2.line(frame, points[i], points[j], (0, 255, 0), 2)
        # Суставы
        for pt in points:
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)

    def close(self) -> None:
        if hasattr(self, 'landmarker'):
            self.landmarker.close()


# ──────────────────────────────────────────────────────────────────────────────
# AR-оверлей: warpPerspective
# ──────────────────────────────────────────────────────────────────────────────

class AROverlay:
    """Накладывает картинку на кадр по 4 точкам через перспективное преобразование."""

    @staticmethod
    def _sort_corners(pts: list) -> list:
        """
        Сортирует 4 точки в порядке: top-left, top-right, bottom-right, bottom-left.
        Без этого картинка может переворачиваться/поворачиваться.
        """
        arr = np.array(pts, dtype=np.float32)
        # top-left = min(x+y), bottom-right = max(x+y)
        s = arr.sum(axis=1)
        d = np.diff(arr, axis=1).flatten()  # y - x
        tl = arr[np.argmin(s)]
        br = arr[np.argmax(s)]
        tr = arr[np.argmin(d)]
        bl = arr[np.argmax(d)]
        return [tl.tolist(), tr.tolist(), br.tolist(), bl.tolist()]

    @staticmethod
    def overlay(frame: np.ndarray, image: np.ndarray, pts_dst: list) -> np.ndarray:
        """
        frame  — кадр камеры (H, W, 3)
        image  — картинка (h, w, 3|4)
        pts_dst — 4 точки назначения [(x,y), ...] в пикселях кадра
        """
        h_img, w_img = image.shape[:2]

        # Если картинка без альфа-канала — добавляем (непрозрачный)
        if image.shape[2] == 3:
            alpha_channel = np.full((h_img, w_img, 1), 255, dtype=np.uint8)
            image = np.concatenate([image, alpha_channel], axis=2)

        # Исходные 4 угла картинки
        pts_src = np.float32([
            [0, 0],
            [w_img, 0],
            [w_img, h_img],
            [0, h_img],
        ])
        # Сортируем точки чтобы картинка не переворачивалась
        pts_dst = AROverlay._sort_corners(pts_dst)
        pts_dst_np = np.float32(pts_dst)

        # Матрица перспективного преобразования
        M = cv2.getPerspectiveTransform(pts_src, pts_dst_np)

        # Варпим картинку и альфа-маску отдельно
        h_frame, w_frame = frame.shape[:2]
        warped_rgb = cv2.warpPerspective(image[:, :, :3], M, (w_frame, h_frame))
        warped_alpha = cv2.warpPerspective(image[:, :, 3], M, (w_frame, h_frame))

        # Альфа-маска: 0..1
        alpha = warped_alpha.astype(np.float32) / 255.0
        alpha = np.stack([alpha] * 3, axis=-1)  # (H, W, 3)

        # Чистое альфа-наложение
        frame_f = frame.astype(np.float32)
        warped_f = warped_rgb.astype(np.float32)
        blended = (warped_f * alpha + frame_f * (1.0 - alpha)).astype(np.uint8)

        return blended


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
        self.smoothed_fingertips: Optional[list] = None
        self.SMOOTH_FACTOR: float = 0.35  # 0=макс плавность, 1=без сглаживания
        self.hand_visible: bool = False  # рука видна прямо сейчас
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
        status = "Рука найдена" if self.hand_visible else "Покажи руку!"
        color = (0, 255, 0) if self.hand_visible else (0, 0, 255)
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
                    self.hand_visible = True
                    # Экспоненциальное сглаживание
                    if self.smoothed_fingertips is None:
                        self.smoothed_fingertips = [list(p) for p in fingertips]
                    else:
                        for i in range(4):
                            old = self.smoothed_fingertips[i]
                            new = fingertips[i]
                            old[0] = old[0] * (1 - self.SMOOTH_FACTOR) + new[0] * self.SMOOTH_FACTOR
                            old[1] = old[1] * (1 - self.SMOOTH_FACTOR) + new[1] * self.SMOOTH_FACTOR
                    self.last_fingertips = [(int(p[0]), int(p[1])) for p in self.smoothed_fingertips]
                    self.hand_lost_time = 0
                else:
                    self.hand_visible = False
                    self.hand_lost_time += 1 / 30
                    if self.hand_lost_time > self.HAND_LOST_TIMEOUT:
                        self.smoothed_fingertips = None

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
