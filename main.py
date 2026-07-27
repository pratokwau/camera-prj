"""
Finger AR Overlay — привязка картинки к кончикам пальцев через камеру.

Стек: OpenCV (камера) + MediaPipe Tasks Vision (HandLandmarker) + warpPerspective (AR-оверлей).

Как работает:
    1. Камера захватывает кадр.
    2. MediaPipe HandLandmarker находит руку и 21 точку.
    3. Кончики 4 пальцев (thumb/index/ring/pinky → точки 4/8/16/20)
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
from dataclasses import dataclass
from typing import Optional

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"


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
    fingertip_ids: tuple = (4, 8, 16, 20)  # thumb, index, ring, pinky
    images_dir: str = "images"
    screenshot_dir: str = "screenshots"
    model_path: str = "hand_landmarker.task"
    model_url: str = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    min_corner_dist: int = 20  # мин расстояние между углами


# ──────────────────────────────────────────────────────────────────────────────
# Загрузчик картинок
# ──────────────────────────────────────────────────────────────────────────────

class ImageLoader:
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
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
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
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._download_model_if_needed(cfg.model_path, cfg.model_url)

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

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
            print(f"[HandDetector] Ошибка: {e}")
            raise

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, Optional[list]]:
        h, w = frame.shape[:2]
        fingertips = None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            fingertips = []
            for idx in self.cfg.fingertip_ids:
                lm = landmarks[idx]
                px, py = int(lm.x * w), int(lm.y * h)
                fingertips.append((px, py))

            self._draw_skeleton(frame, landmarks, w, h)
            for (px, py) in fingertips:
                cv2.circle(frame, (px, py), 10, (0, 255, 255), -1)
                cv2.circle(frame, (px, py), 12, (0, 0, 255), 2)

        return frame, fingertips

    def _draw_skeleton(self, frame, landmarks, w, h):
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for (i, j) in self.HAND_CONNECTIONS:
            cv2.line(frame, points[i], points[j], (0, 255, 0), 2)
        for pt in points:
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)

    def close(self):
        if hasattr(self, 'landmarker'):
            self.landmarker.close()


# ──────────────────────────────────────────────────────────────────────────────
# AR-оверлей
# ──────────────────────────────────────────────────────────────────────────────

class AROverlay:
    @staticmethod
    def _sort_corners(pts: list) -> list:
        arr = np.array(pts, dtype=np.float32)
        s = arr.sum(axis=1)
        d = np.diff(arr, axis=1).flatten()
        tl = arr[np.argmin(s)]
        br = arr[np.argmax(s)]
        tr = arr[np.argmin(d)]
        bl = arr[np.argmax(d)]
        return [tl.tolist(), tr.tolist(), br.tolist(), bl.tolist()]

    @staticmethod
    def _enforce_min_distance(pts: list, min_dist: float) -> list:
        result = [list(p) for p in pts]
        for i in range(4):
            for j in range(i + 1, 4):
                dx = result[j][0] - result[i][0]
                dy = result[j][1] - result[i][1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < min_dist:
                    if dist < 1:
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    shift = (min_dist - dist) / 2
                    nx, ny = dx / dist, dy / dist
                    result[i][0] -= nx * shift
                    result[i][1] -= ny * shift
                    result[j][0] += nx * shift
                    result[j][1] += ny * shift
        return result

    @staticmethod
    def overlay(frame: np.ndarray, image: np.ndarray, pts_dst: list) -> np.ndarray:
        h_img, w_img = image.shape[:2]
        if image.shape[2] == 3:
            alpha_ch = np.full((h_img, w_img, 1), 255, dtype=np.uint8)
            image = np.concatenate([image, alpha_ch], axis=2)

        pts_src = np.float32([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]])
        pts_dst = AROverlay._sort_corners(pts_dst)
        pts_dst_np = np.float32(pts_dst)

        M = cv2.getPerspectiveTransform(pts_src, pts_dst_np)
        h_frame, w_frame = frame.shape[:2]
        warped_rgb = cv2.warpPerspective(image[:, :, :3], M, (w_frame, h_frame))
        warped_alpha = cv2.warpPerspective(image[:, :, 3], M, (w_frame, h_frame))

        alpha = warped_alpha.astype(np.float32) / 255.0
        alpha = np.stack([alpha] * 3, axis=-1)

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
        self.SMOOTH_FACTOR: float = 0.3
        self.hand_visible: bool = False
        self.hand_lost_time: float = 0
        self.HAND_LOST_TIMEOUT: float = 1.0

    def _init_camera(self) -> bool:
        self.cap = cv2.VideoCapture(self.cfg.camera_index)
        if not self.cap.isOpened():
            print("[Ошибка] Не удалось открыть камеру.")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera_height)
        return True

    def _draw_hud(self, frame: np.ndarray) -> None:
        overlay_img = frame.copy()
        cv2.rectangle(overlay_img, (10, 10), (420, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay_img, 0.55, frame, 0.45, 0, frame)

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

    def _smooth_corners(self, new_corners: list) -> list:
        if self.smoothed_fingertips is None:
            self.smoothed_fingertips = [list(p) for p in new_corners]
        else:
            for i in range(4):
                s = self.smoothed_fingertips[i]
                n = new_corners[i]
                s[0] = s[0] * (1 - self.SMOOTH_FACTOR) + n[0] * self.SMOOTH_FACTOR
                s[1] = s[1] * (1 - self.SMOOTH_FACTOR) + n[1] * self.SMOOTH_FACTOR
        return [(int(p[0]), int(p[1])) for p in self.smoothed_fingertips]

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

                frame = cv2.flip(frame, 1)

                frame, fingertips = self.detector.process(frame)

                if fingertips:
                    self.hand_visible = True
                    self.hand_lost_time = 0
                    corners = self.overlay._enforce_min_distance(
                        fingertips, self.cfg.min_corner_dist
                    )
                    self.last_fingertips = self._smooth_corners(corners)
                else:
                    self.hand_visible = False
                    self.hand_lost_time += 1 / 30
                    if self.hand_lost_time > self.HAND_LOST_TIMEOUT:
                        self.smoothed_fingertips = None

                image = self.loader.current()
                if image is not None and self.last_fingertips is not None:
                    if self.hand_lost_time <= self.HAND_LOST_TIMEOUT:
                        frame = self.overlay.overlay(frame, image, self.last_fingertips)

                self._draw_hud(frame)
                cv2.imshow("Finger AR Overlay", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord(' '):
                    self.loader.next()
                elif key == ord('s'):
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


if __name__ == "__main__":
    app = FingerARApp()
    app.run()
