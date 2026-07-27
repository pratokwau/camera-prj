"""
Finger AR Overlay — привязка картинки к кончикам пальцев через камеру.

Стек: OpenCV (камера) + MediaPipe Tasks Vision (HandLandmarker) + warpPerspective (AR-оверлей).

Режимы:
    1 рука  — 4 кончика пальцев = 4 угла картинки (перспектива).
    2 руки  — левая = якорь (угол), правая = растяжение (противоположный угол).
              Картинка растягивается между двумя ладонями, пропорции сохраняются.

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

# Подавляем libpng warning
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"


# ──────────────────────────────────────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    max_num_hands: int = 2  # поддержка двух рук
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # 4 точки: большой палец(4), указательный(8), безымянный(16), мизинец(20)
    fingertip_ids: tuple = (4, 8, 16, 20)
    # Ладонь = среднее между wrist(0) и middle_finger_mcp(9)
    palm_point_ids: tuple = (0, 9)
    images_dir: str = "images"
    screenshot_dir: str = "screenshots"
    model_path: str = "hand_landmarker.task"
    model_url: str = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    # Минимальное расстояние между соседними углами (px) — защита от схлопывания
    min_corner_dist: int = 20
    # Паддинг вокруг двух ладоней (px)
    two_hand_padding: int = 60


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
# Детектор руки (MediaPipe Tasks Vision API)
# ──────────────────────────────────────────────────────────────────────────────

class HandDetector:
    """Обёртка над MediaPipe HandLandmarker. Поддержка 1 и 2 рук."""

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
            print(f"[HandDetector] Ошибка скачивания: {e}")
            print(f"Скачай вручную: {url}")
            raise

    def process(self, frame: np.ndarray) -> dict:
        """
        Возвращает dict:
            {
                'frame': annotated_frame,
                'mode': 'none' | 'single' | 'dual',
                'fingertips': [(x,y)]*4  (single mode),
                'palms': [(x,y), (x,y)]  (dual mode),
                'hands_data': list of raw landmarks (for drawing),
            }
        """
        h, w = frame.shape[:2]
        result_data = {
            'frame': frame,
            'mode': 'none',
            'fingertips': None,
            'palms': None,
            'hands_data': [],
        }

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return result_data

        hands_count = len(result.hand_landmarks)
        result_data['hands_data'] = result.hand_landmarks

        # Рисуем все руки
        for hand_lm in result.hand_landmarks:
            self._draw_skeleton(frame, hand_lm, w, h)

        if hands_count >= 2:
            # === DUAL MODE: 2 руки → пальмы = якорь и растяжение ===
            palm0 = self._get_palm_center(result.hand_landmarks[0], w, h)
            palm1 = self._get_palm_center(result.hand_landmarks[1], w, h)
            result_data['mode'] = 'dual'
            result_data['palms'] = [palm0, palm1]
            # Подсвечиваем центры ладоней
            cv2.circle(frame, palm0, 15, (255, 0, 255), -1)
            cv2.circle(frame, palm1, 15, (255, 0, 255), -1)
            cv2.line(frame, palm0, palm1, (255, 255, 0), 2)
        else:
            # === SINGLE MODE: 1 рука → 4 кончика пальцев ===
            landmarks = result.hand_landmarks[0]
            fingertips = []
            for idx in self.cfg.fingertip_ids:
                lm = landmarks[idx]
                px, py = int(lm.x * w), int(lm.y * h)
                fingertips.append((px, py))
            result_data['mode'] = 'single'
            result_data['fingertips'] = fingertips
            # Подсвечиваем кончики
            for (px, py) in fingertips:
                cv2.circle(frame, (px, py), 10, (0, 255, 255), -1)
                cv2.circle(frame, (px, py), 12, (0, 0, 255), 2)

        return result_data

    def _get_palm_center(self, landmarks, w: int, h: int) -> tuple:
        """Центр ладони = среднее между wrist(0) и middle_mcp(9)."""
        p0 = landmarks[self.cfg.palm_point_ids[0]]
        p1 = landmarks[self.cfg.palm_point_ids[1]]
        cx = int((p0.x + p1.x) / 2 * w)
        cy = int((p0.y + p1.y) / 2 * h)
        return (cx, cy)

    def _draw_skeleton(self, frame: np.ndarray, landmarks, w: int, h: int) -> None:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for (i, j) in self.HAND_CONNECTIONS:
            cv2.line(frame, points[i], points[j], (0, 255, 0), 2)
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
        """Сортирует 4 точки: top-left, top-right, bottom-right, bottom-left."""
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
        """Запрещает углам схлопываться ближе min_dist px."""
        result = [list(p) for p in pts]
        for i in range(4):
            for j in range(i + 1, 4):
                dx = result[j][0] - result[i][0]
                dy = result[j][1] - result[i][1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < min_dist:
                    # Раздвигаем на min_dist
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
    def palms_to_corners(palm1: tuple, palm2: tuple, padding: int,
                         aspect_ratio: float, frame_w: int, frame_h: int) -> list:
        """
        Из двух центров ладоней строит 4 угла прямоугольника
        с сохранением aspect_ratio картинки.
        """
        cx = (palm1[0] + palm2[0]) / 2
        cy = (palm1[1] + palm2[1]) / 2

        # Расстояние между ладонями
        dx = palm2[0] - palm1[0]
        dy = palm2[1] - palm1[1]
        dist = (dx * dx + dy * dy) ** 0.5

        if dist < 10:
            dist = 10

        # Угол поворота между ладонями
        angle = np.arctan2(dy, dx)

        # Ширина = расстояние + паддинг
        img_w = dist + padding * 2
        # Высота = по соотношению сторон картинки
        img_h = img_w / aspect_ratio

        # 4 угла прямоугольника (до поворота)
        half_w = img_w / 2
        half_h = img_h / 2
        corners = np.array([
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h],
        ], dtype=np.float32)

        # Поворот
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        corners = corners @ rot.T

        # Сдвиг к центру
        corners[:, 0] += cx
        corners[:, 1] += cy

        # Ограничиваем кадром
        corners[:, 0] = np.clip(corners[:, 0], 0, frame_w - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, frame_h - 1)

        return corners.tolist()

    @staticmethod
    def overlay(frame: np.ndarray, image: np.ndarray, pts_dst: list) -> np.ndarray:
        """
        frame  — кадр камеры (H, W, 3)
        image  — картинка (h, w, 3|4)
        pts_dst — 4 точки назначения [(x,y), ...]
        """
        h_img, w_img = image.shape[:2]

        # Добавляем альфа-канал если нет
        if image.shape[2] == 3:
            alpha_channel = np.full((h_img, w_img, 1), 255, dtype=np.uint8)
            image = np.concatenate([image, alpha_channel], axis=2)

        pts_src = np.float32([
            [0, 0],
            [w_img, 0],
            [w_img, h_img],
            [0, h_img],
        ])
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
        self.last_corners: Optional[list] = None
        self.smoothed_corners: Optional[list] = None
        self.SMOOTH_FACTOR: float = 0.3
        self.hand_mode: str = 'none'  # 'none' | 'single' | 'dual'
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
        h, w = frame.shape[:2]
        overlay_img = frame.copy()
        cv2.rectangle(overlay_img, (10, 10), (460, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay_img, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, f"Картинка: {self.loader.current_name()}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Картинок загружено: {self.loader.count}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "1-4:выбор SPACE:след S:скриншот Q:выход", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        if self.hand_mode == 'dual':
            status = "Две руки — растяжение"
            color = (255, 0, 255)
        elif self.hand_mode == 'single':
            status = "Одна рука — перспектива"
            color = (0, 255, 0)
        else:
            status = "Покажи руку!"
            color = (0, 0, 255)

        cv2.putText(frame, status, (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if self.hand_mode == 'dual':
            cv2.putText(frame, "Левая=якорь, правая=растяжение", (20, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 200), 1)

    def _save_screenshot(self, frame: np.ndarray) -> None:
        os.makedirs(self.cfg.screenshot_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.cfg.screenshot_dir, f"screenshot_{ts}.png")
        cv2.imwrite(path, frame)
        print(f"[Скриншот] {path}")

    def _smooth_corners(self, new_corners: list) -> list:
        """Экспоненциальное сглаживание 4 углов."""
        if self.smoothed_corners is None:
            self.smoothed_corners = [list(p) for p in new_corners]
        else:
            for i in range(4):
                old = self.smoothed_corners[i]
                new = new_corners[i]
                old[0] = old[0] * (1 - self.SMOOTH_FACTOR) + new[0] * self.SMOOTH_FACTOR
                old[1] = old[1] * (1 - self.SMOOTH_FACTOR) + new[1] * self.SMOOTH_FACTOR
        return [(int(p[0]), int(p[1])) for p in self.smoothed_corners]

    def run(self) -> None:
        if not self._init_camera():
            return

        print("\n" + "=" * 50)
        print("  Finger AR Overlay — запущен!")
        print("  1 рука = перспектива, 2 руки = растяжение")
        print("=" * 50 + "\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)

                # Детекция рук
                det = self.detector.process(frame)
                frame = det['frame']
                mode = det['mode']

                if mode == 'dual':
                    # === ДВЕ РУКИ: растяжение между ладонями ===
                    self.hand_mode = 'dual'
                    self.hand_lost_time = 0
                    image = self.loader.current()
                    if image is not None:
                        h_img, w_img = image.shape[:2]
                        aspect = w_img / h_img if h_img > 0 else 1.0
                        palms = det['palms']
                        corners = self.overlay.palms_to_corners(
                            palms[0], palms[1],
                            self.cfg.two_hand_padding,
                            aspect,
                            frame.shape[1], frame.shape[0],
                        )
                        # Защита от схлопывания
                        corners = self.overlay._enforce_min_distance(
                            corners, self.cfg.min_corner_dist
                        )
                        self.last_corners = self._smooth_corners(corners)
                        frame = self.overlay.overlay(frame, image, self.last_corners)

                elif mode == 'single':
                    # === ОДНА РУКА: 4 кончика пальцев ===
                    self.hand_mode = 'single'
                    self.hand_lost_time = 0
                    fingertips = det['fingertips']
                    # Защита от схлопывания
                    corners = self.overlay._enforce_min_distance(
                        fingertips, self.cfg.min_corner_dist
                    )
                    self.last_corners = self._smooth_corners(corners)

                    image = self.loader.current()
                    if image is not None and self.last_corners is not None:
                        frame = self.overlay.overlay(frame, image, self.last_corners)

                else:
                    # === НЕТ РУК ===
                    self.hand_lost_time += 1 / 30
                    if self.hand_lost_time > self.HAND_LOST_TIMEOUT:
                        self.hand_mode = 'none'
                        self.smoothed_corners = None
                        self.last_corners = None
                    elif self.last_corners is not None:
                        # Держим последнюю позу
                        self.hand_mode = self.hand_mode  # сохраняем
                        image = self.loader.current()
                        if image is not None:
                            frame = self.overlay.overlay(frame, image, self.last_corners)

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


# ──────────────────────────────────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = FingerARApp()
    app.run()
