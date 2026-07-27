"""
Finger AR Overlay — привязка картинки к кончикам пальцев через камеру.

Стек: OpenCV (камера) + MediaPipe Tasks Vision (HandLandmarker) + warpPerspective (AR-оверлей).

Режимы:
    1 рука (правая) — 4 кончика пальцев = 4 угла картинки (перспектива).
    2 руки — правая держит картинку, левая щипком захватывает угол и тянет.

Левая рука: pinch (щипок = большой + указательный пальцы рядом) рядом с углом
            = захват. Двигаешь рукой — угол тянется. Разжал пальцы — отпустил.

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
    max_num_hands: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    fingertip_ids: tuple = (4, 8, 16, 20)  # thumb, index, ring, pinky
    images_dir: str = "images"
    screenshot_dir: str = "screenshots"
    model_path: str = "hand_landmarker.task"
    model_url: str = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    min_corner_dist: int = 20       # мин расстояние между углами
    pinch_threshold: float = 50.0   # px — расстояние thumb-index для щипка
    grab_radius: float = 80.0       # px — радиус захвата угла щипком


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

    def process(self, frame: np.ndarray) -> dict:
        """
        Возвращает dict:
            'frame':      аннотированный кадр
            'hands':      список словарей для каждой руки:
                { 'label': 'Right'|'Left',
                  'fingertips': [(x,y)]*4,
                  'pinch_point': (x,y),  — середина между thumb и index
                  'pinching': bool,
                  'landmarks': raw }
        """
        h, w = frame.shape[:2]
        hands = []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if result.hand_landmarks:
            for i, lm_list in enumerate(result.hand_landmarks):
                # Определяем левая/правая
                label = "Right"
                if result.handedness and i < len(result.handedness):
                    label = result.handedness[i][0].category_name
                    # MediaPipe зеркалит: "Right" на самом деле левая рука
                    # Мы зеркалим кадр → инвертируем
                    label = "Left" if label == "Right" else "Right"

                # Кончики пальцев
                fingertips = []
                for idx in self.cfg.fingertip_ids:
                    lm = lm_list[idx]
                    fingertips.append((int(lm.x * w), int(lm.y * h)))

                # Щипок: расстояние thumb tip (4) — index tip (8)
                tx, ty = lm_list[4].x * w, lm_list[4].y * h
                ix, iy = lm_list[8].x * w, lm_list[8].y * h
                pinch_dist = ((tx - ix) ** 2 + (ty - iy) ** 2) ** 0.5
                pinch_point = (int((tx + ix) / 2), int((ty + iy) / 2))
                pinching = pinch_dist < self.cfg.pinch_threshold

                # Рисуем скелет
                self._draw_skeleton(frame, lm_list, w, h)

                hands.append({
                    'label': label,
                    'fingertips': fingertips,
                    'pinch_point': pinch_point,
                    'pinching': pinching,
                    'landmarks': lm_list,
                })

                # Рисуем pinch indicator
                if pinching:
                    cv2.circle(frame, pinch_point, 15, (0, 255, 0), -1)
                    cv2.circle(frame, pinch_point, 18, (0, 200, 0), 2)
                else:
                    cv2.circle(frame, pinch_point, 10, (0, 0, 255), -1)

        return {'frame': frame, 'hands': hands}

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

        # Текущие 4 угла картинки
        self.corners: Optional[list] = None
        self.smoothed: Optional[list] = None
        self.SMOOTH_FACTOR: float = 0.3

        # Захват щипком: индекс угла (0-3) или None
        self.grabbed_corner: Optional[int] = None

        self.hand_mode: str = 'none'
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
        cv2.rectangle(overlay_img, (10, 10), (460, 170), (0, 0, 0), -1)
        cv2.addWeighted(overlay_img, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, f"Картинка: {self.loader.current_name()}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Картинок загружено: {self.loader.count}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "1-4:выбор SPACE:след S:скриншот Q:выход", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        if self.hand_mode == 'dual':
            if self.grabbed_corner is not None:
                status = f"Захват угла #{self.grabbed_corner + 1}"
                color = (0, 255, 0)
            else:
                status = "Две руки — щипни угол!"
                color = (255, 0, 255)
        elif self.hand_mode == 'single':
            status = "Правая рука — перспектива"
            color = (0, 255, 0)
        else:
            status = "Покажи руку!"
            color = (0, 0, 255)

        cv2.putText(frame, status, (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(frame, "Левая рука: щипок=захват угла, тяни=растяжение", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 200, 200), 1)

    def _save_screenshot(self, frame: np.ndarray) -> None:
        os.makedirs(self.cfg.screenshot_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.cfg.screenshot_dir, f"screenshot_{ts}.png")
        cv2.imwrite(path, frame)
        print(f"[Скриншот] {path}")

    def _smooth_corners(self, new_corners: list) -> list:
        if self.smoothed is None:
            self.smoothed = [list(p) for p in new_corners]
        else:
            for i in range(4):
                s = self.smoothed[i]
                n = new_corners[i]
                s[0] = s[0] * (1 - self.SMOOTH_FACTOR) + n[0] * self.SMOOTH_FACTOR
                s[1] = s[1] * (1 - self.SMOOTH_FACTOR) + n[1] * self.SMOOTH_FACTOR
        return [(int(p[0]), int(p[1])) for p in self.smoothed]

    def _find_nearest_corner(self, point: tuple, corners: list) -> tuple:
        """Находит ближайший угол к точке. Возвращает (индекс, расстояние)."""
        best_i = 0
        best_d = float('inf')
        for i, c in enumerate(corners):
            dx = c[0] - point[0]
            dy = c[1] - point[1]
            d = (dx * dx + dy * dy) ** 0.5
            if d < best_d:
                best_d = d
                best_i = i
        return best_i, best_d

    def run(self) -> None:
        if not self._init_camera():
            return

        print("\n" + "=" * 50)
        print("  Finger AR Overlay — запущен!")
        print("  Правая = перспектива, левая = щипок угол")
        print("=" * 50 + "\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)

                det = self.detector.process(frame)
                frame = det['frame']
                hands = det['hands']

                # Разделяем руки
                right_hand = None
                left_hand = None
                for h_data in hands:
                    if h_data['label'] == 'Right' and right_hand is None:
                        right_hand = h_data
                    elif h_data['label'] == 'Left' and left_hand is None:
                        left_hand = h_data

                image = self.loader.current()

                if right_hand and image is not None:
                    # Правая рука задает 4 угла картинки
                    self.hand_mode = 'dual' if left_hand else 'single'
                    self.hand_lost_time = 0

                    new_corners = self.overlay._enforce_min_distance(
                        right_hand['fingertips'], self.cfg.min_corner_dist
                    )
                    self.corners = self._smooth_corners(new_corners)

                    # Если левая рука есть — обрабатываем щипок
                    if left_hand:
                        pinch_pt = left_hand['pinch_point']

                        if left_hand['pinching']:
                            if self.grabbed_corner is None:
                                # Ищем ближайший угол
                                idx, dist = self._find_nearest_corner(
                                    pinch_pt, self.corners
                                )
                                if dist < self.cfg.grab_radius:
                                    self.grabbed_corner = idx
                                    print(f"[Захват] Угол #{idx + 1} (dist={dist:.0f}px)")

                            # Тянем захваченный угол
                            if self.grabbed_corner is not None:
                                ci = self.grabbed_corner
                                # Обновляем и smooth, и corners
                                self.corners[ci] = list(pinch_pt)
                                self.smoothed[ci] = list(pinch_pt)

                        else:
                            # Отпустили
                            if self.grabbed_corner is not None:
                                print(f"[Отпуск] Угол #{self.grabbed_corner + 1}")
                                self.grabbed_corner = None

                    else:
                        self.grabbed_corner = None

                    # Рисуем углы
                    for i, (cx, cy) in enumerate(self.corners):
                        if i == self.grabbed_corner:
                            cv2.circle(frame, (cx, cy), 16, (0, 255, 0), -1)
                            cv2.circle(frame, (cx, cy), 20, (0, 200, 0), 3)
                        else:
                            cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)
                            cv2.circle(frame, (cx, cy), 13, (0, 0, 255), 2)

                    frame = self.overlay.overlay(frame, image, self.corners)

                elif not right_hand and self.corners is not None:
                    # Рука потеряна — держим позу
                    self.hand_lost_time += 1 / 30
                    if self.hand_lost_time > self.HAND_LOST_TIMEOUT:
                        self.hand_mode = 'none'
                        self.corners = None
                        self.smoothed = None
                        self.grabbed_corner = None
                    else:
                        if image is not None:
                            frame = self.overlay.overlay(frame, image, self.corners)
                else:
                    self.hand_mode = 'none'

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
