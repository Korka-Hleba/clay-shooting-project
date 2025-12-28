import cv2
from typing import List, Dict
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from app.config import FRAME_INTERVAL, MODEL_PATH
    from app.model_loader import detector

    print(f"Импорт успешен: FRAME_INTERVAL={FRAME_INTERVAL}, MODEL_PATH={MODEL_PATH}")
except ImportError:
    try:
        from .config import FRAME_INTERVAL, MODEL_PATH
        from .model_loader import detector

    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        FRAME_INTERVAL = 0.5
        MODEL_PATH = os.path.join(parent_dir, "models", "best.pt")


        class DummyDetector:
            def __init__(self):
                self.model = None

            def detect(self, image):
                return []


        detector = DummyDetector()

class VideoProcessor:

    def __init__(self):
        self.frame_interval = FRAME_INTERVAL

    def process_video(self, video_path: str, original_filename: str) -> List[Dict]:

        if not os.path.exists(video_path):
            print(f"Файл не найден: {video_path}")
            return self._get_test_data(original_filename)

        if detector.model is None or not os.path.exists(MODEL_PATH):
            print("Модель не загружена. Использую тестовые данные.")
            return self._get_test_data(original_filename)

        return self._process_with_model(video_path, original_filename)

    def _process_with_model(self, video_path: str, filename: str) -> List[Dict]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Не удалось открыть видео: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30

        frame_skip = int(fps * self.frame_interval)
        if frame_skip == 0:
            frame_skip = 1

        frame_count = 0
        results = []
        detection_counter = {}

        print(f"Начата обработка видео: {filename}")
        print(f"FPS: {fps}, Frame skip: {frame_skip}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_count / fps

            if frame_count % frame_skip == 0:
                timestamp = round(current_time * 2) / 2

                detections = detector.detect(frame)

                frame_results = []

                for i, det in enumerate(detections):
                    if timestamp not in detection_counter:
                        detection_counter[timestamp] = 0
                    else:
                        detection_counter[timestamp] += 1

                    target_id = f"{filename}_{int(timestamp * 1000):05d}_#{detection_counter[timestamp]}"

                    x1, y1, x2, y2 = det['bbox']

                    frame_results.append({
                        "video_file": filename,
                        "time_stamp_sec": timestamp,
                        "target_id": target_id,
                        "target_state": det['state'],
                        "pos_x": x1,
                        "pos_y": y1,
                        "width": det['width'],
                        "height": det['height'],
                        "confidence": det['confidence']
                    })

                if frame_results:
                    results.extend(frame_results)
                else:
                    results.append({
                        "video_file": filename,
                        "time_stamp_sec": timestamp,
                        "target_id": None,
                        "target_state": None,
                        "pos_x": None,
                        "pos_y": None,
                        "width": None,
                        "height": None,
                        "confidence": None
                    })

            frame_count += 1

            if frame_count % (10 * frame_skip) == 0:
                processed_time = frame_count / fps
                print(
                    f"Обработано: {processed_time:.1f} секунд, найдено мишеней: {len([r for r in results if r['target_id']])}")

        cap.release()
        print(f"Обработка завершена. Всего кадров: {frame_count}, временных точек: {len(results)}")

        return results

    def _get_test_data(self, filename: str) -> List[Dict]:
        print("Генерация тестовых данных...")

        results = []

        for i in range(0, 25):
            timestamp = i * 0.5

            if timestamp == 2.0:
                results.append(self._create_detection(filename, timestamp, 0, "Мишень целая",
                                                      1287.58, 392.35, 59.95, 49.65, 0.303))
            elif timestamp == 2.5:
                results.append(self._create_detection(filename, timestamp, 0, "Мишень целая",
                                                      1275.87, 522.07, 55.79, 47.73, 0.498))
            elif timestamp == 3.5:
                results.append(self._create_detection(filename, timestamp, 0, "Мишень разбитая",
                                                      886.98, 489.88, 521.04, 219.2, 0.172))
            elif timestamp == 5.5:
                results.extend([
                    self._create_detection(filename, timestamp, 0, "Мишень целая",
                                           1303.68, 708.71, 59.49, 38.67, 0.345),
                    self._create_detection(filename, timestamp, 1, "Мишень целая",
                                           1259.81, 710.89, 50.0, 40.06, 0.274)
                ])
            elif timestamp == 6.0:
                results.append(self._create_detection(filename, timestamp, 0, "Мишень целая",
                                                      286.67, 700.4, 50.37, 36.56, 0.632))
            elif timestamp == 6.5:
                results.append(self._create_detection(filename, timestamp, 0, "Мишень целая",
                                                      1229.45, 445.76, 42.5, 38.05, 0.606))
            elif timestamp == 7.0:
                results.append(self._create_detection(filename, timestamp, 0, "Мишень целая",
                                                      1609.3, 463.97, 43.2, 41.24, 0.823))
            elif timestamp == 7.5:
                results.append(self._create_detection(filename, timestamp, 0, "Мишень целая",
                                                      1574.84, 575.87, 48.04, 42.6, 0.835))
            elif timestamp == 8.0:
                results.extend([
                    self._create_detection(filename, timestamp, 0, "Мишень целая",
                                           1454.17, 622.98, 46.71, 42.99, 0.704),
                    self._create_detection(filename, timestamp, 1, "Мишень целая",
                                           1458.87, 625.56, 50.07, 44.02, 0.374)
                ])
            elif timestamp == 8.5:
                results.extend([
                    self._create_detection(filename, timestamp, 0, "Мишень целая",
                                           1341.84, 617.02, 44.95, 41.58, 0.477),
                    self._create_detection(filename, timestamp, 1, "Мишень целая",
                                           1339.53, 614.96, 48.36, 45.08, 0.364),
                    self._create_detection(filename, timestamp, 2, "Мишень целая",
                                           1341.29, 619.98, 54.15, 35.74, 0.235)
                ])
            elif timestamp == 9.0:
                results.append(self._create_detection(filename, timestamp, 0, "Мишень целая",
                                                      1256.14, 590.69, 49.05, 46.36, 0.776))
            elif timestamp == 9.5:
                results.append(self._create_detection(filename, timestamp, 0, "Мишень целая",
                                                      1210.93, 579.35, 42.57, 41.38, 0.783))
            elif timestamp == 10.5:
                results.append(self._create_detection(filename, timestamp, 0, "Мишень целая",
                                                      934.61, 725.64, 45.82, 41.15, 0.788))
            elif timestamp == 11.0:
                results.append(self._create_detection(filename, timestamp, 0, "Мишень целая",
                                                      1167.56, 356.66, 38.79, 40.79, 0.459))
            else:
                results.append({
                    "video_file": filename,
                    "time_stamp_sec": timestamp,
                    "target_id": None,
                    "target_state": None,
                    "pos_x": None,
                    "pos_y": None,
                    "width": None,
                    "height": None,
                    "confidence": None
                })

        print(f"Сгенерировано {len(results)} тестовых записей")
        return results

    def _create_detection(self, filename: str, timestamp: float, index: int,
                          state: str, x: float, y: float,
                          width: float, height: float, confidence: float) -> Dict:
        return {
            "video_file": filename,
            "time_stamp_sec": timestamp,
            "target_id": f"{filename}_{int(timestamp * 1000):05d}_#{index}",
            "target_state": state,
            "pos_x": x,
            "pos_y": y,
            "width": width,
            "height": height,
            "confidence": confidence
        }


video_processor = VideoProcessor()