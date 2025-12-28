import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os

from .config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    CLASS_NAMES,
    CLASS_COLORS,
    BBOX_THICKNESS,
    TEXT_FONT,
    TEXT_THICKNESS
)

class ClayTargetDetector:
    def __init__(self, model_path: str = None):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path or str(MODEL_PATH)
        self._load_model()

    def _load_model(self):
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                self.model.to(self.device)
            else:
                self.model = None
        except Exception as e:
            self.model = None

    def detect(self, image: np.ndarray):
        if self.model is None:
            return []

        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.model(rgb_image, conf=CONFIDENCE_THRESHOLD, verbose=False)
            detections = []

            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())

                    if class_id in CLASS_NAMES:
                        state = CLASS_NAMES[class_id]
                    elif class_id >= 0 and class_id <= 5:
                        state = "Мишень разбитая" if class_id >= 3 else "Мишень целая"
                        class_id = 1 if class_id >= 3 else 0
                    else:
                        state = "Неизвестно"
                        class_id = -1

                    detection = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': confidence,
                        'class_id': class_id,
                        'state': state,
                        'width': float(x2 - x1),
                        'height': float(y2 - y1),
                        'center_x': float((x1 + x2) / 2),
                        'center_y': float((y1 + y2) / 2)
                    }

                    detections.append(detection)

            return detections

        except Exception as e:
            return []

    def save_frame_with_bbox(self, image: np.ndarray, detections, output_path: str) -> bool:
        try:
            img_copy = image.copy()

            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                color = CLASS_COLORS.get(det['class_id'], (255, 255, 0))

                cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, BBOX_THICKNESS)

                label = f"{det['state']} {det['confidence']:.2f}"
                cv2.putText(img_copy, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, TEXT_FONT, color, TEXT_THICKNESS)

            cv2.imwrite(output_path, img_copy)
            return True

        except Exception as e:
            return False

detector = ClayTargetDetector()