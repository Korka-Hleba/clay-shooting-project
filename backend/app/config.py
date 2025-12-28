import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
FRAMES_DIR = BASE_DIR / "frames"
STATIC_DIR = BASE_DIR / "static"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "best.pt"

FRAME_INTERVAL = 0.5
CONFIDENCE_THRESHOLD = 0.1

CLASS_NAMES = {
    0: "Мишень целая",
    1: "Мишень разбитая",
}

CLASS_COLORS = {
    0: (0, 255, 0),
    1: (0, 0, 255),
}
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV'}

BBOX_THICKNESS = 2
TEXT_FONT = 0.5
TEXT_THICKNESS = 2

MAX_FILE_SIZE = 50 * 1024 * 1024

USE_TEST_DATA_IF_NO_MODEL = True