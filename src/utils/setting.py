from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = ROOT / 'data' / 'raw'
PROCESSED_DATA_DIR = ROOT / 'data' / 'processed'

YOLO_CONFIG = ROOT / 'configs' / 'yolo.yaml'
VISDRONE_CONFIG = ROOT / 'configs' / 'visdrone.yaml'

DETECT_FULL_IMAGE_DIR = ROOT / 'data' / 'cache' / 'detect_full_image'
HARD_REGION_DIR = ROOT / 'data' / 'cache' / 'hard_region'

TRAIN_IMAGES_DIR = PROCESSED_DATA_DIR / 'images' / 'train'
VAL_IMAGES_DIR = PROCESSED_DATA_DIR / 'images' / 'val'
TEST_IMAGES_DIR = PROCESSED_DATA_DIR / 'images' / 'test'
TRAIN_LABELS_DIR = PROCESSED_DATA_DIR / 'labels' / 'train'
VAL_LABELS_DIR = PROCESSED_DATA_DIR / 'labels' / 'val'
TEST_LABELS_DIR = PROCESSED_DATA_DIR / 'labels' / 'test'