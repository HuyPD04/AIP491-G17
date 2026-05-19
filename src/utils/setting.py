from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = ROOT / 'data' / 'raw'
PROCESSED_DATA_DIR = ROOT / 'data' / 'processed'

YOLO_CONFIG = ROOT / 'configs' / 'yolo.yaml'
VISDRONE_CONFIG = ROOT / 'configs' / 'visdrone.yaml'