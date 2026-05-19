from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.setting import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.data.visdrone import visdrone2yolo

def main():
    visdrone2yolo(
        root_dir=Path(RAW_DATA_DIR),
        processed_dir=Path(PROCESSED_DATA_DIR),
        split="train",
        source_name="VisDrone2019-DET-train",
        move_images=True,
    )
    visdrone2yolo(
        root_dir=Path(RAW_DATA_DIR),
        processed_dir=Path(PROCESSED_DATA_DIR),
        split="val",
        source_name="VisDrone2019-DET-val",
        move_images=True,
    )
    visdrone2yolo(
        root_dir=Path(RAW_DATA_DIR),
        processed_dir=Path(PROCESSED_DATA_DIR),
        split="test",
        source_name="VisDrone2019-DET-test-dev",
        move_images=True,
    )

if __name__ == "__main__":
    main()