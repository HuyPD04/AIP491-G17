import os
from pathlib import Path
import sys
from ultralytics.utils.downloads import download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.setting import RAW_DATA_DIR

dir = Path(RAW_DATA_DIR)
if not dir.exists():
    os.makedirs(dir)

urls = ['https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip',
        'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip',
        'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-dev.zip',
        'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-challenge.zip']
download(urls, dir=dir, curl=True, threads=4)

for file in ["VisDrone2019-DET-train.zip", "VisDrone2019-DET-val.zip", "VisDrone2019-DET-test-dev.zip", "VisDrone2019-DET-test-challenge.zip"]:
    os.remove(dir / file)