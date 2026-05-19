import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.setting import VISDRONE_CONFIG

def test_load_config():
    config = load_config(VISDRONE_CONFIG)
    print(config["names"][1])
    assert config is not None

if __name__ == "__main__":
    test_load_config()
    print("All tests passed.")