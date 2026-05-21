# Adaptive Slicing for Small Object Detection in High-Resolution Images

## First Run

Follow these steps to get started:

```bash
python scripts/download.py
python scripts/prepare_data.py
python scripts/detect_full_image.py
python scripts/hard_region.py
python scripts/train_rl.py
python scripts/infer.py --image data/processed/images/test/0000006_00159_d_0000001.jpg
python scripts/visualize_policy.py --image data/processed/images/test/0000006_00159_d_0000001.jpg
```