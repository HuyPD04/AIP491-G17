from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.visdrone import load_yolo_label_file
from src.utils.setting import HARD_REGION_DIR, INFER_OUTPUT_DIR, TEST_IMAGES_DIR, VISUALIZATION_OUTPUT_DIR


ROI_COLORS = [
    "lime",
    "yellow",
    "magenta",
    "orange",
    "deepskyblue",
    "white",
    "violet",
    "chartreuse",
]


def main() -> None:
    args = parse_args()
    image_path = args.image or first_image(TEST_IMAGES_DIR)
    pred_path = args.predictions or (INFER_OUTPUT_DIR / f"{image_path.stem}_rl.txt")
    roi_path = args.rois or (INFER_OUTPUT_DIR / f"{image_path.stem}_rois.txt")
    hard_path = args.hard_regions
    if hard_path is None:
        candidate = HARD_REGION_DIR / f"{image_path.stem}.txt"
        hard_path = candidate if candidate.exists() else None
    output_path = args.output or (VISUALIZATION_OUTPUT_DIR / f"{image_path.stem}_rl.jpg")
    visualize(image_path, pred_path, roi_path, hard_path, output_path, draw_predictions=not args.no_predictions)
    print(f"visualization={output_path}")


def visualize(
    image_path: Path,
    pred_path: Path,
    roi_path: Path,
    hard_path: Path | None,
    output_path: Path,
    draw_predictions: bool = True,
) -> None:
    with Image.open(image_path) as image:
        vis = image.convert("RGB")
    draw = ImageDraw.Draw(vis)
    image_size = vis.size

    if hard_path is not None and hard_path.exists():
        for obj in load_yolo_label_file(hard_path, image_size):
            draw.rectangle(obj["bbox_xyxy"], outline="red", width=2)

    for idx, roi in enumerate(load_rois(roi_path), start=1):
        color = ROI_COLORS[(idx - 1) % len(ROI_COLORS)]
        draw.rectangle(roi, outline=color, width=4)
        draw_label(draw, roi, f"ROI {idx}", color)

    if draw_predictions:
        for class_id, score, box in load_predictions(pred_path):
            draw.rectangle(box, outline="cyan", width=2)
            draw_label(draw, box, f"{class_id}:{score:.2f}", "cyan")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(output_path)


def load_rois(path: Path) -> list[list[float]]:
    if not path.exists():
        return []
    rois = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rois.append([float(value) for value in line.split()[:4]])
    return rois


def load_predictions(path: Path) -> list[tuple[int, float, list[float]]]:
    if not path.exists():
        return []
    predictions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id_s, score_s, x1_s, y1_s, x2_s, y2_s = line.split()[:6]
        predictions.append(
            (
                int(float(class_id_s)),
                float(score_s),
                [float(x1_s), float(y1_s), float(x2_s), float(y2_s)],
            )
        )
    return predictions


def draw_label(draw: ImageDraw.ImageDraw, box: list[float], text: str, color: str) -> None:
    x1, y1, _, _ = map(float, box)
    tx = int(max(0, x1 + 3))
    ty = int(max(0, y1 + 3))
    try:
        text_box = draw.textbbox((tx, ty), text)
        draw.rectangle(
            [text_box[0] - 2, text_box[1] - 2, text_box[2] + 2, text_box[3] + 2],
            fill="black",
        )
    except Exception:
        pass
    draw.text((tx, ty), text, fill=color)


def first_image(image_dir: Path) -> Path:
    for image_path in sorted(image_dir.glob("*.jpg")):
        return image_path
    raise FileNotFoundError(f"No .jpg image found in {image_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw RL rois, hard regions, and predictions.")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--rois", type=Path)
    parser.add_argument("--hard-regions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-predictions", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
