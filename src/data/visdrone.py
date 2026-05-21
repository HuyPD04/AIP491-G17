from __future__ import annotations

import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import sys
from dataclasses import dataclass
import shutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.setting import (
    DETECT_FULL_IMAGE_DIR,
    HARD_REGION_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    VISDRONE_CONFIG,
)
from src.utils.config import load_config
from src.detectors.yolo import yolo_to_xyxy
VISDRONE_CLASSES = load_config(VISDRONE_CONFIG)["names"]

@dataclass(frozen=True)
class VisDroneObject:
    object_id: int | str
    class_id: int
    class_name: str
    bbox: tuple[float, float, float, float]  # x y w h 
    tags: tuple[str, ...] = ()

@dataclass(frozen=True)
class VisDroneAnnotation:
    image_id: str
    width: int
    height: int
    objects: tuple[VisDroneObject, ...]
    ignored_regions: int = 0

def parse_visdrone_annotation(
        annotation_path: str | Path,
        image_path: str | Path
) -> VisDroneAnnotation:

    width, height = Image.open(image_path).size
    objects: list[VisDroneObject] = []
    ignored_regions = 0

    text = annotation_path.read_text(encoding="utf-8").strip()

    if not text:
        return VisDroneAnnotation(
            image_id=annotation_path.stem,
            width=width,
            height=height,
            objects=(),
            ignored_regions=0
        )
    for idx, line in enumerate(text.splitlines()):
        row = line.split(",")

        x,y,w,h = map(float, row[:4])
        score = int(row[4])
        category_id = int(row[5])
        truncation = int(row[6])
        occlusion = int(row[7])

        if score == 0 or category_id <= 0 or category_id not in VISDRONE_CLASSES:
            ignored_regions += 1
            continue

        class_id = category_id - 1
        class_name = VISDRONE_CLASSES.get(category_id, f"class_{category_id}")
        x = max(0.0, min(float(width), x))
        y = max(0.0, min(float(height), y))
        w = max(0.0, min(float(width) - x, w))
        h = max(0.0, min(float(height) - y, h))
        if w <= 0.0 or h <= 0.0:
            ignored_regions += 1
            continue

        obj = VisDroneObject(
            object_id=idx,
            class_id=class_id,
            class_name=class_name,
            bbox=(x, y, w, h),
            tags=(
                f"score:{score}",
                f"truncation:{truncation}",
                f"occlusion:{occlusion}"
            ),
        )
        objects.append(obj)

    return VisDroneAnnotation(
        image_id=annotation_path.stem,
        width=width,
        height=height,
        objects=tuple(objects),
        ignored_regions=ignored_regions
    )

def convert_box(
    obj: VisDroneObject,
    img_width: int,
    img_height: int
) -> str:
    x, y, w, h = obj.bbox
    
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return (
        f"{obj.class_id} "
        f"{x_center:.6f} "
        f"{y_center:.6f} "
        f"{w_norm:.6f} "
        f"{h_norm:.6f}\n"
    )

def dump_yolo_labels(
    annotation: VisDroneAnnotation,
    label_path: str | Path
) -> None:
    lines = [
        convert_box(obj, annotation.width, annotation.height)
        for obj in annotation.objects
    ]
    
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("".join(lines), encoding="utf-8")

def visdrone2yolo(
    root_dir: str | Path = RAW_DATA_DIR,
    processed_dir: str | Path = PROCESSED_DATA_DIR,
    split: str = "",
    source_name: str | None = None,
    move_images: bool = False,
) -> None:
    root_dir = Path(root_dir)
    processed_dir = Path(processed_dir)
    
    source_dir = root_dir / (source_name or f"VisDrone2019-DET-{split}")
    images_dir = processed_dir / "images" / split
    labels_dir = processed_dir / "labels" / split

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    source_images_dir = source_dir / "images"
    source_annotations_dir = source_dir / "annotations"

    if source_images_dir.exists():
        for img in source_images_dir.glob("*.jpg"):
            dst = images_dir / img.name
            if move_images:
                img.rename(dst)
            else:
                shutil.copy2(img, dst)
    
    for annotation_path in tqdm(source_annotations_dir.glob("*.txt"), desc=f"Converting {split}"):
        image_path = images_dir / annotation_path.with_suffix(".jpg").name
        annotation = parse_visdrone_annotation(annotation_path, image_path)
        label_path = labels_dir / annotation_path.name
        dump_yolo_labels(annotation, label_path)


def load_yolo_labels(label_path: str | Path) -> list[VisDroneObject]:
    label_path = Path(label_path)
    if not label_path.exists():
        return []
    objects: list[VisDroneObject] = []
    for idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        class_id_s, xc_s, yc_s, bw_s, bh_s = line.split()[:5]
        class_id = int(class_id_s)
        class_name = VISDRONE_CLASSES.get(class_id + 1, f"class_{class_id}")
        objects.append(
            VisDroneObject(
                object_id=idx,
                class_id=class_id,
                class_name=class_name,
                bbox=(float(xc_s), float(yc_s), float(bw_s), float(bh_s)),
            )
        )
    return objects

def load_yolo_label_file(label_path: str | Path, image_size: tuple[int, int]) -> list[dict[str, object]]:
    objects = []
    for obj in load_yolo_labels(label_path):
        x_center, y_center, box_width, box_height = obj.bbox
        objects.append(
            {
                "id": obj.object_id,
                "class_id": obj.class_id,
                "class_name": obj.class_name,
                "bbox_xyxy": yolo_to_xyxy(x_center, y_center, box_width, box_height, image_size),
                "hard_score": 1.0,
            }
        )
    return objects

def build_processed_index(
    processed_dir: str | Path = PROCESSED_DATA_DIR,
    split: str = "train",
) -> list[dict[str, object]]:
    processed_dir = Path(processed_dir)
    image_dir = processed_dir / "images" / split
    label_dir = processed_dir / "labels" / split
    if not image_dir.exists():
        return []
    samples: list[dict[str, object]] = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        with Image.open(image_path) as image:
            width, height = image.size
        samples.append(
            {
                "image_id": image_path.stem,
                "image_path": image_path.as_posix(),
                "label_path": (label_dir / f"{image_path.stem}.txt").as_posix(),
                "prediction_path": (DETECT_FULL_IMAGE_DIR / f"{image_path.stem}.txt").as_posix(),
                "hard_region_path": (HARD_REGION_DIR / f"{image_path.stem}.txt").as_posix(),
                "width": width,
                "height": height,
            }
        )
    return samples
