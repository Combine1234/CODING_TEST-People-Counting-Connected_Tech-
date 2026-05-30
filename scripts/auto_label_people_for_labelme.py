from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_NAME = "person"


def project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    if script_dir.name.lower() == "scripts":
        return script_dir.parent
    return script_dir


def parse_args() -> argparse.Namespace:
    root = project_root()

    parser = argparse.ArgumentParser(
        description=(
            "Auto-label people from captured images with YOLO and write LabelMe JSON "
            "files that can be corrected before YOLO training."
        )
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=root / "Dataset" / "image_capture",
        help="Folder containing captured images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "Dataset" / "labeledandimg",
        help="Folder for copied images, LabelMe JSON, and YOLO training files.",
    )
    parser.add_argument(
        "--model",
        default="yolov8s.pt",
        help=(
            "YOLO model name or path. Default: yolov8s.pt "
            "(Ultralytics downloads/uses cache automatically)."
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold. Default: 0.25.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Detection IoU threshold. Default: 0.45.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size. Default: 640.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example cpu, 0, or cuda:0. Default: auto.",
    )
    parser.add_argument(
        "--person-class-id",
        type=int,
        default=None,
        help="Override the model class id used for person detections.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit how many input images to process. Useful for testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing LabelMe JSON files. Leave off to protect manual edits.",
    )
    parser.add_argument(
        "--convert-yolo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also build a YOLO-format dataset from LabelMe JSON. Default: enabled.",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Skip detection and only rebuild YOLO-format labels from existing JSON files.",
    )
    return parser.parse_args()


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {images_dir}")

    images = [
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda path: path.name.lower())


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def person_class_ids(model: YOLO, override: int | None) -> list[int]:
    if override is not None:
        return [override]

    names = getattr(model, "names", None)
    if isinstance(names, dict):
        items = names.items()
    elif isinstance(names, (list, tuple)):
        items = enumerate(names)
    else:
        return [0]

    exact_matches = [
        int(class_id)
        for class_id, name in items
        if str(name).strip().lower() == LABEL_NAME
    ]
    if exact_matches:
        return exact_matches

    raise RuntimeError(
        f"Could not find a '{LABEL_NAME}' class in model names: {names}. "
        "Use --person-class-id if this model uses a different id."
    )


def make_labelme_json(
    image_name: str,
    width: int,
    height: int,
    boxes: list[dict[str, Any]],
) -> dict[str, Any]:
    shapes: list[dict[str, Any]] = []
    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        x1 = round(clamp(float(x1), 0, width), 2)
        y1 = round(clamp(float(y1), 0, height), 2)
        x2 = round(clamp(float(x2), 0, width), 2)
        y2 = round(clamp(float(y2), 0, height), 2)

        if x2 <= x1 or y2 <= y1:
            continue

        shapes.append(
            {
                "label": LABEL_NAME,
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "description": f"auto_conf={box['confidence']:.4f}",
                "shape_type": "rectangle",
                "flags": {},
            }
        )

    return {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def auto_label(args: argparse.Namespace) -> tuple[int, int, int]:
    image_paths = list_images(args.images.resolve())
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"No images found in: {args.images}")

    model_ref = str(args.model)
    possible_model_path = Path(model_ref).expanduser()
    if possible_model_path.exists():
        model_ref = str(possible_model_path.resolve())

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_ref)
    classes = person_class_ids(model, args.person_class_id)

    predict_kwargs: dict[str, Any] = {
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "classes": classes,
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    processed_count = 0
    detection_count = 0
    skipped_json_count = 0

    for source_path in image_paths:
        results = model.predict(source=str(source_path), **predict_kwargs)
        if not results:
            print(f"Warning: no result returned for {source_path}")
            continue
        result = results[0]

        output_image_path = output_dir / source_path.name
        json_path = output_dir / f"{source_path.stem}.json"

        shutil.copy2(source_path, output_image_path)

        if json_path.exists() and not args.overwrite:
            skipped_json_count += 1
            processed_count += 1
            continue

        height, width = result.orig_shape
        boxes: list[dict[str, Any]] = []
        if result.boxes is not None:
            xyxy_values = result.boxes.xyxy.cpu().tolist()
            conf_values = result.boxes.conf.cpu().tolist()
            class_values = result.boxes.cls.cpu().tolist()

            for xyxy, confidence, class_id in zip(
                xyxy_values,
                conf_values,
                class_values,
            ):
                if int(class_id) not in classes:
                    continue
                boxes.append({"xyxy": xyxy, "confidence": float(confidence)})

        detection_count += len(boxes)
        labelme_payload = make_labelme_json(source_path.name, width, height, boxes)
        write_json(json_path, labelme_payload)
        processed_count += 1

    return processed_count, detection_count, skipped_json_count


def safe_clear_directory(path: Path, allowed_parent: Path) -> None:
    path = path.resolve()
    allowed_parent = allowed_parent.resolve()
    if allowed_parent not in path.parents and path != allowed_parent:
        raise RuntimeError(f"Refusing to clear unexpected folder: {path}")

    if path.exists():
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        path.mkdir(parents=True, exist_ok=True)


def points_to_yolo_bbox(
    points: list[list[float]],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    if not points:
        return None

    xs = [clamp(float(point[0]), 0, width) for point in points]
    ys = [clamp(float(point[1]), 0, height) for point in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0 or box_height <= 0:
        return None

    x_center = (x1 + x2) / 2.0 / width
    y_center = (y1 + y2) / 2.0 / height
    return x_center, y_center, box_width / width, box_height / height


def image_size_from_json_or_file(payload: dict[str, Any], image_path: Path) -> tuple[int, int]:
    width = payload.get("imageWidth")
    height = payload.get("imageHeight")
    if width and height:
        return int(width), int(height)

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image size: {image_path}")
    height, width = image.shape[:2]
    return int(width), int(height)


def convert_labelme_to_yolo(output_dir: Path) -> tuple[int, int]:
    output_dir = output_dir.resolve()
    yolo_dir = output_dir / "yolo_dataset"
    images_train_dir = yolo_dir / "images" / "train"
    labels_train_dir = yolo_dir / "labels" / "train"

    safe_clear_directory(yolo_dir, output_dir)
    images_train_dir.mkdir(parents=True, exist_ok=True)
    labels_train_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(output_dir.glob("*.json"), key=lambda path: path.name.lower())
    image_count = 0
    label_count = 0

    for json_path in json_paths:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        image_name = payload.get("imagePath") or f"{json_path.stem}.jpg"
        source_image_path = (json_path.parent / image_name).resolve()
        if not source_image_path.exists():
            print(f"Warning: image missing for {json_path.name}: {source_image_path}")
            continue

        width, height = image_size_from_json_or_file(payload, source_image_path)
        label_lines: list[str] = []

        for shape in payload.get("shapes", []):
            if str(shape.get("label", "")).strip().lower() != LABEL_NAME:
                continue

            bbox = points_to_yolo_bbox(shape.get("points", []), width, height)
            if bbox is None:
                continue

            x_center, y_center, box_width, box_height = bbox
            label_lines.append(
                f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            )

        shutil.copy2(source_image_path, images_train_dir / source_image_path.name)
        (labels_train_dir / f"{source_image_path.stem}.txt").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )
        image_count += 1
        label_count += len(label_lines)

    data_yaml = yolo_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f'path: "{yolo_dir.as_posix()}"',
                "train: images/train",
                "val: images/train",
                "names:",
                f"  0: {LABEL_NAME}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return image_count, label_count


def main() -> int:
    args = parse_args()

    output_dir = args.output.resolve()
    if args.convert_only:
        if not output_dir.exists():
            raise FileNotFoundError(f"Output folder not found: {output_dir}")
        processed_count = detection_count = skipped_json_count = 0
    else:
        processed_count, detection_count, skipped_json_count = auto_label(args)

    yolo_image_count = yolo_label_count = 0
    if args.convert_yolo:
        yolo_image_count, yolo_label_count = convert_labelme_to_yolo(output_dir)

    print(f"Output: {output_dir}")
    if not args.convert_only:
        print(f"Images processed: {processed_count}")
        print(f"Person boxes found: {detection_count}")
        if skipped_json_count:
            print(f"Existing LabelMe JSON skipped: {skipped_json_count}")
    if args.convert_yolo:
        print(f"YOLO images: {yolo_image_count}")
        print(f"YOLO labels: {yolo_label_count}")
        print(f"YOLO data file: {output_dir / 'yolo_dataset' / 'data.yaml'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
