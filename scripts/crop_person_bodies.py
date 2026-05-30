from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


LABEL_NAME = "person"


def project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    if script_dir.name.lower() == "scripts":
        return script_dir.parent
    return script_dir


def default_model(root: Path) -> str:
    model_path = root / "models" / "yolov8s.pt"
    if model_path.exists():
        return str(model_path)
    legacy_model_path = root / "yolov8s.pt"
    if legacy_model_path.exists():
        return str(legacy_model_path)
    return "yolov8s.pt"


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Crop person/body images from a video for shirt-classifier labeling."
    )
    parser.add_argument("--video", type=Path, default=root / "entrance.mov")
    parser.add_argument(
        "--detector",
        default=default_model(root),
        help="YOLO person detector model path/name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "Dataset" / "shirt_crop_dataset",
        help="Output folder for person and torso crops.",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--process-every", type=int, default=3)
    parser.add_argument(
        "--save-every",
        type=int,
        default=12,
        help="Save at most once every N frames for each visible track.",
    )
    parser.add_argument(
        "--max-crops-per-track",
        type=int,
        default=10,
        help="Maximum saved crop pairs per tracker id.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.06,
        help="Padding around full-person crop as a fraction of box size.",
    )
    parser.add_argument(
        "--person-class-id",
        type=int,
        default=None,
        help="Override the model class id for person.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames. Use 0 for the full video.",
    )
    return parser.parse_args()


def resolve_model_ref(model_arg: str) -> str:
    model_path = Path(model_arg).expanduser()
    if model_path.exists():
        return str(model_path.resolve())
    return model_arg


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

    ids = [
        int(class_id)
        for class_id, name in items
        if str(name).strip().lower() == LABEL_NAME
    ]
    if not ids:
        raise RuntimeError(
            f"Could not find '{LABEL_NAME}' in model names: {names}. "
            "Use --person-class-id for a custom model."
        )
    return ids


def clamp_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    ix1 = max(0, min(int(round(x1)), width - 1))
    iy1 = max(0, min(int(round(y1)), height - 1))
    ix2 = max(0, min(int(round(x2)), width))
    iy2 = max(0, min(int(round(y2)), height))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def person_crop_box(
    xyxy: list[float],
    frame_width: int,
    frame_height: int,
    padding: float,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = xyxy
    box_width = x2 - x1
    box_height = y2 - y1
    return clamp_box(
        x1 - box_width * padding,
        y1 - box_height * padding,
        x2 + box_width * padding,
        y2 + box_height * padding,
        frame_width,
        frame_height,
    )


def torso_crop_box(
    xyxy: list[float],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = xyxy
    box_width = x2 - x1
    box_height = y2 - y1
    return clamp_box(
        x1 + box_width * 0.12,
        y1 + box_height * 0.12,
        x2 - box_width * 0.12,
        y1 + box_height * 0.72,
        frame_width,
        frame_height,
    )


def write_crop(
    frame,
    box: tuple[int, int, int, int] | None,
    output_path: Path,
) -> bool:
    if box is None:
        return False
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), crop))


def main() -> int:
    args = parse_args()
    if args.process_every <= 0:
        raise ValueError("--process-every must be greater than 0")
    if args.save_every <= 0:
        raise ValueError("--save-every must be greater than 0")

    video_path = args.video.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = args.output.resolve()
    person_dir = output_dir / "person_crops"
    torso_dir = output_dir / "torso_crops"
    output_dir.mkdir(parents=True, exist_ok=True)
    person_dir.mkdir(parents=True, exist_ok=True)
    torso_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLO(resolve_model_ref(args.detector))
    classes = person_class_ids(detector, args.person_class_id)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    predict_kwargs: dict[str, Any] = {
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "classes": classes,
        "tracker": args.tracker,
        "persist": True,
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    saved_counts: dict[int, int] = {}
    manifest_rows: list[dict[str, Any]] = []
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and frame_index >= args.max_frames:
            break

        if frame_index % args.process_every != 0:
            frame_index += 1
            continue

        results = detector.track(frame, **predict_kwargs)
        result = results[0] if results else None
        if result is None or result.boxes is None or result.boxes.id is None:
            frame_index += 1
            continue

        boxes = result.boxes.xyxy.cpu().tolist()
        track_ids = result.boxes.id.cpu().int().tolist()
        confidences = result.boxes.conf.cpu().tolist()

        for xyxy, track_id, confidence in zip(boxes, track_ids, confidences):
            track_id = int(track_id)
            if frame_index % args.save_every != 0:
                continue
            if saved_counts.get(track_id, 0) >= args.max_crops_per_track:
                continue

            crop_index = saved_counts.get(track_id, 0) + 1
            stem = f"T{track_id:04d}_f{frame_index:06d}_c{confidence:.2f}_{crop_index:02d}"
            person_path = person_dir / f"{stem}.jpg"
            torso_path = torso_dir / f"{stem}.jpg"

            person_box = person_crop_box(xyxy, frame_width, frame_height, args.padding)
            torso_box = torso_crop_box(xyxy, frame_width, frame_height)
            person_ok = write_crop(frame, person_box, person_path)
            torso_ok = write_crop(frame, torso_box, torso_path)
            if not person_ok and not torso_ok:
                continue

            saved_counts[track_id] = crop_index
            for crop_type, path, box, saved in [
                ("person", person_path, person_box, person_ok),
                ("torso", torso_path, torso_box, torso_ok),
            ]:
                if not saved or box is None:
                    continue
                x1, y1, x2, y2 = box
                manifest_rows.append(
                    {
                        "crop_type": crop_type,
                        "track_id": track_id,
                        "frame": frame_index,
                        "time_sec": round(frame_index / fps, 3),
                        "confidence": round(float(confidence), 4),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "path": str(path),
                    }
                )

        if frame_index and frame_index % 300 == 0:
            print(f"Processed {frame_index}/{total_frames or '?'} frames")
        frame_index += 1

    cap.release()

    manifest_path = output_dir / "crop_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as csv_file:
        fieldnames = [
            "crop_type",
            "track_id",
            "frame",
            "time_sec",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "path",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Output: {output_dir}")
    print(f"Person crops: {len(list(person_dir.glob('*.jpg')))}")
    print(f"Torso crops: {len(list(torso_dir.glob('*.jpg')))}")
    print(f"Tracks with crops: {len(saved_counts)}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
