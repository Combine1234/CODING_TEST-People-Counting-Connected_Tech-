from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASSES = ["Superai_Shirt", "Unknow_Shirt"]


def project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    if script_dir.name.lower() == "scripts":
        return script_dir.parent
    return script_dir


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Build a YOLO classification dataset and train a shirt classifier."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "Dataset" / "shirt_crop_dataset",
        help="Folder containing class folders such as Superai_Shirt and Unknow_Shirt.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "Dataset" / "shirt_cls_dataset",
        help="Output YOLO classification dataset folder.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Class folder names to use.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n-cls.pt",
        help="YOLO classification model. Uses yolov8n-cls.pt by default.",
    )
    parser.add_argument(
        "--fallback-model",
        default="yolov8n-cls.yaml",
        help="Fallback model if pretrained weights are not available.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--project",
        type=Path,
        default=root / "runs" / "classify",
        help="Ultralytics project folder.",
    )
    parser.add_argument("--name", default="shirt_yolov8n_cls")
    parser.add_argument(
        "--best-output",
        type=Path,
        default=root / "models" / "shirt_classifier_best.pt",
        help="Where to copy the trained best.pt.",
    )
    parser.add_argument(
        "--overwrite-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rebuild the train/val split folder.",
    )
    return parser.parse_args()


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: str(path).lower(),
    )


def safe_clear(path: Path, allowed_parent: Path) -> None:
    path = path.resolve()
    allowed_parent = allowed_parent.resolve()
    if allowed_parent not in path.parents and path != allowed_parent:
        raise RuntimeError(f"Refusing to clear unexpected folder: {path}")
    if path.exists():
        shutil.rmtree(path)


def copy_split(
    image_paths: list[Path],
    train_dir: Path,
    val_dir: Path,
    class_name: str,
    val_ratio: float,
    rng: random.Random,
) -> dict[str, int]:
    shuffled = image_paths[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
    val_set = set(shuffled[:val_count])

    counts = {"train": 0, "val": 0}
    for source_path in shuffled:
        split = "val" if source_path in val_set else "train"
        destination_base = val_dir if split == "val" else train_dir
        destination_dir = destination_base / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / source_path.name
        if destination_path.exists():
            destination_path = destination_dir / f"{source_path.stem}_{abs(hash(source_path))}{source_path.suffix}"
        shutil.copy2(source_path, destination_path)
        counts[split] += 1
    return counts


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = args.source.resolve()
    dataset_dir = args.dataset.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")
    if not 0 < args.val_ratio < 0.5:
        raise ValueError("--val-ratio must be > 0 and < 0.5")

    if args.overwrite_split:
        safe_clear(dataset_dir, source_dir.parent)

    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    rng = random.Random(args.seed)
    summary: dict[str, Any] = {"source": str(source_dir), "dataset": str(dataset_dir), "classes": {}}

    for class_name in args.classes:
        class_dir = source_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")
        images = list_images(class_dir)
        if not images:
            raise RuntimeError(f"No images found in: {class_dir}")
        counts = copy_split(images, train_dir, val_dir, class_name, args.val_ratio, rng)
        summary["classes"][class_name] = {"source": len(images), **counts}

    summary_path = dataset_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def load_model(model_name: str, fallback_model: str) -> YOLO:
    try:
        return YOLO(model_name)
    except Exception as exc:
        print(f"Warning: could not load {model_name}: {exc}")
        print(f"Falling back to {fallback_model}")
        return YOLO(fallback_model)


def train(args: argparse.Namespace) -> dict[str, Any]:
    dataset_summary = build_dataset(args)
    dataset_dir = args.dataset.resolve()
    model = load_model(args.model, args.fallback_model)

    results = model.train(
        data=str(dataset_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        seed=args.seed,
        patience=max(5, min(12, args.epochs // 2)),
        verbose=True,
    )

    save_dir = Path(getattr(results, "save_dir", args.project.resolve() / args.name)).resolve()
    best_path = save_dir / "weights" / "best.pt"
    last_path = save_dir / "weights" / "last.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Training finished but best.pt was not found: {best_path}")

    args.best_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, args.best_output)

    metadata = {
        "dataset": dataset_summary,
        "model": args.model,
        "fallback_model": args.fallback_model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "run_dir": str(save_dir),
        "best": str(best_path),
        "last": str(last_path) if last_path.exists() else None,
        "copied_best": str(args.best_output.resolve()),
    }
    metadata_path = args.best_output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    args = parse_args()
    metadata = train(args)
    print(f"Dataset: {metadata['dataset']['dataset']}")
    print(f"Run dir: {metadata['run_dir']}")
    print(f"Best model: {metadata['copied_best']}")
    print(f"Metadata: {Path(metadata['copied_best']).with_suffix('.metadata.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
