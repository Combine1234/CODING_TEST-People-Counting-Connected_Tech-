from __future__ import annotations

import argparse
from pathlib import Path

import cv2


DEFAULT_INTERVAL_SECONDS = 1.0
DEFAULT_JPEG_QUALITY = 95


def project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    if script_dir.name.lower() == "scripts":
        return script_dir.parent
    return script_dir


def parse_args() -> argparse.Namespace:
    root = project_root()

    parser = argparse.ArgumentParser(
        description="Capture still images from entrance.mov at a fixed time interval."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=root / "entrance.mov",
        help="Path to the source video. Default: entrance.mov at the project root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "Dataset" / "image_capture",
        help="Folder to save captured images.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Seconds between captures. Default: 1.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        choices=range(1, 101),
        metavar="1-100",
        help="JPEG quality. Default: 95.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files if the same capture names already exist.",
    )
    return parser.parse_args()


def metadata_duration_seconds(capture: cv2.VideoCapture) -> float:
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps and fps > 0 and frame_count and frame_count > 0:
        return frame_count / fps
    return 0.0


def write_frame(
    frame,
    output_path: Path,
    quality: int,
) -> bool:
    return bool(
        cv2.imwrite(
            str(output_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
    )


def main() -> int:
    args = parse_args()

    video_path = args.video.resolve()
    output_dir = args.output.resolve()

    if args.interval <= 0:
        raise ValueError("--interval must be greater than 0")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        raise RuntimeError("Could not read video FPS from the file.")

    metadata_duration = metadata_duration_seconds(capture)
    saved_count = 0
    skipped_count = 0
    frame_index = 0
    target_index = 0
    next_capture_time = 0.0
    last_readable_time = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            current_time = frame_index / fps
            last_readable_time = current_time
            target_frame_index = int(round(next_capture_time * fps))

            if frame_index >= target_frame_index:
                timestamp_ms = int(round(next_capture_time * 1000))
                output_path = (
                    output_dir / f"entrance_{target_index:05d}_{timestamp_ms:010d}ms.jpg"
                )

                if output_path.exists() and not args.overwrite:
                    skipped_count += 1
                elif write_frame(frame, output_path, args.quality):
                    saved_count += 1
                else:
                    print(f"Warning: could not save image: {output_path}")

                target_index += 1
                next_capture_time = target_index * args.interval

            frame_index += 1
    finally:
        capture.release()

    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    print(f"Interval: {args.interval:g} seconds")
    if metadata_duration > 0:
        print(f"Metadata duration: {metadata_duration:.2f} seconds")
    print(f"Readable through: {last_readable_time:.2f} seconds")
    print(f"Saved: {saved_count} images")
    if skipped_count:
        print(f"Skipped existing files: {skipped_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
