from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


LABEL_NAME = "person"
UNKNOWN_SHIRT = "unknown_shirt"


@dataclass
class Zone:
    name: str
    label: str
    color_bgr: tuple[int, int, int]
    points: np.ndarray


@dataclass
class PersonProfile:
    person_id: int
    first_frame: int
    last_frame: int
    hits: int = 0
    last_point: tuple[float, float] = (0.0, 0.0)
    hist: np.ndarray | None = None
    trail: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=50))


def project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    if script_dir.name.lower() == "scripts":
        return script_dir.parent
    return script_dir


def parse_args() -> argparse.Namespace:
    root = project_root()
    default_output_dir = root / "Dataset" / "counting_output"

    parser = argparse.ArgumentParser(
        description="Track people, count zone events, and export an annotated video."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=root / "entrance.mov",
        help="Input video path.",
    )
    parser.add_argument(
        "--model",
        default="yolov8s.pt",
        help="YOLO model name or path. Example: yolov8s.pt or runs/detect/train/weights/best.pt",
    )
    parser.add_argument(
        "--zones",
        type=Path,
        default=root / "configs" / "counting_zones.json",
        help="JSON file containing polygon zones.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Folder for annotated video, summary, and event CSV.",
    )
    parser.add_argument(
        "--output-video",
        default="entrance_counted_with_shirts.mp4",
        help="Annotated output video file name.",
    )
    parser.add_argument(
        "--shirt-classifier",
        default=None,
        help="Optional YOLO classification model for shirt class, e.g. models/shirt_classifier_best.pt.",
    )
    parser.add_argument(
        "--shirt-imgsz",
        type=int,
        default=224,
        help="Input size for shirt classification.",
    )
    parser.add_argument(
        "--shirt-min-observations",
        type=int,
        default=3,
        help="Minimum shirt classifier observations before locking a shirt label.",
    )
    parser.add_argument(
        "--shirt-confidence-threshold",
        type=float,
        default=0.55,
        help="Minimum averaged confidence before showing a shirt class instead of unknown.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=512, help="YOLO inference image size.")
    parser.add_argument(
        "--tracker",
        default="botsort.yaml",
        help="Ultralytics tracker config, for example botsort.yaml or bytetrack.yaml.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example cpu, 0, or cuda:0. Default: auto.",
    )
    parser.add_argument(
        "--process-every",
        type=int,
        default=3,
        help="Run YOLO every N frames. Lower is more accurate; higher is faster.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames. Use 0 for the full video.",
    )
    parser.add_argument(
        "--person-class-id",
        type=int,
        default=None,
        help="Override the model class id for person.",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="Minimum tracked detections before a person is counted in total.",
    )
    parser.add_argument(
        "--event-cooldown-sec",
        type=float,
        default=1.25,
        help="Minimum seconds before the same person can trigger the same event again.",
    )
    parser.add_argument(
        "--enable-reid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge short track-id breaks using a conservative color-histogram ReID.",
    )
    parser.add_argument(
        "--reid-sim-threshold",
        type=float,
        default=0.93,
        help="Higher is stricter. Used only when --enable-reid is on.",
    )
    parser.add_argument(
        "--reid-max-gap-sec",
        type=float,
        default=2.5,
        help="Max time gap for ReID merge.",
    )
    parser.add_argument(
        "--reid-max-distance",
        type=float,
        default=360.0,
        help="Max pixel distance for ReID merge.",
    )
    parser.add_argument(
        "--draw-zones",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw translucent counting zones on the output video.",
    )
    return parser.parse_args()


def resolve_model_ref(model_arg: str) -> str:
    model_path = Path(model_arg).expanduser()
    if model_path.exists():
        return str(model_path.resolve())
    return model_arg


def torso_crop(frame: np.ndarray, xyxy: list[float]) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    box_width = x2 - x1
    box_height = y2 - y1
    tx1 = max(0, min(int(round(x1 + box_width * 0.12)), width - 1))
    tx2 = max(0, min(int(round(x2 - box_width * 0.12)), width))
    ty1 = max(0, min(int(round(y1 + box_height * 0.12)), height - 1))
    ty2 = max(0, min(int(round(y1 + box_height * 0.72)), height))
    if tx2 <= tx1 or ty2 <= ty1:
        return None

    crop = frame[ty1:ty2, tx1:tx2]
    if crop.size == 0:
        return None
    return crop


def classify_shirt(
    classifier: YOLO | None,
    crop: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[str, float]:
    if classifier is None or crop is None:
        return UNKNOWN_SHIRT, 0.0

    predict_kwargs: dict[str, Any] = {
        "imgsz": args.shirt_imgsz,
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    results = classifier.predict(crop, **predict_kwargs)
    if not results or results[0].probs is None:
        return UNKNOWN_SHIRT, 0.0

    result = results[0]
    class_id = int(result.probs.top1)
    confidence = float(result.probs.top1conf)
    label = str(result.names[class_id])
    return label, confidence


def update_shirt_votes(
    shirt_votes: dict[int, dict[str, float]],
    shirt_observations: dict[int, int],
    person_id: int,
    label: str,
    confidence: float,
) -> None:
    if label == UNKNOWN_SHIRT:
        return
    shirt_votes.setdefault(person_id, {})
    shirt_votes[person_id][label] = shirt_votes[person_id].get(label, 0.0) + confidence
    shirt_observations[person_id] = shirt_observations.get(person_id, 0) + 1


def shirt_status(
    shirt_votes: dict[int, dict[str, float]],
    shirt_observations: dict[int, int],
    person_id: int,
    min_observations: int,
    confidence_threshold: float,
) -> tuple[str, float]:
    observations = shirt_observations.get(person_id, 0)
    votes = shirt_votes.get(person_id, {})
    if observations < min_observations or not votes:
        return UNKNOWN_SHIRT, 0.0

    label, score_sum = max(votes.items(), key=lambda item: item[1])
    score = score_sum / max(1, observations)
    if score < confidence_threshold:
        return UNKNOWN_SHIRT, score
    return label, score


def shirt_color(label: str) -> tuple[int, int, int]:
    normalized = label.lower()
    if "superai" in normalized:
        return (40, 240, 70)
    if "unknow" in normalized or "unknown" in normalized:
        return (120, 120, 120)
    return (40, 220, 255)


def short_shirt_label(label: str) -> str:
    normalized = label.lower()
    if "superai" in normalized:
        return "SAI"
    if "unknow" in normalized:
        return "UNK"
    return "?"


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
            "Use --person-class-id if your model uses a custom class id."
        )
    return ids


def load_zones(path: Path, frame_width: int, frame_height: int) -> dict[str, Zone]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_width, source_height = payload.get("frame_size", [frame_width, frame_height])
    scale_x = frame_width / float(source_width)
    scale_y = frame_height / float(source_height)

    zones: dict[str, Zone] = {}
    for name, zone_payload in payload["zones"].items():
        points = np.array(zone_payload["points"], dtype=np.float32)
        points[:, 0] *= scale_x
        points[:, 1] *= scale_y
        color = tuple(int(value) for value in zone_payload.get("color_bgr", [255, 255, 255]))
        zones[name] = Zone(
            name=name,
            label=zone_payload.get("label", name),
            color_bgr=color,
            points=points.astype(np.int32),
        )
    return zones


def point_in_zone(point: tuple[float, float], zone: Zone) -> bool:
    return cv2.pointPolygonTest(zone.points, point, False) >= 0


def draw_zone_overlay(frame: np.ndarray, zones: dict[str, Zone], alpha: float = 0.22) -> None:
    overlay = frame.copy()
    for zone in zones.values():
        cv2.fillPoly(overlay, [zone.points], zone.color_bgr)
        cv2.polylines(frame, [zone.points], True, zone.color_bgr, 3)
        label_point = tuple(zone.points[0].tolist())
        cv2.putText(
            frame,
            zone.label,
            (label_point[0] + 8, label_point[1] + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            zone.color_bgr,
            2,
            cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)


def box_center_bottom(xyxy: list[float]) -> tuple[float, float]:
    x1, _y1, x2, y2 = xyxy
    return (float(x1 + x2) / 2.0, float(y2))


def crop_histogram(frame: np.ndarray, xyxy: list[float]) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def hist_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-9))


def person_color(person_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(person_id * 9973)
    color = rng.integers(80, 255, size=3).tolist()
    return int(color[0]), int(color[1]), int(color[2])


def event_allowed(
    last_event_frame: dict[tuple[int, str], int],
    person_id: int,
    event_name: str,
    frame_index: int,
    cooldown_frames: int,
) -> bool:
    key = (person_id, event_name)
    previous_frame = last_event_frame.get(key, -10**9)
    if frame_index - previous_frame < cooldown_frames:
        return False
    last_event_frame[key] = frame_index
    return True


def write_events_csv(path: Path, events: list[dict[str, Any]]) -> None:
    fieldnames = ["frame", "time_sec", "person_id", "event", "from_zone", "to_zone", "x", "y"]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)


def write_shirt_csv(
    path: Path,
    profiles: dict[int, PersonProfile],
    shirt_votes: dict[int, dict[str, float]],
    shirt_observations: dict[int, int],
    args: argparse.Namespace,
) -> None:
    fieldnames = ["person_id", "shirt_label", "shirt_score", "observations", "first_frame", "last_frame", "hits"]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for person_id in sorted(profiles):
            profile = profiles[person_id]
            label, score = shirt_status(
                shirt_votes,
                shirt_observations,
                person_id,
                args.shirt_min_observations,
                args.shirt_confidence_threshold,
            )
            writer.writerow(
                {
                    "person_id": person_id,
                    "shirt_label": label,
                    "shirt_score": round(score, 4),
                    "observations": shirt_observations.get(person_id, 0),
                    "first_frame": profile.first_frame,
                    "last_frame": profile.last_frame,
                    "hits": profile.hits,
                }
            )


def draw_stats_panel(frame: np.ndarray, stats: dict[str, int], frame_index: int, fps: float) -> None:
    lines = [
        f"Time: {frame_index / fps:06.2f}s",
        f"People total: {stats['unique_people']}",
        f"Bathroom IN/OUT: {stats['bathroom_in']} / {stats['bathroom_out']}",
        f"Door IN/OUT: {stats['door_in']} / {stats['door_out']}",
        f"SuperAI/Unknown: {stats.get('superai_shirt', 0)} / {stats.get('unknown_shirt', 0)}",
        f"Active tracks: {stats['active_tracks']}",
    ]

    x, y = 18, 24
    width, height = 440, 202
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.56, frame, 0.44, 0, dst=frame)

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x + 18, y + 34 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def update_hist(existing: np.ndarray | None, new_hist: np.ndarray | None) -> np.ndarray | None:
    if new_hist is None:
        return existing
    if existing is None:
        return new_hist
    merged = existing * 0.85 + new_hist * 0.15
    norm = np.linalg.norm(merged)
    if norm > 0:
        merged = merged / norm
    return merged.astype(np.float32)


def assign_person_id(
    tracker_id: int,
    track_to_person: dict[int, int],
    profiles: dict[int, PersonProfile],
    next_person_id: int,
    point: tuple[float, float],
    hist: np.ndarray | None,
    frame_index: int,
    args: argparse.Namespace,
    fps: float,
) -> tuple[int, int]:
    if tracker_id in track_to_person:
        return track_to_person[tracker_id], next_person_id

    best_person_id: int | None = None
    best_score = 0.0
    max_gap_frames = int(round(args.reid_max_gap_sec * fps))

    if args.enable_reid:
        for person_id, profile in profiles.items():
            gap = frame_index - profile.last_frame
            if gap <= 0 or gap > max_gap_frames:
                continue

            distance = math.dist(point, profile.last_point)
            if distance > args.reid_max_distance:
                continue

            score = hist_similarity(hist, profile.hist)
            if score > best_score:
                best_score = score
                best_person_id = person_id

    if best_person_id is not None and best_score >= args.reid_sim_threshold:
        track_to_person[tracker_id] = best_person_id
        return best_person_id, next_person_id

    person_id = next_person_id
    next_person_id += 1
    track_to_person[tracker_id] = person_id
    profiles[person_id] = PersonProfile(
        person_id=person_id,
        first_frame=frame_index,
        last_frame=frame_index,
        hits=0,
        last_point=point,
        hist=hist,
    )
    return person_id, next_person_id


def main() -> int:
    args = parse_args()
    if args.process_every <= 0:
        raise ValueError("--process-every must be greater than 0")

    video_path = args.video.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    zones = load_zones(args.zones.resolve(), width, height)

    output_video_path = output_dir / args.output_video
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {output_video_path}")

    model = YOLO(resolve_model_ref(args.model))
    classes = person_class_ids(model, args.person_class_id)
    shirt_classifier = YOLO(resolve_model_ref(args.shirt_classifier)) if args.shirt_classifier else None
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

    track_to_person: dict[int, int] = {}
    profiles: dict[int, PersonProfile] = {}
    next_person_id = 1
    last_point_by_person: dict[int, tuple[float, float]] = {}
    in_bathroom_by_person: dict[int, bool] = {}
    door_zone_by_person: dict[int, str | None] = {}
    last_event_frame: dict[tuple[int, str], int] = {}
    events: list[dict[str, Any]] = []
    last_detections: list[dict[str, Any]] = []
    counts = defaultdict(int)
    shirt_votes: dict[int, dict[str, float]] = {}
    shirt_observations: dict[int, int] = {}
    cooldown_frames = int(round(args.event_cooldown_sec * fps))

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and frame_index >= args.max_frames:
            break

        active_person_ids: set[int] = set()
        if frame_index % args.process_every == 0:
            results = model.track(frame, **predict_kwargs)
            result = results[0] if results else None
            detections: list[dict[str, Any]] = []

            if result is not None and result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().tolist()
                tracker_ids = result.boxes.id.cpu().int().tolist()
                confidences = result.boxes.conf.cpu().tolist()

                for xyxy, tracker_id, confidence in zip(boxes, tracker_ids, confidences):
                    point = box_center_bottom(xyxy)
                    hist = crop_histogram(frame, xyxy)
                    person_id, next_person_id = assign_person_id(
                        int(tracker_id),
                        track_to_person,
                        profiles,
                        next_person_id,
                        point,
                        hist,
                        frame_index,
                        args,
                        fps,
                    )

                    profile = profiles[person_id]
                    profile.hits += 1
                    profile.last_frame = frame_index
                    profile.last_point = point
                    profile.hist = update_hist(profile.hist, hist)
                    profile.trail.append((int(point[0]), int(point[1])))
                    active_person_ids.add(person_id)

                    if shirt_classifier is not None:
                        crop = torso_crop(frame, xyxy)
                        shirt_label, shirt_confidence = classify_shirt(
                            shirt_classifier,
                            crop,
                            args,
                        )
                        update_shirt_votes(
                            shirt_votes,
                            shirt_observations,
                            person_id,
                            shirt_label,
                            shirt_confidence,
                        )

                    previous_point = last_point_by_person.get(person_id)
                    current_bathroom = point_in_zone(point, zones["bathroom"])
                    previous_bathroom = in_bathroom_by_person.get(person_id)
                    if previous_bathroom is None:
                        in_bathroom_by_person[person_id] = current_bathroom
                    elif current_bathroom != previous_bathroom:
                        dx = 0.0 if previous_point is None else point[0] - previous_point[0]
                        if current_bathroom and dx < -8:
                            event_name = "bathroom_in"
                            if event_allowed(
                                last_event_frame,
                                person_id,
                                event_name,
                                frame_index,
                                cooldown_frames,
                            ):
                                counts[event_name] += 1
                                events.append(
                                    {
                                        "frame": frame_index,
                                        "time_sec": round(frame_index / fps, 3),
                                        "person_id": person_id,
                                        "event": event_name,
                                        "from_zone": "outside",
                                        "to_zone": "bathroom",
                                        "x": round(point[0], 2),
                                        "y": round(point[1], 2),
                                    }
                                )
                        elif (not current_bathroom) and dx > 8:
                            event_name = "bathroom_out"
                            if event_allowed(
                                last_event_frame,
                                person_id,
                                event_name,
                                frame_index,
                                cooldown_frames,
                            ):
                                counts[event_name] += 1
                                events.append(
                                    {
                                        "frame": frame_index,
                                        "time_sec": round(frame_index / fps, 3),
                                        "person_id": person_id,
                                        "event": event_name,
                                        "from_zone": "bathroom",
                                        "to_zone": "outside",
                                        "x": round(point[0], 2),
                                        "y": round(point[1], 2),
                                    }
                                )
                        in_bathroom_by_person[person_id] = current_bathroom

                    current_door_zone = None
                    if point_in_zone(point, zones["door_inside"]):
                        current_door_zone = "inside"
                    elif point_in_zone(point, zones["door_outside"]):
                        current_door_zone = "outside"

                    previous_door_zone = door_zone_by_person.get(person_id)
                    if (
                        current_door_zone is not None
                        and previous_door_zone is not None
                        and current_door_zone != previous_door_zone
                    ):
                        if previous_door_zone == "outside" and current_door_zone == "inside":
                            event_name = "door_in"
                        elif previous_door_zone == "inside" and current_door_zone == "outside":
                            event_name = "door_out"
                        else:
                            event_name = ""

                        if event_name and event_allowed(
                            last_event_frame,
                            person_id,
                            event_name,
                            frame_index,
                            cooldown_frames,
                        ):
                            counts[event_name] += 1
                            events.append(
                                {
                                    "frame": frame_index,
                                    "time_sec": round(frame_index / fps, 3),
                                    "person_id": person_id,
                                    "event": event_name,
                                    "from_zone": previous_door_zone,
                                    "to_zone": current_door_zone,
                                    "x": round(point[0], 2),
                                    "y": round(point[1], 2),
                                }
                            )

                    if current_door_zone is not None:
                        door_zone_by_person[person_id] = current_door_zone

                    last_point_by_person[person_id] = point
                    current_shirt_label, current_shirt_score = shirt_status(
                        shirt_votes,
                        shirt_observations,
                        person_id,
                        args.shirt_min_observations,
                        args.shirt_confidence_threshold,
                    )
                    detections.append(
                        {
                            "xyxy": xyxy,
                            "person_id": person_id,
                            "tracker_id": int(tracker_id),
                            "confidence": float(confidence),
                            "shirt_label": current_shirt_label,
                            "shirt_score": current_shirt_score,
                        }
                    )

            last_detections = detections

        annotated = frame.copy()
        if args.draw_zones:
            draw_zone_overlay(annotated, zones)

        for detection in last_detections:
            x1, y1, x2, y2 = [int(round(value)) for value in detection["xyxy"]]
            person_id = detection["person_id"]
            color = (
                shirt_color(detection["shirt_label"])
                if shirt_classifier is not None
                else person_color(person_id)
            )
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            shirt_text = ""
            if shirt_classifier is not None:
                shirt_text = f" {short_shirt_label(detection['shirt_label'])} {detection['shirt_score']:.2f}"
            label = f"P{person_id} T{detection['tracker_id']} {detection['confidence']:.2f}{shirt_text}"
            cv2.putText(
                annotated,
                label,
                (x1, max(26, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

            trail = profiles.get(person_id).trail if person_id in profiles else []
            for p1, p2 in zip(list(trail), list(trail)[1:]):
                cv2.line(annotated, p1, p2, color, 2)

        confirmed_people = sum(1 for profile in profiles.values() if profile.hits >= args.min_hits)
        shirt_counts = defaultdict(int)
        for person_id, profile in profiles.items():
            if profile.hits < args.min_hits:
                continue
            label, _score = shirt_status(
                shirt_votes,
                shirt_observations,
                person_id,
                args.shirt_min_observations,
                args.shirt_confidence_threshold,
            )
            shirt_counts[label] += 1
        stats = {
            "unique_people": confirmed_people,
            "bathroom_in": counts["bathroom_in"],
            "bathroom_out": counts["bathroom_out"],
            "door_in": counts["door_in"],
            "door_out": counts["door_out"],
            "superai_shirt": shirt_counts["Superai_Shirt"],
            "unknown_shirt": shirt_counts["Unknow_Shirt"] + shirt_counts[UNKNOWN_SHIRT],
            "active_tracks": len(active_person_ids),
        }
        draw_stats_panel(annotated, stats, frame_index, fps)
        writer.write(annotated)

        if frame_index and frame_index % 300 == 0:
            print(f"Processed {frame_index}/{total_frames or '?'} frames")

        frame_index += 1

    cap.release()
    writer.release()

    confirmed_people = sum(1 for profile in profiles.values() if profile.hits >= args.min_hits)
    final_shirt_counts = defaultdict(int)
    shirt_details: dict[str, dict[str, Any]] = {}
    for person_id, profile in sorted(profiles.items()):
        if profile.hits < args.min_hits:
            continue
        label, score = shirt_status(
            shirt_votes,
            shirt_observations,
            person_id,
            args.shirt_min_observations,
            args.shirt_confidence_threshold,
        )
        final_shirt_counts[label] += 1
        shirt_details[str(person_id)] = {
            "shirt_label": label,
            "shirt_score": round(score, 4),
            "shirt_observations": shirt_observations.get(person_id, 0),
            "hits": profile.hits,
        }

    summary = {
        "video": str(video_path),
        "output_video": str(output_video_path),
        "model": args.model,
        "shirt_classifier": args.shirt_classifier,
        "tracker": args.tracker,
        "processed_frames": frame_index,
        "fps": fps,
        "process_every": args.process_every,
        "imgsz": args.imgsz,
        "unique_people": confirmed_people,
        "raw_tracker_or_reid_profiles": len(profiles),
        "bathroom_in": counts["bathroom_in"],
        "bathroom_out": counts["bathroom_out"],
        "door_in": counts["door_in"],
        "door_out": counts["door_out"],
        "events": len(events),
        "zones_file": str(args.zones.resolve()),
        "shirts": {
            "Superai_Shirt": final_shirt_counts["Superai_Shirt"],
            "Unknow_Shirt": final_shirt_counts["Unknow_Shirt"],
            UNKNOWN_SHIRT: final_shirt_counts[UNKNOWN_SHIRT],
        },
        "shirt_persons": shirt_details,
    }
    (output_dir / "count_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_events_csv(output_dir / "count_events.csv", events)
    if shirt_classifier is not None:
        write_shirt_csv(
            output_dir / "shirt_person_summary.csv",
            profiles,
            shirt_votes,
            shirt_observations,
            args,
        )

    print(f"Output video: {output_video_path}")
    print(f"Summary: {output_dir / 'count_summary.json'}")
    print(f"Events: {output_dir / 'count_events.csv'}")
    if shirt_classifier is not None:
        print(f"Shirts: {output_dir / 'shirt_person_summary.csv'}")
    print(f"Unique people: {confirmed_people}")
    print(f"Bathroom IN/OUT: {counts['bathroom_in']} / {counts['bathroom_out']}")
    print(f"Door IN/OUT: {counts['door_in']} / {counts['door_out']}")
    if shirt_classifier is not None:
        print(
            "Shirt SuperAI / Unknow / Unknown: "
            f"{final_shirt_counts['Superai_Shirt']} / "
            f"{final_shirt_counts['Unknow_Shirt']} / "
            f"{final_shirt_counts[UNKNOWN_SHIRT]}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
