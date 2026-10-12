import json
import math
import os
from typing import Optional, Set

# Allowed row-count gap vs video frame count. Override via BITBOX_FRAME_TOLERANCE.
DEFAULT_FRAME_TOLERANCE_ABS = 10
DEFAULT_FRAME_TOLERANCE_REL = 0.001  # 0.1% of frame count


def frame_tolerance(frame_count: int) -> int:
    override = os.environ.get("BITBOX_FRAME_TOLERANCE")
    if override is not None:
        try:
            return max(0, int(override))
        except ValueError:
            pass
    return max(DEFAULT_FRAME_TOLERANCE_ABS,
               math.ceil(frame_count * DEFAULT_FRAME_TOLERANCE_REL))


def count_data_rows(file_path: str) -> Optional[int]:
    try:
        with open(file_path, "r", errors="ignore") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return None


def probe_video_frame_count(video_path: str) -> Optional[int]:
    try:
        import cv2
    except Exception:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if frame_count <= 0:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        count = 0
        try:
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                count += 1
        finally:
            cap.release()
        frame_count = count

    if frame_count <= 0:
        return None
    return frame_count


def expected_row_counts(frame_count: int) -> Set[int]:
    tol = frame_tolerance(frame_count)
    lo = max(0, frame_count - tol)
    hi = frame_count + tol
    return set(range(lo, hi + 1))


def resolve_frame_count_source(data_path: str, fallback_input_path: Optional[str]) -> Optional[str]:
    metadata_path = data_path + ".json"
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as meta_file:
                metadata = json.load(meta_file)
        except (OSError, json.JSONDecodeError):
            metadata = None
        if metadata:
            input_path = metadata.get("input")
            if input_path and os.path.exists(input_path):
                return input_path

    if fallback_input_path and os.path.exists(fallback_input_path):
        return fallback_input_path
    return None


def frame_rows_match(data_path: str, fallback_input_path: Optional[str]) -> bool:
    if not data_path or not os.path.exists(data_path):
        return False

    input_path = resolve_frame_count_source(data_path, fallback_input_path)
    if not input_path:
        return True

    frame_count = probe_video_frame_count(input_path)
    if not frame_count:
        return True

    row_count = count_data_rows(data_path)
    if row_count is None:
        return False

    return abs(row_count - frame_count) <= frame_tolerance(frame_count)


def should_check_frame_rows(file_path: Optional[str]) -> bool:
    if not file_path:
        return False
    base = os.path.basename(file_path)
    tokens = (
        "_rects",
        "_landmarks",
        "_expression",
        "_pose",
    )
    return any(token in base for token in tokens)
