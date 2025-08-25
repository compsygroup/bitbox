"""Bitbox visualization utilities.

Provides utilities to render interactive visualizations used across Bitbox:
- Frame extraction and robust cropping around rectangles or landmarks
- Grids of cropped images with rectangle and/or landmark overlays
- Video + canonicalized landmark plots
- Video + 3D pose axes plots with export helpers
- Expressions-over-time 3D plot synchronized with video frames

"""
from __future__ import annotations
# -----------------------------------------------------------------------------
# Imports 
# -----------------------------------------------------------------------------
import base64
import json
import os
from string import Template
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os, traceback

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
TARGET_SIZE: Tuple[int, int] = (240, 300)

# -----------------------------------------------------------------------------
# Common Helpers (I/O, cropping, sampling)
# -----------------------------------------------------------------------------

def get_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Load a single RGB frame by 0-based index from a video on disk.

    Raises RuntimeError if the frame cannot be read.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Frame {frame_idx} not found in {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def crop_and_scale(
    frame: np.ndarray,
    x: Optional[int] = None,
    y: Optional[int] = None,
    w: Optional[int] = None,
    h: Optional[int] = None,
    xs: Optional[np.ndarray] = None,
    ys: Optional[np.ndarray] = None,
    cushion_ratio: float = 0.0,
    target_size: Tuple[int, int] = TARGET_SIZE,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Tuple[int, int, float, float]]:
    """Crop frame around a rectangle or landmarks with cushion and scale to target size.

    If (x, y, w, h) are provided, crops around the rectangle.
    If (xs, ys) are provided, crops around the tight bounding box of landmarks.
    Returns: (resized crop, xs_crop or scaled box, ys_crop or None, (x1, y1, scale_x, scale_y))
    """
    if xs is not None and ys is not None:
        # Crop around landmarks
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        w, h = max_x - min_x, max_y - min_y

        # Guard degenerate extents
        if w <= 0 or h <= 0:
            return cv2.resize(frame, target_size), xs.copy(), ys.copy(), (0, 0, target_size[0] / frame.shape[1], target_size[1] / frame.shape[0])

        cushion_w = int(cushion_ratio * w)
        cushion_h = int(cushion_ratio * h)
        x1 = max(int(min_x) - cushion_w, 0)
        y1 = max(int(min_y) - cushion_h, 0)
        x2 = min(int(max_x) + cushion_w, frame.shape[1])
        y2 = min(int(max_y) + cushion_h, frame.shape[0])

        frame_crop = frame[y1:y2, x1:x2]
        orig_h, orig_w = frame_crop.shape[:2]
        if orig_h == 0 or orig_w == 0:
            frame_crop = frame
            x1, y1 = 0, 0
            orig_h, orig_w = frame.shape[:2]

        scale_x = target_size[0] / orig_w
        scale_y = target_size[1] / orig_h
        frame_crop_resized = cv2.resize(frame_crop, target_size)

        xs_crop = (xs - x1) * scale_x
        ys_crop = (ys - y1) * scale_y
        return frame_crop_resized, xs_crop, ys_crop, (x1, y1, scale_x, scale_y)

    elif x is not None and y is not None and w is not None and h is not None:
        # Crop around rectangle
        cushion = int(cushion_ratio * w)
        x1 = max(x - cushion, 0)
        y1 = max(y - cushion, 0)
        x2 = min(x + w + cushion, frame.shape[1])
        y2 = min(y + h + cushion, frame.shape[0])
        frame_crop = frame[y1:y2, x1:x2]

        orig_h, orig_w = frame_crop.shape[:2]
        if orig_h == 0 or orig_w == 0:
            # Fallback to full frame to avoid errors
            frame_crop = frame
            x1, y1 = 0, 0
            orig_h, orig_w = frame.shape[:2]

        scale_x = target_size[0] / orig_w
        scale_y = target_size[1] / orig_h
        frame_crop_resized = cv2.resize(frame_crop, target_size)

        # rect in crop-relative coordinates, then scaled
        box = (x - x1, y - y1, w, h)
        box_scaled = tuple(np.array(box) * np.array([scale_x, scale_y, scale_x, scale_y]))
        return frame_crop_resized, box_scaled, None, (x1, y1, scale_x, scale_y)

    else:
        raise ValueError("Either (x, y, w, h) or (xs, ys) must be provided.")

# -----------------------------------------------------------------------------
# Rectangle/landmark sampling helpers
# -----------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, candidates):
    """Case-insensitive column resolver; candidates can be a str or list/tuple of names."""
    if isinstance(candidates, str):
        candidates = [candidates]
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        hit = lower_map.get(str(name).lower())
        if hit is not None:
            return hit
    return None

def select_diverse_frames_maxmin(
    pose_df: pd.DataFrame,
    k: int,
    x_col_candidates=("Rx", "rx", "pitch"),
    y_col_candidates=("Ry", "ry", "yaw"),
    prefer_extremes=True,
    random_fallback=False,
) -> np.ndarray:
    """
    Greedy farthest-point sampling on (x,y) to pick k diverse rows from pose_df.
    Returns sorted row indices (frame numbers).
    """
    xcol = _find_col(pose_df, x_col_candidates)
    ycol = _find_col(pose_df, y_col_candidates)
    if xcol is None or ycol is None:
        raise KeyError(
            f"Could not find angle columns among {x_col_candidates} and {y_col_candidates}. "
            f"Have: {list(pose_df.columns)}"
        )

    xy = pose_df[[xcol, ycol]].to_numpy(dtype=float)
    n = xy.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    if k >= n:
        return np.arange(n, dtype=int)

    # z-score so pitch and yaw have equal weight
    mu = xy.mean(axis=0, keepdims=True)
    sd = xy.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    z = (xy - mu) / sd

    # seed
    r2 = (z ** 2).sum(axis=1)
    seed = int(np.argmax(r2) if prefer_extremes else np.argmin(r2))

    selected = [seed]
    dists = np.linalg.norm(z - z[seed], axis=1)

    # greedy add
    for _ in range(1, k):
        nxt = int(np.argmax(dists))
        if dists[nxt] == 0 and random_fallback:
            # degenerate duplicates case
            unselected = np.setdiff1d(np.arange(n), np.array(selected))
            if len(unselected) == 0:
                break
            nxt = int(np.random.choice(unselected))
        selected.append(nxt)
        dists = np.minimum(dists, np.linalg.norm(z - z[nxt], axis=1))

    return np.array(sorted(selected), dtype=int)


def euler_to_rotmat(rx: float, ry: float, rz: float) -> np.ndarray:
    """Compose rotation matrix from Euler angles in radians.

    The angles correspond to intrinsic rotations about X (pitch), then Y (yaw),
    then Z (roll); composition order: Rx @ Ry @ Rz. Returns a 3x3 ndarray.
    """
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz

# -----------------------------------------------------------------------------
# image grid visualizations (no/with 3D)
# -----------------------------------------------------------------------------

def visualize_and_export(
    num_frames: int,
    video_path: Optional[str],
    out_dir: str,
    random_seed: int,
    rects: Optional[dict] = None,          #  OPTIONAL
    cushion_ratio: float = 0.35,
    overlay: Optional[dict] = None,        # Can be landmarks OR rectangles
    pose: Optional[dict] = None,
    video: bool = False,                   # NEW: when True, embed video with overlays instead of frame grid
):
    """Flexible rectangles/landmarks plotter.

    Uses pose to pick diverse frames for the image grid (falls back to random if pose is missing).
    When video=True and a video_path is given, renders a single video player with overlays instead of a grid.
    """
    # -------------------------
    # Helpers
    # -------------------------
    def _dtype(d: Optional[dict]) -> Optional[str]:
        return d.get("type") if isinstance(d, dict) else None

    def _safe_df(d: Optional[dict]):
        return d["data"] if isinstance(d, dict) and "data" in d else None

    def _pick(a, b, want: str) -> Optional[dict]:
        return a if _dtype(a) == want else (b if _dtype(b) == want else None)

    # Resolve sources
    rects_src = _pick(rects, overlay, "rectangle")
    lands_src = _pick(overlay, rects, "landmark")

    df = _safe_df(rects_src)              # rectangles df, may be None
    overlay_df = _safe_df(lands_src)      # landmarks df, may be None

    pose_df = _safe_df(pose)

    #  accept landmark-only input (no rectangles)
    if (df is None or len(df) == 0) and (overlay_df is None or len(overlay_df) == 0):
        raise ValueError("No rectangles or landmarks found. Provide rectangles via `rects`/`overlay` or landmarks via `overlay`/`rects`.")

    # -------------------------
    # NEW: Video mode (embed HTML5 video + canvas overlays)
    # -------------------------
    if video and video_path:
        try:
            # Collect overlay maps (frame -> data)
            rects_map = {}
            if df is not None and len(df) > 0:
                for fid, row in df.iterrows():
                    # Support different rectangle column naming; skip on failure
                    try:
                        if all(k in row.index for k in ("x", "y", "w", "h")):
                            x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
                        elif all(k in row.index for k in ("left", "top", "width", "height")):
                            x, y, w, h = int(row["left"]), int(row["top"]), int(row["width"]), int(row["height"])
                        elif all(k in row.index for k in ("x1", "y1", "x2", "y2")):
                            x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
                            x, y, w, h = x1, y1, max(1, x2 - x1), max(1, y2 - y1)
                        else:
                            # fallback: first four numeric values
                            vals = [v for v in row.values if np.issubdtype(type(v), np.number)]
                            if len(vals) >= 4:
                                x, y, w, h = map(int, vals[:4])
                            else:
                                continue
                        rects_map.setdefault(int(fid), []).append({"x": x, "y": y, "w": w, "h": h})
                    except Exception:
                        continue  # skip malformed row

            lands_map = {}
            if overlay_df is not None and len(overlay_df) > 0:
                for fid, row in overlay_df.iterrows():
                    try:
                        vals = np.asarray(row.values, dtype=float)
                        if vals.size % 2 != 0 or vals.size == 0:
                            continue
                        xs = vals[::2]
                        ys = vals[1::2]
                        lands_map[int(fid)] = [[float(a), float(b)] for a, b in zip(xs, ys)]
                    except Exception:
                        continue

            # Probe video properties (fps, size)
            fps = 30.0
            vw = vh = None
            try:
                cap = cv2.VideoCapture(video_path)
                fps_cap = cap.get(cv2.CAP_PROP_FPS)
                if fps_cap and fps_cap > 1e-3:
                    fps = float(fps_cap)
                vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
                vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
                cap.release()
            except Exception:
                pass

            html_path = os.path.join(out_dir, "bitbox_viz.html")
            os.makedirs(out_dir or ".", exist_ok=True)
            write_video_overlay_html(
                video_src=video_path,
                out_path=html_path,
                rects_map=rects_map if rects_map else None,
                lands_map=lands_map if lands_map else None,
                fps=fps,
                video_w=vw,
                video_h=vh,
            )
            return
        except Exception as e:
            video_exists = bool(video_path and os.path.isfile(video_path))
            video_size = os.path.getsize(video_path) if video_exists else 0
            rect_rows = int(len(df)) if df is not None else 0
            land_rows = int(len(overlay_df)) if overlay_df is not None else 0
            rect_sample = []
            land_sample = []
            try:
                rect_sample = list(sorted(rects_map.keys()))[:5] if 'rects_map' in locals() and rects_map else []
            except Exception:
                pass
            try:
                land_sample = list(sorted(lands_map.keys()))[:5] if 'lands_map' in locals() and lands_map else []
            except Exception:
                pass

            context = {
            "video_path": video_path,
            "video_exists": video_exists,
            "video_size_bytes": video_size,
            "fps_detected": fps,
            "rect_rows": rect_rows,
            "landmark_rows": land_rows,
            "rect_frame_sample": rect_sample,
            "landmark_frame_sample": land_sample,
            "out_dir": out_dir,
            }
            print(
            "[bitbox.visualize_and_export] Video overlay mode failed "
            f"({type(e).__name__}: {e}). Falling back to frame grid. Context: {context}"
            )
            if os.environ.get("BITBOX_DEBUG"):
                traceback.print_exc()
            # (Fallback proceeds to grid logic)

    # -------------------------
    # Choose frames: use pose if provided, else random
    # -------------------------
    src_df = df if (df is not None and len(df) > 0) else overlay_df
    frame_ids: List[int] = []

    if pose_df is not None and len(pose_df) > 0:
        k = min(num_frames, len(src_df), len(pose_df))
        if k > 0:
            try:
                pos_idxs = select_diverse_frames_maxmin(
                    pose_df, k=k,
                    x_col_candidates=("Rx", "rx", "pitch"),
                    y_col_candidates=("Ry", "ry", "yaw"),
                    prefer_extremes=True
                )
                candidate_fids = [int(pose_df.index[i]) for i in pos_idxs]
                src_index_set = set(map(int, src_df.index.tolist()))
                frame_ids = [fid for fid in candidate_fids if fid in src_index_set]
            except Exception:
                frame_ids = []

    if not frame_ids:
        n = min(num_frames, len(src_df))
        if n <= 0:
            return
        sampled = src_df.sample(n=n, random_state=random_seed)
        frame_ids = [int(i) for i in sampled.index.tolist()]

    # -------------------------
    # Prep crops + overlays
    # -------------------------
    crops: List[np.ndarray] = []
    blurred_crops: List[np.ndarray] = []
    rel_boxes: List[Tuple[float, float, float, float]] = []  # only filled when rectangles exist
    overlay_landmarks: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []

    use_rectangles = df is not None and len(df) > 0

    for fid in frame_ids:
        if not video_path:
            overlay_landmarks.append(None)
            continue
        try:
            frame = get_frame(video_path, fid)
        except Exception:
            overlay_landmarks.append(None)
            continue

        if use_rectangles:
            # Rectangle-driven crop
            if fid in df.index:
                row = df.loc[fid]
            else:
                try:
                    row = df.iloc[fid]
                except Exception:
                    overlay_landmarks.append(None)
                    continue
            try:
                x, y, w, h = map(int, [row["x"], row["y"], row["w"], row["h"]])
            except Exception:
                try:
                    x, y, w, h = map(int, row.values[:4])
                except Exception:
                    overlay_landmarks.append(None)
                    continue

            crop, box_scaled, _unused, (x1, y1, scale_x, scale_y) = crop_and_scale(
                frame, x=x, y=y, w=w,h=h, cushion_ratio=cushion_ratio
            )
            crops.append(crop)
            blurred_crops.append(cv2.GaussianBlur(crop, (max(31, (crop.shape[1] // 10) * 2 + 1), max(31, (crop.shape[0] // 10) * 2 + 1)), 0))
            rel_boxes.append(box_scaled)

            # Landmarks overlay if available
            if overlay_df is not None and fid in overlay_df.index:
                lmk_row = overlay_df.loc[fid]
                xs = lmk_row.values[::2].astype(float)
                ys = lmk_row.values[1::2].astype(float)
                xs_crop = (xs - x1) * scale_x
                ys_crop = (ys - y1) * scale_y
                overlay_landmarks.append((xs_crop, ys_crop))
            else:
                overlay_landmarks.append(None)
        else:
            # Landmark-driven crop (no rectangles)
            if overlay_df is None or fid not in overlay_df.index:
                overlay_landmarks.append(None)
                continue
            lmk_row = overlay_df.loc[fid]
            xs = lmk_row.values[::2].astype(float)
            ys = lmk_row.values[1::2].astype(float)
            crop, xs_crop, ys_crop, (_x1, _y1, _sx, _sy) = crop_and_scale(
                frame, xs=xs, ys=ys, cushion_ratio=cushion_ratio
            )
            crops.append(crop)
            blurred_crops.append(cv2.GaussianBlur(crop, (max(31, (crop.shape[1] // 10) * 2 + 1), max(31, (crop.shape[0] // 10) * 2 + 1)), 0))
            overlay_landmarks.append((xs_crop, ys_crop))
            # No rectangles, so do not append to rel_boxes

    if not crops:
        return

    # -------------------------
    # Images-only grid (no 3D plotting)
    # -------------------------
    ncols = len(crops)
    if use_rectangles:
        # rectangles as main, landmarks as overlay
        fig = make_centered_subplot_with_overlay(
            crops=crops,
            main_items=rel_boxes,
            overlay_items=overlay_landmarks,
            main_type="rect",
            ncols=ncols,
            blurred_crops=blurred_crops,
        )
    else:
        # landmarks as main, no rectangle overlays
        fig = make_centered_subplot_with_overlay(
            crops=crops,
            main_items=overlay_landmarks,
            overlay_items=None,
            main_type="landmark",
            ncols=ncols,
            blurred_crops=blurred_crops,
        )

    html_path = os.path.join(out_dir, "bitbox_viz.html")
    try:
        # No pose-only export button for this plot
        write_centered_html(fig, html_path, export_filename="bitbox visualizations")
    except Exception:
        pass

def visualize_and_export_can_land(
    num_frames: int,
    out_dir: str,
    overlay: Optional[dict] = None,    # can be landmarks, rectangles, or [landmark, rectangle, {...privacy...}]
    video_path: Optional[str] = None,
    pose: Optional[dict] = None,       # used only to diversify selection in static grid
    land_can: Optional[dict] = None,   # canonicalized landmarks (frame-indexed DataFrame expected)
    video: bool = False,               # when True: left = video, right = live 3D canonicalized landmarks
):
    """
    When video=False (default): multi-column static frame row + 3D canonicalized landmarks row.
    When video=True  : synchronized video (left) + a single dynamic 3D canonical landmark plot (right),
                       generated by write_video_overlay_html().
    """
    # ---- helpers ----
    def _safe_df(d: Optional[dict]):
        return d["data"] if isinstance(d, dict) and "data" in d else None

    def _dtype(d: Optional[dict]) -> Optional[str]:
        return d.get("type") if isinstance(d, dict) else None

    def _find_col(df, candidates):
        # graceful local fallback if caller doesn't define _find_col elsewhere
        try:
            # exact match first
            for c in candidates:
                if c in df.columns:
                    return c
            # case-insensitive fallback
            low = {c.lower(): c for c in df.columns}
            for c in candidates:
                if c.lower() in low:
                    return low[c.lower()]
        except Exception:
            pass
        return None

    # Resolve inputs
    can_df = _safe_df(land_can)
    pose_df = _safe_df(pose)

    # ---- resolve overlay variants into two DataFrames (optional) ----
    overlay_land_df = None
    overlay_rect_df = None
    blur_default = False  # static grid only

    if isinstance(overlay, list):
        for value in overlay:
            if isinstance(value, dict) and "type" in value:
                if _dtype(value) == "landmark":
                    overlay_land_df = _safe_df(value)
                elif _dtype(value) == "rectangle":
                    overlay_rect_df = _safe_df(value)
            elif isinstance(value, dict):
                if bool(value.get("blur")) or (value.get("privacy") in (True, "blur", "on")):
                    blur_default = True
    elif isinstance(overlay, dict) and "type" in overlay:
        if _dtype(overlay) == "landmark":
            overlay_land_df = _safe_df(overlay)
        elif _dtype(overlay) == "rectangle":
            overlay_rect_df = _safe_df(overlay)
    elif isinstance(overlay, dict):
        if bool(overlay.get("blur")) or (overlay.get("privacy") in (True, "blur", "on")):
            blur_default = True

    # =========================
    # VIDEO MODE: delegate to write_video_overlay_html with can3d_map
    # =========================
    if video and video_path:
        # Build per-frame maps for rectangles, 2D landmarks, and canonical 3D
        rects_map = {}
        if overlay_rect_df is not None and len(overlay_rect_df) > 0:
            for fid, row in overlay_rect_df.iterrows():
                try:
                    if all(k in row.index for k in ("x", "y", "w", "h")):
                        x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
                    elif all(k in row.index for k in ("left", "top", "width", "height")):
                        x, y, w, h = int(row["left"]), int(row["top"]), int(row["width"]), int(row["height"])
                    elif all(k in row.index for k in ("x1", "y1", "x2", "y2")):
                        x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
                        x, y, w, h = x1, y1, max(1, x2 - x1), max(1, y2 - y1)
                    else:
                        vals = [v for v in row.values if np.issubdtype(type(v), np.number)]
                        if len(vals) < 4:
                            continue
                        x, y, w, h = map(int, vals[:4])
                    rects_map.setdefault(int(fid), []).append({"x": x, "y": y, "w": w, "h": h})
                except Exception:
                    continue

        lands_map = {}
        if overlay_land_df is not None and len(overlay_land_df) > 0:
            for fid, row in overlay_land_df.iterrows():
                try:
                    vals = np.asarray(row.values, dtype=float)
                    if vals.size % 2 != 0 or vals.size == 0:
                        continue
                    xs = vals[::2]
                    ys = vals[1::2]
                    lands_map[int(fid)] = [[float(a), float(b)] for a, b in zip(xs, ys)]
                except Exception:
                    continue

        can3d_map = {}
        if can_df is not None and len(can_df) > 0:
            for fid, row in can_df.iterrows():
                vals = np.asarray(getattr(row, "values", row), dtype=float)
                if vals.size % 3 == 0:
                    xyz = vals.reshape(-1, 3)
                elif vals.size % 2 == 0:  # pad Z with zeros if only XY provided
                    xy = vals.reshape(-1, 2)
                    xyz = np.c_[xy, np.zeros((xy.shape[0],), dtype=float)]
                else:
                    continue
                can3d_map[int(fid)] = xyz.tolist()

        # Probe video properties
        fps = 30.0
        vw = vh = None
        try:
            cap = cv2.VideoCapture(video_path)
            fps_cap = cap.get(cv2.CAP_PROP_FPS)
            if fps_cap and fps_cap > 1e-3:
                fps = float(fps_cap)
            vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
            vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
            cap.release()
        except Exception:
            pass

        os.makedirs(out_dir or ".", exist_ok=True)
        html_path = os.path.join(out_dir, "bitbox_viz.html")

        # NOTE: requires the upgraded write_video_overlay_html that accepts can3d_map
        write_video_overlay_html(
            video_src=video_path,
            out_path=html_path,
            rects_map=rects_map or None,
            lands_map=lands_map or None,
            can3d_map=can3d_map or None,
            fps=fps,
            video_w=vw,
            video_h=vh,
            title="Video + 3D Canonicalized Landmarks",
            cushion_ratio=0.35,
            fixed_portrait=True,
            portrait_w=360,
            portrait_h=480,
            can3d_decimate=1,   # bump to 2/3 if perf is tight
        )
        return

    # =========================
    # STATIC GRID MODE
    # =========================
    # Require canonicalized landmarks
    if can_df is None or len(can_df) == 0:
        return

    # resolve pose angle columns (for labels under land_can when pose is provided)
    rx_col = ry_col = rz_col = None
    if pose_df is not None and len(pose_df) > 0:
        rx_col = _find_col(pose_df, ("Rx", "rx", "pitch"))
        ry_col = _find_col(pose_df, ("Ry", "ry", "yaw"))
        rz_col = _find_col(pose_df, ("Rz", "rz", "roll"))

    # Select frames by pose diversity if possible, else evenly across canonical rows
    selected_fids: List[int] = []
    if pose_df is not None and len(pose_df) > 0:
        k = max(1, min(num_frames, len(pose_df)))
        try:
            pos_idxs = select_diverse_frames_maxmin(
                pose_df, k=k,
                x_col_candidates=("Rx", "rx", "pitch"),
                y_col_candidates=("Ry", "ry", "yaw"),
                prefer_extremes=True
            )
            candidate_fids = [int(pose_df.index[i]) for i in pos_idxs]
        except Exception:
            idxs = np.linspace(0, len(pose_df) - 1, num=k, dtype=int)
            candidate_fids = [int(pose_df.index[i]) for i in idxs]
        idx_can = set(map(int, can_df.index.tolist()))
        selected_fids = [fid for fid in candidate_fids if fid in idx_can]

    if not selected_fids:
        total = len(can_df)
        k = max(1, min(num_frames, total))
        ilocs = np.unique(np.linspace(0, total - 1, num=k, dtype=int)).tolist()
        selected_fids = [int(can_df.index[i]) for i in ilocs]

    # need a video to show top row frames in static grid path
    if not selected_fids or not video_path:
        return

    # Build 2-row figure
    ncols = len(selected_fids)
    nrows = 2
    cell_w, cell_h = TARGET_SIZE
    h_gap_px = 10
    v_gap_px = 20
    lr_margin = 30
    tb_margin = 80
    fig_w = int(ncols * cell_w + max(0, ncols - 1) * h_gap_px + 2 * lr_margin)
    fig_h = int(2 * cell_h + v_gap_px + 2 * tb_margin)

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        specs=[[{"type": "image"}] * ncols, [{"type": "scene"}] * ncols],
        vertical_spacing=0.03,
        horizontal_spacing=0.01,
        row_heights=[0.5, 0.5],
    )

    # Top row: frames and optional overlays
    overlay_present = (overlay_land_df is not None) or (overlay_rect_df is not None)
    overlay_trace_indices: List[int] = []
    rect_shape_indices: List[int] = []
    image_trace_indices: List[int] = []
    privacy_pairs: List[Tuple[int, int]] = []  # for blur toggle
    cushion_ratio = 0.35

    for i, fid in enumerate(selected_fids):
        # Load frame i
        try:
            frame = get_frame(video_path, fid)
        except Exception:
            continue

        crop = None
        x1 = y1 = 0
        sx = sy = 1.0

        # Prefer rectangle-driven crop
        if overlay_rect_df is not None and fid in overlay_rect_df.index:
            r = overlay_rect_df.loc[fid]
            try:
                rx, ry, rw, rh = map(int, [r["x"], r["y"], r["w"], r["h"]])
            except Exception:
                try:
                    rx, ry, rw, rh = map(int, r.values[:4])
                except Exception:
                    rx = ry = rw = rh = None

            if None not in (rx, ry, rw, rh):
                crop, box_scaled, _unused, (x1, y1, sx, sy) = crop_and_scale(
                    frame, x=int(rx), y=int(ry), w=int(rw), h=int(rh), cushion_ratio=cushion_ratio
                )
                # Add image + blur pair (optional)
                if overlay_present:
                    fig.add_trace(go.Image(z=crop, visible=True), row=1, col=i + 1)
                    orig_idx = len(fig.data) - 1
                    image_trace_indices.append(orig_idx)
                    kx = max(31, (crop.shape[1] // 10) * 2 + 1)
                    ky = max(31, (crop.shape[0] // 10) * 2 + 1)
                    blur_img = cv2.GaussianBlur(crop, (kx, ky), 0)
                    fig.add_trace(go.Image(z=blur_img, visible=False), row=1, col=i + 1)
                    blur_idx = len(fig.data) - 1
                    image_trace_indices.append(blur_idx)
                    privacy_pairs.append((orig_idx, blur_idx))
                else:
                    fig.add_trace(go.Image(z=crop), row=1, col=i + 1)
                    image_trace_indices.append(len(fig.data) - 1)

                # Rectangle overlay (scaled)
                if box_scaled is not None:
                    bx, by, bw, bh = box_scaled
                    fig.add_shape(
                        type="rect",
                        x0=bx, y0=by, x1=bx + bw, y1=by + bh,
                        line=dict(color="red", width=2),
                        row=1, col=i + 1,
                    )
                    if fig.layout.shapes:
                        rect_shape_indices.append(len(fig.layout.shapes) - 1)

                # Landmarks overlay if present too
                if overlay_land_df is not None and fid in overlay_land_df.index:
                    lmk_row = overlay_land_df.loc[fid]
                    xs = lmk_row.values[::2].astype(float)
                    ys = lmk_row.values[1::2].astype(float)
                    xs_crop = (xs - x1) * sx
                    ys_crop = (ys - y1) * sy
                    fig.add_trace(
                        go.Scatter(
                            x=xs_crop, y=ys_crop,
                            mode="markers",
                            marker=dict(color="blue", size=5, symbol="circle"),
                            name="Landmarks",
                            showlegend=False,
                        ),
                        row=1, col=i + 1,
                    )
                    overlay_trace_indices.append(len(fig.data) - 1)

        elif overlay_land_df is not None and fid in overlay_land_df.index:
            # Landmark-driven crop (fallback)
            lmk_row = overlay_land_df.loc[fid]
            xs = lmk_row.values[::2].astype(float)
            ys = lmk_row.values[1::2].astype(float)
            crop, xs_crop, ys_crop, (x1, y1, sx, sy) = crop_and_scale(
                frame, xs=xs, ys=ys, cushion_ratio=cushion_ratio
            )
            if overlay_present:
                fig.add_trace(go.Image(z=crop, visible=True), row=1, col=i + 1)
                orig_idx = len(fig.data) - 1
                image_trace_indices.append(orig_idx)
                kx = max(31, (crop.shape[1] // 10) * 2 + 1)
                ky = max(31, (crop.shape[0] // 10) * 2 + 1)
                blur_img = cv2.GaussianBlur(crop, (kx, ky), 0)
                fig.add_trace(go.Image(z=blur_img, visible=False), row=1, col=i + 1)
                blur_idx = len(fig.data) - 1
                image_trace_indices.append(blur_idx)
                privacy_pairs.append((orig_idx, blur_idx))
            else:
                fig.add_trace(go.Image(z=crop), row=1, col=i + 1)
                image_trace_indices.append(len(fig.data) - 1)

            fig.add_trace(
                go.Scatter(
                    x=xs_crop, y=ys_crop,
                    mode="markers",
                    marker=dict(color="blue", size=5, symbol="circle"),
                    name="Landmarks",
                    showlegend=False,
                ),
                row=1, col=i + 1,
            )
            overlay_trace_indices.append(len(fig.data) - 1)

            # Companion rectangle overlay (if any)
            if overlay_rect_df is not None and fid in overlay_rect_df.index:
                try:
                    r = overlay_rect_df.loc[fid]
                    rx, ry, rw, rh = float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])
                    x_rel, y_rel = (rx - x1) * sx, (ry - y1) * sy
                    w_rel, h_rel = rw * sx, rh * sy
                    fig.add_shape(
                        type="rect",
                        x0=x_rel, y0=y_rel, x1=x_rel + w_rel, y1=y_rel + h_rel,
                        line=dict(color="red", width=2),
                        row=1, col=i + 1,
                    )
                    if fig.layout.shapes:
                        rect_shape_indices.append(len(fig.layout.shapes) - 1)
                except Exception:
                    pass

        # No overlay or failed crop: show resized raw frame
        if crop is None:
            crop = cv2.resize(frame, TARGET_SIZE)
            if overlay_present:
                fig.add_trace(go.Image(z=crop, visible=True), row=1, col=i + 1)
                orig_idx = len(fig.data) - 1
                image_trace_indices.append(orig_idx)
                kx = max(31, (crop.shape[1] // 10) * 2 + 1)
                ky = max(31, (crop.shape[0] // 10) * 2 + 1)
                blur_img = cv2.GaussianBlur(crop, (kx, ky), 0)
                fig.add_trace(go.Image(z=blur_img, visible=False), row=1, col=i + 1)
                blur_idx = len(fig.data) - 1
                image_trace_indices.append(blur_idx)
                privacy_pairs.append((orig_idx, blur_idx))
            else:
                fig.add_trace(go.Image(z=crop), row=1, col=i + 1)
                image_trace_indices.append(len(fig.data) - 1)

        # Lock axes per cell
        fig.update_xaxes(range=[0, TARGET_SIZE[0]], fixedrange=True, visible=False, row=1, col=i + 1)
        fig.update_yaxes(range=[TARGET_SIZE[1], 0], fixedrange=True, visible=False, autorange=False, row=1, col=i + 1)

    # Bottom row: 3D canonicalized landmarks
    column_titles: List[str] = []
    pose_eulers_meta: List[dict] = []
    for i, fid in enumerate(selected_fids):
        if fid not in can_df.index:
            continue
        row = can_df.loc[fid]
        vals = getattr(row, "values", np.asarray(row))
        if len(vals) % 3 == 0:
            xyz = np.asarray(vals, dtype=float).reshape(-1, 3)
        elif len(vals) % 2 == 0:
            xy = np.asarray(vals, dtype=float).reshape(-1, 2)
            xyz = np.concatenate([xy, np.zeros((xy.shape[0], 1))], axis=1)
        else:
            continue

        fig.add_trace(
            go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode="markers",
                marker=dict(size=3, color="blue"),
                name=f"Frame {fid}",
            ),
            row=2, col=i + 1,
        )
        column_titles.append(f"Frame {fid}")

        # Add Yaw/Pitch/Roll labels under can_land plots when pose is available
        if pose_df is not None and len(pose_df) > 0 and (rx_col or ry_col or rz_col):
            def _get(s, col, default=0.0):
                try:
                    return float(s[col]) if (col is not None and col in s.index) else float(default)
                except Exception:
                    return float(default)
            try:
                if fid in pose_df.index:
                    s = pose_df.loc[fid]
                else:
                    try:
                        s = pose_df.iloc[fid]
                    except Exception:
                        continue
                pitch = _get(s, rx_col, 0.0)
                yaw   = _get(s, ry_col, 0.0)
                roll  = _get(s, rz_col, 0.0)
                pose_eulers_meta.append({"frame_idx": int(fid), "pitch": float(pitch), "yaw": float(yaw), "roll": float(roll)})

                scene_id = f"scene{i + 1}" if i > 0 else "scene"
                dom = fig.layout[scene_id].domain
                xmid = (dom.x[0] + dom.x[1]) / 2.0
                ybelow2 = dom.y[0] - 0.16
                fig.add_annotation(
                    text=f"Yaw: {yaw:.2f}, Pitch: {pitch:.2f}, Roll: {roll:.2f}",
                    x=xmid, y=ybelow2, xref="paper", yref="paper",
                    showarrow=False, align="center", xanchor="center", yanchor="top",
                    font=dict(size=12),
                )
            except Exception:
                pass

    if len(column_titles) == 0:
        return

    # Layout and styling
    fig.update_layout(
        autosize=False,
        width=fig_w, height=fig_h,
        showlegend=False,
        title={"text": "Video (top) and Canonicalized Landmarks (bottom)", "x": 0.5, "xanchor": "center"},
        margin=dict(t=70, l=20, r=20, b=120),
        font=dict(family="Roboto, sans-serif", size=14, color="#111"),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    # Configure 3D scenes
    camera = dict(eye=dict(x=0, y=0, z=2.5), up=dict(x=0, y=1, z=0))
    for i in range(1, len(selected_fids) + 1):
        scene_id = f"scene{i}" if i > 1 else "scene"
        fig.layout[scene_id].camera = camera
        fig.layout[scene_id].aspectmode = "data"
        fig.layout[scene_id].xaxis.showticklabels = False
        fig.layout[scene_id].yaxis.showticklabels = False
        fig.layout[scene_id].zaxis.showticklabels = False
        fig.layout[scene_id].xaxis.color = "gray"
        fig.layout[scene_id].yaxis.color = "gray"
        fig.layout[scene_id].zaxis.color = "gray"

    # Labels under 3D row
    try:
        for i in range(1, len(selected_fids) + 1):
            scene_id = f"scene{i}" if i > 1 else "scene"
            dom = fig.layout[scene_id].domain
            xmid = (dom.x[0] + dom.x[1]) / 2.0
            ybelow = dom.y[0] - 0.09
            fig.add_annotation(
                text=column_titles[i - 1],
                x=xmid, y=ybelow, xref="paper", yref="paper",
                showarrow=False, align="center", xanchor="center", yanchor="top",
                font=dict(size=12),
            )
    except Exception:
        pass

    # Toolbar meta for your write_centered_html toolbar
    meta = {}
    if overlay_trace_indices: meta["overlay_trace_indices"] = overlay_trace_indices
    if rect_shape_indices:    meta["rectangle_shape_indices"] = rect_shape_indices
    if image_trace_indices:   meta["image_trace_indices"] = image_trace_indices
    if privacy_pairs:         meta["privacy_pairs"] = privacy_pairs
    if blur_default:          meta["blur_initial"] = True
    if column_titles:         meta["pose_column_titles"] = column_titles
    if pose_eulers_meta:      meta["pose_eulers"] = pose_eulers_meta
    if meta: fig.update_layout(meta=meta)

    # Export
    html_path = os.path.join(out_dir, "bitbox_viz.html")
    try:
        write_centered_html(fig, html_path, export_filename="bitbox visualizations", add_pose_only_export=True)
    except Exception:
        pass
    return


def visualize_and_export_pose(pose: dict, out_dir: str, video_path: Optional[str] = None, num_frames: int = 5, overlay: Optional[dict] = None):
    """Render a 3D pose axis visualization (optionally with top-row video frames).
    Adds camera presets and a Pose-only export button. If video is provided, frames are
    cropped/overlaid similarly to other grid helpers.
    """
    def _safe_df(d: Optional[dict]):
        return d["data"] if isinstance(d, dict) and "data" in d else None

    def _dtype(d: Optional[dict]) -> Optional[str]:
        return d.get("type") if isinstance(d, dict) else None

    pose_df = _safe_df(pose)
    if pose_df is None or len(pose_df) == 0:
        raise ValueError("No pose data found. Provide a valid pose dictionary with 'data' key.")

    # --- Resolve Euler columns (case-insensitive, allow pitch/yaw/roll fallbacks)
    rx_col = _find_col(pose_df, ("Rx", "rx", "pitch"))
    ry_col = _find_col(pose_df, ("Ry", "ry", "yaw"))
    rz_col = _find_col(pose_df, ("Rz", "rz", "roll"))
    if rx_col is None or ry_col is None or rz_col is None:
        raise KeyError(f"Pose DataFrame must contain Rx/Ry/Rz (or pitch/yaw/roll). Have: {list(pose_df.columns)}")

    # --- Select diverse frames by yaw/pitch
    k = max(1, min(int(num_frames), len(pose_df)))
    try:
        idxs = select_diverse_frames_maxmin(
            pose_df,
            k=k,
            x_col_candidates=("Rx", "rx", "pitch"),
            y_col_candidates=("Ry", "ry", "yaw"),
            prefer_extremes=True,
        )
        frame_ids = [int(pose_df.index[i]) for i in idxs]
    except Exception:
        # Fallback: evenly spaced
        ilocs = np.unique(np.linspace(0, len(pose_df) - 1, num=k, dtype=int)).tolist()
        frame_ids = [int(pose_df.index[i]) for i in ilocs]

    if not frame_ids:
        return None

    # --- Helper: ensure radians if input appears to be degrees
    def ensure_radians(vals: np.ndarray) -> np.ndarray:
        vals = np.asarray(vals, dtype=float)
        if np.max(np.abs(vals)) > np.pi * 1.25:  # likely degrees
            return np.deg2rad(vals)
        return vals

    # --- Parse overlay inputs (optional)
    overlay_land = None
    overlay_rect = None
    blur_default = False

    if isinstance(overlay, list):
        for value in overlay:
            if isinstance(value, dict) and "type" in value:
                if _dtype(value) == "landmark":
                    overlay_land = value
                elif _dtype(value) == "rectangle":
                    overlay_rect = value
            elif isinstance(value, dict):
                if bool(value.get("blur")) or (value.get("privacy") in (True, "blur", "on")):
                    blur_default = True
    elif isinstance(overlay, dict) and "type" in overlay:
        if _dtype(overlay) == "landmark":
            overlay_land = overlay
        elif _dtype(overlay) == "rectangle":
            overlay_rect = overlay
    elif isinstance(overlay, dict):
        if bool(overlay.get("blur")) or (overlay.get("privacy") in (True, "blur", "on")):
            blur_default = True

    overlay_land_df = _safe_df(overlay_land)
    overlay_rect_df = _safe_df(overlay_rect)
    overlay_present = (overlay_land_df is not None) or (overlay_rect_df is not None)

    # --- Figure grid
    have_video = bool(video_path)
    ncols = len(frame_ids)
    cell_w, cell_h = TARGET_SIZE

    if have_video:
        nrows = 2
        specs = [[{"type": "image"}] * ncols, [{"type": "scene"}] * ncols]
        row_heights = [0.5, 0.5]
        v_gap_px = 20
        tb_margin = 80
        fig_h = int(2 * cell_h + v_gap_px + 2 * tb_margin)
    else:
        nrows = 1
        specs = [[{"type": "scene"}] * ncols]
        row_heights = [1.0]
        v_gap_px = 10
        tb_margin = 80
        fig_h = int(1 * cell_h + v_gap_px + 2 * tb_margin)

    h_gap_px = 10
    lr_margin = 30
    fig_w = int(ncols * cell_w + max(0, ncols - 1) * h_gap_px + 2 * lr_margin)

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        specs=specs,
        vertical_spacing=0.03,
        horizontal_spacing=0.01,
        row_heights=row_heights,
    )

    image_trace_indices: List[int] = []
    column_titles: List[str] = []
    pose_eulers_meta: List[dict] = []
    overlay_trace_indices: List[int] = []  # landmark scatters
    rect_shape_indices: List[int] = []     # rectangle shapes
    privacy_pairs: List[Tuple[int, int]] = []  # (orig_img_idx, blur_img_idx)

    # --- NEW: indices for toggles + helpers
    label_trace_indices: List[int] = []
    plane_trace_indices: List[int] = []
    follow_cameras: List[dict] = []

    def _plane_surface(R: np.ndarray, plane: str = "xy", L: float = 1.05, opacity: float = 0.18):
        # small 2x2 grid in local plane, rotated by R
        S, T = np.meshgrid(np.array([-L, L]), np.array([-L, L]))
        if plane == "xy":
            Xl, Yl, Zl = S, T, np.zeros_like(S)
        elif plane == "yz":
            Xl, Yl, Zl = np.zeros_like(S), S, T
        else:  # "zx"
            Xl, Yl, Zl = S, np.zeros_like(S), T

        Xw = np.zeros_like(Xl, dtype=float)
        Yw = np.zeros_like(Yl, dtype=float)
        Zw = np.zeros_like(Zl, dtype=float)
        for i in range(Xl.shape[0]):
            for j in range(Xl.shape[1]):
                v = R @ np.array([Xl[i, j], Yl[i, j], Zl[i, j]])
                Xw[i, j], Yw[i, j], Zw[i, j] = v[0], v[1], v[2]

        sc = np.ones_like(Xw)
        return go.Surface(
            x=Xw, y=Yw, z=Zw,
            surfacecolor=sc,
            colorscale=[[0, "#DCDCDC"], [1, "#DCDCDC"]],
            showscale=False, opacity=opacity, visible=False, hoverinfo="skip"
        )

    def _camera_follow_pose(R: np.ndarray, dist: float = 1.35):
        base_eye = np.array([dist, dist, dist])
        base_up  = np.array([0.0, 1.0, 0.0])
        eye = R @ base_eye
        up  = R @ base_up
        return dict(eye=dict(x=float(eye[0]), y=float(eye[1]), z=float(eye[2])),
                    up=dict(x=float(up[0]),  y=float(up[1]),  z=float(up[2])))

    # --- Top row: frames (with optional crops/overlays)
    cushion_ratio = 0.35

    def _extract_rect_row(row) -> Optional[Tuple[float, float, float, float]]:
        try:
            if all(c in row.index for c in ("x", "y", "w", "h")):
                return float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])
            if all(c in row.index for c in ("left", "top", "width", "height")):
                return float(row["left"]), float(row["top"]), float(row["width"]), float(row["height"])
            if all(c in row.index for c in ("x1", "y1", "x2", "y2")):
                x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
                return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)
        except Exception:
            return None
        return None
    
    def _scene_id(i: int) -> str:
        return "scene" if i == 0 else f"scene{i+1}"

    if have_video:
        for i, fid in enumerate(frame_ids):
            # Load frame
            try:
                frame = get_frame(video_path, fid)
            except Exception:
                frame = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

            crop = None
            x1 = y1 = 0.0
            sx = sy = 1.0

            # Prefer rectangle-driven crop, then landmark-driven
            if overlay_present and (overlay_rect_df is not None) and (fid in overlay_rect_df.index):
                r = overlay_rect_df.loc[fid]
                rect = _extract_rect_row(r)
                if rect is not None:
                    rx, ry, rw, rh = rect
                    try:
                        crop, box_scaled, _, (x1, y1, sx, sy) = crop_and_scale(
                            frame, x=int(rx), y=int(ry), w=int(rw), h=int(rh), cushion_ratio=cushion_ratio
                        )
                    except Exception:
                        crop = None
            if crop is None and overlay_present and (overlay_land_df is not None) and (fid in overlay_land_df.index):
                l = overlay_land_df.loc[fid]
                vals = getattr(l, "values", np.asarray(l))
                xs = vals[::2].astype(float)
                ys = vals[1::2].astype(float)
                try:
                    crop, xs_crop, ys_crop, (x1, y1, sx, sy) = crop_and_scale(
                        frame, xs=xs, ys=ys, cushion_ratio=cushion_ratio
                    )
                except Exception:
                    crop = None

            # If no crop possible, just resize
            if crop is None:
                try:
                    crop = cv2.resize(frame, TARGET_SIZE)
                except Exception:
                    crop = frame

            # Add image and optional blur pair if any overlay provided (so toolbar can offer privacy)
            if overlay_present:
                fig.add_trace(go.Image(z=crop, visible=True), row=1, col=i + 1)
                orig_idx = len(fig.data) - 1
                image_trace_indices.append(orig_idx)
                try:
                    kx = max(31, (crop.shape[1] // 10) * 2 + 1)
                    ky = max(31, (crop.shape[0] // 10) * 2 + 1)
                    blur_img = cv2.GaussianBlur(crop, (kx, ky), 0)
                except Exception:
                    blur_img = crop
                fig.add_trace(go.Image(z=blur_img, visible=False), row=1, col=i + 1)
                blur_idx = len(fig.data) - 1
                image_trace_indices.append(blur_idx)
                privacy_pairs.append((orig_idx, blur_idx))
            else:
                fig.add_trace(go.Image(z=crop), row=1, col=i + 1)
                image_trace_indices.append(len(fig.data) - 1)

            # Overlays
            if overlay_present and (overlay_rect_df is not None) and (fid in overlay_rect_df.index):
                try:
                    r = overlay_rect_df.loc[fid]
                    rect = _extract_rect_row(r)
                    if rect is not None:
                        rx, ry, rw, rh = rect
                        xr = (float(rx) - x1) * sx
                        yr = (float(ry) - y1) * sy
                        wr = float(rw) * sx
                        hr = float(rh) * sy
                        fig.add_shape(
                            type="rect",
                            x0=xr, y0=yr, x1=xr + wr, y1=yr + hr,
                            line=dict(color="red", width=2),
                            row=1, col=i + 1,
                        )
                        try:
                            rect_shape_indices.append(len(fig.layout.shapes) - 1)
                        except Exception:
                            pass
                except Exception:
                    pass

            if overlay_present and (overlay_land_df is not None) and (fid in overlay_land_df.index):
                try:
                    l = overlay_land_df.loc[fid]
                    vals = getattr(l, "values", np.asarray(l))
                    xs = vals[::2].astype(float)
                    ys = vals[1::2].astype(float)
                    xs_c = (xs - x1) * sx
                    ys_c = (ys - y1) * sy
                    fig.add_trace(
                        go.Scatter(
                            x=xs_c, y=ys_c,
                            mode="markers",
                            marker=dict(color="blue", size=5, symbol="circle"),
                            name="Landmarks",
                            showlegend=False,
                        ),
                        row=1, col=i + 1,
                    )
                    overlay_trace_indices.append(len(fig.data) - 1)
                except Exception:
                    pass

            fig.update_xaxes(range=[0, TARGET_SIZE[0]], fixedrange=True, visible=False, row=1, col=i + 1)
            fig.update_yaxes(range=[TARGET_SIZE[1], 0], fixedrange=True, visible=False, autorange=False, row=1, col=i + 1)

    # --- Bottom row: 3D pose axes (with arrowheads + world triad)
    row3d = 2 if have_video else 1

    axis_len = 1.0
    shaft_width = 6
    cone_size = 0.18

    origin = np.zeros((3,))
    camera_iso = dict(eye=dict(x=1.35, y=1.35, z=1.35), up=dict(x=0, y=1, z=0))

    def add_world_guides(col_idx: int):
        fig.add_trace(
            go.Scatter3d(
                x=[0, 1.1], y=[0, 0], z=[0, 0],
                mode="lines", line=dict(color="#A0A0A0", width=2),
                showlegend=False, hoverinfo="skip",
            ),
            row=row3d, col=col_idx
        )
        fig.add_trace(
            go.Scatter3d(
                x=[0, 0], y=[0, 1.1], z=[0, 0],
                mode="lines", line=dict(color="#A0A0A0", width=2),
                showlegend=False, hoverinfo="skip",
            ),
            row=row3d, col=col_idx
        )
        fig.add_trace(
            go.Scatter3d(
                x=[0, 0], y=[0, 0], z=[0, 1.1],
                mode="lines", line=dict(color="#A0A0A0", width=2),
                showlegend=False, hoverinfo="skip",
            ),
            row=row3d, col=col_idx
        )
        L = 1.2
        edges = [
            ((-L,-L,-L),( L,-L,-L)), ((-L, L,-L),( L, L,-L)), ((-L,-L, L),( L,-L, L)), ((-L, L, L),( L, L, L)),
            ((-L,-L,-L),(-L, L,-L)), (( L,-L,-L),( L, L,-L)), ((-L,-L, L),(-L, L, L)), (( L,-L, L),( L, L, L)),
            ((-L,-L,-L),(-L,-L, L)), (( L,-L,-L),( L,-L, L)), ((-L, L,-L),(-L, L, L)), (( L, L,-L),( L, L, L)),
        ]
        for (a, b) in edges:
            fig.add_trace(
                go.Scatter3d(
                    x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]],
                    mode="lines", line=dict(color="#E0E0E0", width=1),
                    showlegend=False, hoverinfo="skip",
                ),
                row=row3d, col=col_idx
            )

    scene_ids = []

    for i, fid in enumerate(frame_ids):
        col = i + 1
        if fid in pose_df.index:
            s = pose_df.loc[fid]
        else:
            try:
                s = pose_df.iloc[fid]
            except Exception:
                continue

        try:
            rx = float(s[rx_col]); ry = float(s[ry_col]); rz = float(s[rz_col])
        except Exception:
            continue

        rx, ry, rz = ensure_radians([rx, ry, rz])
        R = euler_to_rotmat(rx, ry, rz)

        x_axis = R @ np.array([axis_len, 0.0, 0.0])
        y_axis = R @ np.array([0.0, axis_len, 0.0])
        z_axis = R @ np.array([0.0, 0.0, axis_len])

        add_world_guides(col_idx=col)

        fig.add_trace(
            go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode="markers",
                marker=dict(size=3, color="#404040"),
                showlegend=False,
                hovertext=[f"Frame {fid}"],
                hoverinfo="text",
            ),
            row=row3d, col=col
        )

        # Degrees + hover template
        pitch_deg = float(np.rad2deg(rx))
        yaw_deg   = float(np.rad2deg(ry))
        roll_deg  = float(np.rad2deg(rz))
        hovertemplate = (
            "Frame %{customdata[0]:.0f}<br>"
            "pitch %{customdata[1]:.1f}°, yaw %{customdata[2]:.1f}°, roll %{customdata[3]:.1f}°<br>"
            "Point (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>"
        )
        cd = np.array([[fid, pitch_deg, yaw_deg, roll_deg]] * 2, dtype=float)

        # Shafts (lines) with hover
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], x_axis[0]], y=[origin[1], x_axis[1]], z=[origin[2], x_axis[2]],
                mode="lines",
                line=dict(color="red", width=shaft_width),
                showlegend=False,
                customdata=cd, hovertemplate=hovertemplate,
            ),
            row=row3d, col=col
        )
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], y_axis[0]], y=[origin[1], y_axis[1]], z=[origin[2], y_axis[2]],
                mode="lines",
                line=dict(color="green", width=shaft_width),
                showlegend=False,
                customdata=cd, hovertemplate=hovertemplate,
            ),
            row=row3d, col=col
        )
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], z_axis[0]], y=[origin[1], z_axis[1]], z=[origin[2], z_axis[2]],
                mode="lines",
                line=dict(color="blue", width=shaft_width),
                showlegend=False,
                customdata=cd, hovertemplate=hovertemplate,
            ),
            row=row3d, col=col
        )

        # Arrowheads (cones) at the tips (no hover)
        def _cone_at_tip(tip_vec, color):
            return go.Cone(
                x=[tip_vec[0]], y=[tip_vec[1]], z=[tip_vec[2]],
                u=[-tip_vec[0]], v=[-tip_vec[1]], w=[-tip_vec[2]],
                sizemode="absolute", sizeref=cone_size,
                showscale=False, colorscale=[[0, color], [1, color]],
                anchor="tip", hoverinfo="skip"
            )

        fig.add_trace(_cone_at_tip(x_axis, "red"),   row=row3d, col=col)
        fig.add_trace(_cone_at_tip(y_axis, "green"), row=row3d, col=col)
        fig.add_trace(_cone_at_tip(z_axis, "blue"),  row=row3d, col=col)

        # Tip labels
        label_scale = 1.08
        def _label_at_tip(tip_vec, text):
            p = tip_vec * label_scale
            return go.Scatter3d(
                x=[p[0]], y=[p[1]], z=[p[2]],
                mode="text", text=[text],
                textfont=dict(size=10),
                showlegend=False, hoverinfo="skip", visible=True,
            )
        for text, vec in (("pitch (X)", x_axis), ("yaw (Y)", y_axis), ("roll (Z)", z_axis)):
            fig.add_trace(_label_at_tip(vec, text), row=row3d, col=col)
            label_trace_indices.append(len(fig.data) - 1)

        # Translucent planes (default hidden)
        for plane in ("xy", "yz", "zx"):
            surf = _plane_surface(R, plane=plane, L=1.05, opacity=0.18)
            fig.add_trace(surf, row=row3d, col=col)
            plane_trace_indices.append(len(fig.data) - 1)

        # Per-column labels + meta
        column_titles.append(f"Frame {fid}")
        pose_eulers_meta.append({
            "frame_idx": int(fid),
            "pitch": float(rx),
            "yaw": float(ry),
            "roll": float(rz),
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg,
            "roll_deg": roll_deg,
        })

        # Configure each 3D scene
        scene_id = _scene_id(i)
        scene_ids.append(scene_id)
        fig.layout[scene_id].camera = camera_iso
        fig.layout[scene_id].aspectmode = "cube"
        rng = 1.3
        fig.layout[scene_id].xaxis.update(range=[-rng, rng], showgrid=True, zeroline=False,
                                          showticklabels=False, title="", gridcolor="#E6E6E6",
                                          backgroundcolor="#FAFAFA")
        fig.layout[scene_id].yaxis.update(range=[-rng, rng], showgrid=True, zeroline=False,
                                          showticklabels=False, title="", gridcolor="#E6E6E6",
                                          backgroundcolor="#FAFAFA")
        fig.layout[scene_id].zaxis.update(range=[-rng, rng], showgrid=True, zeroline=False,
                                          showticklabels=False, title="", gridcolor="#E6E6E6",
                                          backgroundcolor="#FAFAFA")

        # Store follow camera for this column
        follow_cameras.append(_camera_follow_pose(R, dist=1.35))

    if not column_titles:
        return None

    # --- Titles and layout
    title_text = "Video (top) and 3D Pose (bottom)" if have_video else "3D Pose"
    fig.update_layout(
        autosize=False,
        width=fig_w,
        height=fig_h,
        showlegend=False,
        title={"text": title_text, "x": 0.5, "xanchor": "center"},
        margin=dict(t=70, l=20, r=20, b=120 if have_video else 110),
        font=dict(family="Roboto, system-ui, -apple-system, Arial, sans-serif", size=14, color="#111"),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    # Bottom-row labels: frame and Euler values under each scene
    try:
        for i in range(len(frame_ids)):
            scene_id = _scene_id(i)

            dom = fig.layout[scene_id].domain
            xmid = (dom.x[0] + dom.x[1]) / 2.0
            e = pose_eulers_meta[i]
            txt1 = column_titles[i]
            txt2 = f"pitch {e['pitch_deg']:.1f}°, yaw {e['yaw_deg']:.1f}°, roll {e['roll_deg']:.1f}°"
            fig.add_annotation(
                text=txt1,
                x=xmid, y=dom.y[0] - 0.06, xref="paper", yref="paper",
                showarrow=False, align="center", xanchor="center", yanchor="top",
                font=dict(size=12),
            )
            fig.add_annotation(
                text=txt2,
                x=xmid, y=dom.y[0] - 0.095, xref="paper", yref="paper",
                showarrow=False, align="center", xanchor="center", yanchor="top",
                font=dict(size=11, color="#444"),
            )
    except Exception:
        pass

    # --- Camera preset buttons (apply to all scenes) + toggles
    def _relayout_for(camera):
        d = {}
        for sid in scene_ids:
            d[f"{sid}.camera"] = camera
        return d

    cam_iso  = dict(eye=dict(x=1.35, y=1.35, z=1.35), up=dict(x=0, y=1, z=0))
    cam_front = dict(eye=dict(x=0.001, y=0.001, z=2.2), up=dict(x=0, y=1, z=0))
    cam_left  = dict(eye=dict(x=2.2, y=0.0, z=0.001), up=dict(x=0, y=1, z=0))
    cam_top   = dict(eye=dict(x=0.0, y=2.2, z=0.001), up=dict(x=0, y=0, z=-1))

    # Build follow layout mapping (per-scene)
    follow_layout = {}
    for sid, cam in zip(scene_ids, follow_cameras):
        follow_layout[f"{sid}.camera"] = cam

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.50, xanchor="center",
                y=-0.08, yanchor="top",
                pad=dict(t=40, l=0, r=0),
                buttons=[
                    dict(label="ISO",    method="relayout", args=[_relayout_for(cam_iso)]),
                    dict(label="Front",  method="relayout", args=[_relayout_for(cam_front)]),
                    dict(label="Left",   method="relayout", args=[_relayout_for(cam_left)]),
                    dict(label="Top",    method="relayout", args=[_relayout_for(cam_top)]),
                    dict(label="Follow", method="relayout", args=[follow_layout]),
                ]
            )
        ]
    )

    # --- Toolbar meta for HTML helper
    meta = {}
    if image_trace_indices:
        meta["image_trace_indices"] = image_trace_indices
    if overlay_trace_indices:
        meta["overlay_trace_indices"] = overlay_trace_indices
    if rect_shape_indices:
        meta["rectangle_shape_indices"] = rect_shape_indices
    if privacy_pairs:
        meta["privacy_pairs"] = privacy_pairs
    if blur_default:
        meta["blur_initial"] = True
    if column_titles:
        meta["pose_column_titles"] = column_titles
    if pose_eulers_meta:
        meta["pose_eulers"] = pose_eulers_meta
    if meta:
        fig.update_layout(meta=meta)

    # --- Write HTML with Pose-only export button enabled
    html_path = os.path.join(out_dir, "bitbox_viz.html")
    os.makedirs(out_dir or ".", exist_ok=True)
    try:
        write_centered_html(fig, html_path, export_filename="bitbox visualizations", add_pose_only_export=True)
    except Exception:
        pass

    return None


# -----------------------------------------------------------------------------
# Subplot builder for rectangle/landmark grids
# -----------------------------------------------------------------------------

def make_centered_subplot_with_overlay(
    crops: Sequence[np.ndarray],
    main_items: Sequence,
    overlay_items: Optional[Sequence] = None,
    main_type: str = "rect",
    ncols: int = 0,
    main_title: Optional[str] = None,
    add_overlay_toggle: bool = True,
    blurred_crops: Optional[Sequence[np.ndarray]] = None,
):
    """Build a tightly packed grid of cropped images with overlays.

    Parameters:
    - crops: image crops, each sized to TARGET_SIZE (w, h)
    - main_items: rectangles (x,y,w,h) when main_type='rect' or (xs, ys) landmarks when 'landmark'
    - overlay_items: optional companion overlay per cell (landmarks for rect main, rectangle for landmark main)
    - main_type: 'rect' or 'landmark'
    - ncols: number of columns (if 0, uses len(crops))
    - main_title: title text above the grid
    - add_overlay_toggle: kept for compatibility (toggle handled in HTML layer)
    - blurred_crops: optional blurred versions to support privacy toggling

    Returns:
    - Plotly Figure with layout.meta containing indices used by the HTML toolbar
    """
    num = len(crops)
    ncols = len(crops)
    nrows = (num + ncols - 1) // ncols

    font_family = "Roboto, Helvetica, Arial, sans-serif"
    title_map = {"rect": "Face Rectangles with Landmark Overlay", "landmark": "Face Landmarks with Rectangle Overlay"}
    if main_title is None:
        main_title = title_map.get(main_type, "Plot")

    # Compute a compact figure size so frames sit close together (small gaps)
    cell_w, cell_h = TARGET_SIZE  # pixel size of each crop cell
    h_gap_px = 15                # horizontal gap between cells in pixels
    v_gap_px = 18                 # vertical gap between rows in pixels
    lr_margin = 20
    t_margin = 100
    b_margin = 30
    fig_w = int(ncols * cell_w + max(0, ncols - 1) * h_gap_px + 2 * lr_margin)
    fig_h = int(nrows * cell_h + max(0, nrows - 1) * v_gap_px + t_margin + b_margin)

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        # Convert pixel gaps to fraction of figure size to tightly pack subplots
        horizontal_spacing=(h_gap_px / max(fig_w, 1)),
        vertical_spacing=(v_gap_px / max(fig_h, 1)),
        subplot_titles=[f"{chr(65 + i)}" for i in range(num)],
    )

    overlay_trace_indices: List[int] = []
    privacy_pairs: List[Tuple[int, int]] = []
    image_trace_indices: List[int] = []
    rect_shape_indices: List[int] = []  #  collect rectangle shape indices
    rectangle_overlay_trace_indices: List[int] = []  #  rectangles drawn as traces (landmark main)

    for idx, crop in enumerate(crops):
        row = idx // ncols + 1
        col = idx % ncols + 1

        # Add original image trace
        fig.add_trace(go.Image(z=crop), row=row, col=col)
        orig_trace_idx = len(fig.data) - 1
        image_trace_indices.append(orig_trace_idx)

        # Add blurred image trace (if provided), hidden by default, added immediately so overlays/landmarks stay on top
        if (blurred_crops is not None and idx < len(blurred_crops) and blurred_crops[idx] is not None):
            fig.add_trace(go.Image(z=blurred_crops[idx], visible=False), row=row, col=col)
            blur_trace_idx = len(fig.data) - 1
            privacy_pairs.append((orig_trace_idx, blur_trace_idx))
            image_trace_indices.append(blur_trace_idx)
        else:
            blur_trace_idx = -1

        if main_type == "rect":
            x_rel, y_rel, w, h = main_items[idx]
            fig.add_shape(
                type="rect",
                x0=x_rel,
                y0=y_rel,
                x1=x_rel + w,
                y1=y_rel + h,
                line=dict(color="red", width=2),
                row=row,
                col=col,
            )
            #  record this rectangle shape index
            try:
                if fig.layout.shapes:
                    rect_shape_indices.append(len(fig.layout.shapes) - 1)
            except Exception:
                pass
        elif main_type == "landmark":
            xs_crop, ys_crop = main_items[idx]
            fig.add_trace(
                go.Scatter(
                    x=xs_crop,
                    y=ys_crop,
                    mode="markers",
                    marker=dict(color="blue", size=5, symbol="circle"),
                    name="Landmarks",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        # Optional overlay
        if overlay_items and overlay_items[idx] is not None:
            if main_type == "rect":
                xs_overlay, ys_overlay = overlay_items[idx]
                fig.add_trace(
                    go.Scatter(
                        x=xs_overlay,
                        y=ys_overlay,
                        mode="markers",
                        name="Overlay",
                        marker=dict(color="blue", size=5, symbol="circle"),
                        visible=True,
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
                overlay_trace_indices.append(len(fig.data) - 1)
            elif main_type == "landmark":
                ox_rel, oy_rel, ow, oh = overlay_items[idx]
                x_seq = [ox_rel, ox_rel + ow, ox_rel + ow, ox_rel, ox_rel]
                y_seq = [oy_rel, oy_rel, oy_rel + oh, oy_rel + oh, oy_rel]
                fig.add_trace(
                    go.Scatter(
                        x=x_seq,
                        y=y_seq,
                        mode="lines",
                        name="Overlay",
                        line=dict(color="red", width=2, dash="dash"),
                        visible=True,
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
                rectangle_overlay_trace_indices.append(len(fig.data) - 1)  

        # Hide ticks and flip Y; then lock ranges to prevent autorange flips
        fig.update_xaxes(showticklabels=False, visible=False, row=row, col=col)
        fig.update_yaxes(showticklabels=False, visible=False, autorange="reversed", row=row, col=col)
        # FIX: lock axes to the image size so toggling traces doesn't change orientation
        fig.update_xaxes(range=[0, cell_w], fixedrange=True, row=row, col=col)
        fig.update_yaxes(range=[cell_h, 0], fixedrange=True, autorange=False, row=row, col=col)

    fig.update_layout(
        title=dict(text=main_title, x=0.5, xanchor="center"),  # top title
        autosize=False,               # fixed sizing to keep gaps compact
        width=fig_w,
        height=fig_h,
        margin=dict(t=t_margin, l=lr_margin, r=lr_margin, b=b_margin),
        showlegend=False,
        font=dict(size=16, family=font_family, color="#222"),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    # Style subplot titles (A, B, ...)
    for anno in fig["layout"]["annotations"]:
        if "text" in anno:
            anno["font"] = dict(size=17, family=font_family, color="#222")
            anno["yshift"] = 2

    # Store overlay trace indices and privacy helpers in layout meta for external HTML controls
    meta_dict = {}
    if overlay_trace_indices:
        meta_dict["overlay_trace_indices"] = overlay_trace_indices
    if privacy_pairs:
        meta_dict["privacy_pairs"] = privacy_pairs
    if image_trace_indices:
        meta_dict["image_trace_indices"] = image_trace_indices
    if rect_shape_indices:
        meta_dict["rectangle_shape_indices"] = rect_shape_indices
    if rectangle_overlay_trace_indices:                                  # 
        meta_dict["rectangle_trace_indices"] = rectangle_overlay_trace_indices
    if meta_dict:
        fig.update_layout(meta=meta_dict)

    return fig


# ----------------------------------------------------------------------------
# HTML writer with centered layout and custom toolbar
# ----------------------------------------------------------------------------

def write_centered_html(fig: go.Figure, out_path: str, export_filename: str = "figure", add_pose_only_export: bool = False) -> None:
    """Write a self-contained HTML file that centers the figure and adds a toolbar.

    Toolbar includes Landmarks/Rectangles toggles (when available), Blur/Remove face toggles
    for privacy (when image stacks exist), and export buttons. For pose plots, also exposes
    a Pose-only export option that excludes 2D overlays.
    """
    overlay_indices = []
    privacy_pairs = []
    image_indices = []
    rectangle_shape_indices = []  
    rectangle_trace_indices = []  

    try:
        meta = fig.layout.meta or {}
        if isinstance(meta, dict):
            overlay_indices = meta.get("overlay_trace_indices", []) or []
            privacy_pairs = meta.get("privacy_pairs", []) or []           # may be present for blur
            image_indices = meta.get("image_trace_indices", []) or []
            rectangle_shape_indices = meta.get("rectangle_shape_indices", []) or []
            rectangle_trace_indices = meta.get("rectangle_trace_indices", []) or []
    except Exception:
        overlay_indices = []

    overlay_json = json.dumps(overlay_indices)
    privacy_pairs_json = json.dumps(privacy_pairs)
    image_indices_json = json.dumps(image_indices)
    rectangles_json = json.dumps(rectangle_shape_indices)          # shapes
    rectangle_traces_json = json.dumps(rectangle_trace_indices)    #  traces

    has_landmarks = bool(overlay_indices)
    has_rectangles = bool(rectangle_shape_indices) or bool(rectangle_trace_indices)  
    has_privacy = bool(privacy_pairs)
    has_remove = bool(image_indices)
    export_filename_safe = export_filename.replace(" ", "_")

    # Generate Plotly div + script with a version-pinned plotly.js reference
    inner = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True, "displayModeBar": True},
        div_id="plot",
    )

    # Conditionally include buttons (two separate toggles)
    buttons_html = (
        (("<button id=\"landmarksBtn\" class=\"btn\">Landmarks: Off</button>") if has_landmarks else "")
        + (("\n    " + "<button id=\"rectanglesBtn\" class=\"btn\">Rectangles: Off</button>") if has_rectangles else "")
        + (("\n    " + "<button id=\"blurBtn\" class=\"btn\">Blur Face</button>") if has_privacy else "")
        + (("\n    " + "<button id=\"removeBtn\" class=\"btn\">Remove Face</button>") if has_remove else "")
        + "\n    "
        + '<button id="exportBtn" class="btn export">Export (PNG)</button>'
        + (("\n    " + '<button id="exportPoseOnlyBtn" class="btn export">Export Pose Only (PNG)</button>') if add_pose_only_export else "")
    )

    html_template = Template("""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>$export_filename_safe</title>
  <style>
    body {
    margin: 0;
    background: #ffffff;
    font-family: 'Roboto', sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    }
    .toolbar {
      margin: 12px 0 18px; /* moved below figure: add top margin */
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: center;
      width: 100%;
    }
    .btn {
      padding: 6px 12px;
      border-radius: 6px;
      border: 2px solid #2ecc71; /* updated in JS */
      background: #f8f8f8;
      color: #222;
      cursor: pointer;
      user-select: none;
      transition: border-color 120ms ease, background 120ms ease, opacity 120ms ease;
    }
    .btn:hover { background: #eee; }
    .btn.export { border-color: #555; }
    .wrapper {
      width: 100%;
      display: flex;
      justify-content: center; /* center horizontally */
    }
    #plot {
      max-width: 95vw;
    }
  </style>
</head>
<body>
  <div class=\"wrapper\">
    $inner
  </div>
  <div class=\"toolbar\">$buttons_html</div>
  <script>
    (function() {
      const overlayIdx = $overlay_json;          // landmark trace indices
      const rectShapeIdx = $rectangles_json;     // rectangle shape indices
      const rectTraceIdx = $rectangle_traces_json; //  rectangle traces
      const privacyPairs = $privacy_pairs_json;
      const imageIdxs = $image_indices_json;

      const gd = document.getElementById('plot');
      const exportBtn = document.getElementById('exportBtn');
      const landmarksBtn = document.getElementById('landmarksBtn');
      const rectanglesBtn = document.getElementById('rectanglesBtn');
      const blurBtn = document.getElementById('blurBtn');
      const removeBtn = document.getElementById('removeBtn');
      const exportPoseBtn = document.getElementById('exportPoseOnlyBtn');

      const hasLandmarks = Array.isArray(overlayIdx) && overlayIdx.length > 0;
      const hasRectangles = (Array.isArray(rectShapeIdx) && rectShapeIdx.length > 0) || (Array.isArray(rectTraceIdx) && rectTraceIdx.length > 0);
      const hasPrivacy = Array.isArray(privacyPairs) && privacyPairs.length > 0;
      const hasRemove = Array.isArray(imageIdxs) && imageIdxs.length > 0;

      function exportFilename(suffix) { return '$export_filename_safe' + (suffix ? ('_' + suffix) : '') + '.png'; }

      //  read initial blur preference from layout meta
      function readBlurInitial() {
        try {
          const m = (gd && gd.layout && gd.layout.meta) ? gd.layout.meta : {};
          return !!m.blur_initial;
        } catch (e) { return false; }
      }
      const blurInitial = readBlurInitial();

      //  Capture and re-apply original image axes to prevent flips on toggles
      const originalAxisRanges = {}; // e.g., { xaxis: [0, W], yaxis: [H, 0], xaxis2: [...], yaxis2: [...] }
      function axisKeyFromId(id) { // 'x' -> 'xaxis', 'x2' -> 'xaxis2', 'y' -> 'yaxis', ...
        if (!id || typeof id !== 'string') return '';
        return (id[0] === 'x' ? 'xaxis' : 'yaxis') + (id.length > 1 ? id.slice(1) : '');
      }
      function captureImageAxisRanges() {
        try {
          const full = gd && gd._fullData ? gd._fullData : [];
          const seen = new Set();
          for (const idx of (imageIdxs || [])) {
            const f = full[idx];
            if (!f || !f.xaxis || !f.yaxis) continue;
            const xaId = f.xaxis._id || 'x';
            const yaId = f.yaxis._id || 'y';
            const keys = [axisKeyFromId(xaId), axisKeyFromId(yaId)];
            for (const key of keys) {
              if (!key || seen.has(key)) continue;
              const ax = (gd.layout && gd.layout[key]) || {};
              if (Array.isArray(ax.range) && ax.range.length === 2) {
                originalAxisRanges[key] = ax.range.slice();
              }
              seen.add(key);
            }
          }
        } catch (e) { /* no-op */ }
      }
      async function reapplyImageAxisRanges() {
        if (!gd) return;
        const keys = Object.keys(originalAxisRanges);
        if (!keys.length) return;
        const payload = {};
        for (const key of keys) {
          const rng = originalAxisRanges[key];
          payload[`${key}.autorange`] = false;
          payload[`${key}.range`] = rng.slice();
          payload[`${key}.fixedrange`] = true;
        }
        try { await Plotly.relayout(gd, payload); } catch (e) { console.warn(e); }
      }

      // Landmarks toggle
      let landmarksOn = true;
      function updateLandmarksBtn() {
        if (landmarksBtn) {
          landmarksBtn.textContent = 'Landmarks: ' + (landmarksOn ? 'Off' : 'On');
          landmarksBtn.style.borderColor = landmarksOn ? '#2ecc71' : '#999';
        }
      }
      async function setLandmarks(on) {
        landmarksOn = on;
        if (hasLandmarks) {
          try { await Plotly.restyle(gd, {visible: on}, overlayIdx); } catch (e) { console.warn(e); }
        }
        updateLandmarksBtn();
        //  enforce original axes after visibility changes
        reapplyImageAxisRanges();
      }

      // Rectangles toggle (shapes + traces)
      let rectanglesOn = true;
      function updateRectanglesBtn() {
        if (rectanglesBtn) {
          rectanglesBtn.textContent = 'Rectangles: ' + (rectanglesOn ? 'Off' : 'On');
          rectanglesBtn.style.borderColor = rectanglesOn ? '#2ecc71' : '#999';
        }
      }
      async function setRectangles(on) {
        rectanglesOn = on;
        // Toggle rectangle shapes
        if (Array.isArray(rectShapeIdx) && rectShapeIdx.length) {
          const relayoutPayload = {};
          for (const si of rectShapeIdx) relayoutPayload[`shapes[${si}].visible`] = on;
          try { await Plotly.relayout(gd, relayoutPayload); } catch (e) { console.warn(e); }
        }
        // Toggle rectangle traces
        if (Array.isArray(rectTraceIdx) && rectTraceIdx.length) {
          try { await Plotly.restyle(gd, {visible: on}, rectTraceIdx); } catch (e) { console.warn(e); }
        }
        updateRectanglesBtn();
        reapplyImageAxisRanges();
      }

      // Blur helpers
      let blurOn = blurInitial;  //  honor initial request
      function updateBlurBtn() {
        if (blurBtn) {
          blurBtn.textContent = blurOn ? 'Unblur Face' : 'Blur Face';
          blurBtn.style.borderColor = blurOn ? '#2ecc71' : '#999';
        }
      }
      function setBlur(on) {
        blurOn = on;
        if (hasPrivacy) {
          const origIdx = [], blurIdx = [];
          for (const pair of privacyPairs) {
            if (Array.isArray(pair) && pair.length === 2) {
              origIdx.push(pair[0]); blurIdx.push(pair[1]);
            }
          }
          try {
            if (removeOn) {
              Plotly.restyle(gd, {visible: false}, origIdx);
              Plotly.restyle(gd, {visible: false}, blurIdx);
            } else {
              Plotly.restyle(gd, {visible: on}, blurIdx);
              Plotly.restyle(gd, {visible: !on}, origIdx);
            }
          } catch(e) { console.warn(e); }
        }
        updateBlurBtn();
        reapplyImageAxisRanges();
      }

      // Remove helpers
      let removeOn = false;
      function updateRemoveBtn() {
        if (removeBtn) {
          removeBtn.textContent = removeOn ? 'Show Face' : 'Remove Face';
          removeBtn.style.borderColor = removeOn ? '#2ecc71' : '#999';
        }
      }
      function setRemove(on) {
        removeOn = on;
        if (hasRemove) {
          try {
            Plotly.restyle(gd, {visible: !on}, imageIdxs);
            if (!on && hasPrivacy) setBlur(blurOn); // restore blur state
          } catch(e) { console.warn(e); }
        }
        updateRemoveBtn();
        //  enforce original axes after visibility changes
        reapplyImageAxisRanges();
      }

      function afterPlotted() {
        captureImageAxisRanges();     //  capture once
        reapplyImageAxisRanges();     //  enforce immediately
        if (hasLandmarks) setLandmarks(true);
        if (hasRectangles) setRectangles(true);
        if (hasPrivacy) setBlur(blurOn);  //  apply requested initial blur
        if (hasRemove) setRemove(false);
      }

      if (gd && gd.addEventListener) {
        gd.addEventListener('plotly_afterplot', afterPlotted, { once: true });
      }
      setTimeout(afterPlotted, 150);

      if (landmarksBtn) landmarksBtn.addEventListener('click', function() { setLandmarks(!landmarksOn); });
      if (rectanglesBtn) rectanglesBtn.addEventListener('click', function() { setRectangles(!rectanglesOn); });
      if (blurBtn) blurBtn.addEventListener('click', function() { setBlur(!blurOn); });
      if (removeBtn) removeBtn.addEventListener('click', function() { setRemove(!removeOn); });

      exportBtn.addEventListener('click', async function() {
        try {
          // Render to an offscreen div without updatemenus (ISO/Front/etc) so buttons don't appear in export
          const tmp = document.createElement('div');
          tmp.style.position = 'fixed';
          tmp.style.left = '-10000px';
          tmp.style.top = '-10000px';
          document.body.appendChild(tmp);

          const dataCopy = JSON.parse(JSON.stringify(gd.data || []));
          const layoutCopy = JSON.parse(JSON.stringify(gd.layout || {}));
          layoutCopy.updatemenus = []; // remove camera preset buttons
          // Keep same size if present
          if (gd.layout && typeof gd.layout.width === 'number') layoutCopy.width = gd.layout.width;
          if (gd.layout && typeof gd.layout.height === 'number') layoutCopy.height = gd.layout.height;

          await Plotly.newPlot(tmp, dataCopy, layoutCopy, {displayModeBar: false, staticPlot: true});
          const url = await Plotly.toImage(tmp, {format: 'png', scale: 4});
          Plotly.purge(tmp);
          document.body.removeChild(tmp);

          const a = document.createElement('a');
          a.href = url;
          a.download = exportFilename('full');
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
        } catch (e) { console.error(e); }
      });

      if (exportPoseBtn) {
        exportPoseBtn.addEventListener('click', async function() {
          try {
            const srcGd = gd;
            const poseData = (srcGd.data || []).filter(t => (t && (t.type === 'scatter3d' || t.type === 'mesh3d' || t.type === 'surface')));
            if (!poseData.length) { console.warn('No 3D pose traces found for export'); return; }
            const baseLayout = JSON.parse(JSON.stringify(srcGd.layout || {}));
            baseLayout.updatemenus = []; // ensure ISO/Front/etc buttons not included in pose-only export
            // Remove annotations to avoid misplacement below scenes
            baseLayout.annotations = [];
            //  Remove any 2D shapes (e.g., red rectangles from top images) and images
            baseLayout.shapes = [];
            baseLayout.images = [];
            //  Drop 2D cartesian axes (xaxis*, yaxis*) not needed in pose-only export
            try {
              const axisKeys = Object.keys(baseLayout).filter(k => /^xaxis\d*$|^yaxis\d*$/i.test(k));
              for (const k of axisKeys) delete baseLayout[k];
            } catch (e) { /* no-op */ }

            // Normalize scene Y domains to occupy full height
            const sceneKeys = Object.keys(baseLayout).filter(k => (k === 'scene' || /^scene\d+$/.test(k)));
            // Sort scenes left-to-right by domain.x
            sceneKeys.sort((a, b) => {
              const da = (baseLayout[a] && baseLayout[a].domain && baseLayout[a].domain.x) ? baseLayout[a].domain.x[0] : 0;
              const db = (baseLayout[b] && baseLayout[b].domain && baseLayout[b].domain.x) ? baseLayout[b].domain.x[0] : 0;
              return da - db;
            });
            for (const key of sceneKeys) {
              if (baseLayout[key] && baseLayout[key].domain) {
                baseLayout[key].domain.y = [0, 1];
              }
              if (baseLayout[key]) {
                baseLayout[key].dragmode = false;
                // Hide axis numbering for pose-only export too
                baseLayout[key].xaxis = Object.assign({}, baseLayout[key].xaxis, {showticklabels:false, ticks:'', tickvals:[], showgrid:false, title:{text:''}});
                baseLayout[key].yaxis = Object.assign({}, baseLayout[key].yaxis, {showticklabels:false, ticks:'', tickvals:[], showgrid:false, title:{text:''}});
                baseLayout[key].zaxis = Object.assign({}, baseLayout[key].zaxis, {showticklabels:false, ticks:'', tickvals:[], showgrid:false, title:{text:''}});
              }
            }
            // Prepare per-column titles from meta
            const titles = (srcGd.layout && srcGd.layout.meta && srcGd.layout.meta.pose_column_titles) ? srcGd.layout.meta.pose_column_titles : [];
            // Euler angles for bottom labels (from meta if present)
            const eulers = (srcGd.layout && srcGd.layout.meta && srcGd.layout.meta.pose_eulers) ? srcGd.layout.meta.pose_eulers : [];
            // Add annotations above each scene with Frame text only (no Euler values)
            const anns = [];
            for (let i = 0; i < sceneKeys.length; i++) {
              const key = sceneKeys[i];
              const dom = (baseLayout[key] && baseLayout[key].domain) ? baseLayout[key].domain : {x: [i / sceneKeys.length, (i + 1) / sceneKeys.length], y: [0,1]};
              const xmid = (dom.x[0] + dom.x[1]) / 2.0;
              let topText = 'Frame ' + (i + 1);
              if (Array.isArray(eulers) && i < eulers.length && eulers[i] && typeof eulers[i].frame_idx === 'number') {
                topText = 'Frame ' + eulers[i].frame_idx;
              } else if (Array.isArray(titles) && i < titles.length && typeof titles[i] === 'string') {
                const mF = /Frame\s*(\d+)/i.exec(titles[i]);
                if (mF) topText = 'Frame ' + mF[1];
              }
              anns.push({
                text: topText,
                x: xmid,
                y: 1.06,
                xref: 'paper',
                yref: 'paper',
                showarrow: false,
                align: 'center',
                xanchor: 'center',
                yanchor: 'bottom',
                font: {size: 12}
              });
            }
            // Add bottom labels with Yaw/Pitch/Roll
            const bottomAnns = [];
            for (let i = 0; i < sceneKeys.length; i++) {
              const key = sceneKeys[i];
              const dom = (baseLayout[key] && baseLayout[key].domain) ? baseLayout[key].domain : {x: [i / sceneKeys.length, (i + 1) / sceneKeys.length], y: [0,1]};
              const xmid = (dom.x[0] + dom.x[1]) / 2.0;
              let label = '';
              if (Array.isArray(eulers) && i < eulers.length && eulers[i]) {
                const e = eulers[i];
                const yaw = (typeof e.yaw === 'number') ? e.yaw.toFixed(2) : '';
                const pitch = (typeof e.pitch === 'number') ? e.pitch.toFixed(2) : '';
                const roll = (typeof e.roll === 'number') ? e.roll.toFixed(2) : '';
                label = `Yaw: ${yaw}, Pitch: ${pitch}, Roll: ${roll}`;
              } else if (Array.isArray(titles) && i < titles.length && typeof titles[i] === 'string') {
                // Fallback: best-effort parse from title string
                const mYaw = /Yaw:\s*([-+]?\d*\.?\d+)/i.exec(titles[i]);
                const mPitch = /Pitch:\s*([-+]?\d*\.?\d+)/i.exec(titles[i]);
                const mRoll = /Roll:\s*([-+]?\d*\.?\d+)/i.exec(titles[i]);
                const yaw = mYaw ? mYaw[1] : '';
                const pitch = mPitch ? mPitch[1] : '';
                const roll = mRoll ? mRoll[1] : '';
                label = `Yaw: ${yaw}, Pitch: ${pitch}, Roll: ${roll}`;
              } else {
                label = 'Yaw: -, Pitch: -, Roll: -';
              }
              bottomAnns.push({
                text: label,
                x: xmid,
                y: -0.08,
                xref: 'paper',
                yref: 'paper',
                showarrow: false,
                align: 'center',
                xanchor: 'center',
                yanchor: 'top',
                font: {size: 12}
              });
            }
            baseLayout.annotations = anns.concat(bottomAnns);
            // Adjust size and margins for pose-only view (extra bottom space for labels)
            baseLayout.height = Math.max(380, Math.round(((srcGd.layout && srcGd.layout.height) ? srcGd.layout.height : 600) / 2) + 60);
            const currentB = (baseLayout.margin && typeof baseLayout.margin.b === 'number') ? baseLayout.margin.b : 40;
            baseLayout.margin = {t: 80, l: 20, r: 20, b: Math.max(110, currentB)};
            baseLayout.title = {text: '3D Pose', x: 0.5, xanchor: 'center'};
            baseLayout.paper_bgcolor = 'white';
            baseLayout.plot_bgcolor = 'white';
            // Create offscreen container
            const tmp = document.createElement('div');
            tmp.style.position = 'fixed';
            tmp.style.left = '-10000px';
            tmp.style.top = '-10000px';
            document.body.appendChild(tmp);
            await Plotly.newPlot(tmp, poseData, baseLayout, {displayModeBar: false, staticPlot: true});
            const url = await Plotly.toImage(tmp, {format: 'png', scale: 4});
            Plotly.purge(tmp);
            document.body.removeChild(tmp);
            const a = document.createElement('a');
            a.href = url;
            a.download = exportFilename('pose_only');
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
          } catch (e) { console.error(e); }
        });
      }
    })();
  </script>
</body>
</html>
""")

    html = html_template.safe_substitute(
        export_filename_safe=export_filename_safe,
        buttons_html=buttons_html,
        inner=inner,
        overlay_json=overlay_json,          # landmark traces
        privacy_pairs_json=privacy_pairs_json,
        image_indices_json=image_indices_json,
        rectangles_json=rectangles_json,    # rectangle shapes
        rectangle_traces_json=rectangle_traces_json,  #  rectangle traces
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# -----------------------------------------------------------------------------
# expressions visualization 
# -----------------------------------------------------------------------------

def visualize_expressions_3d(
    expressions,
    out_dir: str = "bitbox_viz.html",
    title: str = "Global Expressions over Time (3D)",
    video_path: Optional[str] = None,   # optional: show frames alongside the plot
    smooth: int = 0,                    # rolling window; 0/1 disables
    downsample: int = 1,                # keep every k-th frame (set 10 to speed up)
    play_fps: int = 5,                  # toolbar playback speed
    max_frames: int = 360,              # cap for HTML weight
    target_size: Tuple[int, int] = (400, 300),  # (w, h) video frame size for display
    overlay: Optional[object] = None,   # NEW: can be list/dict with type 'landmark'/'rectangle'
    cushion_ratio: float = 0.35,        # NEW: crop cushion ratio
):
    """Expressions-over-time 3D visualization with optional synchronized video.

    Left pane shows the current (cropped) frame and overlays; right pane shows 79 GE signals
    as progressive 3D polylines with a moving marker. Prev/Play/Next navigate both panes.
    """
    # ---------------------------- data prep ----------------------------
    def _to_df(obj: object) -> pd.DataFrame:
        if isinstance(obj, pd.DataFrame):
            return obj.copy()
        if isinstance(obj, dict) and isinstance(obj.get("data"), pd.DataFrame):
            return obj["data"].copy()
        raise ValueError("expressions must be a DataFrame or {'data': DataFrame}")

    df = _to_df(expressions)
    if df.empty:
        raise ValueError("expressions DataFrame is empty")

    # prefer GE* columns, else all numeric
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    ge_cols = [c for c in num_cols if str(c).lower().startswith("ge")] or num_cols
    if not ge_cols:
        raise ValueError("No numeric (or GE*) columns found")

    df = df.sort_index()

    if isinstance(smooth, int) and smooth >= 2:
        df[ge_cols] = df[ge_cols].rolling(window=int(smooth), min_periods=1).mean()

    stride = int(max(1, downsample or 1))
    df_ds = df.iloc[::stride].copy()
    if len(df_ds) == 0:
        raise ValueError("No rows after downsampling")

    if len(df_ds) > max_frames:
        df_ds = df_ds.iloc[:max_frames]

    orig_indices = df_ds.index.to_numpy(dtype=int)       # original frame ids
    ge_vals = df_ds[ge_cols].to_numpy(dtype=float)       # (T, 79)
    n_frames, ge_count = ge_vals.shape
    x_full = np.arange(n_frames, dtype=float)
    z_by_ge = ge_vals.T                                  # (79, T)

    # ---------------------------- overlay parsing ----------------------
    def _safe_df(d: Optional[dict]):
        return d["data"] if isinstance(d, dict) and "data" in d else None

    def _dtype(d: Optional[dict]) -> Optional[str]:
        return d.get("type") if isinstance(d, dict) else None

    overlay_land = None
    overlay_rect = None
    blur_default = False

    if isinstance(overlay, list):
        for value in overlay:
            if isinstance(value, dict) and "type" in value:
                if _dtype(value) == "landmark":
                    overlay_land = value
                elif _dtype(value) == "rectangle":
                    overlay_rect = value
            elif isinstance(value, dict):
                if bool(value.get("blur")) or (value.get("privacy") in (True, "blur", "on")):
                    blur_default = True
    elif isinstance(overlay, dict) and "type" in overlay:
        if _dtype(overlay) == "landmark":
            overlay_land = overlay
        elif _dtype(overlay) == "rectangle":
            overlay_rect = overlay
    elif isinstance(overlay, dict):
        if bool(overlay.get("blur")) or (overlay.get("privacy") in (True, "blur", "on")):
            blur_default = True

    overlay_land_df = _safe_df(overlay_land)
    overlay_rect_df = _safe_df(overlay_rect)
    overlay_present = (overlay_land_df is not None) or (overlay_rect_df is not None)



    # ---------------------------- figure ------------------------------
    has_video = bool(video_path and os.path.exists(video_path))
    if has_video:
        # equal widths so both panels feel the same size
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "xy"}, {"type": "scene"}]],
            column_widths=[0.50, 0.50],
            horizontal_spacing=0.06,
        )
        fig.update_layout(
            yaxis=dict(domain=[0, 0.8]),  # Align the video frame to the bottom
            scene=dict(domain=dict(y=[0.1, 0.9])),  # Center the 3D plot vertically within its subplot
        )
    else:
        fig = go.Figure()

    colorway = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    ]

    # left: stacked Image traces (raw or cropped) + caption annotation beneath
    nav_trace_indices: List[int] = []
    caption_anno_index = -1

    # NEW: keep per-frame overlay indices to toggle with nav
    overlay_trace_indices_by_frame: List[List[int]] = []
    rect_shape_indices_by_frame: List[List[int]] = []

    if has_video:
        W, H = int(target_size[0]), int(target_size[1])
        if not overlay_present:
            # Legacy/raw mode: use base64 for robust browser display
            blank = np.zeros((H, W, 3), dtype=np.uint8)
            ok_b, buf_b = cv2.imencode(".jpg", blank, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            blank_uri = "data:image/jpeg;base64," + (base64.b64encode(buf_b).decode("ascii") if ok_b else "")

            cap = cv2.VideoCapture(video_path)
            try:
                first = True
                for fid in orig_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
                    ok, bgr = cap.read()
                    if not ok or bgr is None:
                        data_uri = blank_uri
                    else:
                        if (bgr.shape[1], bgr.shape[0]) != (W, H):
                            bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
                        ok2, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        data_uri = "data:image/jpeg;base64," + (base64.b64encode(buf).decode("ascii") if ok2 else "")
                    fig.add_trace(go.Image(source=data_uri, visible=True, opacity=1.0 if first else 0.0), row=1, col=1)
                    nav_trace_indices.append(len(fig.data) - 1)
                    overlay_trace_indices_by_frame.append([])
                    rect_shape_indices_by_frame.append([])
                    first = False
            finally:
                cap.release()

            # lock axes to image extents
            fig.update_xaxes(range=[0, W], visible=False, fixedrange=True, row=1, col=1)
            fig.update_yaxes(range=[H, 0], visible=False, fixedrange=True, row=1, col=1)
        else:
            # Overlay mode: crop and draw overlays per frame
            cap = cv2.VideoCapture(video_path)
            try:
                first = True
                for t, fid in enumerate(orig_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
                    ok, bgr = cap.read()
                    if not ok or bgr is None:
                        # fallback to blank frame
                        rgb = np.zeros((H, W, 3), dtype=np.uint8)
                        crop = rgb
                        x1 = y1 = 0
                        sx = sy = 1.0
                        used_rect = False
                    else:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        crop = None
                        x1 = y1 = 0
                        sx = sy = 1.0
                        used_rect = False
                        # Prefer rectangle-driven crop
                        if overlay_rect_df is not None and (int(fid) in overlay_rect_df.index):
                            r = overlay_rect_df.loc[int(fid)]
                            try:
                                rx, ry, rw, rh = map(int, [r["x"], r["y"], r["w"], r["h"]])
                            except Exception:
                                try:
                                    rx, ry, rw, rh = map(int, r.values[:4])
                                except Exception:
                                    rx = ry = rw = rh = None
                            if None not in (rx, ry, rw, rh):
                                crop, box_scaled, _unused, (x1, y1, sx, sy) = crop_and_scale(
                                    rgb, x=rx, y=ry, w=rw, h=rh, cushion_ratio=cushion_ratio, target_size=(W, H)
                                )
                                used_rect = True
                                # add image
                                fig.add_trace(go.Image(z=crop, visible=True, opacity=1.0 if first else 0.0), row=1, col=1)
                                nav_trace_indices.append(len(fig.data) - 1)
                                # overlays for this frame
                                per_frame_traces: List[int] = []
                                per_frame_shapes: List[int] = []
                                # Rectangle shape overlay
                                if box_scaled is not None:
                                    bx, by, bw, bh = box_scaled
                                    fig.add_shape(
                                        type="rect",
                                        x0=bx, y0=by, x1=bx + bw, y1=by + bh,
                                        line=dict(color="red", width=2),
                                        visible=True if first else False,
                                        row=1, col=1,
                                    )
                                    if fig.layout.shapes:
                                        per_frame_shapes.append(len(fig.layout.shapes) - 1)
                                # Landmarks overlay if available
                                if overlay_land_df is not None and (int(fid) in overlay_land_df.index):
                                    lmk_row = overlay_land_df.loc[int(fid)]
                                    xs = lmk_row.values[::2].astype(float)
                                    ys = lmk_row.values[1::2].astype(float)
                                    xs_crop = (xs - x1) * sx
                                    ys_crop = (ys - y1) * sy
                                    fig.add_trace(
                                        go.Scatter(
                                            x=xs_crop, y=ys_crop,
                                            mode="markers",
                                            marker=dict(color="blue", size=5, symbol="circle"),
                                            name="Landmarks",
                                            showlegend=False,
                                            visible=True if first else False,
                                        ),
                                        row=1, col=1,
                                    )
                                    per_frame_traces.append(len(fig.data) - 1)
                                overlay_trace_indices_by_frame.append(per_frame_traces)
                                rect_shape_indices_by_frame.append(per_frame_shapes)
                        # Landmark-driven crop (fallback)
                        if crop is None and overlay_land_df is not None and (int(fid) in overlay_land_df.index):
                            lmk_row = overlay_land_df.loc[int(fid)]
                            xs = lmk_row.values[::2].astype(float)
                            ys = lmk_row.values[1::2].astype(float)
                            crop, xs_crop, ys_crop, (x1, y1, sx, sy) = crop_and_scale(
                                rgb, xs=xs, ys=ys, cushion_ratio=cushion_ratio, target_size=(W, H)
                            )
                            used_rect = False
                            # add image
                            fig.add_trace(go.Image(z=crop, visible=True, opacity=1.0 if first else 0.0), row=1, col=1)
                            nav_trace_indices.append(len(fig.data) - 1)
                            # overlays for this frame
                            per_frame_traces = []
                            per_frame_shapes = []
                            # Landmarks overlay (main)
                            fig.add_trace(
                                go.Scatter(
                                    x=xs_crop, y=ys_crop,
                                    mode="markers",
                                    marker=dict(color="blue", size=5, symbol="circle"),
                                    name="Landmarks",
                                    showlegend=False,
                                    visible=True if first else False,
                                ),
                                row=1, col=1,
                            )
                            per_frame_traces.append(len(fig.data) - 1)
                            # Rectangle overlay if present as companion
                            if overlay_rect_df is not None and (int(fid) in overlay_rect_df.index):
                                try:
                                    r = overlay_rect_df.loc[int(fid)]
                                    rx, ry, rw, rh = float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])
                                    x_rel, y_rel = (rx - x1) * sx, (ry - y1) * sy
                                    w_rel, h_rel = rw * sx, rh * sy
                                    fig.add_shape(
                                        type="rect",
                                        x0=x_rel, y0=y_rel, x1=x_rel + w_rel, y1=y_rel + h_rel,
                                        line=dict(color="red", width=2),
                                        visible=True if first else False,
                                        row=1, col=1,
                                    )
                                    if fig.layout.shapes:
                                        per_frame_shapes.append(len(fig.layout.shapes) - 1)
                                except Exception:
                                    pass
                            overlay_trace_indices_by_frame.append(per_frame_traces)
                            rect_shape_indices_by_frame.append(per_frame_shapes)
                        # If still no crop, show raw resized frame
                        if crop is None:
                            frame_resized = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
                            fig.add_trace(go.Image(z=frame_resized, visible=True, opacity=1.0 if first else 0.0), row=1, col=1)
                            nav_trace_indices.append(len(fig.data) - 1)
                            overlay_trace_indices_by_frame.append([])
                            rect_shape_indices_by_frame.append([])
                    first = False
            finally:
                cap.release()

            # lock axes to image extents
            fig.update_xaxes(range=[0, W], visible=False, fixedrange=True, row=1, col=1)
            fig.update_yaxes(range=[H, 0], visible=False, fixedrange=True, row=1, col=1)

        # caption under the video (centered under the left subplot)
        cap_text = f"Frame: 1 / {n_frames} (orig {int(orig_indices[0])})"
        fig.add_annotation(
            text=cap_text,
            x=0.5, xref="x domain",  # center of left subplot
            y=-0.14, yref="paper",   # below the subplot area
            showarrow=False,
            align="center",
        )
        caption_anno_index = len(fig.layout.annotations) - 1

    # right: 79 lines (start with 1 point each so they build up) + current marker
    line_trace_indices: List[int] = []
    for gi in range(ge_count):
        trace = go.Scatter3d(
            x=[0.0], y=[float(gi)], z=[z_by_ge[gi, 0]],
            mode="lines",
            line=dict(width=2, color=colorway[gi % len(colorway)]),
            showlegend=False,
            hovertemplate="t=%{x}<br>GE %{y}: %{z}<extra></extra>",
        )
        if has_video:
            fig.add_trace(trace, row=1, col=2)
        else:
            fig.add_trace(trace)
        line_trace_indices.append(len(fig.data) - 1)

    y_idx = np.arange(ge_count, dtype=float)
    z0 = ge_vals[0, :] if ge_count else np.array([])
    marker_trace = go.Scatter3d(
        x=np.zeros_like(y_idx),
        y=y_idx,
        z=z0,
        mode="markers",
        marker=dict(size=4, symbol="circle", color=[colorway[i % len(colorway)] for i in range(ge_count)]),
        showlegend=False,
        name="Current",
    )
    if has_video:
        fig.add_trace(marker_trace, row=1, col=2)
    else:
        fig.add_trace(marker_trace)
    marker_trace_index = len(fig.data) - 1

    # 3D layout: zoomed-in camera + aspect so it fills the right pane
    scene_kwargs = dict(
            xaxis=dict(title=f"Frame (x{stride})", range=[0, max(1, float(n_frames - 1))]),
            yaxis=dict(title="GE Index", range=[-1, ge_count]),
            zaxis=dict(title="Value"),

            camera=dict(
                eye=dict(x=1.75, y=-1.25, z=0.35),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),  # ensure camera is centered on the scene
                projection=dict(type="perspective"),
            ),
        )
    if has_video:
        fig.update_scenes(scene_kwargs, row=1, col=2)
    else:
        fig.update_scenes(scene_kwargs)

    # Build meta including overlay mappings
    meta_frame_nav = dict(
        type="expressions",
        frame_indices=list(range(n_frames)),
        active_index=0,
        play_fps=int(play_fps),
        trace_indices=nav_trace_indices if nav_trace_indices else [],
    )
    meta_extra = dict(
        expr_marker_trace_index=int(marker_trace_index),
        expr_marker_y=y_idx.tolist(),
        expr_marker_z_by_frame=[ge_vals[t, :].astype(float).tolist() for t in range(n_frames)],
        expr_line_trace_indices=line_trace_indices,
        expr_line_z_by_ge=[z_by_ge[g, :].astype(float).tolist() for g in range(ge_count)],
        expr_line_x_full=x_full.astype(float).tolist(),
        expr_line_y_const=[float(g) for g in range(ge_count)],
        expr_orig_indices=orig_indices.astype(int).tolist(),
        frame_caption_anno_index=int(caption_anno_index),
    )
    # Attach per-frame overlay meta (only when video and overlay present)
    if has_video and overlay_present:
        meta_extra["overlay_traces_by_frame"] = overlay_trace_indices_by_frame
        meta_extra["rectangle_shapes_by_frame"] = rect_shape_indices_by_frame
        if blur_default:
            # For future privacy toggles; no button here but keep state
            meta_extra["blur_initial"] = True

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        font={"family": "Roboto, sans-serif", "size": 14, "color": "#111"},
        height=760 if has_video else 700,
        margin=dict(t=40, l=30, r=30, b=110 if has_video else 60),  # extra bottom for caption
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        uirevision=True,
        meta=dict(frame_nav=meta_frame_nav, **meta_extra),
    )

    # ---------------------------- HTML (toolbar) ----------------------
    inner = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True, "displayModeBar": True},
        div_id="plot",
    )
    export_filename_safe = title.replace(" ", "_")

    buttons_html = (
        '<button id="prevBtn" class="btn">&#8592; Prev</button>'
        '<button id="playBtn" class="btn">Play</button>'
        '<button id="nextBtn" class="btn">Next &#8594;</button>'
        ' &nbsp; '
        '<button id="exportBtn" class="btn export">Export (PNG)</button>'
        ' '
        '<button id="export3DBtn" class="btn export">Export 3D (PNG)</button>'
    )

    html_template = Template(r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>$export_filename_safe</title>
  <style>
    body { margin: 0; background: #fff; font-family: Roboto, Helvetica, Arial, sans-serif; display: flex; flex-direction: column; align-items: center; }
    #plot-wrapper { margin-top: 38px; } /* add space above the title */
    .toolbar { margin: 14px 0 22px; display: flex; gap: 10px; align-items: center; justify-content: center; width: 100%; flex-wrap: wrap; }
    .btn { padding: 6px 12px; border-radius: 6px; border: 2px solid #000; background: #f8f8f8; color: #222; cursor: pointer; user-select: none; }
    .btn:hover { background: #eee; }
    .btn.export { border-color: #000; }
    .btn:active, .btn.active { border-color: #2ecc71; }
    #plot { max-width: 1200px; }
  </style>
</head>
<body>
  <div id="plot-wrapper">$inner</div>
  <div class="toolbar">$buttons_html</div>
  <script>
    (function(){
      const gd = document.getElementById('plot') || document.querySelector('.js-plotly-plot');
      const prevBtn = document.getElementById('prevBtn');
      const nextBtn = document.getElementById('nextBtn');
      const playBtn = document.getElementById('playBtn');
      const exportBtn = document.getElementById('exportBtn');
      const export3DBtn = document.getElementById('export3DBtn');

      let playing = false, timer = null, activeIndex = 0;

      function nav() { try { return (gd && gd.layout && gd.layout.meta && gd.layout.meta.frame_nav) || {}; } catch(e) { return {}; } }
      function frames(){ const m = nav(); return Array.isArray(m.frame_indices) ? m.frame_indices.slice() : []; }
      function fps(){ const m = nav(); return Math.max(1, Number(m.play_fps)||5); }
      function navTraceIdxs(){ const m = nav(); return (m.trace_indices || []).slice(); }

      function meta(){
        const m = (gd && gd.layout && gd.layout.meta) || {};
        return {
          markerIdx: (typeof m.expr_marker_trace_index === 'number') ? m.expr_marker_trace_index : -1,
          markerY:   Array.isArray(m.expr_marker_y) ? m.expr_marker_y : [],
          markerZs:  Array.isArray(m.expr_marker_z_by_frame) ? m.expr_marker_z_by_frame : [],
          lineIdxs:  Array.isArray(m.expr_line_trace_indices) ? m.expr_line_trace_indices : [],
          lineZbyGe: Array.isArray(m.expr_line_z_by_ge) ? m.expr_line_z_by_ge : [],
          lineX:     Array.isArray(m.expr_line_x_full) ? m.expr_line_x_full : [],
          lineYc:    Array.isArray(m.expr_line_y_const) ? m.expr_line_y_const : [],
          origIdx:   Array.isArray(m.expr_orig_indices) ? m.expr_orig_indices : [],
          capIdx:    (typeof m.frame_caption_anno_index === 'number') ? m.frame_caption_anno_index : -1,
          overTrByF: Array.isArray(m.overlay_traces_by_frame) ? m.overlay_traces_by_frame : [],
          rectShByF: Array.isArray(m.rectangle_shapes_by_frame) ? m.rectangle_shapes_by_frame : [],
          blurInit: !!m.blur_initial,
        };
      }

      function setActive(i){
        const n = frames().length;
        if (!n) return;
        activeIndex = ((i % n) + n) % n;
      }

      async function initImageStack(){
        const idxs = navTraceIdxs();
        if (!idxs.length) return;
        await Plotly.restyle(gd, {visible: true, opacity: 0}, idxs);
        await Plotly.restyle(gd, {opacity: 1}, [idxs[activeIndex]]);
        await updateOverlaysForFrame(activeIndex);
      }
      async function applyActiveImage(){
        const idxs = navTraceIdxs();
        if (!idxs.length) return;
        const tIdx = idxs[activeIndex];
        await Plotly.restyle(gd, {opacity: 0}, idxs);
        await Plotly.restyle(gd, {opacity: 1}, [tIdx]);
        await updateOverlaysForFrame(activeIndex);
      }

      async function updateLinesAndMarkerAndCaption(){
        const m = meta();
        const t = activeIndex;

        // marker (one point per GE at frame t)
        if (m.markerIdx >= 0 && m.markerY.length && Array.isArray(m.markerZs[t])) {
          const y = m.markerY.slice();
          const x = new Array(y.length).fill(t);
          const z = m.markerZs[t].slice();
          await Plotly.restyle(gd, {x: [x], y: [y], z: [z]}, [m.markerIdx]);
        }

        // progressive lines
        const idxs = m.lineIdxs;
        if (idxs.length && m.lineZbyGe.length) {
          const xs = [], ys = [], zs = [];
          for (let g = 0; g < idxs.length; g++) {
            const zfull = m.lineZbyGe[g] || [];
            const xfull = m.lineX || [];
            const k = Math.min(t + 1, Math.min(zfull.length, xfull.length));
            xs.push([].slice.call(xfull, 0, k));
            ys.push(new Array(k).fill(m.lineYc[g] || g));
            zs.push([].slice.call(zfull, 0, k));
          }
          await Plotly.restyle(gd, {x: xs, y: ys, z: zs}, idxs);
        }

        // caption under the video (annotation text)
        const total = frames().length;
        const orig = (Array.isArray(m.origIdx) && m.origIdx[t] !== undefined) ? m.origIdx[t] : t;
        if (m.capIdx >= 0) {
          const payload = {};
          payload['annotations[' + m.capIdx + '].text'] = 'Frame: ' + (t+1) + ' / ' + total + ' (orig ' + orig + ')';
          await Plotly.relayout(gd, payload);
        }
      }

      // NEW: toggle overlays per active frame
      async function updateOverlaysForFrame(t){
        const m = meta();
        const over = m.overTrByF || [];
        const rects = m.rectShByF || [];
        // traces
        const allTr = [].concat.apply([], over);
        if (allTr.length) {
          try { await Plotly.restyle(gd, {visible: false}, allTr); } catch(e) {}
        }
        const currTr = (Array.isArray(over[t]) ? over[t] : []);
        if (currTr.length) {
          try { await Plotly.restyle(gd, {visible: true}, currTr); } catch(e) {}
        }
        // shapes
        const allSh = [].concat.apply([], rects);
        const payload = {};
        for (const si of allSh) payload['shapes['+si+'].visible'] = false;
        const currSh = (Array.isArray(rects[t]) ? rects[t] : []);
        for (const si of currSh) payload['shapes['+si+'].visible'] = true;
        if (Object.keys(payload).length) {
          try { await Plotly.relayout(gd, payload); } catch(e) {}
        }
      }

      function start(){
        if (playing) return;
        playing = true; if (playBtn) playBtn.textContent = 'Pause';
        timer = setInterval(async function(){
          setActive(activeIndex + 1);
          await applyActiveImage();
          await updateLinesAndMarkerAndCaption();
        }, Math.round(1000 / fps()));
      }
      function stop(){
        playing = false; if (playBtn) playBtn.textContent = 'Play';
        if (timer) { clearInterval(timer); timer = null; }
      }

      function flash(el){
        if (!el) return;
        el.classList.add('active');
        setTimeout(() => el.classList.remove('active'), 180);
      }

      // NEW: export only the 3D scene (right subplot or full figure if no video)
      function exportSceneOnly(){
        try{
          const m = meta();
          const idxs = (m.lineIdxs || []).slice();
          if (m.markerIdx >= 0) idxs.push(m.markerIdx);
          if (!idxs.length) return;
          // Clone current traces (so we capture current progressive lines and marker state)
          const traces = [];
          for (const i of idxs){
            const tr = gd.data && gd.data[i];
            if (tr) traces.push(JSON.parse(JSON.stringify(tr)));
          }
          if (!traces.length) return;
          // Clone current scene layout and reset domain so it fills the figure
          const scene = (gd.layout && gd.layout.scene) ? JSON.parse(JSON.stringify(gd.layout.scene)) : {};
          if (scene && scene.domain) delete scene.domain; // remove subplot domain from original figure
          if (scene && scene.anchor) delete scene.anchor;

          // Determine export size from current figure if possible
          const dfl = (gd && gd._fullLayout) || {};
          const width  = Math.max(800, Math.min(1600, Number(dfl.width)  || 1200));
          const height = Math.max(600, Math.min(1200, Number(dfl.height) || 800));

          // Pull over base figure theming
          const baseLayout = (gd && gd.layout) || {};
          const titleText = (baseLayout.title && (typeof baseLayout.title === 'string' ? baseLayout.title : baseLayout.title.text)) || '3D Expressions';
          const font = baseLayout.font ? JSON.parse(JSON.stringify(baseLayout.font)) : null;
          const paperBG = baseLayout.paper_bgcolor || 'white';
          const plotBG = baseLayout.plot_bgcolor || 'white';

          // Build layout with explicit size and title so Plotly centers correctly
          const lay = {
            scene: scene,
            paper_bgcolor: paperBG,
            plot_bgcolor: plotBG,
            showlegend: false,
            margin: {t: 70, l: 2, r: 2, b: 2},
            width: width,
            height: height,
            autosize: false,
            title: { text: titleText, x: 0.5, xanchor: 'center' }
          };
          if (font) lay.font = font;

          const tmp = document.createElement('div');
          tmp.style.position = 'fixed';
          tmp.style.left = '-2000px';
          tmp.style.top = '-2000px';
          tmp.style.width = width + 'px';
          tmp.style.height = height + 'px';
          document.body.appendChild(tmp);
          Plotly.newPlot(tmp, traces, lay, {displayModeBar: false, responsive: false})
            .then(function(){
              return Plotly.downloadImage(tmp, {format: 'png', width: width, height: height, filename: 'bitbox_viz_3d'});
            })
            .then(function(){
              Plotly.purge(tmp); document.body.removeChild(tmp);
            })
            .catch(function(){
              try { Plotly.purge(tmp); document.body.removeChild(tmp); } catch(e) {}
            });
        } catch(e) { /* no-op */ }
      }

      // Lightweight full export: current frame image + its overlays + current 3D scene
      function exportFullLightweight(){
        try{
          const hasVideo = (navTraceIdxs().length > 0);
          // If no video, reuse the 3D-only exporter
          if (!hasVideo) { exportSceneOnly(); return; }

          const dfl = (gd && gd._fullLayout) || {};
          const width  = Math.max(800, Math.min(1920, Number(dfl.width)  || 1280));
          const height = Math.max(600, Math.min(1200, Number(dfl.height) || 760));

          // Collect left image + overlays for current frame
          const t = activeIndex;
          const imgIdxs = navTraceIdxs();
          if (!imgIdxs.length) return;
          const activeImgIdx = imgIdxs[((t % imgIdxs.length)+imgIdxs.length)%imgIdxs.length];

          const traces = [];
          const pushClone = (idx) => {
            const tr = gd.data && gd.data[idx];
            if (!tr) return;
            const clone = JSON.parse(JSON.stringify(tr));
            clone.visible = true;
            if (typeof clone.opacity !== 'undefined') clone.opacity = 1;
            traces.push(clone);
          };

          // image
          pushClone(activeImgIdx);

          // overlays (points/lines) for this frame
          const m = meta();
          const overlaysThis = (Array.isArray(m.overTrByF) && Array.isArray(m.overTrByF[t])) ? m.overTrByF[t] : [];
          overlaysThis.forEach(pushClone);

          // 3D traces (lines + marker)
          const sceneIdxs = (m.lineIdxs || []).slice();
          if (m.markerIdx >= 0) sceneIdxs.push(m.markerIdx);
          sceneIdxs.forEach(pushClone);

          // Clone scene layout, clear subplot anchoring so it takes right half
          const scene = (gd.layout && gd.layout.scene) ? JSON.parse(JSON.stringify(gd.layout.scene)) : {};
          if (scene && scene.domain) delete scene.domain;
          if (scene && scene.anchor) delete scene.anchor;

          // Build axes for left image pane using existing xaxis/yaxis settings if present
          const xax = (gd.layout && gd.layout.xaxis) ? JSON.parse(JSON.stringify(gd.layout.xaxis)) : {visible:false, fixedrange:true};
          const yax = (gd.layout && gd.layout.yaxis) ? JSON.parse(JSON.stringify(gd.layout.yaxis)) : {visible:false, fixedrange:true};

          // Place subplots side-by-side
          const spacing = 0.06; // match original
          const leftMax = 0.5 - spacing/2;
          const rightMin = 0.5 + spacing/2;
          xax.domain = [0, Math.max(0.1, leftMax)];
          // Full height
          // y domain by default [0,1]
          scene.domain = {x: [Math.min(0.9, rightMin), 1], y: [0, 1]};

          // Copy only the shapes for this frame (rectangles)
          let shapes = [];
          const allShapes = (gd.layout && Array.isArray(gd.layout.shapes)) ? gd.layout.shapes : [];
          const shIdxs = (Array.isArray(m.rectShByF) && Array.isArray(m.rectShByF[t])) ? m.rectShByF[t] : [];
          if (allShapes && shIdxs.length) {
            shapes = shIdxs.map(si => {
              const s = allShapes[si];
              return s ? JSON.parse(JSON.stringify(s)) : null;
            }).filter(Boolean);
            // ensure shapes are visible and bound to left axes
            shapes.forEach(s => { s.visible = true; if (!s.xref) s.xref = 'x'; if (!s.yref) s.yref = 'y'; });
          }

          // Theming and title
          const baseLayout = (gd && gd.layout) || {};
          const titleText = (baseLayout.title && (typeof baseLayout.title === 'string' ? baseLayout.title : baseLayout.title.text)) || '';
          const font = baseLayout.font ? JSON.parse(JSON.stringify(baseLayout.font)) : null;
          const paperBG = baseLayout.paper_bgcolor || 'white';
          const plotBG = baseLayout.plot_bgcolor || 'white';

          const lay = {
            xaxis: xax,
            yaxis: yax,
            scene: scene,
            shapes: shapes,
            paper_bgcolor: paperBG,
            plot_bgcolor: plotBG,
            showlegend: false,
            margin: {t: titleText ? 70 : 10, l: 10, r: 10, b: 10},
            width: width,
            height: height,
            autosize: false,
          };
          if (titleText) lay.title = { text: titleText, x: 0.5, xanchor: 'center' };
          if (font) lay.font = font;

          const tmp = document.createElement('div');
          tmp.style.position = 'fixed';
          tmp.style.left = '-2000px';
          tmp.style.top = '-2000px';
          tmp.style.width = width + 'px';
          tmp.style.height = height + 'px';
          document.body.appendChild(tmp);

          Plotly.newPlot(tmp, traces, lay, {displayModeBar: false, responsive: false})
            .then(function(){
              return Plotly.downloadImage(tmp, {format: 'png', width: width, height: height, filename: 'bitbox_viz'});
            })
            .then(function(){ Plotly.purge(tmp); document.body.removeChild(tmp); })
            .catch(function(){ try { Plotly.purge(tmp); document.body.removeChild(tmp); } catch(e) {} });
        } catch(e) { /* no-op */ }
      }

      function bind(){
        const n = frames().length;
        const m = nav();
        activeIndex = (typeof m.active_index === 'number') ? m.active_index : 0;
        if (n) { initImageStack(); }
        updateLinesAndMarkerAndCaption();

        if (prevBtn) prevBtn.onclick = async function(){ flash(this); stop(); setActive(activeIndex - 1); await applyActiveImage(); await updateLinesAndMarkerAndCaption(); };
        if (nextBtn) nextBtn.onclick = async function(){ flash(this); stop(); setActive(activeIndex + 1); await applyActiveImage(); await updateLinesAndMarkerAndCaption(); };
        if (playBtn) playBtn.onclick = function(){ flash(this); playing ? stop() : start(); };
        if (exportBtn) exportBtn.onclick = function(){ flash(this); exportFullLightweight(); };
        if (export3DBtn) export3DBtn.onclick = function(){ flash(this); exportSceneOnly(); };
      }

      (function waitReady(){
        if (gd && gd._fullLayout) { bind(); }
        else { setTimeout(waitReady, 50); }
      })();
    })();
  </script>
</body>
</html>
""")

    # Resolve output path (file or directory)
    if os.path.isdir(out_dir):
        out_dir = os.path.join(out_dir, "bitbox_viz.html")

    html = html_template.safe_substitute(
        export_filename_safe=export_filename_safe,
        inner=inner,
        buttons_html=buttons_html,
    )
    os.makedirs(os.path.dirname(out_dir) or ".", exist_ok=True)
    with open(out_dir, "w", encoding="utf-8") as f:
        f.write(html)

    return fig



def write_video_overlay_html(
    video_src: str,
    out_path: str,
    rects_map: Optional[dict] = None,
    lands_map: Optional[dict] = None,
    can3d_map: Optional[dict] = None,      # frame → [[x,y,z], ...]
    fps: float = 30.0,
    video_w: Optional[int] = None,
    video_h: Optional[int] = None,
    title: str = "Video Overlay",
    cushion_ratio: float = 0.25,
    fixed_portrait: bool = True,
    portrait_w: int = 360,
    portrait_h: int = 480,
    can3d_decimate: int = 1,               # keep every k-th point (perf)
) -> None:
    """Write an HTML page showing a cropped video with overlays + a synchronized, interactive 3D canonical-landmark plot.

    Implemented:
      - No border around 3D plot.
      - No "3D only" toggle; 3D shows when data present.
      - No "reset camera" button.
      - Export PNG (title + video pane + 3D snapshot).
      - Face Blur and Hide Face toggles (whole frame), overlays always visible.
      - Buttons centered under the page.
    """
    import json, os
    from typing import List

    # Fallback target size if not defined by caller module
    try:
        TS_W, TS_H = TARGET_SIZE  # type: ignore[name-defined]
    except Exception:
        TS_W, TS_H = 320, 240

    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    try:
        rel_src = os.path.relpath(video_src, start=out_dir)
    except Exception:
        rel_src = video_src
    rel_src = rel_src.replace(os.sep, "/")

    rects_map = rects_map or {}
    lands_map = lands_map or {}
    can3d_map = can3d_map or {}

    # Optional decimation for can3d points (performance)
    if can3d_decimate and can3d_decimate > 1 and can3d_map:
        decimated = {}
        for f, pts in can3d_map.items():
            try:
                k = int(can3d_decimate)
                decimated[int(f)] = [p for i, p in enumerate(pts) if i % k == 0]
            except Exception:
                decimated[int(f)] = pts
        can3d_map = decimated

    total_frame_keys = sorted(set(rects_map.keys()) | set(lands_map.keys()) | set(can3d_map.keys()))
    MAX_INLINE_FRAMES = 300
    CHUNK_SIZE = 500
    use_chunks = len(total_frame_keys) > MAX_INLINE_FRAMES

    rects_json = "{}"
    lands_json = "{}"
    can3d_json = "{}"
    rect_chunk_starts: List[int] = []
    land_chunk_starts: List[int] = []
    can3d_chunk_starts: List[int] = []

    if not use_chunks:
        if rects_map: rects_json = json.dumps(rects_map, separators=(",", ":"))
        if lands_map: lands_json = json.dumps(lands_map, separators=(",", ":"))
        if can3d_map: can3d_json = json.dumps(can3d_map, separators=(",", ":"))
    else:
        def _make_chunks(src: dict, prefix: str) -> List[int]:
            if not src: return []
            starts: List[int] = []
            keys = sorted(int(k) for k in src.keys())
            if not keys: return []
            start = (keys[0] // CHUNK_SIZE) * CHUNK_SIZE
            while start <= keys[-1]:
                end = start + CHUNK_SIZE - 1
                chunk = {int(k): src[int(k)] for k in keys if start <= int(k) <= end}
                if chunk:
                    starts.append(start)
                    js_path = os.path.join(out_dir, f"{prefix}_chunk_{start}.js")
                    func = "registerRectsChunk" if prefix == "rects" else \
                           "registerLandsChunk" if prefix == "lands" else \
                           "registerCan3dChunk"
                    with open(js_path, "w", encoding="utf-8") as jf:
                        jf.write(f"{func}({start},{json.dumps(chunk, separators=(',', ':'))});")
                start += CHUNK_SIZE
            return starts

        rect_chunk_starts = _make_chunks(rects_map, "rects")
        land_chunk_starts = _make_chunks(lands_map, "lands")
        can3d_chunk_starts = _make_chunks(can3d_map, "can3d")

    fps_val = float(fps or 30.0)
    vw = int(video_w) if (video_w and video_w > 0) else None
    vh = int(video_h) if (video_h and video_h > 0) else None

    has_rects = bool(rects_map) or bool(rect_chunk_starts)
    has_lands = bool(lands_map) or bool(land_chunk_starts)
    has_can3d = bool(can3d_map) or bool(can3d_chunk_starts)

    if title == "Video Overlay":
        if has_can3d:
            computed_title = "Video + 3D Canonicalized Landmarks"
        elif has_lands and not has_rects:
            computed_title = "Video with Landmarks"
        elif has_rects and not has_lands:
            computed_title = "Video with Rectangles"
        elif has_rects and has_lands:
            computed_title = "Video with Rectangle and Landmark Overlay"
        else:
            computed_title = title
    else:
        computed_title = title

    # HTML template builder
    def _build_html(box_w: int, box_h: int) -> str:
        html_base = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>__TITLE__</title>
<style>
  body { margin:0; background:#fff; font-family:Roboto, Helvetica, Arial, sans-serif; color:#111; display:flex; flex-direction:column; align-items:center; }
  h2.title { margin:40px 0 14px; font-weight:400; font-size:20px; text-align:center; }
  .container { display:flex; gap:24px; justify-content:center; align-items:flex-start; padding:10px 16px 0; flex-wrap:wrap; }
  .video-wrap, .plot-wrap { position:relative; width:__BOX_W__px; height:__BOX_H__px; }
  /* 3D pane interactive, no border */
  .plot-wrap, #plot3d { pointer-events: auto; }
  .plot-wrap { border:none; border-radius:0; box-sizing:border-box; background:transparent; }
  video#vid { position:absolute; width:1px; height:1px; opacity:0; pointer-events:none; left:-9999px; top:-9999px; }
  .stage { position:relative; width:100%; height:100%; background:transparent; }
  canvas#view, canvas#overlay { position:absolute; left:0; top:0; width:100%; height:100%; pointer-events:none; }
  #plot3d { width:100%; height:100%; }
  #plot3d .gl-container canvas, #plot3d .draglayer, #plot3d .nsewdrag { pointer-events:auto !important; }
  /* Page-wide centered toolbar */
  .toolbar-page { width:100%; display:flex; justify-content:center; }
  .toolbar { margin:16px 0 24px; display:flex; gap:10px; justify-content:center; align-items:center; flex-wrap:wrap; white-space:nowrap; }
  .btn { padding:6px 12px; border-radius:6px; border:2px solid #2ecc71; background:#f8f8f8; cursor:pointer; }
  .seek { width:280px; accent-color:#2ecc71; }
  .time { font:12px/1.2 monospace; color:#333; }
</style>
</head>
<body>
  <h2 class="title">__TITLE__</h2>
  <script>window.__EXPORT_TITLE__ = "__TITLE__";</script>

  <div class="container">
    <div class="video-wrap">
      <video id="vid" src="__REL_SRC__" muted playsinline preload="metadata"></video>
      <div id="stage" class="stage">
        <canvas id="view"></canvas>
        <canvas id="overlay"></canvas>
      </div>
    </div>
    __PLOT_WRAP__
  </div>

  <div class="toolbar-page">
    <div class="toolbar" id="toolbar">
      <button id="playBtn" class="btn">Play</button>
      <input id="seek" class="seek" type="range" min="0" max="0" step="0.01" value="0"/>
      <span id="time" class="time">0:00 / 0:00</span>
      __RECT_BTN____LAND_BTN____BLUR_BTN____HIDE_BTN____EXPORT_BTN__
    </div>
  </div>

__PLOTLY_TAG__

<script>
  const FPS = __FPS__;
  const USE_CHUNKS = __USE_CHUNKS__;
  const CHUNK_SIZE = __CHUNK_SIZE__;
  const RECT_CHUNK_STARTS = __RECT_CHUNK_STARTS__;
  const LAND_CHUNK_STARTS = __LAND_CHUNK_STARTS__;
  const CAN3D_CHUNK_STARTS = __CAN3D_CHUNK_STARTS__;
  const CUSHION = __CUSHION__;
  const BOX_W = __BOX_W__;
  const BOX_H = __BOX_H__;
  const rectsByFrame = __RECTS_JSON__;
  const landsByFrame = __LANDS_JSON__;
  const can3dByFrame = __CAN3D_JSON__;
  const loadedRectChunks = new Set(), loadedLandChunks = new Set(), loadedCan3dChunks = new Set();
  const pendingRectChunks = new Set(), pendingLandChunks = new Set(), pendingCan3dChunks = new Set();

  function chunkStartFor(f){
    if(!CHUNK_SIZE||CHUNK_SIZE<=0) return 0;
    return Math.floor(Math.max(0,f)/CHUNK_SIZE)*CHUNK_SIZE;
  }
  function registerRectsChunk(start,data){try{Object.assign(rectsByFrame,data||{});loadedRectChunks.add(start);pendingRectChunks.delete(start);}catch(e){}}
  function registerLandsChunk(start,data){try{Object.assign(landsByFrame,data||{});loadedLandChunks.add(start);pendingLandChunks.delete(start);}catch(e){}}
  function registerCan3dChunk(start,data){try{Object.assign(can3dByFrame,data||{});loadedCan3dChunks.add(start);pendingCan3dChunks.delete(start);}catch(e){}}

  function loadChunk(p,start){
    if(!USE_CHUNKS) return;
    if(!CHUNK_SIZE||CHUNK_SIZE<=0) return;
    const starts = p==='rects'?RECT_CHUNK_STARTS : p==='lands'?LAND_CHUNK_STARTS : CAN3D_CHUNK_STARTS;
    const loaded = p==='rects'?loadedRectChunks : p==='lands'?loadedLandChunks : loadedCan3dChunks;
    const pending= p==='rects'?pendingRectChunks: p==='lands'?pendingLandChunks: pendingCan3dChunks;
    if(!starts.includes(start)||loaded.has(start)||pending.has(start)) return;
    pending.add(start);
    const s=document.createElement('script');
    s.src=`${p}_chunk_${start}.js`;
    s.async=true;
    s.onerror=()=>pending.delete(start);
    document.head.appendChild(s);
  }
  function ensureChunksForFrame(f){
    if(!USE_CHUNKS) return;
    const st=chunkStartFor(f);
    for(const p of ['rects','lands','can3d']){
      loadChunk(p,st); loadChunk(p,st+CHUNK_SIZE);
      if(st-CHUNK_SIZE>=0) loadChunk(p,st-CHUNK_SIZE);
    }
  }

  const vid=document.getElementById('vid');
  const view=document.getElementById('view'), overlay=document.getElementById('overlay');
  const vctx=view.getContext('2d'), octx=overlay.getContext('2d');
  const playBtn=document.getElementById('playBtn'), seek=document.getElementById('seek'), timeLbl=document.getElementById('time');
  const rectBtn=document.getElementById('rectBtn'), landBtn=document.getElementById('landBtn');
  const blurBtn=document.getElementById('blurBtn'), hideBtn=document.getElementById('hideBtn');
  const exportBtn=document.getElementById('exportBtn');
  const HAS_RECTS=__HAS_RECTS__, HAS_LANDS=__HAS_LANDS__, HAS_CAN3D=__HAS_CAN3D__;
  let showRects=HAS_RECTS, showLands=HAS_LANDS;
  let enableBlur=false, enableHide=false; // hide wins over blur
  let lastCrop=null;

  function updateBtns(){
    if(rectBtn){rectBtn.textContent='Rectangles: '+(showRects?'On':'Off');rectBtn.style.borderColor=showRects?'#2ecc71':'#999';}
    if(landBtn){landBtn.textContent='Landmarks: '+(showLands?'On':'Off');landBtn.style.borderColor=showLands?'#2ecc71':'#999';}
    if(blurBtn){blurBtn.textContent='Blur Face: '+(enableBlur?'On':'Off');blurBtn.style.borderColor=enableBlur?'#2ecc71':'#999';}
    if(hideBtn){hideBtn.textContent='Hide Face: '+(enableHide?'On':'Off');hideBtn.style.borderColor=enableHide?'#2ecc71':'#999';}
    playBtn.textContent=vid.paused?'Play':'Pause';
  }
  if(rectBtn)rectBtn.addEventListener('click',()=>{showRects=!showRects;updateBtns();});
  if(landBtn)landBtn.addEventListener('click',()=>{showLands=!showLands;updateBtns();});
  if(blurBtn)blurBtn.addEventListener('click',()=>{enableBlur=!enableBlur; if(enableBlur) enableHide=false; updateBtns();});
  if(hideBtn)hideBtn.addEventListener('click',()=>{enableHide=!enableHide; if(enableHide) enableBlur=false; updateBtns();});

  playBtn.addEventListener('click',()=>{vid.paused?vid.play():vid.pause();});
  seek.addEventListener('input',()=>{vid.currentTime=parseFloat(seek.value)||0;});
  function fmt(t){if(!isFinite(t))return'0:00';const m=Math.floor(t/60);const s=Math.floor(t%60).toString().padStart(2,'0');return m+':'+s;}
  function tickUI(){if(vid.readyState>=1){seek.max=vid.duration||seek.max;seek.value=vid.currentTime||0;timeLbl.textContent=`${fmt(vid.currentTime)} / ${fmt(vid.duration)}`;}requestAnimationFrame(tickUI);}

  function resizeCanvases(){
    const dpr=window.devicePixelRatio||1;
    for(const c of [view,overlay]){c.style.width=BOX_W+'px';c.style.height=BOX_H+'px';c.width=Math.round(BOX_W*dpr);c.height=Math.round(BOX_H*dpr);}
    vctx.setTransform(dpr,0,0,dpr,0,0);octx.setTransform(dpr,0,0,dpr,0,0);
    if(plotDiv && HAS_CAN3D){ try{Plotly.Plots.resize(plotDiv);}catch(_){/* no-op */} }
  }

  function cropBoxFromLandmarks(points,vw,vh){
    if(!points||!points.length)return null;
    let minx=Infinity,miny=Infinity,maxx=-Infinity,maxy=-Infinity;
    for(const p of points){const x=p[0],y=p[1];if(x<minx)minx=x;if(y<miny)miny=y;if(x>maxx)maxx=x;if(y>maxy)maxy=y;}
    let w=Math.max(1,maxx-minx),h=Math.max(1,maxy-miny);
    const padX=w*CUSHION,padY=h*CUSHION;
    let sx=Math.max(0,Math.floor(minx-padX));
    let sy=Math.max(0,Math.floor(miny-padY));
    let ex=Math.min(vw,Math.ceil(maxx+padX));
    let ey=Math.min(vh,Math.ceil(maxy+padY));
    return[sx,sy,Math.max(1,ex-sx),Math.max(1,ey-sy)];
  }
  function cropBoxFromRects(rects,vw,vh){
    if(!rects||!rects.length)return null;
    let minx=Infinity,miny=Infinity,maxx=-Infinity,maxy=-Infinity;
    for(const r of rects){const x1=r.x,y1=r.y,x2=r.x+r.w,y2=r.y+r.h;if(x1<minx)minx=x1;if(y1<miny)miny=y1;if(x2>maxx)maxx=x2;if(y2>maxy)maxy=y2;}
    let w=Math.max(1,maxx-minx),h=Math.max(1,maxy-miny);
    const padX=w*CUSHION,padY=h*CUSHION;
    let sx=Math.max(0,Math.floor(minx-padX));
    let sy=Math.max(0,Math.floor(miny-padY));
    let ex=Math.min(vw,Math.ceil(maxx+padX));
    let ey=Math.min(vh,Math.ceil(maxy+padY));
    return[sx,sy,Math.max(1,ex-sx),Math.max(1,ey-sy)];
  }

  function ensureAndGetDataFor(frame){
    if(USE_CHUNKS)ensureChunksForFrame(frame);
    return {lands:landsByFrame[frame]||null,rects:rectsByFrame[frame]||null,can3d:can3dByFrame[frame]||null};
  }

  // ---- 3D setup (Plotly) ----
  const plotDiv = document.getElementById('plot3d');
  // Frontal view by default
  let camera = { eye:{x:0, y:0, z:2.5}, up:{x:0, y:1, z:0} };

  function init3d(){
    if(!HAS_CAN3D || !plotDiv) return;

    const data = [{
      type: 'scatter3d',
      mode: 'markers',
      x: [0], y: [0], z: [0],
      marker: { size: 3 }
    }];

    const layout = {
      margin:{l:0,r:0,t:0,b:0},
      scene:{
        dragmode:'orbit',
        aspectmode:'data',
        camera: camera,
        xaxis:{visible:false, showgrid:false, zeroline:false},
        yaxis:{visible:false, showgrid:false, zeroline:false},
        zaxis:{visible:false, showgrid:false, zeroline:false}
      }
    };

    const config = {
      staticPlot: false,
      displayModeBar: true,
      scrollZoom: true,
      responsive: true,
      modeBarButtonsToRemove: ['toImage']
    };

    Plotly.newPlot(plotDiv, data, layout, config);
    plotDiv.on('plotly_relayout', (e) => { if (e && e['scene.camera']) camera = e['scene.camera']; });
  }

  function update3d(frame){
    if(!HAS_CAN3D || !plotDiv) return;
    const pts = can3dByFrame[frame];
    if(!pts) return;
    const x=[],y=[],z=[];
    for(const p of pts){ x.push(p[0]||0); y.push(p[1]||0); z.push(p[2]||0); }
    try { Plotly.restyle(plotDiv, {x:[x], y:[y], z:[z]}); } catch(_){}
  }

  // ---- Export (single PNG: title + left pane + right 3D) ----
  if (exportBtn) exportBtn.addEventListener('click', () => { exportComposite().catch(()=>{}); });

  async function exportComposite(){
    // Prefer injected title to avoid DOM/webfont timing issues
    let titleText = (typeof window.__EXPORT_TITLE__ === 'string' && window.__EXPORT_TITLE__.trim())
        ? window.__EXPORT_TITLE__.trim()
        : ((document.querySelector('h2.title')?.textContent || '').trim());

    // Ensure fonts are ready (best-effort)
    try { if (document.fonts && document.fonts.ready) await document.fonts.ready; } catch(_) {}

    // Left pane = base + overlays (already rendered to canvases)
    const left = document.createElement('canvas');
    left.width  = view.width;
    left.height = view.height;
    const lctx = left.getContext('2d');
    lctx.drawImage(view, 0, 0);
    lctx.drawImage(overlay, 0, 0);

    // Right pane = Plotly to PNG, if present
    let rightImg = null;
    if (HAS_CAN3D && typeof Plotly !== 'undefined' && plotDiv) {
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(1, Math.round(plotDiv.clientWidth  * dpr));
      const h = Math.max(1, Math.round(plotDiv.clientHeight * dpr));
      try {
        const dataUrl = await Plotly.toImage(plotDiv, { format:'png', width:w, height:h });
        rightImg = await new Promise((resolve) => { const img = new Image(); img.onload = () => resolve(img); img.src = dataUrl; });
      } catch(_) {}
    }

    // Layout with title
    const dpr = window.devicePixelRatio || 1;
    const GAP_X = Math.round(24 * dpr);
    const PAD_X = Math.round(24 * dpr);
    const PAD_Y = Math.round(20 * dpr);

    const titleSize = Math.round(20 * dpr);
    const lineH = Math.round(titleSize * 1.4);
    const titleH = titleText ? (PAD_Y + lineH) : 0;

    const contentW = left.width + (rightImg ? GAP_X + rightImg.width : 0);
    const contentH = Math.max(left.height, rightImg ? rightImg.height : 0);

    const outW = PAD_X + contentW + PAD_X;
    const outH = PAD_Y + titleH + contentH + PAD_Y;

    const out = document.createElement('canvas');
    out.width = outW; out.height = outH;
    const octx = out.getContext('2d');

    // white background
    octx.fillStyle = '#fff';
    octx.fillRect(0, 0, outW, outH);

    // title
    if (titleText) {
      octx.fillStyle = '#111';
      octx.textAlign = 'center';
      octx.textBaseline = 'top';
      // Use a system font stack to avoid CORS issues with webfonts
      octx.font = `${titleSize}px Helvetica, Arial, sans-serif`;
      octx.fillText(titleText, outW/2, PAD_Y);
    }

    // panes
    const contentTop = PAD_Y + titleH;
    const leftY = contentTop + Math.floor((contentH - left.height) / 2);
    octx.drawImage(left, PAD_X, leftY);

    if (rightImg) {
      const rightX = PAD_X + left.width + GAP_X;
      const rightY = contentTop + Math.floor((contentH - rightImg.height) / 2);
      octx.drawImage(rightImg, rightX, rightY);
    }

    // download
    const a = document.createElement('a');
    a.href = out.toDataURL('image/png');
    a.download = 'video_3d_with_title.png';
    document.body.appendChild(a); a.click(); a.remove();
  }

  // ---- Main draw loop (privacy + overlays) ----
  function draw(){
    const dpr = window.devicePixelRatio || 1;
    const targetW = view.width / dpr, targetH = view.height / dpr;

    const durFrames = (isFinite(vid.duration) && vid.duration>0 && FPS>0)
        ? Math.max(1, Math.floor(vid.duration*FPS)) : null;
    const rawFrame = Math.floor((vid.currentTime||0)*FPS + 0.0001);
    const frame = durFrames ? Math.max(0, Math.min(rawFrame, durFrames-1)) : Math.max(0, rawFrame);

    const vw = vid.videoWidth || 1, vh = vid.videoHeight || 1;
    const {lands, rects} = ensureAndGetDataFor(frame);

    // crop selection
    let crop = null;
    if (rects && rects.length) { crop = cropBoxFromRects(rects, vw, vh); if (crop) lastCrop = crop; }
    else if (lands && lands.length) { crop = cropBoxFromLandmarks(lands, vw, vh); if (crop) lastCrop = crop; }
    else if (lastCrop) { crop = lastCrop; }
    else { crop = [0,0,vw,vh]; }

    const [sx,sy,sw,sh] = crop;
    // COVER scaling
    const s = Math.max(targetW/sw, targetH/sh);
    const drawW = Math.ceil(sw * s), drawH = Math.ceil(sh * s);
    const dx = Math.floor((targetW - drawW) / 2), dy = Math.floor((targetH - drawH) / 2);

    vctx.clearRect(0,0,view.width,view.height);
    octx.clearRect(0,0,overlay.width,overlay.height);

    // base frame
    try { vctx.drawImage(vid, sx, sy, sw, sh, dx, dy, drawW, drawH); } catch(e){}

    // privacy
    if (enableHide) {
      vctx.save();
      vctx.globalAlpha = 1;
      vctx.fillStyle = '#fff';      // white background when hidden
      vctx.fillRect(0, 0, targetW, targetH);
      vctx.restore();
    } else if (enableBlur) {
      vctx.save();
      vctx.filter = 'blur(18px)';
      try { vctx.drawImage(vid, sx, sy, sw, sh, dx, dy, drawW, drawH); } catch(e){}
      vctx.filter = 'none';
      vctx.restore();
    }

    // overlays always visible
    if (HAS_RECTS && showRects && rects) {
      octx.strokeStyle = 'red'; octx.lineWidth = 2; octx.globalAlpha = 0.9;
      for (const r of rects) {
        octx.strokeRect((r.x - sx)*s + dx, (r.y - sy)*s + dy, r.w*s, r.h*s);
      }
    }
    if (HAS_LANDS && showLands && lands) {
      octx.fillStyle = 'rgba(0,102,255,0.95)';
      for (const p of lands) {
        const px = (p[0]-sx)*s + dx, py = (p[1]-sy)*s + dy;
        octx.beginPath(); octx.arc(px, py, 2.2, 0, 2*Math.PI); octx.fill();
      }
    }

    if (HAS_CAN3D) { update3d(frame); }
    requestAnimationFrame(draw);
  }

  // Keys: space play/pause, r rects, l lands, b blur, h hide, e export
  window.addEventListener('keydown',e=>{
    const k=(e.key||'').toLowerCase();
    if(k===' '){e.preventDefault();vid.paused?vid.play():vid.pause();}
    else if(k==='r'){showRects=!showRects;updateBtns();}
    else if(k==='l'){showLands=!showLands;updateBtns();}
    else if(k==='b'){enableBlur=!enableBlur; if(enableBlur) enableHide=false; updateBtns();}
    else if(k==='h'){enableHide=!enableHide; if(enableHide) enableBlur=false; updateBtns();}
    else if(k==='e'){exportComposite().catch(()=>{});}
  });

  vid.addEventListener('loadedmetadata',()=>{
    if(FPS>0 && isFinite(FPS)) { try { seek.step=(1.0/Number(FPS)).toFixed(3); } catch(_){} }
    try{ vid.play().catch(()=>{}); }catch(_){}
    resizeCanvases(); updateBtns();
    if(HAS_CAN3D) init3d();
    if(USE_CHUNKS) ensureChunksForFrame(0);
    requestAnimationFrame(draw);
    requestAnimationFrame(tickUI);
  });

  vid.addEventListener('timeupdate',()=>{
    seek.value=vid.currentTime||0;
    timeLbl.textContent=`${fmt(vid.currentTime)} / ${fmt(vid.duration)}`;
  });

  window.addEventListener('resize',resizeCanvases);
</script>
</body></html>
"""
        rect_btn   = "<button id='rectBtn' class='btn'>Rectangles: On</button>" if has_rects else ""
        land_btn   = "<button id='landBtn' class='btn'>Landmarks: On</button>" if has_lands else ""
        blur_btn   = "<button id='blurBtn' class='btn'>Blur Face: Off</button>"
        hide_btn   = "<button id='hideBtn' class='btn'>Hide Face: Off</button>"
        export_btn = "<button id='exportBtn' class='btn'>Export PNG</button>"
        plot_wrap  = "<div class='plot-wrap'><div id='plot3d'></div></div>" if has_can3d else ""
        plotly_tag = "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>" if has_can3d else ""

        return (
            html_base
            .replace("__TITLE__", computed_title)
            .replace("__REL_SRC__", rel_src)
            .replace("__BOX_W__", str(int(box_w)))
            .replace("__BOX_H__", str(int(box_h)))
            .replace("__FPS__", f"{fps_val:.6f}")
            .replace("__USE_CHUNKS__", str(use_chunks).lower())
            .replace("__CHUNK_SIZE__", str(CHUNK_SIZE if use_chunks else 0))
            .replace("__RECT_CHUNK_STARTS__", json.dumps(rect_chunk_starts))
            .replace("__LAND_CHUNK_STARTS__", json.dumps(land_chunk_starts))
            .replace("__CAN3D_CHUNK_STARTS__", json.dumps(can3d_chunk_starts))
            .replace("__CUSHION__", f"{float(cushion_ratio):.6f}")
            .replace("__RECTS_JSON__", rects_json)
            .replace("__LANDS_JSON__", lands_json)
            .replace("__CAN3D_JSON__", can3d_json)
            .replace("__HAS_RECTS__", str(has_rects).lower())
            .replace("__HAS_LANDS__", str(has_lands).lower())
            .replace("__HAS_CAN3D__", str(has_can3d).lower())
            .replace("__RECT_BTN__", rect_btn)
            .replace("__LAND_BTN__", land_btn)
            .replace("__BLUR_BTN__", blur_btn)
            .replace("__HIDE_BTN__", hide_btn)
            .replace("__EXPORT_BTN__", export_btn)
            .replace("__PLOT_WRAP__", plot_wrap)
            .replace("__PLOTLY_TAG__", plotly_tag)
        )

    html = _build_html(portrait_w, portrait_h) if fixed_portrait else _build_html(TS_W, TS_H)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
