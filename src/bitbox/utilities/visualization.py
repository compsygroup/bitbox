import os
from typing import List, Optional, Sequence, Tuple
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import json
import plotly.io as pio
from string import Template

TARGET_SIZE: Tuple[int, int] = (240, 300)

# -----------------------------------------------------------------------------
# Common Helpers
# -----------------------------------------------------------------------------

def get_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Load a single RGB frame by index from a video on disk.

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
# Plot Dispatch
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Rectangle plots
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
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def visualize_and_export(
    num_frames: int,
    video_path: Optional[str],
    out_dir: str,
    random_seed: int,
    rects: Optional[dict] = None,          # NOW OPTIONAL
    cushion_ratio: float = 0.35,
    overlay: Optional[dict] = None,        # Can be landmarks OR rectangles
    pose: Optional[dict] = None,
):
    """
    Flexible rectangles/landmarks plotter.

    - Uses pose to pick diverse frames for the image grid (falls back to random if pose is missing).
    - No 3D plotting; only images with optional rectangle/landmark overlays.
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
                frame, x=x, y=y, w=w, h=h, cushion_ratio=cushion_ratio
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
    overlay: Optional[dict] = None,  # can be landmarks, rectangles, or {"landmark": {...}, "rectangle": {...}}
    video_path: Optional[str] = None,    
    pose: Optional[dict] = None,           # used only to diversify selection
    land_can: Optional[dict] = None,
):
    """
    Top row: video frames (raw if no overlay; cropped + overlays if overlay present)
    Bottom row: 3D canonicalized landmarks.

    - Uses pose to pick diverse frames if available, else evenly from land_can.
    - When overlays are provided:
      * Prefer cropping by rectangles when available; else crop by landmarks.
      * Overlay landmarks (blue) and rectangles (red).
      * Blur toggle is available when any overlay is provided.
    """
    def _safe_df(d: Optional[dict]):
        return d["data"] if isinstance(d, dict) and "data" in d else None

    def _dtype(d: Optional[dict]) -> Optional[str]:
        return d.get("type") if isinstance(d, dict) else None

    # Inputs
    if land_can is None or _safe_df(land_can) is None or len(_safe_df(land_can)) == 0:
        return
    can_df = _safe_df(land_can)

    pose_df = _safe_df(pose)

    #  resolve pose angle columns (for labels under land_can when pose is provided)
    rx_col = ry_col = rz_col = None
    if pose_df is not None and len(pose_df) > 0:
        rx_col = _find_col(pose_df, ("Rx", "rx", "pitch"))
        ry_col = _find_col(pose_df, ("Ry", "ry", "yaw"))
        rz_col = _find_col(pose_df, ("Rz", "rz", "roll"))

    # Resolve overlay variants: dict with type, or a list containing both
    overlay_land = None
    overlay_rect = None
    blur_default = False  #  default blur state, can be enabled by overlay

    if isinstance(overlay, list):
        for value in overlay:
            if isinstance(value, dict) and "type" in value:
                if _dtype(value) == "landmark":
                    overlay_land = value
                elif _dtype(value) == "rectangle":
                    overlay_rect = value
            elif isinstance(value, dict):
                # Accept {"blur": True} or {"privacy": True/"blur"} anywhere in the list
                if bool(value.get("blur")) or (value.get("privacy") in (True, "blur", "on")):
                    blur_default = True
    elif isinstance(overlay, dict) and "type" in overlay:
        if _dtype(overlay) == "landmark":
            overlay_land = overlay
        elif _dtype(overlay) == "rectangle":
            overlay_rect = overlay
    elif isinstance(overlay, dict):
        # Accept single dict form: {"blur": True}
        if bool(overlay.get("blur")) or (overlay.get("privacy") in (True, "blur", "on")):
            blur_default = True

    overlay_land_df = _safe_df(overlay_land)
    overlay_rect_df = _safe_df(overlay_rect)
    overlay_present = (overlay_land_df is not None) or (overlay_rect_df is not None)  # 

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

    # Top row: frames (raw vs cropped) and optional overlays
    overlay_trace_indices: List[int] = []
    rect_shape_indices: List[int] = []
    image_trace_indices: List[int] = []
    privacy_pairs: List[Tuple[int, int]] = []  #  for blur toggle

    cushion_ratio = 0.35

    for i, fid in enumerate(selected_fids):
        # Load frame
        try:
            frame = get_frame(video_path, fid)
        except Exception:
            # Skip this column if frame missing
            continue

        added_overlay = False
        crop = None
        x1 = y1 = 0
        sx = sy = 1.0

        # Prefer rectangle-driven crop when both exist
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
                    frame, x=rx, y=ry, w=rw, h=rh, cushion_ratio=cushion_ratio
                )
                # ADD image + optional blur pair
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

                # Rectangle overlay
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

                # Landmarks overlay (if provided too)
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
                added_overlay = True

        elif overlay_land_df is not None and fid in overlay_land_df.index:
            # Landmark-driven crop (fallback when rectangles are absent)
            lmk_row = overlay_land_df.loc[fid]
            xs = lmk_row.values[::2].astype(float)
            ys = lmk_row.values[1::2].astype(float)
            crop, xs_crop, ys_crop, (x1, y1, sx, sy) = crop_and_scale(
                frame, xs=xs, ys=ys, cushion_ratio=cushion_ratio
            )
            # ADD image + optional blur pair
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
            added_overlay = True

            # Rectangle overlay (if present as companion)
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

        # No overlay or failed crop: show raw frame (resized to target cell)
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
    pose_eulers_meta: List[dict] = []  #  to store pose eulers for meta
    for i, fid in enumerate(selected_fids):
        if fid not in can_df.index:
            continue
        row = can_df.loc[fid]
        vals = getattr(row, "values", np.asarray(row))
        if len(vals) % 3 == 0:
            xyz = vals.reshape(-1, 3).astype(float)
        elif len(vals) % 2 == 0:
            xy = vals.reshape(-1, 2).astype(float)
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

        #  Add Yaw/Pitch/Roll labels under land_can plots when pose is available
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
                    # best-effort fallback
                    try:
                        s = pose_df.iloc[fid]
                    except Exception:
                        continue
                pitch = _get(s, rx_col, 0.0)
                yaw   = _get(s, ry_col, 0.0)
                roll  = _get(s, rz_col, 0.0)
                pose_eulers_meta.append({"frame_idx": int(fid), "pitch": float(pitch), "yaw": float(yaw), "roll": float(roll)})

                # Place a second line of labels beneath the existing frame label
                scene_id = f"scene{i + 1}" if i > 0 else "scene"
                dom = fig.layout[scene_id].domain
                xmid = (dom.x[0] + dom.x[1]) / 2.0
                ybelow2 = dom.y[0] - 0.16  # a bit lower to avoid overlap
                fig.add_annotation(
                    text=f"Yaw: {yaw:.2f}, Pitch: {pitch:.2f}, Roll: {roll:.2f}",
                    x=xmid, y=ybelow2, xref="paper", yref="paper",
                    showarrow=False, align="center", xanchor="center", yanchor="top",
                    font=dict(size=12),
                )
            except Exception:
                pass

    # Layout and styling
    if len(column_titles) == 0:
        return

    fig.update_layout(
        autosize=False,
        width=fig_w, height=fig_h,
        showlegend=False,
        title={"text": "Video (top) and Canonical Landmarks (bottom)", "x": 0.5, "xanchor": "center"},
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

    # Toolbar meta: enable toggles if overlays exist
    meta = {}
    if overlay_trace_indices: meta["overlay_trace_indices"] = overlay_trace_indices
    if rect_shape_indices:    meta["rectangle_shape_indices"] = rect_shape_indices
    if image_trace_indices:   meta["image_trace_indices"] = image_trace_indices
    if privacy_pairs:         meta["privacy_pairs"] = privacy_pairs   #  enable Blur button
    if blur_default:          meta["blur_initial"] = True             # 
    #  persist titles and eulers for export consistency
    if column_titles:         meta["pose_column_titles"] = column_titles
    if pose_eulers_meta:      meta["pose_eulers"] = pose_eulers_meta
    if meta: fig.update_layout(meta=meta)

    html_path = os.path.join(out_dir, "bitbox_viz.html")
    try:
        write_centered_html(fig, html_path, export_filename="bitbox visualizations", add_pose_only_export=False)
    except Exception:
        pass
    return

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
    """Write a self-contained HTML file that centers the figure and adds
    a small toolbar with:
      - Overlay toggle (only shown if overlays exist)
      - Blur Face toggle (only for rect/landmark plots when privacy pairs exist)
      - Remove Face toggle (hides all image traces)
      - Export for publication (PNG)
      - Export Pose Only (PNG) when requested by pose plots

    Uses Plotly's version-pinned CDN bundle via to_html to avoid plotly-latest
    incompatibilities that can hide images or break overlays.

    The overlay toggle uses indices stored in fig.layout.meta["overlay_trace_indices"].
    The blur toggle uses pairs in fig.layout.meta["privacy_pairs"].
    The remove toggle uses image indices in fig.layout.meta["image_trace_indices"].
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
          const seen =  Set();
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
          const url = await Plotly.toImage(gd, {format: 'png', scale: 4});
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
            // Remove annotations to avoid misplacement below scenes
            baseLayout.annotations = [];
            //  Remove any 2D shapes (e.g., red rectangles from top images) and images
            baseLayout.shapes = [];
            baseLayout.images = [];
            //  Drop 2D cartesian axes (xaxis*, yaxis*) not needed in pose-only export
            try {
              const axisKeys = Object.keys(baseLayout).filter(k => /^xaxis\\d*$|^yaxis\\d*$/i.test(k));
              for (const k of axisKeys) delete baseLayout[k];
            } catch (e) { /* no-op */ }

            // Normalize scene Y domains to occupy full height
            const sceneKeys = Object.keys(baseLayout).filter(k => (k === 'scene' || /^scene\\d+$$/.test(k)));
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
            await Plotly.Plot(tmp, poseData, baseLayout, {displayModeBar: false, staticPlot: true});
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

