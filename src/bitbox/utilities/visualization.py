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

def plot(
    data: dict,
    video_path: Optional[str] = None,
    output_dir: str = "output",
    random_seed: int = 42,
    num_frames: int = 5,
    overlay: Optional[dict] = None,
    land_can: Optional[dict] = None,
    rect: Optional[dict] = None,
):
    """Dispatch to appropriate plotter based on data['type'].

    - rectangle: plots face crops with rectangle, optionally overlay landmarks
    - landmark: plots face crops with landmarks (overlays disabled intentionally)
    - pose: plots top row images (cropped by rect) and bottom row 3D scatter of mean face under pose
    - landmark-can: plots 2D crop (from 2D landmarks) and 3D canonicalized landmarks
    """
    cushion_ratio=0.35
    out_dir = os.path.join(output_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    data_type = data.get("type", None)

    if data_type == "rectangle":
        return plot_rects(data, num_frames, video_path, out_dir, random_seed, cushion_ratio, overlay)
    elif data_type == "landmark":
        return plot_landmarks(data, num_frames, video_path, out_dir, random_seed, cushion_ratio, overlay=overlay)
    elif data_type == "pose":
        return plot_pose(data, land_can, rect, video_path, num_frames, out_dir)
    else:
        raise ValueError(f"Unknown data type: {data_type}")


# -----------------------------------------------------------------------------
# Rectangle plots
# -----------------------------------------------------------------------------

def plot_rects(
    rects: dict,
    num_frames: int, 
    video_path: Optional[str],
    out_dir: str,
    random_seed: int,
    cushion_ratio: float = 0.35,
    overlay: Optional[dict] = None,
):
    df = rects["data"]
    n = min(num_frames, len(df))
    if n <= 0:
        return
    frames = df.sample(n=n, random_state=random_seed)

    crops: List[np.ndarray] = []
    blurred_crops: List[np.ndarray] = []
    rel_boxes: List[Tuple[float, float, float, float]] = []
    overlay_landmarks: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
    overlay_df = overlay["data"] if overlay is not None else None

    for idx, row in frames.iterrows():
        if video_path:
            frame = get_frame(video_path, idx)
            x, y, w, h = map(int, [row["x"], row["y"], row["w"], row["h"]])
            # Use keyword args and unpack 4 return values; ignore the placeholder third value
            crop, box_scaled, _unused, (x1, y1, scale_x, scale_y) = crop_and_scale(
                frame, x=x, y=y, w=w, h=h, cushion_ratio=cushion_ratio
            )
            crops.append(crop)
            # generate a reasonably strong blur (kernel size proportional to crop)
            kx = max(15, (crop.shape[1] // 15) * 2 + 1)
            ky = max(15, (crop.shape[0] // 15) * 2 + 1)
            blurred = cv2.GaussianBlur(crop, (kx, ky), 0)
            blurred_crops.append(blurred)

            rel_boxes.append(box_scaled)

            # Optional overlay of landmarks
            if overlay_df is not None and idx in overlay_df.index:
                lmk_row = overlay_df.loc[idx]
                xs = lmk_row.values[::2].astype(int)
                ys = lmk_row.values[1::2].astype(int)
                xs_crop = (xs - x1) * scale_x
                ys_crop = (ys - y1) * scale_y
                overlay_landmarks.append((xs_crop, ys_crop))
            else:
                overlay_landmarks.append(None)
        else:
            overlay_landmarks.append(None)

    if not crops:
        return

    ncols = len(crops)
    fig = make_centered_subplot_with_overlay(
        crops,
        rel_boxes,
        overlay_landmarks,
        main_type="rect",
        ncols=ncols,
        blurred_crops=blurred_crops,
    )
    # fig.show(config={"responsive": True})

    # Persist an HTML snapshot for later viewing (centered with custom toolbar)
    html_path = os.path.join(out_dir, "rectangles.html")
    try:
        write_centered_html(fig, html_path, export_filename="rectangles")
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Landmark plots (2D)
# -----------------------------------------------------------------------------

def plot_landmarks(
    lands: dict,
    num_frames: int,
    video_path: Optional[str],
    out_dir: str,
    random_seed: int,
    cushion_ratio: float = 0.2,
    overlay: Optional[dict] = None,  # ignored: overlays disabled for landmarks
):
    df = lands["data"]
    n = min(num_frames, len(df))
    if n <= 0:
        return

    frames = df.sample(n=n, random_state=random_seed)
    crops: List[np.ndarray] = []
    landmarks: List[Tuple[np.ndarray, np.ndarray]] = []


    for idx, row in frames.iterrows():
        xs = row.values[::2].astype(int)
        ys = row.values[1::2].astype(int)
        if video_path:
            frame = get_frame(video_path, idx)
            # Use keyword args to route to the landmark branch correctly
            crop, xs_crop, ys_crop, (_x1, _y1, _scale_x, _scale_y) = crop_and_scale(
                frame, xs=xs, ys=ys, cushion_ratio=cushion_ratio
            )
            crops.append(crop)
            landmarks.append((xs_crop, ys_crop))

    if not crops:
        return

    ncols = min(4, len(crops))
    fig = make_centered_subplot_with_overlay(
        crops,
        landmarks,
        overlay_items=None,  # no overlays
        main_type="landmark",
        ncols=ncols,
        add_overlay_toggle=False,
    )
    # fig.show(config={"responsive": True})

    html_path = os.path.join(out_dir, "landmarks.html")
    try:
        write_centered_html(fig, html_path, export_filename="landmarks")
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Pose plots (3D scatter per sampled frame) with top-row images
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


def plot_pose(
    pose_dict: dict,
    land_can_dict: Optional[dict],
    rect_dict: Optional[dict],
    video_path: Optional[str],
    num_frames: int = 5,
    output_dir: str = "output/plots",
    target_size: Tuple[int, int] = (300, 300),
) -> str:
    """
    NOTE: expects your existing `get_frame(...)` and `write_centered_html(...)` to be available.
    """
    os.makedirs(output_dir, exist_ok=True)

    pose_df = pose_dict["data"]
    mean_face = land_can_dict["data"].iloc[0].values.reshape(-1, 3)

    T = pose_df.shape[0]
    if T == 0:
        raise ValueError("Empty pose dataframe")

    # Helper: robustly extract Tx,Ty,Tz,Rx,Ry,Rz regardless of column casing or order
    def _pose_tuple_from_series(s):
        try:
            idx = s.index
        except Exception:
            idx = []
        lower = ["tx", "ty", "tz", "rx", "ry", "rz"]
        upper = ["Tx", "Ty", "Tz", "Rx", "Ry", "Rz"]
        if all(k in idx for k in lower):
            return tuple(float(s[k]) for k in lower)
        if all(k in idx for k in upper):
            return tuple(float(s[k]) for k in upper)
        vals = getattr(s, "values", s)
        if len(vals) >= 6:
            return tuple(float(v) for v in vals[:6])
        raise KeyError(
            "pose_df must contain pose columns Tx,Ty,Tz,Rx,Ry,Rz (any case) or have first 6 columns as pose values"
        )

    # select frames by pose diversity on (Rx, Ry) instead of random/linspace
    k = min(num_frames, T)
    frame_idxs = select_diverse_frames_maxmin(
        pose_df,
        k=k,
        x_col_candidates=("Rx", "rx", "pitch"),  # change to ("Tx","tx") if you want translations
        y_col_candidates=("Ry", "ry", "yaw"),
        prefer_extremes=True
    )
    ncols = len(frame_idxs)
    nrows = 2

    # Column titles under 3D plots
    column_titles = []
    for idx in frame_idxs:
        _tx, _ty, _tz, _rx, _ry, _rz = _pose_tuple_from_series(pose_df.iloc[idx])
        column_titles.append(f"Frame {idx}<br>Pitch: {_rx:.2f}, Yaw: {_ry:.2f}, Roll: {_rz:.2f}")

    # Fixed figure sizing
    cell_w, cell_h = target_size
    h_gap_px = 10
    v_gap_px = 20
    left_right_margin = 30
    top_bottom_margin = 80
    fig_w = int(ncols * cell_w + (ncols - 1) * h_gap_px + 2 * left_right_margin)
    fig_h = int(2 * cell_h + v_gap_px + 2 * top_bottom_margin)

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        specs=[[{"type": "image"}] * ncols, [{"type": "scene"}] * ncols],
        vertical_spacing=0.03,
        horizontal_spacing=0.01,
        row_heights=[0.5, 0.5],
    )

    # Top row: images
    for i, idx in enumerate(frame_idxs):
        if video_path is None:
            continue
        img = get_frame(video_path, idx)

        # crop using rects when available
        if rect_dict is not None and "data" in rect_dict and idx < len(rect_dict["data"]):
            rect_row = rect_dict["data"].iloc[idx]
            try:
                x, y, w, h = (int(rect_row["x"]), int(rect_row["y"]), int(rect_row["w"]), int(rect_row["h"]))
            except Exception:
                x, y, w, h = map(int, rect_row.values[:4])

            cushion = 20
            H, W = img.shape[:2]
            x1 = max(int(x) - cushion, 0)
            y1 = max(int(y) - cushion, 0)
            x2 = min(int(x + w) + cushion, W)
            y2 = min(int(y + h) + cushion, H)
            cropped_img = img[y1:y2, x1:x2]
        else:
            cropped_img = img

        if target_size:
            cropped_img = cv2.resize(cropped_img, target_size)

        fig.add_trace(go.Image(z=cropped_img), row=1, col=i + 1)
        fig.update_xaxes(showticklabels=False, row=1, col=i + 1, visible=False)
        fig.update_yaxes(showticklabels=False, row=1, col=i + 1, visible=False)

    # Bottom row: 3D faces
    camera = dict(eye=dict(x=0, y=0, z=2), up=dict(x=0, y=1, z=0))  # Adjusted zoom level
    for i, frame_idx in enumerate(frame_idxs):
        tx, ty, tz, rx, ry, rz = _pose_tuple_from_series(pose_df.iloc[frame_idx])
        R = euler_to_rotmat(rx, ry, rz)
        face_xyz = (R @ mean_face.T).T + np.array([tx, ty, tz])
        face_xyz[:, 2] *= -1  # optional flip for orientation

        fig.add_trace(
            go.Scatter3d(
                x=face_xyz[:, 0],
                y=face_xyz[:, 1],
                z=face_xyz[:, 2],
                mode="markers",
                marker=dict(color='blue', size=4), 
                name=f"Frame {frame_idx}",
            ),
            row=2,
            col=i + 1,
        )

    fig.update_layout(
        autosize=False,
        width=fig_w,
        height=fig_h,
        showlegend=False,
        title={"text": "Video Frame (top) and 3D Pose (bottom) for Each Selected Frame", "x": 0.5, "xanchor": "center"},
        margin=dict(t=70, l=20, r=20, b=80),
    )

    # same camera + aspect for all scenes
    for i in range(1, ncols + 1):
        scene_id = f"scene{i}" if i > 1 else "scene"
        fig.layout[scene_id].camera = camera  # Apply zoomed camera settings
        fig.layout[scene_id].aspectmode = "data"

    # Labels under the 3D row
    try:
        for i in range(1, ncols + 1):
            scene_id = f"scene{i}" if i > 1 else "scene"
            dom = fig.layout[scene_id].domain
            xmid = (dom.x[0] + dom.x[1]) / 2.0
            ybelow = dom.y[0] - 0.09  # Adjusted to add more space above the text
            fig.add_annotation(
                text=column_titles[i - 1],
                x=xmid,
                y=ybelow,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="center",
                xanchor="center",
                yanchor="top",  # Ensures the text is positioned above
                font=dict(size=12),
            )
        fig.update_layout(margin=dict(b=max(fig.layout.margin.b, 120)))  # Adjusted bottom margin
    except Exception:
        pass

    # Store pose titles + raw eulers in meta
    try:
        existing_meta = fig.layout.meta if isinstance(fig.layout.meta, dict) else {}
        meta = dict(existing_meta) if isinstance(existing_meta, dict) else {}
        meta["pose_column_titles"] = column_titles
        try:
            pose_eulers = []
            for idx in frame_idxs:
                _tx, _ty, _tz, _rx, _ry, _rz = _pose_tuple_from_series(pose_df.iloc[idx])
                pose_eulers.append({"frame_idx": int(idx), "pitch": float(_rx), "yaw": float(_ry), "roll": float(_rz)})
            meta["pose_eulers"] = pose_eulers
        except Exception:
            pass
        fig.update_layout(meta=meta)
    except Exception:
        pass

    out_file = os.path.join(output_dir, "pose_and_video_subplots.html")
    write_centered_html(fig, out_file, export_filename="pose_and_video_subplots", add_pose_only_export=True)
    # fig.show(config={"responsive": True})
    return out_file


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
    title_map = {"rect": "Face Rectangles", "landmark": "Face Landmarks"}
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
    privacy_pairs: List[Tuple[int, int]] =[] # (orig_image_trace_idx, blurred_image_trace_idx)
    image_trace_indices: List[int]= [] # track all image traces to enable remove face

    for idx, crop in enumerate(crops):
        row = idx // ncols + 1
        col = idx % ncols + 1

        # Add original image trace
        fig.add_trace(go.Image(z=crop), row=row, col=col)
        orig_trace_idx = len(fig.data) - 1
        image_trace_indices.append(orig_trace_idx)

        # Add blurred image trace (if provided), hidden by default, added immediately so overlays/landmarks stay on top
        if blurred_crops is not None and idx < len(blurred_crops) and blurred_crops[idx] is not None:
            fig.add_trace(go.Image(z=blurred_crops[idx], visible=False), row=row, col=col)
            blur_trace_idx = len(fig.data) - 1
            privacy_pairs.append((orig_trace_idx, blur_trace_idx))
            image_trace_indices.append(blur_trace_idx)
        else:
            blur_trace_idx = -1

        if main_type == "rect":
            x_rel, y_rel, w, h = main_items[idx]
            # Outline rectangle shape (always visible)
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
                overlay_trace_indices.append(len(fig.data) - 1)

        # Hide ticks and flip Y so image + overlays align consistently across Plotly versions
        fig.update_xaxes(showticklabels=False, visible=False, row=row, col=col)
        fig.update_yaxes(showticklabels=False, visible=False, autorange="reversed", row=row, col=col)

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
    try:
        meta = fig.layout.meta or {}
        if isinstance(meta, dict):
            overlay_indices = meta.get("overlay_trace_indices", []) or []
            privacy_pairs = meta.get("privacy_pairs", []) or []
            image_indices = meta.get("image_trace_indices", []) or []
    except Exception:
        overlay_indices = []

    overlay_json = json.dumps(overlay_indices)
    privacy_pairs_json = json.dumps(privacy_pairs)
    image_indices_json = json.dumps(image_indices)
    has_overlay = bool(overlay_indices)
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

    # Conditionally include buttons
    buttons_html = (
        ("<button id=\"overlayBtn\" class=\"btn\">Overlay</button>" if has_overlay else "")
        + ("\n    " + "<button id=\"blurBtn\" class=\"btn\">Blur Face</button>" if has_privacy else "")
        + ("\n    " + "<button id=\"removeBtn\" class=\"btn\">Remove Face</button>" if has_remove else "")
        + "\n    "
        + '<button id="exportBtn" class="btn export">Export (PNG)</button>'
        + ("\n    " + '<button id="exportPoseOnlyBtn" class="btn export">Export Pose Only (PNG)</button>' if add_pose_only_export else "")
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
      const overlayIdx = $overlay_json;
      const privacyPairs = $privacy_pairs_json;
      const imageIdxs = $image_indices_json;
      const gd = document.getElementById('plot');
      const exportBtn = document.getElementById('exportBtn');
      const overlayBtn = document.getElementById('overlayBtn');
      const blurBtn = document.getElementById('blurBtn');
      const removeBtn = document.getElementById('removeBtn');
      const exportPoseBtn = document.getElementById('exportPoseOnlyBtn');
      const hasOverlay = Array.isArray(overlayIdx) && overlayIdx.length > 0;
      const hasPrivacy = Array.isArray(privacyPairs) && privacyPairs.length > 0;
      const hasRemove = Array.isArray(imageIdxs) && imageIdxs.length > 0;

      function exportFilename(suffix) { return '$export_filename_safe' + (suffix ? ('_' + suffix) : '') + '.png'; }

      // Overlay helpers are no-ops unless overlays exist
      let overlayOn = true;
      function setOverlay(on) {
        overlayOn = on;
        if (hasOverlay) {
          try { Plotly.restyle(gd, {visible: on}, overlayIdx); } catch(e) { console.warn(e); }
          if (overlayBtn) overlayBtn.style.borderColor = on ? '#2ecc71' : '#999';
        }
      }

      // Blur helpers (toggle between original and blurred traces)
      let blurOn = false;
      function setBlur(on) {
        blurOn = on;
        if (hasPrivacy) {
          const origIdx = [], blurIdx = [];
          for (const pair of privacyPairs) {
            if (Array.isArray(pair) && pair.length === 2) {
              origIdx.push(pair[0]);
              blurIdx.push(pair[1]);
            }
          }
          try {
            if (removeOn) {
              // when removed, keep all images hidden
              Plotly.restyle(gd, {visible: false}, origIdx);
              Plotly.restyle(gd, {visible: false}, blurIdx);
            } else {
              // show blurred if on, hide originals (or vice versa)
              Plotly.restyle(gd, {visible: on}, blurIdx);
              Plotly.restyle(gd, {visible: !on}, origIdx);
            }
          } catch(e) { console.warn(e); }
          if (blurBtn) blurBtn.style.borderColor = on ? '#2ecc71' : '#999';
        }
      }

      // Remove helpers (hide/show all image traces)
      let removeOn = false;
      function setRemove(on) {
        removeOn = on;
        if (hasRemove) {
          try {
            Plotly.restyle(gd, {visible: !on}, imageIdxs);
            // if we just re-enabled images, re-apply current blur state
            if (!on) {
              if (hasPrivacy) setBlur(blurOn);
            }
          } catch(e) { console.warn(e); }
          if (removeBtn) removeBtn.style.borderColor = on ? '#2ecc71' : '#999';
        }
      }

      function afterPlotted() {
        if (hasOverlay) { setOverlay(true); }
        if (hasPrivacy) { setBlur(false); }
        if (hasRemove) { setRemove(false); }
      }

      if (gd && gd.addEventListener) {
        gd.addEventListener('plotly_afterplot', afterPlotted, { once: true });
      }
      setTimeout(afterPlotted, 150);

      if (hasOverlay && overlayBtn) {
        overlayBtn.addEventListener('click', function() { setOverlay(!overlayOn); });
      }
      if (hasPrivacy && blurBtn) {
        blurBtn.addEventListener('click', function() { setBlur(!blurOn); });
      }
      if (hasRemove && removeBtn) {
        removeBtn.addEventListener('click', function() { setRemove(!removeOn); });
      }

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
        overlay_json=overlay_json,
        privacy_pairs_json=privacy_pairs_json,
        image_indices_json=image_indices_json,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
