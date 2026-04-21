import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# .OF file readers (read from split files, mirrors reader3DI pattern)
# ---------------------------------------------------------------------------

def read_confidence(file):
    """Read per-frame confidence and success flags from a .OF file."""
    _data = np.loadtxt(file)
    data = pd.DataFrame(_data, columns=['frame', 'face_id', 'timestamp', 'confidence', 'success'])

    return {
        'backend': 'OF',
        'frame count': data.shape[0],
        'type': 'confidence',
        'format': 'for each frame (rows) [frame, face_id, timestamp, confidence, success]',
        'dimension': 1,
        'data': data,
    }


def read_landmarks(file):
    """Read 2D facial landmarks (68-point Multi-PIE schema) from a .OF file."""
    _data = np.loadtxt(file)
    num_landmarks = _data.shape[1] // 2

    if num_landmarks == 68:
        schema = 'multipie68'
    else:
        raise ValueError(f"Unrecognized landmark schema: {num_landmarks} landmarks.")

    # stored as [x_0..x_67, y_0..y_67], label accordingly
    column_list = [f'x{i}' for i in range(num_landmarks)] + [f'y{i}' for i in range(num_landmarks)]
    data = pd.DataFrame(_data, columns=column_list)

    return {
        'backend': 'OF',
        'frame count': data.shape[0],
        'type': 'landmark',
        'format': 'for each frame (rows) [x, y] values of the detected landmarks',
        'schema': schema,
        'dimension': 2,
        'data': data,
    }


def read_canonical_landmarks(file):
    """Read 3D facial landmarks (68-point) from a .OF file."""
    _data = np.loadtxt(file)
    num_landmarks = _data.shape[1] // 3

    if num_landmarks == 68:
        schema = 'multipie68'
    else:
        raise ValueError(f"Unrecognized landmark schema: {num_landmarks} landmarks.")

    # stored as [X_0..X_67, Y_0..Y_67, Z_0..Z_67], label accordingly
    column_list = ([f'x{i}' for i in range(num_landmarks)]
                   + [f'y{i}' for i in range(num_landmarks)]
                   + [f'z{i}' for i in range(num_landmarks)])
    data = pd.DataFrame(_data, columns=column_list)

    return {
        'backend': 'OF',
        'frame count': data.shape[0],
        'type': 'landmark',
        'format': 'for each frame (rows) [x, y, z] values of the detected 3D landmarks',
        'schema': schema,
        'dimension': 3,
        'data': data,
    }


def read_pose(file):
    """Read head pose (translation + rotation) from a .OF file."""
    _data = np.loadtxt(file)
    data = pd.DataFrame(_data, columns=['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz'])

    return {
        'backend': 'OF',
        'frame count': data.shape[0],
        'type': 'pose',
        'format': 'for each frame (rows) [Tx, Ty, Tz, Rx, Ry, Rz] values of the detected face pose',
        'dimension': 3,
        'data': data,
    }


def read_rectangles(file):
    """Read face bounding boxes from a .OF file.

    Rectangles are derived from 2D landmark extents during splitting.
    """
    _data = np.loadtxt(file)
    data = pd.DataFrame(_data, columns=['x', 'y', 'w', 'h'])

    return {
        'backend': 'OF',
        'frame count': data.shape[0],
        'type': 'rectangle',
        'format': 'for each frame (rows) [x, y, w, h] values of the detected rectangles',
        'dimension': 2,
        'data': data,
    }


def read_gaze(file):
    """Read gaze direction vectors and angles from a .OF file."""
    _data = np.loadtxt(file)
    columns = [
        'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
        'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
        'gaze_angle_x', 'gaze_angle_y',
    ]
    data = pd.DataFrame(_data, columns=columns)

    return {
        'backend': 'OF',
        'frame count': data.shape[0],
        'type': 'gaze',
        'format': 'for each frame (rows) gaze vectors for left/right eye and gaze angles',
        'dimension': 3,
        'data': data,
    }


def read_eye_landmarks(file):
    """Read 2D and 3D eye landmarks (56 points) from a .OF file."""
    _data = np.loadtxt(file)
    # stored as [x_0..x_55, y_0..y_55, X_0..X_55, Y_0..Y_55, Z_0..Z_55]
    cols = ([f'eye_lmk_x_{i}' for i in range(56)]
            + [f'eye_lmk_y_{i}' for i in range(56)]
            + [f'eye_lmk_X_{i}' for i in range(56)]
            + [f'eye_lmk_Y_{i}' for i in range(56)]
            + [f'eye_lmk_Z_{i}' for i in range(56)])
    data = pd.DataFrame(_data, columns=cols)

    return {
        'backend': 'OF',
        'frame count': data.shape[0],
        'type': 'eye_landmark',
        'format': 'for each frame (rows) 2D and 3D eye landmark coordinates (56 points)',
        'dimension': 3,
        'data': data,
    }


_AU_INTENSITY = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
    'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
    'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r',
]
_AU_PRESENCE = [
    'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c',
    'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c',
    'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c',
]


def read_expression(file):
    """Read Action Unit intensities and presences from a .OF file."""
    _data = np.loadtxt(file)
    au_cols = _AU_INTENSITY + _AU_PRESENCE
    data = pd.DataFrame(_data, columns=au_cols)

    return {
        'backend': 'OF',
        'frame count': data.shape[0],
        'type': 'expression',
        'format': 'for each frame (rows) Action Unit intensities (_r) and presences (_c)',
        'schema': 'FACS',
        'intensity_columns': list(_AU_INTENSITY),
        'presence_columns': list(_AU_PRESENCE),
        'dimension': 1,
        'data': data,
    }


def read_shape_params(file):
    """Read rigid and non-rigid shape parameters from a .OF file."""
    _data = np.loadtxt(file)
    rigid_cols = ['p_scale', 'p_rx', 'p_ry', 'p_rz', 'p_tx', 'p_ty']
    nonrigid_cols = [f'p_{i}' for i in range(34)]
    data = pd.DataFrame(_data, columns=rigid_cols + nonrigid_cols)

    return {
        'backend': 'OF',
        'frame count': data.shape[0],
        'type': 'shape_params',
        'format': 'for each frame (rows) rigid (scale, rotation, translation) and non-rigid shape parameters',
        'schema': 'PDM',
        'dimension': 1,
        'data': data,
    }


# ---------------------------------------------------------------------------
# CSV splitter (converts monolithic OpenFace CSV to individual .OF files)
# ---------------------------------------------------------------------------

def _load_csv(file):
    """Load an OpenFace CSV, stripping whitespace from column names."""
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    return df


def split_csv_to_of(csv_file, output_dir):
    """Split a monolithic OpenFace CSV into separate .OF files.

    Produces the following files in output_dir:
        {base}_confidence.OF
        {base}_landmarks_2d.OF
        {base}_landmarks_3d.OF
        {base}_pose.OF
        {base}_gaze.OF
        {base}_eye_landmarks.OF
        {base}_action_units.OF
        {base}_shape_params.OF

    Args:
        csv_file: Path to the OpenFace CSV file.
        output_dir: Directory where .OF files will be written.

    Returns:
        Dictionary mapping output type names to file paths.
    """
    df = _load_csv(csv_file)
    base = os.path.splitext(os.path.basename(csv_file))[0]
    os.makedirs(output_dir, exist_ok=True)

    outputs = {}

    # Confidence / metadata
    meta_cols = ['frame', 'face_id', 'timestamp', 'confidence', 'success']
    _write(df, meta_cols, output_dir, base, 'confidence', outputs)

    # Rectangles derived from 2D landmarks
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    x_vals = df[x_cols].values
    y_vals = df[y_cols].values
    x_min = x_vals.min(axis=1)
    y_min = y_vals.min(axis=1)
    w = x_vals.max(axis=1) - x_min
    h = y_vals.max(axis=1) - y_min
    rects = np.column_stack([x_min, y_min, w, h])
    rect_path = os.path.join(output_dir, f'{base}_rects.OF')
    np.savetxt(rect_path, rects)
    outputs['rects'] = rect_path

    # 2D face landmarks (68 points)
    lmk2d_cols = [f'x_{i}' for i in range(68)] + [f'y_{i}' for i in range(68)]
    _write(df, lmk2d_cols, output_dir, base, 'landmarks_2d', outputs)

    # 3D face landmarks (68 points)
    lmk3d_cols = [f'X_{i}' for i in range(68)] + [f'Y_{i}' for i in range(68)] + [f'Z_{i}' for i in range(68)]
    _write(df, lmk3d_cols, output_dir, base, 'landmarks_3d', outputs)

    # Head pose
    pose_cols = ['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz']
    _write(df, pose_cols, output_dir, base, 'pose', outputs)

    # Gaze
    gaze_cols = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z',
                 'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
                 'gaze_angle_x', 'gaze_angle_y']
    _write(df, gaze_cols, output_dir, base, 'gaze', outputs)

    # Eye landmarks (2D + 3D)
    eye_cols = ([f'eye_lmk_{c}_{i}' for c in 'xy' for i in range(56)]
                + [f'eye_lmk_{c}_{i}' for c in 'XYZ' for i in range(56)])
    _write(df, eye_cols, output_dir, base, 'eye_landmarks', outputs)

    # Action Units
    au_cols = _AU_INTENSITY + _AU_PRESENCE
    _write(df, au_cols, output_dir, base, 'action_units', outputs)

    # Shape parameters
    rigid_cols = ['p_scale', 'p_rx', 'p_ry', 'p_rz', 'p_tx', 'p_ty']
    nonrigid_cols = [f'p_{i}' for i in range(34)]
    _write(df, rigid_cols + nonrigid_cols, output_dir, base, 'shape_params', outputs)

    return outputs


def _write(df, cols, output_dir, base, name, outputs):
    """Write a subset of columns to an .OF file."""
    path = os.path.join(output_dir, f'{base}_{name}.OF')
    np.savetxt(path, df[cols].values)
    outputs[name] = path
