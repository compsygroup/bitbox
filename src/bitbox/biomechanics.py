from .utilities import get_data_values, check_data_type

import numpy as np
from typing import Union, Sequence

def _convert_to_coords(data: dict, angular: bool = False) -> np.ndarray:
    coords = get_data_values(data)
    d = data['dimension']
    
    if check_data_type(data, ['landmark', 'landmark-can']):
        # convert the shape
        # (N,M,d) array. N: number of frames, M: number of landmarks, d: dimension (2 for 2D, 3 for 3D).
        N, D = coords.shape
        M = D // d
        coords = coords.reshape(N, M, d)
    elif check_data_type(data, 'rectangle'):
        # compute the center of the bounding box
        x, y, w, h = coords.T  # each is (N,)
        cx = x + w / 2
        cy = y + h / 2
        coords = np.stack((cx, cy), axis=1) # (N,2)
    elif check_data_type(data, 'pose'):
        if angular:
            # just keep the rotation part
            coords = coords[:, 3:]  # (N,3)
        else:
            # just keep the translation part
            coords = coords[:, :3]  # (N,3)
            
    return coords


def _check_reference(reference: Sequence[int]) -> bool:
    if not isinstance(reference, (list, np.ndarray)):
        return False
    if len(reference) == 0:
        return False
    if not all(isinstance(idx, int) for idx in reference):
        return False
    return True


def motion_kinematics(data: dict, fps: int = 30, angular: bool = False) -> list:
    """
    Total path length, range, speed, acceleration, and jerk of the motion.

    Parameters:
        data: data dictionary including coordinates of landmarks, poses, or rectangles.
        fps: frames per second of the video.
        angular: whether to use translation (default) or angles of pose. Only used when data includes poses

    Returns:
        list containing total path length, range, average speed, acceleration, and jerk.
    """
    
    # https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0169019
    
    # check data type
    if not check_data_type(data, ['landmark', 'landmark-can', 'pose', 'rectangle']):
        raise ValueError("Only 'landmark', 'pose', or 'rectangle' data can be used for kinematics calculations. Make sure to use the correct data type.")
    
    # convert data to coordinates
    coords = _convert_to_coords(data, angular=angular)

    if coords.shape[0] < 4:
        raise ValueError("Not enough frames to calculate motion profile. At least 4 frames are required.")
    
    # compute range of motion
    max_xy = coords.max(axis=0)             # (M,d) | (d,)
    min_xy = coords.min(axis=0)             # (M,d) | (d,)
    diff = max_xy - min_xy                  # (M,d) | (d,)
    mrange = np.linalg.norm(diff, axis=-1)   # (M,) | (1,)
    
    # calculate the total path length
    dx = np.diff(coords, axis=0)            # (N-1, M, d)
    path = np.linalg.norm(dx, axis=-1).sum(axis=0) # (,M)

    # speed in units/sec
    v = dx * fps      # (N-1, M, d)
    speed = np.linalg.norm(v, axis=-1).mean(axis=0)

    # acceleration in units/sec²
    a = np.diff(v, axis=0)                 # (N-2, M, d)
    accelaration = np.linalg.norm(a, axis=-1).mean(axis=0)

    return [mrange, path, speed, accelaration]


def motion_smoothness(data: dict, fps: int = 30, angular: bool = False) -> list:
    """
    Average jerk and log dimensionless jerk.

    Parameters:
        data: data dictionary including coordinates of landmarks, poses, or rectangles.
        fps: frames per second of the video.
        angular: whether to use translation (default) or angles of pose. Only used when data includes poses

    Returns:
        list containing jerk and log dimensionless jerk.
    """
    
    # LDJ: https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2018.00615/full
    # SPARC: https://www.nature.com/articles/s41598-022-09149-1
    
    # check data type
    if not check_data_type(data, ['landmark', 'landmark-can', 'pose', 'rectangle']):
        raise ValueError("Only 'landmark', 'pose', or 'rectangle' data can be used for kinematics calculations. Make sure to use the correct data type.")
    
    # convert data to coordinates
    coords = _convert_to_coords(data, angular=angular)

    if coords.shape[0] < 4:
        raise ValueError("Not enough frames to calculate motion profile. At least 4 frames are required.")
   
    # x and time parameters
    dx = np.diff(coords, axis=0)            # (N-1, M, d)
    dt = 1.0 / fps
    T = coords.shape[0] * dt # total duration in seconds

    # speed in units/sec
    v = dx * fps      # (N-1, M, d)

    # acceleration in units/sec²
    a = np.diff(v, axis=0)                 # (N-2, M, d)

    # jerk in units/sec3
    j = np.diff(a, axis=0)                 # (N-3, M, d)
    
    # average jerk
    jerk = np.linalg.norm(j, axis=-1).mean(axis=0)

    # ∫₀ᵀ ‖a(t)‖² dt ≈ Σ ‖aᵢ‖² * dt
    a_sq = np.linalg.norm(a, axis=-1)**2   # (N-2, M)
    J_int = a_sq.sum(axis=0) * dt          # (,M)

    # peak velocity squared
    v_mag = np.linalg.norm(v, axis=-1)     # (N-1, M)
    v_peak2 = v_mag.max(axis=0)**2         # (,M)

    # log dimensionless jerk
    ldj = -np.log((T**3 * J_int) / v_peak2)
    
    return [jerk, ldj]


def relative_motion(data: dict, reference: Union[Sequence[int], str] = "mean") -> list:
    """
    Calculate displacement statistics with respect to a set of reference coordinates.

    Parameters:
        data: data dictionary including coordinates of landmarks, poses, or rectangles.
        reference: sequence of indices for reference coordinates or 'mean' to use the average coordinate as reference.

    Returns:
        list containing minimum, average, standard deviation, and maximum displacement.
    """
    
    # check data type
    if not check_data_type(data, ['landmark', 'landmark-can', 'pose', 'rectangle']):
        raise ValueError("Only 'landmark', 'pose', or 'rectangle' data can be used for relative motion stats. Make sure to use the correct data type.")

    # convert data to coordinates
    coords = _convert_to_coords(data)
    
    mind = 0
    avgd = 0
    stdd = 0
    maxd = 0

    if isinstance(reference, str) and reference == 'mean':
        # use the mean of the coordinates as reference
        r_frame = coords.mean(axis=0)
        o_frames = coords
        reference = [-100000] # so that loop runs once
    elif _check_reference(reference):
        o_frames = np.delete(coords, reference, axis=0)  # (N-k,M,d) | (N-k,d)
    else:
        raise ValueError("Reference must be a list of indices or 'mean'. If using indices, ensure they are valid integers.")
    
    for idx in reference:
        if idx != -100000:
            if idx >= coords.shape[0]:
                raise ValueError(f"Reference frame index {idx} is out of bounds for the number of frames {coords.shape[0]}.")
            r_frame = coords[idx, ...]           # (1,M,d) | (1,d)
        
        disp = o_frames - r_frame                # (N-1,M,d) | (N-1,d)
        dist = np.linalg.norm(disp, axis=-1)     # (N-1,M) | (N-1,)
        
        mind += dist.min(axis=0)                 # (M,) | (1,)
        avgd += dist.mean(axis=0)                # (M,) | (1,)
        stdd += dist.std(axis=0)                 # (M,) | (1,)
        maxd += dist.max(axis=0)                 # (M,) | (1,)
    mind /= len(reference)
    avgd /= len(reference)
    stdd /= len(reference)
    maxd /= len(reference)
    
    return [mind, avgd, stdd, maxd]