import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import transform as trans

def rodrigues_to_euler(er, axes='yxz', frame='intrinsic', unwrap=False, is_opencv_rotvec=True):
    """
    Convert T x 3 ER vectors to T x 3 Euler angles (radians).

    Parameters
    ----------
    er : ndarray, shape (T,3)
        Either true Euler–Rodrigues/Gibbs vector (axis * tan(theta/2)) or
        OpenCV rotation vector (axis * theta), chosen by `is_opencv_rotvec`.
    axes : str
        A permutation of 'xyz' (e.g., 'zyx', 'xyz', ...).
    frame : {'intrinsic','extrinsic'}
        Intrinsic (rotating axes) or extrinsic (static axes).
    unwrap : bool
        Enforce temporal continuity (unwrap + nearest-branch selection).
    is_opencv_rotvec : bool
        If True, interpret `er` as OpenCV rotation vectors (axis * theta).
        If False, interpret `er` as Euler–Rodrigues/Gibbs (axis * tan(theta/2)).
    """
    er = np.asarray(er, dtype=np.float64)
    T = er.shape[0]

    # ER/RotVec -> quaternion [x, y, z, w]
    if is_opencv_rotvec:
        # OpenCV Rodrigues rotation vector: rho = axis * theta
        theta = np.linalg.norm(er, axis=1, keepdims=True)        # (T,1)
        half = 0.5 * theta
        # sin(θ/2)/θ with stable small-angle limit -> 1/2
        s = np.where(theta > 1e-12, np.sin(half) / (theta + 1e-15), 0.5)
        q = np.empty((T, 4), dtype=np.float64)                   # (x, y, z, w)
        q[:, :3] = er * s                                        # v = axis * sin(θ/2)
        q[:, 3]  = np.cos(half)[:, 0]                            # w = cos(θ/2)
    else:
        # Euler–Rodrigues/Gibbs vector: r = axis * tan(theta/2)
        r2 = np.sum(er * er, axis=1, keepdims=True)
        w = 1.0 / np.sqrt(1.0 + r2)                              # w = 1/sqrt(1+||r||^2)
        q = np.empty((T, 4), dtype=np.float64)                   # (x, y, z, w)
        q[:, :3] = er * w
        q[:, 3]  = w[:, 0]

    order = axes if frame == 'intrinsic' else axes.upper()

    try:
        from scipy.spatial.transform import Rotation as R
        eul = R.from_quat(q).as_euler(order, degrees=False)
    except Exception:
        # Fallback supports only intrinsic 'zyx'
        if not (frame == 'intrinsic' and axes == 'zyx'):
            raise ValueError("Fallback (no SciPy) supports only intrinsic 'zyx'. Install SciPy for other orders.")
        x, y, z, ww = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = ww*x, ww*y, ww*z
        # Rotation matrix from quaternion
        R00 = 1 - 2*(yy + zz);  R01 = 2*(xy - wz);  R10 = 2*(xy + wz);  R11 = 1 - 2*(xx + zz)
        R20 = 2*(xz - wy);      R21 = 2*(yz + wx);  R22 = 1 - 2*(xx + yy)
        # Stable ZYX extraction (atan2 form for middle angle)
        y_mid = np.arctan2(-R20, np.hypot(R00, R10))   # Y
        z_1st = np.arctan2(R10, R00)                   # Z
        x_last= np.arctan2(R21, R22)                   # X
        eul = np.column_stack([z_1st, y_mid, x_last])

    if not unwrap or T == 0:
        return eul

    # Temporal continuity: unwrap + nearest-branch selection (handle gimbal-equivalent branch)
    eul = np.unwrap(eul, axis=0)  # per-angle 2π unwrap

    two_pi = 2.0 * np.pi

    def nearest_to(ref, a):
        # bring 'a' near 'ref' via 2π shifts, per component
        return ref + ((a - ref + np.pi) % two_pi - np.pi)

    for t in range(1, T):
        prev = eul[t-1]
        cur  = eul[t].copy()

        # Candidate 1: current (after unwrap), snapped near previous
        c1 = nearest_to(prev, cur)

        # Candidate 2: gimbal-equivalent: add π to first & last, flip sign of middle; then snap near previous
        cand2 = cur.copy()
        cand2[0] += np.pi
        cand2[1]  = -cand2[1]
        cand2[2] += np.pi
        c2 = nearest_to(prev, cand2)

        # Choose the closer candidate
        eul[t] = c2 if np.linalg.norm(c2 - prev) < np.linalg.norm(c1 - prev) else c1

    return eul


############################################
# Functions for converting between 3DI-lite pose and 3DI pose
############################################

# Helper: extract 5 keypoints from 51p
def _extract_5p(lm_51p):
    """
    lm_51p: (51,2) landmarks from your detector (ibug51 style after dropping first 17 if needed)
    returns lm5p: (5,2) in the order [left eye, right eye, nose, left mouth, right mouth]
    This matches your utils.py logic.
    """
    lm_idx = np.array([31-17, 37-17, 40-17, 43-17, 46-17, 49-17, 55-17]) - 1
    # grab 7 points
    pts = lm_51p[lm_idx, :]  # shape (7,2)

    # average eye corners and mouth corners the same way utils.py does
    lm5p = np.stack([
        pts[0, :],                                # 31
        np.mean(pts[[1, 2], :], axis=0),          # avg(37,40)
        np.mean(pts[[3, 4], :], axis=0),          # avg(43,46)
        pts[5, :],                                # 49
        pts[6, :],                                # 55
    ], axis=0)

    # reorder to [left eye, right eye, nose, left mouth, right mouth]
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p  # (5,2)


# Recreate estimate_norm from utils.py
def _estimate_norm(lm_51p, scale=1.5, off=(25,25)):
    """
    lm_51p: (51,2) landmarks in ORIGINAL frame coords (before alignment)
    scale, off: must match what you used in video_fitter.py (1.5, [25,25])

    returns:
        M: (2,3) affine matrix that maps *normalized crop coords* -> *original image coords*
    """
    lm = _extract_5p(lm_51p)  # (5,2)

    tform = trans.SimilarityTransform()

    # canonical template points (same as utils.py)
    src = scale * np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32
    ) + np.array(off, dtype=np.float32).reshape(1, 2)

    # skimage SimilarityTransform.estimate(src, dst)
    # learns transform T such that  src @ T  ~= dst
    # Here: template (src) -> actual detected lm (lm)
    tform.estimate(src, lm)
    M_full = tform.params  # 3x3
    M = M_full[0:2, :]     # 2x3

    return M  # used later to get the in-plane rotation


# Recover roll angle from M
def _extract_roll_from_M(M):
    """
    M is 2x3 affine from estimate_norm.
    For a similarity transform (scale s, rotation θ, translation),
    M = [[ s cosθ , -s sinθ , tx ],
         [ s sinθ ,  s cosθ , ty ]]

    We can solve θ = atan2(M[1,0], M[0,0]).
    θ is the in-plane rotation (roll) that was originally present
    BEFORE alignment. Alignment effectively rotates the face by -θ
    so that the network sees a level, upright face.

    return θ (radians)
    """
    a = M[0,0]
    c = M[1,0]
    theta = np.arctan2(c, a)
    return theta


# Undo alignment on predicted Rodrigues
def _rodrigues_to_matrix(rvec):
    """
    rvec: (3,) Rodrigues rotation vector
    return: (3,3) rotation matrix
    """
    R, _ = cv2.Rodrigues(rvec.reshape(3,1))
    return R  # (3,3)

def _matrix_to_rodrigues(R):
    """
    R: (3,3) rotation matrix
    return: (3,) Rodrigues rotation vector
    """
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3,)

def _undo_alignment_single(lm_51p, rvec_pred, scale=1.5, off=(25,25)):
    """
    lm_51p: (51,2) landmarks in ORIGINAL frame coords (before alignment)
    rvec_pred: (3,) predicted Rodrigues from the DL model
               (this was learned on aligned / cropped face)
    returns:
        rvec_recovered: (3,) Rodrigues in original (un-aligned) image coordinates
    """

    # 1. Recompute the same normalization transform used at training
    M = _estimate_norm(lm_51p, scale=scale, off=off)

    # 2. Extract the roll angle theta that was removed
    theta = _extract_roll_from_M(M)  # radians

    # 3. Build rotation matrix that ADDS BACK that roll
    #    Note: alignment rotated the face by -theta so that roll ~ 0.
    #    To go back, we rotate by +theta around the camera Z axis.
    R_realign = np.array([
        [ np.cos(theta), -np.sin(theta), 0.0],
        [ np.sin(theta),  np.cos(theta), 0.0],
        [ 0.0,            0.0,           1.0],
    ], dtype=np.float32)

    # 4. Convert predicted Rodrigues to matrix
    R_pred = _rodrigues_to_matrix(rvec_pred)  # model output in aligned coords

    # 5. Compose: first apply the predicted head rotation, then undo crop alignment
    #    We want rotation in the original image coordinate frame.
    #    Original_frame_R = R_realign @ R_pred
    R_recovered = R_realign @ R_pred

    # 6. Convert back to Rodrigues
    rvec_recovered = _matrix_to_rodrigues(R_recovered)
    
    return rvec_recovered  # (3,)


# Batch version over T frames
def recover_full_rodrigues(landmarks_2d, rodrigues_pred, scale=1.5, off=(25,25)):
    """
    landmarks_2d: shape (T, 51*2)
                  columns [x0,y0,x1,y1,...,x50,y50] in ORIGINAL frames
    rodrigues_pred: shape (T, 3)
                    network predictions from aligned crops

    returns:
        rodrigues_recovered: shape (T,3)
    """

    T = landmarks_2d.shape[0]
    rodrigues_out = np.zeros((T,3), dtype=np.float32)

    for t in range(T):
        lm_t = landmarks_2d[t].reshape(51,2).astype(np.float32)
        rvec_t = rodrigues_pred[t].astype(np.float32)

        rvec_rec_t = _undo_alignment_single(
            lm_51p = lm_t,
            rvec_pred = rvec_t,
            scale=scale,
            off=off
        )
        rodrigues_out[t] = rvec_rec_t

    return rodrigues_out