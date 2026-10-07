import numpy as np
import pandas as pd

def landmarks_left_right(schema='ibug51'):
    idx = {}
    
    # ordering of right is to mirror left
    if schema == 'ibug51':
        idx = {'lb': np.array([0,1,2,3,4]),
               'rb': np.array([9,8,7,6,5]),
               'lno': np.array([14,15]), # excluding landmarks shared by left/right [10-13,16]
               'rno': np.array([18,17]),
               'le': np.array([19,20,21,22,23,24]),
               're': np.array([28,27,26,25,30,29]),
               'lm': np.array([31,32,33,43,44,50,42,41]), # excluding shared ones [34,45,49,40]   #ul: list(range(31, 37))+list(range(43, 47))
               'rm': np.array([37,36,35,47,46,48,38,39]) #ll: list(range(37, 43))+list(range(47, 51))
              } 
    # elif schema == 'ibug51_mirrored':
    #     idx = {'lb': np.array([9,8,7,6,5]),
    #            'rb': np.array([4,3,2,1,0]),
    #            'no': np.array([10, 11, 12, 13, 18, 17, 16, 15, 14]),
    #            'le': np.array([28, 27, 26, 25, 30, 29]),
    #            're': np.array([22, 21, 20, 19, 24, 23]),
    #            'ul': np.array([37, 36, 35, 34, 33, 32, 47, 46, 45, 44]),
    #            'll': np.array([31, 42, 41, 40, 39, 38, 43, 50, 49, 48])}
    else:
        raise ValueError(f"Landmark schema {schema} not recognized")
    
    return idx


# Landmarks used to fit the whole-face mirror plane. Both sets sit on or near the midline and move
# little with expression. ibug51: 22/25 are the inner eye corners, 10-13 the nose bridge.
_MIRROR_ANCHOR_PAIRS = {'ibug51': [(22, 25)]}
_MIRROR_ANCHOR_MID = {'ibug51': [10, 11, 12, 13]}

def face_mirror_plane(coords, schema='ibug51', smooth=5):
    """Whole-face mirror plane (the face's median/midsagittal plane), one per frame.

    The normal is the inner-eye-corner axis, and the plane passes through the midpoint of the inner
    eye corners together with the nose-bridge points, i.e. it is the perpendicular bisector of the
    inner-ocular segment.

    Args:
        coords: (T, L, D) landmark array, 2D or 3D, canonical ok.
        schema: landmark schema; only 'ibug51' is defined.
        smooth: frames of centered median smoothing applied to the plane (0 or 1 disables it). The
            plane is an estimate, so without this its own jitter is added to every score.

    Returns:
        (normal, point): unit normal (T, D) and a point on the plane (T, D).
    """
    if schema not in _MIRROR_ANCHOR_PAIRS:
        raise ValueError(f"No mirror plane anchors defined for landmark schema {schema}")
    pairs = _MIRROR_ANCHOR_PAIRS[schema]
    mids = _MIRROR_ANCHOR_MID[schema]

    n = np.sum([coords[:, r] - coords[:, l] for l, r in pairs], axis=0)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)

    # offset of the plane along the normal: average over the anchor midpoints and the midline points
    pts = np.stack([(coords[:, l] + coords[:, r]) / 2 for l, r in pairs] + [coords[:, i] for i in mids], axis=1)
    d = np.einsum('tkd,td->tk', pts, n).mean(axis=1)

    # the plane is estimated per frame, so smooth it over time to keep its own jitter out of the scores
    if smooth and smooth > 1 and coords.shape[0] > smooth:
        n = pd.DataFrame(n).rolling(smooth, center=True, min_periods=1).median().to_numpy()
        n = n / np.linalg.norm(n, axis=1, keepdims=True)
        d = pd.Series(d).rolling(smooth, center=True, min_periods=1).median().to_numpy()

    return n, n * d[:, None]
