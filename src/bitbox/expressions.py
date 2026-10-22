from .utilities import get_data_values, check_data_type
from .signal_processing import peak_detection, outlier_detectionIQR, log_transform
from .utilities import landmarks_left_right, face_mirror_plane
import numpy as np
import pandas as pd

# Calculate asymmetry scores using mirror error approach
def asymmetry(landmarks, axis=0, smooth=5):
    """Per-frame facial asymmetry using mirrored landmark error.

    Every feature is mirrored across one whole-face plane per frame, fitted to the inner eye
    corners and nose bridge, so positional asymmetry is measured.

    Args:
        landmarks: Landmark dict from the face backend (2D or 3D, canonical ok).
        axis: 0 if rows are frames (default); 1 if rows are signals.
        smooth: frames of median smoothing applied to the fitted plane.

    Returns:
        pandas.DataFrame with per-frame asymmetry for eye, brow, nose, mouth, and overall.
    """
    # check data type
    if not check_data_type(landmarks, ['landmark', 'landmark-can']):
        raise ValueError("Only 'landmark' data can be used for asymmetry calculation. Make sure to use the correct data type.")
    
    # read actual values
    data = get_data_values(landmarks)
    
    # whether rows are time points (axis=0) or signals (axis=1)
    if axis == 1:
        data = data.T
    
    dimension = landmarks['dimension']
    schema = landmarks['schema']
    
    land_ids = landmarks_left_right(schema=schema)
    
    feature_idx_left = {
        'eye': land_ids['le'],
        'brow': land_ids['lb'],
        'nose': land_ids['lno'],
        'mouth': land_ids['lm']
    }
    
    feature_idx_right = {
        'eye': land_ids['re'],
        'brow': land_ids['rb'],
        'nose': land_ids['rno'],
        'mouth': land_ids['rm']
    }
   
    T = data.shape[0]
    
    if data.shape[1] % dimension != 0:
        raise ValueError(f"Landmarks are not {dimension} dimensional. Please set the correct dimension.")
    
    all_coords = data.reshape((T, data.shape[1] // dimension, dimension))
    
    # the whole-face plane is smoothed over time, so it is fitted once for the entire series
    plane_n, plane_c = face_mirror_plane(all_coords, schema=schema, smooth=smooth)

    # for each frame, compute asymmetry scores for each feature, plus the overall score (average of all)
    # use per-frame plane reflection (no fixed x-flip)
    asymmetry_scores = np.full((T, len(feature_idx_left.keys())+1), np.nan)

    for t in range(T):
        coords = all_coords[t]

        # Compute mirrored error for each feature, reflecting right across the whole-face plane
        # shared by all features in this frame
        n = plane_n[t]
        c = plane_c[t]
        for i, feat in enumerate(feature_idx_left.keys()):
            xl = coords[feature_idx_left[feat], :]
            xr = coords[feature_idx_right[feat], :]
            xrm = xr - 2.0 * ((xr - c) @ n)[:, None] * n  # Householder reflection

            score = np.mean(np.sqrt(np.sum((xl-xrm)**2, axis=1)))
            asymmetry_scores[t, i] = score
        asymmetry_scores[t, -1] = np.mean(asymmetry_scores[t, 0:-1])

    column_names = list(feature_idx_left.keys())+['overall']
    return pd.DataFrame(data=asymmetry_scores, columns=column_names)


def expressivity(activations, axis=0, scales=None, aggregate=False, robust=True, fps=30, verbose=False):
    """Peak-based expressivity statistics across time scales.

    Args:
        activations: Expression activations dict from the backend.
        axis: 0 if rows are frames (default); 1 if rows are signals.
        scales: Number of time scales, explicit list of scales (sec), or None for raw signal.
        aggregate: If True, aggregate peaks across scales.
        robust: Remove outliers (IQR) before computing stats.
        fps: Frames per second for the time base.
        verbose: If True, log when no peaks are found.

    Returns:
        List of pandas.DataFrame per signal with columns scale, frequency, density, mean, std, min, max.
    """
    
    # check data type
    if not check_data_type(activations, 'expression'):
        raise ValueError("Only 'expression' data can be used for expressivity calculation. Make sure to use the correct data type.")
         
    # make sure data is in the right format
    data = get_data_values(activations)
    
    # whether rows are time points (axis=0) or signals (axis=1)
    if axis == 1:
        data = data.T
    
    num_signals = data.shape[1]
       
    expresivity_stats = []    
    # for each signal
    for i in range(num_signals):
        signal = data[:,i]
        
        # detect peaks at multiple scales
        durations, peaks = peak_detection(signal, scales=scales, aggregate=aggregate, fps=fps, smooth=True, noise_removal=False)
        num_scales = peaks.shape[0]
        
        # number of peaks, density (average across entire signal), mean (across peak activations), std, min, max
        _stats = pd.DataFrame(index=range(num_scales), columns=['scale', 'frequency', 'density', 'mean', 'std', 'min', 'max'])
        for s in range(num_scales):
            _peaks = peaks[s, :]
            
            # use only positive peaks for expressivity calculation
            idx = np.where(_peaks==1)[0]
            
            # extract the peaked signal
            # if robust, we only consider inliers (removing outliers)
            peaked_signal = signal[idx]
            if robust and len(idx) > 5:
                outliers = outlier_detectionIQR(peaked_signal)
                peaked_signal = np.delete(peaked_signal, outliers)
                
            # calculate the statistics
            if len(peaked_signal) == 0:
                if verbose:
                    print("No peaks detected for signal %d at scale %d" % (i, s))
                results = [durations[s],0,0,0,0,0,0]
            else:
                _number = len(peaked_signal)
                _density = peaked_signal.sum() / len(signal)
                _mean = peaked_signal.mean()
                _std = peaked_signal.std()
                _min = peaked_signal.min()
                _max = peaked_signal.max()
                results = [durations[s], _number, _density, _mean, _std, _min, _max]
        
            _stats.loc[s] = results
        expresivity_stats.append(_stats)
        
    return expresivity_stats


def diversity(activations, axis=0, magnitude=True, scales=None, aggregate=False, robust=True, fps=30):
    """Entropy-style diversity of expression peaks across time scales.

    Args:
        activations: Expression activations dict from the backend.
        axis: 0 if rows are frames (default); 1 if rows are signals.
        magnitude: If True, weight peaks by activation magnitude; otherwise use binary peaks.
        scales: Number of time scales (default 6) or list of scales (sec).
        aggregate: If True, aggregate peaks across scales.
        robust: Remove outliers (IQR) before computing stats.
        fps: Frames per second for the time base.

    Returns:
        pandas.DataFrame with scale-wise overall entropy and frame-wise average entropy.
    """
    
    # check data type
    if not check_data_type(activations, 'expression'):
        raise ValueError("Only 'expression' data can be used for diversity calculation. Make sure to use the correct data type.")
           
    # make sure data is in the right format
    data = get_data_values(activations)
    
    # whether rows are time points (axis=0) or signals (axis=1)
    if axis == 1:
        data = data.T
        
    num_frames, num_signals = data.shape
    
    #STEP 1: Detect peaks at multiple scales
    #---------------------------------------
    data_peaked = [] # will have shape (num_signals, num_frames, num_scales)
    for i in range(num_signals):
        signal = data[:, i]
        
        # detect peaks at multiple scales
        durations, peaks = peak_detection(signal, scales=scales, aggregate=aggregate, fps=fps, smooth=True, noise_removal=False)
        num_scales = peaks.shape[0]
        
        data_peaked.append(np.zeros((num_frames, num_scales)))
        for s in range(num_scales):
            _peaks = peaks[s, :]
            
            # use only positive peaks for expressivity calculation
            idx = np.where(_peaks==1)[0]
            
            # extract the peaked signal
            # if robust, we only consider inliers (removing outliers)
            if robust and len(idx) > 5:
                outliers = outlier_detectionIQR(signal[idx])
                idx = np.delete(idx, outliers)
            signal_peaked = np.zeros_like(signal)
            if magnitude:
                signal_peaked[idx] = signal[idx]
            else:
                signal_peaked[idx] = _peaks[idx]
            
            data_peaked[-1][:, s] = signal_peaked
    data_peaked = np.array(data_peaked) # shape (num_signals, num_frames, num_scales)
    data_peaked = np.transpose(data_peaked, (2,1,0)) # shape (num_scales, num_frames, num_signals)
    
    #STEP 2: Compute diversity at each scale
    #---------------------------------------
    diversity = pd.DataFrame(index=range(num_scales), columns=['scale', 'overall', 'frame_wise'])
    for s in range(num_scales):    
        data_s = data_peaked[s, :, :] # shape (num_frames, num_signals)
        #TODO: make sure each signal has the same range. Otherwise, we need to normalize the probabilities
                
        # type 1: compute for the entire time period
        prob = np.abs(data_s).sum(axis=0)
        prob = np.divide(prob, prob.sum(), out=np.zeros_like(prob), where=prob.sum() > 0)
        base = num_signals#2
        log_prob = log_transform(prob, base)
        
        # type 2: compute for each frame separately and take the average
        prob_frame = np.divide(np.abs(data_s), np.abs(data_s).sum(axis=1, keepdims=True), 
                       out=np.zeros_like(data_s), where=np.abs(data_s).sum(axis=1, keepdims=True) > 0)
        
        log_prob_frame = log_transform(prob_frame, base)
        
        entropy = -1 * np.sum(prob * log_prob)
        entropy_frame = -1 * np.sum(prob_frame * log_prob_frame, axis=1)

        diversity.loc[s] = [durations[s], entropy, entropy_frame.mean()]
    
    return diversity
            
