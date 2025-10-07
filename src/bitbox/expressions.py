from .utilities import get_data_values, check_data_type
from .signal_processing import peak_detection, outlier_detectionIQR, log_transform
from .utilities import landmarks_left_right
import numpy as np
import pandas as pd

# Calculate asymmetry scores using mirror error approach
def asymmetry(landmarks, axis=0, normalize=True):
    # check data type
    if not check_data_type(landmarks, ['landmark', 'landmark-can']):
        raise ValueError("Only 'landmark' data can be used for asymmetry calculation. Make sure to use the correct data type.")
    
    # read actual values
    data = get_data_values(landmarks)
    
    # whether rows are time points (axis=0) or signals (axis=1)
    if axis == 1:
        data = data.T
    
    processor = landmarks['backend']
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
    
    # for each frame, compute asymmetry scores for each feature, plus the overall score (average of all)
    # use per-frame, per-feature plane reflection (no fixed x-flip)
    asymmetry_scores = np.full((T, len(feature_idx_left.keys())+1), np.nan)
    
    for t in range(T):
        coords = data[t, :]
        
        if len(coords) % dimension != 0:
            raise ValueError(f"Landmarks are not {dimension} dimensional. Please set the correct dimension.")
        
        num_landmarks = int(len(coords) / dimension)
        coords = coords.reshape((num_landmarks, dimension))

        # Compute mirrored error for each feature
        for i, feat in enumerate(feature_idx_left.keys()):
            xl = coords[feature_idx_left[feat], :]
            xr = coords[feature_idx_right[feat], :]
            
            # reflect right across the perpendicular-bisector plane between centroids
            cL = xl.mean(axis=0)
            cR = xr.mean(axis=0)
            n = cR - cL
            nn = np.linalg.norm(n)
            if nn > 1e-8: 
                n = n / nn
                c = 0.5 * (cL + cR)
                xrm = xr - 2.0 * ((xr - c) @ n)[:, None] * n  # Householder reflection
            else: # fallback if features coincide (degenerate)
                xrm = xr
            
            score = np.mean(np.sqrt(np.sum((xl-xrm)**2, axis=1)))
            asymmetry_scores[t, i] = score
        asymmetry_scores[t, -1] = np.mean(asymmetry_scores[t, 0:-1])
    
    column_names = list(feature_idx_left.keys())+['overall']
    _scores = pd.DataFrame(data=asymmetry_scores, columns=column_names)
    
    # normalze scores based on expected landmark errors for perfectly symmetric faces
    # and extreme values generated from Jim Carrey videos
    if normalize:
        if dimension == 3: # 3D canonicalized landmarks
            if processor == '3DI':
                min_sym = pd.Series({"eye":0.5801, "brow":0.7857, "nose":0.0965, "mouth":1.0076, "overall":0.6172}) # 50 perc (median) of sym
                max_jim = pd.Series({"eye":3.1310, "brow":2.2180, "nose":1.4357, "mouth":5.5638, "overall":2.4260}) # 99 perc of jim
            elif processor == '3DIl':
                min_sym = pd.Series({"eye":0.5321, "brow":0.8457, "nose":0.1585, "mouth":1.0731, "overall":0.6547}) # 50 perc (median) of sym
                max_jim = pd.Series({"eye":1.4455, "brow":2.3044, "nose":0.7789, "mouth":3.1907, "overall":1.4160}) # 99 perc of jim
            else:
                raise ValueError("Data is from an unsupported backend processor. Normalization cannot be applied.")
                
            asymmetry_scores = ((_scores - min_sym) / (max_jim - min_sym)).clip(lower=0)
        elif dimension == 2: # 2D landmarks
            min_sym = pd.Series({"eye":0.9330, "brow":1.1110, "nose":0.0001, "mouth":0.7021, "overall":0.7689})
            asymmetry_scores = (_scores - min_sym).clip(lower=0)
        else:
            raise ValueError("Unsupported dimension for asymmetry normalization. Use 2D or 3D landmarks.")
    else:
        asymmetry_scores = _scores
        
    return asymmetry_scores


# use_negatives: whether to use negative peaks
# 0: only positive peaks, 1: only negative peaks, 2: both
def expressivity(activations, axis=0, use_negatives=0, scales=6, robust=True, fps=30):
    """
    scales:   either the number of time scales to be considered (default, 6) or a list of time scales in seconds
    """
    
    # check data type
    if not check_data_type(activations, 'expression'):
        raise ValueError("Only 'expression' data can be used for expressivity calculation. Make sure to use the correct data type.")
    
    # determine time scales
    if isinstance(scales, list):
        num_scales = len(scales)
    elif isinstance(scales, int):
        if scales == 0:
            num_scales = 1
        else:
            num_scales = scales
    else:
        raise ValueError("scales must be either an integer or a list")
        
    # make sure data is in the right format
    data = get_data_values(activations)
    
    # whether rows are time points (axis=0) or signals (axis=1)
    if axis == 1:
        data = data.T
    
    num_signals = data.shape[1]
    
    expresivity_stats = []
    # define dataframes for each scale
    for s in range(num_scales):
         # number of peaks, density (average across entire signal), mean (across peak activations), std, min, max
        _data = pd.DataFrame(columns=['number', 'density', 'mean', 'std', 'min', 'max'])
        expresivity_stats.append(_data)
    
    # for each signal
    for i in range(num_signals):
        signal = data[:,i]
        
        # detect peaks at multiple scales
        peaks = peak_detection(signal, scales=scales, fps=fps, smooth=True, noise_removal=False)
        
        for s in range(num_scales):
            _peaks = peaks[s, :]
            
            # whether we use negative peaks
            if use_negatives == 0:
                idx = np.where(_peaks==1)[0]
            elif use_negatives == 1:
                idx = np.where(_peaks==-1)[0]
            elif use_negatives == 2:
                idx = np.where(_peaks!=0)[0]
            else:
                raise ValueError("Invalid value for use_negatives")
            
            # extract the peaked signal
            # if robust, we only consider inliers (removing outliers)
            peaked_signal = signal[idx]
            if robust and len(idx) > 5:
                outliers = outlier_detectionIQR(peaked_signal)
                peaked_signal = np.delete(peaked_signal, outliers)
                
            # calculate the statistics
            if len(peaked_signal) == 0:
                print("No peaks detected for signal %d at scale %d" % (i, s))
                results = np.zeros(6)
            else:
                _number = len(peaked_signal)
                _density = peaked_signal.sum() / len(signal)
                _mean = peaked_signal.mean()
                _std = peaked_signal.std()
                _min = peaked_signal.min()
                _max = peaked_signal.max()
                results = [_number, _density, _mean, _std, _min, _max]
        
            expresivity_stats[s].loc[i] = results
        
    return expresivity_stats


def diversity(activations, axis=0, use_negatives=0, scales=6, robust=True, fps=30):
    """
    scales:   either the number of time scales to be considered (default, 6) or a list of time scales in seconds
    """
    
    # check data type
    if not check_data_type(activations, 'expression'):
        raise ValueError("Only 'expression' data can be used for diversity calculation. Make sure to use the correct data type.")
    
    # determine time scales
    if isinstance(scales, list):
        num_scales = len(scales)
    elif isinstance(scales, int):
        num_scales = scales
    else:
        raise ValueError("scales must be either an integer or a list")
        
    # make sure data is in the right format
    data = get_data_values(activations)
    
    # whether rows are time points (axis=0) or signals (axis=1)
    if axis == 1:
        data = data.T
        
    num_frames, num_signals = data.shape
    
    #STEP 1: Detect peaks at multiple scales
    #---------------------------------------

    # peak data will have shape (num_scales, num_frames, num_signals)
    # we will compute diversity for pos and neg separately and take the average
    data_peaked_pos = [0] * num_scales
    data_peaked_neg = [0] * num_scales
    for s in range(num_scales):
        data_peaked_pos[s] = np.zeros((num_frames, num_signals))
        data_peaked_neg[s] = np.zeros((num_frames, num_signals))

    for i in range(num_signals):
        signal = data[:, i]
        
        # detect peaks at multiple scales
        peaks = peak_detection(signal, scales=scales, fps=fps, smooth=True, noise_removal=False)
        
        for s in range(num_scales):
            _peaks = peaks[s, :]
            
            # whether we use negative peaks or not
            if use_negatives == 0: # only use positives
                _peaks[_peaks==-1] = 0
            elif use_negatives == 1: # only use negatives
                _peaks[_peaks==1] = 0
            
            # if robust, we only consider inliers (removing outliers)
            idx = np.where(_peaks!=0)[0]
            if robust and len(idx) > 5:
                outliers = outlier_detectionIQR(signal[idx])
                idx = np.delete(idx, outliers)
            tmp = _peaks[idx]
            _peaks[:] = 0
            _peaks[idx] = tmp
            
            # store the peaked signal
            signal_pos = np.zeros_like(signal)
            signal_pos[_peaks==1] = signal[_peaks==1]
            signal_neg = np.zeros_like(signal)
            signal_neg[_peaks==-1] = signal[_peaks==-1]
            
            data_peaked_pos[s][:, i] = signal_pos
            data_peaked_neg[s][:, i] = signal_neg
            
    #STEP 2: Compute diversity at each scale
    #---------------------------------------
    diversity = pd.DataFrame(index=range(num_scales), columns=['overall', 'frame_wise'])
    for s in range(num_scales):
        
        data_final = []
        if use_negatives == 0: # only use positives
            data_final = [data_peaked_pos[s]]
        elif use_negatives == 1: # only use negatives
            data_final = [data_peaked_neg[s]]
        elif use_negatives == 2: # use both
            data_final = [data_peaked_pos[s], data_peaked_neg[s]]
        else:
            raise ValueError("Invalid value for use_negatives")
        
        # compute entropy for pos and neg separately and take the average
        entropy = 0
        entropy_frame = 0
        for data_peaked in data_final:            
            #TODO: make sure each signal has the same range. Otherwise, we need to normalize the probabilities
            
            base = num_signals#2
            
            # type 1: compute for the entire time period
            prob = np.abs(data_peaked).sum(axis=0)
            normalizer = prob.sum()
            if normalizer > 0:
                prob /= normalizer
            
            log_prob = log_transform(prob, base)
            
            # type 2: compute for each frame separately and take the average
            prob_frame = np.zeros_like(data_peaked)
            for f in range(num_frames):
                normalizer = np.abs(data_peaked[f, :]).sum()
                if normalizer > 0:
                    prob_frame[f, :] = np.abs(data_peaked[f, :]) / normalizer
            prob_frame[np.isinf(prob_frame)] = 0
            prob_frame[np.isnan(prob_frame)] = 0
            
            log_prob_frame = log_transform(prob_frame, base)
            
            entropy += -1 * np.sum(prob * log_prob)
            entropy_frame += -1 * np.sum(prob_frame * log_prob_frame, axis=1)
            
        entropy /= len(data_final)
        entropy_frame /= len(data_final)
        
        diversity.loc[s, :] = [entropy, entropy_frame.mean()]
    
    return diversity
            
