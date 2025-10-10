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


def expressivity(activations, axis=0, scales=None, aggregate=False, robust=True, fps=30, verbose=False):
    """
    scales:   either the number of time scales to be considered or a list of time scales in seconds, or None (using the original signal)
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
    """
    scales:   either the number of time scales to be considered (default, 6) or a list of time scales in seconds
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
            
