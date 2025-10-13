import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
import matplotlib.pylab as plt

import pywt

def _value_at_percentile(data, percentile):
    # Sort the data
    sorted_data = np.sort(data)
    # Calculate the index. Note: numpy uses 0-based indexing
    index = int(np.floor(len(sorted_data) * percentile / 100.0))
    # Ensure the index is within the bounds of the list
    index = min(max(index, 0), len(sorted_data) - 1)
    # Return the value at the calculated index
    return sorted_data[index]


def _peak_detector(signal, noise_removal=True):
    # Treat positive and negative peaks separately
    # output is 0,1, or -1, with 1 for peaks and -1 for valleys
    positives = signal.copy()
    negatives = signal.copy()
    positives[positives < 0] = 0
    negatives[negatives > 0] = 0
    
    _data = [positives, negatives]
    output = np.zeros_like(signal)
    
    for s in range(2):
        # detect all peaks/valleys where derivative changes sign
        _d = np.abs(_data[s])
        jumps = np.diff(np.sign(np.diff(_d)))
        peaks = np.where(jumps == -2)[0] + 1
        
        # remove tiny peaks
        if noise_removal:
            N = len(signal)
            # robust noise sigma (use first differences to avoid slow trends)
            dif = np.diff(signal)
            sigma = 1.4826 * np.median(np.abs(dif - np.median(dif))) + 1e-12
            tau = sigma * np.sqrt(2*np.log(N))          # universal threshold
            mags = np.abs(signal[peaks])
            peaks = peaks[mags >= tau]
            
            # magnitudes = np.abs(signal[peaks])
            # thresholds = _value_at_percentile(magnitudes, 97.5) * 0.1
            # idx = np.where(magnitudes < thresholds)[0]
            # peaks = np.delete(peaks, idx)
        
        output[peaks] = 1 if s == 0 else -1
        
    return output


def _wavelet_decomposition(signal, scales):
    # decomposition
    cwtmatr, freqs = pywt.cwt(signal, scales, 'mexh')
    
    # smooth coefficients
    cwtmatr = np.array([gaussian_filter(cwtmatr[s, :], sigma=1) for s in range(cwtmatr.shape[0])])
       
    return cwtmatr
        
        
def _aggregate_peaks(peaks, wavelets, fps=30, t_suppress=0.1, scale_win=3):
    """
    peaks: (S,T) in {0,1,-1}
    wavelets: (S,T)
    returns: (T,) in {-1,0,1}
    """
    S, T = peaks.shape
    W = wavelets

    # robust per-scale normalization
    med = np.median(W, axis=1, keepdims=True)
    mad = np.median(np.abs(W - med), axis=1, keepdims=True)
    mad = np.where(mad == 0, 1e-12, mad)
    score = (W - med) / mad

    t_win = max(1, int(round(t_suppress * fps)))

    # positives: 2D NMS on score
    mask_pos = (peaks == 1)
    score_pos = np.where(mask_pos, score, -np.inf)
    nms_pos = (score_pos == maximum_filter(score_pos, size=(scale_win, 2*t_win+1), mode="nearest"))
    s_pos, t_pos = np.where(nms_pos & mask_pos)
    v_pos = score[s_pos, t_pos]
    sign_pos = np.ones_like(v_pos, dtype=int)

    # negatives: 2D NMS on -score (equiv. min on score)
    mask_neg = (peaks == -1)
    score_neg_for_nms = np.where(mask_neg, -score, -np.inf)
    nms_neg = (score_neg_for_nms == maximum_filter(score_neg_for_nms, size=(scale_win, 2*t_win+1), mode="nearest"))
    s_neg, t_neg = np.where(nms_neg & mask_neg)
    v_neg = score[s_neg, t_neg]          # negative values (more negative = stronger)
    sign_neg = -np.ones_like(v_neg, dtype=int)

    # merge candidates, rank by absolute strength
    cand_t = np.concatenate([t_pos, t_neg])
    cand_v = np.concatenate([v_pos, v_neg])
    cand_sign = np.concatenate([sign_pos, sign_neg])

    # Greedy 1D NMS over time
    order = np.argsort(-np.abs(cand_v))
    taken = np.zeros(T, dtype=bool)
    out = np.zeros(T, dtype=int)

    for i in order:
        t = cand_t[i]
        lo, hi = max(0, t - t_win), min(T, t + t_win + 1)
        if not taken[lo:hi].any():
            out[t] = cand_sign[i]
            taken[lo:hi] = True

    return out



def _visualize_peaks(signal, wavelets, peaks, fps):
    num_scales = wavelets.shape[0]
    dt = 1. / fps
    seconds = dt * np.arange(len(signal))
    
    fig, ax = plt.subplots(num_scales+1, 1, figsize=(25, num_scales*3))
    x = [signal] + [wavelets[s,:] for s in range(num_scales)]
    
    for s in range(num_scales+1):
        if s == 0:
            if num_scales == 1:
                _peaks = peaks[0, :]
            else:
                _peaks = _aggregate_peaks(peaks, wavelets, fps=fps)
            # _peaks = peaks.sum(axis=0)
            # _peaks[_peaks>0] = 1
            # _peaks[_peaks<0] = 0#-1
        else:
            _peaks = peaks[s-1, :]
            
        ax[s].plot(seconds, x[s])
        ax[s].hlines(x[s].mean(), xmin=seconds.min(), xmax=seconds.max(), ls='--')
        
        ax[s].vlines(seconds[np.where(_peaks==1)[0]], ymin=x[s].min(), ymax=x[s].max(), colors='blue', linewidth=1)
        # ax[s].vlines(seconds[np.where(_peaks==-1)[0]], ymin=x[s].min(), ymax=x[s].max(), colors='red', linewidth=1)
        
        dx = round(seconds.max() / 40, 2)
        ax[s].set_xticks(np.arange(0, seconds.max()+dx, dx))


def peak_detection(data, scales=None, aggregate=False, fps=30, smooth=True, noise_removal=True, visualize=False):
    """
    scales:   either the number of time scales to be considered or a list of time scales in seconds, or None (using the original signal)
    """
    
    # check if the data is a list
    if not isinstance(data, list):
        datal = [data]
    else:
        datal = data
        
    # determine time scales
    # For PyWavelets 'mexh', pseudo-freq f≈0.25/(scale⋅dt). To target events of duration d seconds, use scale ≈ (fps * d) / 4
    if scales is None:
        num_scales = 1
        durations = [None]
    elif isinstance(scales, list):
        num_scales = len(scales)
        durations = np.array(scales)
        # map duration->mexh scale
        scales = (fps * durations) / 4.0
    elif isinstance(scales, int):
        if scales < 0:
            raise ValueError("scales must be a positive integer or a list")
        num_scales = scales
        durations = np.linspace(0.1, 4.0, num=num_scales)  # seconds
        # map duration->mexh scale
        scales = (fps * durations) / 4.0
        #scales = (fps/30) * np.geomspace(1, 18, num=num_scales)
    else:
        raise ValueError("scales must be either an integer or a list")
    
    if num_scales == 1:
        aggregate = False
        
    # for each signal in the list
    peaksl = []
    for i in range(len(datal)):
        signal_org = datal[i]
        
        if (scales is not None) and (np.max(durations) >= len(signal_org)/fps):
            raise ValueError(f"The maximum scale cannot be larger than the signal length. Signal length: {len(signal_org)/fps:.1f} s, max scale: {np.max(durations):.1f} s")
        
        # zero mean the signal
        signal = signal_org - signal_org.mean()
        
        # smooth the signal
        if smooth:
            signal = gaussian_filter(signal, sigma=1)
        
        if scales is None:
            peaks = _peak_detector(signal, noise_removal=noise_removal).reshape([1,-1])
            if visualize:
                _visualize_peaks(signal_org, signal.reshape([1,-1]), peaks, fps)
        else:
            # wavelet decomposition at multiple scales
            wavelets = _wavelet_decomposition(signal, scales)
            
            # peak detection at different scales
            peaks = np.zeros_like(wavelets)
            for s in range(num_scales):
                peaks[s, :] = _peak_detector(wavelets[s, :], noise_removal=noise_removal)
            
            if visualize:
                _visualize_peaks(signal_org, wavelets, peaks, fps)
                
            if aggregate:
                peaks = _aggregate_peaks(peaks, wavelets, fps=fps, t_suppress=np.min(durations-0.01), scale_win=3).reshape([1,-1])
                durations = [f"Agg {durations.min():.1f}-{durations.max():.1f}"]
            
        peaksl.append(peaks)
            
    if not isinstance(data, list):
        peaksl = peaksl[0]
        
    return durations, peaksl