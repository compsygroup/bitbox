from scipy.stats import pearsonr, spearmanr
import math
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def windowed_cross_correlation_lagged_one_sided(X, Y, width=0.5, lag=None, step=None, fps=30.0, eps=1e-12, polarity=True, causality=False):
    """
    Windowed cross-correlation between 2D arrays X and Y with bidirectional lags on the second array (Y) only.
    
    Parameters
    ----------
    X : array_like, shape (Tx, Nx)
        Time (axis=0) by signal (axis=1) matrix for source 1.
    Y : array_like, shape (Ty, Ny)
        Time (axis=0) by signal (axis=1) matrix for source 2.
    width : float, default 0.5
        Window width in seconds.
    lag : float or None, default None
        Maximum absolute lag allowed (in seconds). If None, defaults to width/4.
    step : float or None, default None
        Step between consecutive windows (in seconds). If None, defaults to width/2.
    fps : float, default 30.0
        Frames (samples) per second.
    eps : float, default 1e-12
        Small constant to avoid division by zero.
    polarity: bool, default True
        If True, treat positive and negative correlations equally by selecting maximum absolute correlation.
    causality: bool, default False
        If True, only past lags (L<=0) are considered. A window of X at time t can only correlate with the
        windows of Y at time t or earlier, but not later

    Returns
    -------
    Xcorr : ndarray, shape (Nwin, Nx, Ny)
        Maximum Pearson correlation coefficient over lags for each window and pair.
    Xlag : ndarray, shape (Nwin, Nx, Ny)
        Selected lag (in seconds) that maximizes the correlation for each window and pair.

    Notes
    -----
    - Windows are defined on X and Y over the *same* base start times, but Y is
      additionally shifted by all integer lags in [-L, ..., +L].
    - Only windows for which *all* lags are valid are considered:
        start ∈ [L, T - W - L], where T = min(Tx, Ty).
    - Pearson correlation is computed (zero-mean across the window) and normalized
      by (W * std_x * std_y), consistent with ddof=0.
    """
    
    # Cast to float32 to save memory and speed up computations
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    Tx, Nx = X.shape
    Ty, Ny = Y.shape

    # Convert seconds to frames
    W = int(round(width * fps))
    if W < 1:
        raise ValueError("width too small after converting to frames.")
    if step is None:
        S = max(1, W // 2)
    else:
        S = int(round(step * fps))
        if S < 1:
            S = 1
    if lag is None:
        L = max(0, W // 4)
    else:
        L = int(round(lag * fps))
        if L < 0:
            L = 0

    # Use only the overlapping time span
    T = min(Tx, Ty)
    Xc = X[:T]
    Yc = Y[:T]

    # Require full lag coverage at both ends: start ∈ [L, T-W-L]
    max_start = T-W-L
    if max_start < L:
        # No valid windows with full ±L lag
        return (np.zeros((0, Nx, Ny), dtype=np.float32), np.zeros((0, Nx, Ny), dtype=np.float32))

    starts = np.arange(L, max_start + 1, S)
    Nwin = starts.size

    # Gather windows
    Xsw = sliding_window_view(Xc, window_shape=(W,), axis=0) # (T-W+1, Nx, W)
    Ysw = sliding_window_view(Yc, window_shape=(W,), axis=0) # (T-W+1, Ny, W)
    # Lag blocks for Y; center at each 'start' so that idx L == zero lag
    Ysw_lag = sliding_window_view(Ysw, window_shape=(2*L+1,), axis=0)  # ((T-W+1)-2L, Ny, W, 2L+1)
    
    # final windows to consider
    # for Ywin, index [:, :, :, L] corresponds to zero lag
    Xwin = Xsw[starts] # (Nwin, Nx, W) 
    Ywin = Ysw_lag[starts - L] # (Nwin, Ny, W, 2L+1)
    
    
    # --- restrict to NON-POSITIVE lags only (indices 0..L; idx L == zero lag) ---
    # if this is used, then in the remaining of the code every dimension of 2L+1
    # becomes L+1
    if causality:
        Ywin = Ywin[..., :L+1]              # (Nwin, Ny, W, L+1)

    # Zero-mean / std along time (axis=2)
    Xmean = Xwin.mean(axis=2, keepdims=True) # (Nwin, Nx, 1)
    Xstd = Xwin.std(axis=2, keepdims=True) # (Nwin, Nx, 1)
    X0 = Xwin - Xmean # (Nwin, Nx, W)
    
    Ymean = Ywin.mean(axis=2, keepdims=True) # (Nwin, Ny, 1, 2L+1)
    Ystd = Ywin.std(axis=2, keepdims=True) # (Nwin, Ny, 1, 2L+1)
    Y0 = Ywin - Ymean # (Nwin, Ny, W, 2L+1)
    
    # Correlation numerator over time
    # X0: (Nwin, Nx, W), Y0: (Nwin, Ny, W, 2L+1)
    num = np.einsum('niw,njwl->nijl', X0, Y0, optimize=True) # (Nwin, Nx, Ny, 2L+1)
    
    # Denominator: W * std_x * std_y  (W cancels the 1/W in stds)
    Xstd_b = np.maximum(Xstd, eps)[:, None] # (Nwin, Nx, 1, 1)
    Ystd_b = np.maximum(Ystd, eps) # (Nwin, Ny, 1, 2L+1)
    denom = W * Xstd_b * Ystd_b # (Nwin, Ny, Nx, 2L+1)
    denom = denom.transpose(0,2,1,3) # (Nwin, Nx, Ny, 2L+1)
    
    r = num / denom # (Nwin, Nx, Ny, 2L+1)
    
    # Max over lag and argmax
    if polarity:
        lag_idx = np.argmax(np.abs(r), axis=3) # (Nwin, Nx, Ny)
    else:
        lag_idx = np.argmax(r, axis=3) # (Nwin, Nx, Ny)
    # r_max = np.take_along_axis(r, lag_idx[:, None], axis=3)[:, :, :, 0]  # (Nwin, Nx, Ny)
    lags_sec = (lag_idx - L) / float(fps)
    
    r_max = np.take_along_axis(r, lag_idx[..., None], axis=3)[..., 0]

    return r_max.astype(np.float32), lags_sec.astype(np.float32)


def windowed_cross_correlation_lagged_two_sided(X, Y, width=0.5, lag=None, step=None, fps=30.0, eps=1e-12, polarity=True):
    """
    Windowed cross-correlation between 2D arrays X and Y with bidirectional lags on both arrays.
    
    Parameters
    ----------
    X : array_like, shape (Tx, Nx)
        Time (axis=0) by signal (axis=1) matrix for source 1.
    Y : array_like, shape (Ty, Ny)
        Time (axis=0) by signal (axis=1) matrix for source 2.
    width : float, default 0.5
        Window width in seconds.
    lag : float or None, default None
        Maximum absolute lag allowed (in seconds). If None, defaults to width/4.
    step : float or None, default None
        Step between consecutive windows (in seconds). If None, defaults to width/2.
    fps : float, default 30.0
        Frames (samples) per second.
    eps : float, default 1e-12
        Small constant to avoid division by zero.
    polarity: bool, default True
        If True, treat positive and negative correlations equally by selecting maximum absolute correlation.

    Returns
    -------
    Xcorr : ndarray, shape (Nwin, Nx, Ny)
        Maximum Pearson correlation coefficient over lags for each window and pair.
    Xlag : ndarray, shape (Nwin, Nx, Ny)
        Selected lag (in seconds) that maximizes the correlation for each window and pair.
    """
    
    # Cast to float32 to save memory and speed up computations
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    Tx, Nx = X.shape
    Ty, Ny = Y.shape
    
    # Convert seconds to frames
    W = int(round(width * fps))
    if W < 1:
        raise ValueError("width too small after converting to frames.")
    if step is None:
        S = max(1, W // 2)
    else:
        S = int(round(step * fps))
        if S < 1:
            S = 1
    if lag is None:
        L = max(0, W // 4)
    else:
        L = int(round(lag * fps))
        if L < 0:
            L = 0
    
    # Use only the overlapping time span
    T = min(Tx, Ty)
    Xc = X[:T]
    Yc = Y[:T]
    
    # Require full lag coverage at both ends: start ∈ [L, T-W-L]
    max_start = T-W-L
    if max_start < L:
        # No valid windows with full ±L lag
        return (np.zeros((0, Nx, Ny), dtype=np.float32), np.zeros((0, Nx, Ny), dtype=np.float32))

    starts = np.arange(L, max_start + 1, S)
    Nwin = starts.size
    
    # Gather windows
    Xsw = sliding_window_view(Xc, window_shape=(W,), axis=0)  # (T-W+1, Nx, W)
    Ysw = sliding_window_view(Yc, window_shape=(W,), axis=0)  # (T-W+1, Ny, W)
    
    # Precompute all Y lag windows relative to base 'starts'
    Ysw_lag = sliding_window_view(Ysw, window_shape=(2*L+1,), axis=0)   # ((T-W+1)-2L, Ny, W, 2L+1)
    Ywin = Ysw_lag[starts - L]                                       # (Nwin, Ny, W, 2L+1)
    
    # Zero-mean/std for Y once (time axis=2)
    Ymean = Ywin.mean(axis=2, keepdims=True)                            # (Nwin, Ny, 1, 2L+1)
    Ystd = Ywin.std(axis=2, keepdims=True)                             # (Nwin, Ny, 1, 2L+1)
    Y0 = Ywin - Ymean                                                # (Nwin, Ny, W, 2L+1)
    
    # Pre-broadcast Y std: (Nwin, 1, Ny, 2L+1)
    Ystd_b = np.maximum(Ystd, eps).reshape(Nwin, 1, Ny, 2*L+1).astype(np.float32)
    
    if polarity:
        best_r = np.full((Nwin, Nx, Ny), 0, dtype=np.float32)
    else:
        best_r = np.full((Nwin, Nx, Ny), -np.inf, dtype=np.float32)
    best_ly_idx = np.zeros((Nwin, Nx, Ny), dtype=np.int16)   # 0..2L
    best_lx_val = np.zeros((Nwin, Nx, Ny), dtype=np.int16)   # [-L..L]
    
    for lx in range(-L, L + 1):
        idx_x = starts + lx                                  # (Nwin,)
        Xwin = Xsw[idx_x]                                   # (Nwin, Nx, W)
    
        # Zero-mean/std for X at this lx  (time axis=2)
        Xmean = Xwin.mean(axis=2, keepdims=True)             # (Nwin, Nx, 1)
        Xstd = Xwin.std(axis=2, keepdims=True)              # (Nwin, Nx, 1)
        X0 = Xwin - Xmean                                 # (Nwin, Nx, W)
    
        # Numerator over time
        # X0: (Nwin, Nx, W), Y0: (Nwin, Ny, W, 2L+1) -> (Nwin, Nx, Ny, 2L+1)
        num = np.einsum('niw,njwl->nijl', X0, Y0, optimize=True).astype(np.float32)
    
        # Denominator: W * std_x * std_y
        Xstd_b = np.maximum(Xstd, eps).reshape(Nwin, Nx, 1, 1).astype(np.float32)  # (Nwin, Nx, 1, 1)
        denom = (W * Xstd_b * Ystd_b)                                             # (Nwin, Nx, Ny, 2L+1)
    
        r = num / denom                                                            # (Nwin, Nx, Ny, 2L+1)
    
        # Best over ly for this lx
        if polarity:
            ly_idx = np.argmax(np.abs(r), axis=3)                                # (Nwin, Nx, Ny)
        else:
            ly_idx = np.argmax(r, axis=3)                                            # (Nwin, Nx, Ny)
        r_max_ly = np.take_along_axis(r, ly_idx[..., None], axis=3)[..., 0]        # (Nwin, Nx, Ny)
    
        # Update global best over (lx, ly)
        if polarity:
            better = np.abs(r_max_ly) > np.abs(best_r)
        else:
            better = r_max_ly > best_r
        best_r[better] = r_max_ly[better]
        best_ly_idx[better] = ly_idx[better]
        best_lx_val[better] = lx
    
    # Net lag (seconds): (ly - lx) / fps, with ly = best_ly_idx - L
    lags_sec = ((best_ly_idx.astype(np.int32) - L) - best_lx_val.astype(np.int32)).astype(np.float32) / float(fps)

    return best_r.astype(np.float32), lags_sec


def windowed_cross_correlation(X, Y, width=0.5, lag=None, step=None, fps=30):
    width = int(round(fps*width))
    
    if step is None:
        step = int(width/2.)
    else:
        step = int(round(fps*step))
    
    if lag is None:
        lag = int(width/4.)
    else:
        lag = int(round(fps*lag))
    
    T = min((X.shape[0], Y.shape[0]))
    
    pairs = [(i1, i2) for i1 in range(0,X.shape[1]) for i2 in range(0,Y.shape[1])]
    
    window_offsets = range(0, T-width, step)
    Nwindows = len(window_offsets)
    Xcorr = np.zeros((Nwindows, len(pairs)))
    Xlag  = np.zeros((Nwindows, len(pairs)))
    
    pidx = 0
    for pidx in range(0, len(pairs)):
        pair = pairs[pidx]
        
        # if X and Y are different vectors, diagonal is meaningful
        # if pair[0] == pair[1]:
        #     continue
        
        for tidx in range(0, Nwindows):
            t1 = window_offsets[tidx]
            t2 = t1
            
            x = X[range(t1,t1+width), pair[0]]
            y = Y[range(t2,t2+width), pair[1]]
        
            nx = (x-np.mean(x))/(np.std(x)*len(x)+np.finfo(float).eps)
            ny = (y-np.mean(y))/(np.std(y)+np.finfo(float).eps)
        
            corr = np.correlate(nx, ny, 'full')
        
            zero_out_frames = round((width - lag))
            corr[0:zero_out_frames] = -1
            corr[0:round(len(corr)/2.0)] = -1
            corr[-zero_out_frames:] = -1
                                
            maxcorr = np.max(corr)
            maxlag = (np.argmax(corr) - round(width / 2)) / fps # in seconds
        
            Xcorr[tidx, pidx] = maxcorr
            Xlag[tidx, pidx] = maxlag
            
    return Xcorr, Xlag
