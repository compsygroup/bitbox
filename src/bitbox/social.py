from .signal_processing import windowed_cross_correlation_lagged_one_sided, windowed_cross_correlation_lagged_two_sided
from .utilities import check_data_type, convert_to_activations

import numpy as np


def coordination(X, Y, width=0.5, lag=None, step=None, fps=30, stats=True, polarity=True, angular=True):
    """Windowed cross-correlation between two motion/expression streams.

    Args:
        X: Data dict for stream X (landmark, landmark-can, pose, or expression).
        Y: Data dict for stream Y (same accepted types as X).
        width: Window width in seconds.
        lag: Max lag in seconds (None uses width).
        step: Step size in seconds between windows (None defaults to width/2).
        fps: Frames per second for both streams.
        stats: If True, return mean/std/lag summaries; otherwise return per-window arrays.
        polarity: Whether to allow both positive/negative correlations.
        angular: Treat poses as angular signals if True.

    Returns:
        If stats is True: tuple of (corr_mean, corr_std, corr_lag) arrays.
        Otherwise: tuple of (corrs, lags) per window.
    """
    # check data type
    if not check_data_type(X, ['landmark', 'landmark-can', 'pose', 'expression']):
        raise ValueError("Only 'landmark', 'pose', or 'expression' data can be used for coordination analysis. Make sure to use the correct data type.")
    if not check_data_type(Y, ['landmark', 'landmark-can', 'pose', 'expression']):
        raise ValueError("Only 'landmark', 'pose', or 'expression' data can be used for coordination analysis. Make sure to use the correct data type.")

    # convert data to activations if not already
    signal_X = convert_to_activations(X, angular=angular)
    signal_Y = convert_to_activations(Y, angular=angular)

    # both corrs and lags have shape (Nwindows, num_signals, num_signals)
    corrs, lags = windowed_cross_correlation_lagged_two_sided(signal_X, signal_Y, width=width, lag=lag, step=step, fps=fps, polarity=polarity)

    if stats:
        # identify zero (from windows with no data) and nan values in corrs and ignore them in all calculations
        mask = np.isfinite(corrs) & (corrs != 0)
        corrs_m = np.where(mask, corrs, np.nan)
        lags_m  = np.where(mask, lags,  np.nan)

        corr_mean = np.nanmean(corrs_m, axis=0) # (num_signals, num_signals)
        corr_std  = np.nanstd(corrs_m,  axis=0) # (num_signals, num_signals)
        corr_lag  = np.nanmean(lags_m,  axis=0) # (num_signals, num_signals)
        
        return corr_mean, corr_std, corr_lag
    else:
        return corrs, lags


def imitation(X, Y, width=0.5, lag=None, step=None, fps=30, stats=True, polarity=True, casuality=True, angular=True):
    """One-sided cross-correlation to quantify imitation/lead-lag dynamics.

    Args:
        X: Data dict for stream X (landmark, landmark-can, pose, or expression).
        Y: Data dict for stream Y (same accepted types as X).
        width: Window width in seconds.
        lag: Max lag in seconds (None uses width).
        step: Step size in seconds between windows (None defaults to width/2).
        fps: Frames per second for both streams.
        stats: If True, return mean/std/lag summaries; otherwise return per-window arrays.
        polarity: Whether to allow both positive/negative correlations.
        casuality: If True, enforce causal direction (Y follows X).
        angular: Treat poses as angular signals if True.

    Returns:
        If stats is True: tuple of (corr_mean, corr_std, corr_lag) arrays.
        Otherwise: tuple of (corrs, lags) per window.
    """
    # check data type
    if not check_data_type(X, ['landmark', 'landmark-can', 'pose', 'expression']):
        raise ValueError("Only 'landmark', 'pose', or 'expression' data can be used for coordination analysis. Make sure to use the correct data type.")
    if not check_data_type(Y, ['landmark', 'landmark-can', 'pose', 'expression']):
        raise ValueError("Only 'landmark', 'pose', or 'expression' data can be used for coordination analysis. Make sure to use the correct data type.")

    # convert data to activations if not already
    signal_X = convert_to_activations(X, angular=angular)
    signal_Y = convert_to_activations(Y, angular=angular)

    # both corrs and lags have shape (Nwindows, num_signals, num_signals)
    corrs, lags = windowed_cross_correlation_lagged_one_sided(signal_X, signal_Y, width=width, lag=lag, step=step, fps=fps, polarity=polarity, casuality=casuality)

    if stats:
        # identify zero (from windows with no data) and nan values in corrs and ignore them in all calculations
        mask = np.isfinite(corrs) & (corrs != 0)
        corrs_m = np.where(mask, corrs, np.nan)
        lags_m  = np.where(mask, lags,  np.nan)

        corr_mean = np.nanmean(corrs_m, axis=0) # (num_signals, num_signals)
        corr_std  = np.nanstd(corrs_m,  axis=0) # (num_signals, num_signals)
        corr_lag  = np.nanmean(lags_m,  axis=0) # (num_signals, num_signals)
    
        return corr_mean, corr_std, corr_lag
    else:
        return corrs, lags
