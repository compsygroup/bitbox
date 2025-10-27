from .signal_processing import  windowed_cross_correlation_lagged_one_sided, windowed_cross_correlation_lagged_two_sided
from .utilities import check_data_type, convert_to_activations

import numpy as np


def coordination(X, Y, width=0.5, lag=None, step=None, fps=30, stats=True, polarity=True, angular=True):       
    # check data type
    if not check_data_type(X, ['landmark', 'landmark-can', 'pose', 'expression']):
        raise ValueError("Only 'landmark', 'pose', or 'expression' data can be used for coordination analysis. Make sure to use the correct data type.")
    if not check_data_type(Y, ['landmark', 'landmark-can', 'pose', 'expression']):
        raise ValueError("Only 'landmark', 'pose', or 'expression' data can be used for coordination analysis. Make sure to use the correct data type.")

    # convert data to activations if not already
    signal_X = convert_to_activations(X, angular=angular)
    signal_Y = convert_to_activations(Y, angular=angular)

    # TODO: in documentation (and warning in code) make sure with 3DI-lite, pose is not pose
        
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
    # check data type
    if not check_data_type(X, ['landmark', 'landmark-can', 'pose', 'expression']):
        raise ValueError("Only 'landmark', 'pose', or 'expression' data can be used for coordination analysis. Make sure to use the correct data type.")
    if not check_data_type(Y, ['landmark', 'landmark-can', 'pose', 'expression']):
        raise ValueError("Only 'landmark', 'pose', or 'expression' data can be used for coordination analysis. Make sure to use the correct data type.")

    # convert data to activations if not already
    signal_X = convert_to_activations(X, angular=angular)
    signal_Y = convert_to_activations(Y, angular=angular)

    # TODO: in documentation (and warning in code) make sure with 3DI-lite, pose is not pose
        
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