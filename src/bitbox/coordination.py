from .signal_processing import  windowed_cross_correlation, windowed_cross_correlation_2S
from .utilities import get_data_values,check_data_type

import numpy as np


def intra_person_coordination(data, axis=0, width=0.5, lag=None, step=None, fps=30):
    # check data type
    if not check_data_type(data, ['expression', 'pose']):
        raise ValueError("Only 'expression' or 'pose' data can be used for coordination calculation. Make sure to use the correct data type.")
        
    # make sure data is in the right format
    data = get_data_values(data)
    
    # whether rows are time points (axis=0) or signals (axis=1)
    if axis == 1:
        data = data.T
        
    num_signals = data.shape[1]
    
    corr_mean = np.zeros((num_signals, num_signals))
    corr_std = np.zeros((num_signals, num_signals))
    corr_lag = np.zeros((num_signals, num_signals))
    
    corrs, lags = windowed_cross_correlation(data, data, width=width, lag=lag, step=step)
    
    pairs = [(i1, i2) for i1 in range(num_signals) for i2 in range(num_signals)]
    
    # calculate the average correlation and lag
    for idx in range(corrs.shape[1]):
        i, j = pairs[idx]
        corr_mean[i,j] = corrs[:,idx].mean()
        corr_std[i,j] = corrs[:,idx].std()
        corr_lag[i,j] = lags[:,idx].mean()
            
    return corr_mean, corr_std, corr_lag


def intra_person_coordination_2S(data, axis=0, width=0.5, lag=None, step=None, fps=30, ordinal=False):
    # check data type
    if not check_data_type(data, ['expression', 'pose']):
        raise ValueError("Only 'expression' or 'pose' data can be used for coordination calculation. Make sure to use the correct data type.")
        
    # make sure data is in the right format
    data = get_data_values(data)
    
    # whether rows are time points (axis=0) or signals (axis=1)
    if axis == 1:
        data = data.T
        
    num_signals = data.shape[1]
    
    corr_mean = np.zeros((num_signals, num_signals))
    corr_std = np.zeros((num_signals, num_signals))
    
    # for each pair of signals
    for i in range(num_signals-1):
        for j in range(i+1, num_signals):
            print(i, j)
            
            x = data[:,i]
            y = data[:,j]
            
            # calculate the windowed cross-correlation
            corrs = windowed_cross_correlation_2S(x, y, width=width, lag=lag, step=step, ordinal=ordinal)
            
            # calculate the average correlation
            corr_mean[i,j] = corr_mean[j,i] = np.mean(corrs)
            corr_std[i,j] = corr_std[j,i] = np.std(corrs)
            
    return corr_mean, corr_std




            
            