from importlib import resources as ir
from .. import data
from ..utilities import check_data_type

import numpy as np
import pandas as pd

def normalizeExpressions3DI(exps, signed=False):
    # check data type
    if not check_data_type(exps, 'expression'):
        raise ValueError("Only 'expression' data can be used for normalization. Make sure to use the correct data type.")
    
    # get actual data
    exp = exps['data']
    
    # read stats
    with ir.open_text(data, "expression_noise_signal_stats_3DI.csv", encoding="utf-8") as stats_path:
        stats = np.loadtxt(stats_path, delimiter=',')
        L = stats[:,0]    # (1,79)
        U = stats[:,1]    # (1,79)
        sigma = stats[:,2]    # (1,79)
    
    # normalize
    # distance to nearest band edge (signed)
    d = np.where(exp > U, exp - U, np.where(exp < L, L - exp, 0.0))
    if signed:
        sign = np.where(exp > U, 1.0, np.where(exp < L, -1.0, 0.0)) # signed version
    else:
        sign = np.where(exp > U, 1.0, np.where(exp < L, 1.0, 0.0)) # absolute version
    z = sign * (np.abs(d) / sigma)
    
    exps['data'] = pd.DataFrame(z, columns=exp.columns, index=exp.index)
        
    return exps