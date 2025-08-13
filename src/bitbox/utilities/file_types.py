import numpy as np
import pandas as pd

def check_data_type(data, typ):
    if isinstance(data, dict):
        if 'type' in data:
            if isinstance(typ, list):
                return (data['type'] in typ)
            else:
                return (data['type'] == typ)
        else:
            return False
    
    raise ValueError("Unrecognized data type. Please only use Bitbox outputs as the input data")


def get_data_values(data):
    # check if data is a dictionary
    if isinstance(data, dict):
        data = dictionary_to_array(data)
    
    return data


def dictionary_to_array(data):
    if isinstance(data, dict):
        if 'data' in data and isinstance(data['data'], pd.DataFrame):
            return data['data'].values
    
    raise ValueError("Unrecognized data type. Please only use Bitbox outputs as the input data")