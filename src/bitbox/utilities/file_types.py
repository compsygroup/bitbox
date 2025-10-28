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


def convert_to_coords(data: dict, angular: bool = False) -> np.ndarray:
    coords = get_data_values(data)
    d = data['dimension']
    
    if check_data_type(data, ['landmark', 'landmark-can']):
        # convert the shape
        # (N,M,d) array. N: number of frames, M: number of landmarks, d: dimension (2 for 2D, 3 for 3D).
        N, D = coords.shape
        M = D // d
        coords = coords.reshape(N, M, d)
    elif check_data_type(data, 'rectangle'):
        # compute the center of the bounding box
        x, y, w, h = coords.T  # each is (N,)
        cx = x + w / 2
        cy = y + h / 2
        coords = np.stack((cx, cy), axis=1) # (N,2)
    elif check_data_type(data, 'pose'):
        if coords.shape[1] == 3:
            # already only rotations
            return coords
        elif angular:
            # just keep the rotation part
            coords = coords[:, 3:]  # (N,3)
        else:
            # just keep the translation part
            coords = coords[:, :3]  # (N,3)
            
    return coords


def convert_to_activations(data: dict, angular: bool = False) -> np.ndarray:
    coords = get_data_values(data)
    d = data['dimension']
    
    if check_data_type(data, ['landmark', 'landmark-can']):
        # convert the shape
        # (N,M,d) array. N: number of frames, M: number of landmarks, d: dimension (2 for 2D, 3 for 3D).
        N, D = coords.shape
        M = D // d
        coords = coords.reshape(N, M, d)
        # for each landmark, compute the activation as the Euclidean norm of the coordinates
        activations = np.linalg.norm(coords, axis=-1)  # (N,M)
        return activations
    elif check_data_type(data, 'expression'):
        # expression data is already in activation format
        return coords
    elif check_data_type(data, 'pose'):
        if coords.shape[1] == 3:
            # already only rotations
            coords = coords # (N,3)
        elif angular:
            # just keep the rotation part
            coords = coords[:, 3:]  # (N,3)
        else:
            # just keep the translation part
            coords = coords[:, :3]  # (N,3)
        # compute the activation as the Euclidean norm of the pose components
        activations = np.linalg.norm(coords, axis=-1, keepdims=True)  # (N,1)
        return activations
            
    return None