import pandas as pd
import numpy as np
from ..utilities import recover_full_rodrigues, rodrigues_to_euler

def read_rectangles(file):
    ext = file.split(".")[-1]
    _data = np.loadtxt(file)
    data = pd.DataFrame(_data, columns=['x', 'y', 'w', 'h'])  

    dict = {
        'backend': ext,
        'frame count': data.shape[0],
        'type': 'rectangle',
        'format': 'for each frame (rows) [x, y, w, h] values of the detected rectangles',
        'dimension': 2,
        'data': data
    }
    
    return dict


def read_pose(file):
    ext = file.split(".")[-1]
    _data = np.loadtxt(file)
    # first three are translation, ignore middle three (Euclid-Rodrigues parameterization), last three are angles in radians
    # we change the order to have pitch (Rx), yaw (Ry), roll (Rz)
    _data = _data[:, [0, 1, 2, 7, 6, 8]]
    data = pd.DataFrame(_data, columns=['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz'])  

    dict = {
        'backend': ext,
        'frame count': data.shape[0],
        'type': 'pose',
        'format': 'for each frame (rows) [Tx, Ty, Tz, Rx, Ry, Rz] values of the detected face pose',
        'dimension': 3,
        'data': data
    }

    return dict


def read_pose_lite(file, landmark_file):
    ext = file.split(".")[-1]
    rodrigues = np.loadtxt(file)
    landmarks_2d = np.loadtxt(landmark_file)
    # only rotations are meaningful in 3DI-lite
    rodrigues = rodrigues[:, 3:]
    
    # recover full Rodrigues in original image coords
    rodrigues[:,0] = -rodrigues[:,0]    
    
    rodrigues_corr = recover_full_rodrigues(landmarks_2d, rodrigues)
    # convert to Euler angles
    euler = rodrigues_to_euler(rodrigues_corr)

    # we change the order to have pitch (Rx), yaw (Ry), roll (Rz)
    euler = euler[:, [1, 0, 2]]
    
    # filter out implausible angles
    # pitch: +-45 [−0.785,0.785]
    # yaw: +-90 [−1.57,1.57]
    # roll: +-45 [−0.785,0.785]
    euler[np.abs(euler[:,0])>0.785, 0] = np.nan
    euler[np.abs(euler[:,1])>1.57, 1] = np.nan
    euler[np.abs(euler[:,2])>0.785, 2] = np.nan
    
    data = pd.DataFrame(euler, columns=['Rx', 'Ry', 'Rz'])

    dict = {
        'backend': ext,
        'frame count': data.shape[0],
        'type': 'pose',
        'format': 'for each frame (rows) [Rx, Ry, Rz] values of the detected face pose',
        'dimension': 3,
        'data': data
    }

    return dict
    
    
def read_landmarks(file):
    ext = file.split(".")[-1]
    _data = np.loadtxt(file)
    
    num_landmarks = _data.shape[1] // 2
    if num_landmarks == 51:
        schema = 'ibug51'
        column_list = [f'{c}{i}' for i in range(num_landmarks) for c in 'xy']
    else:
        raise ValueError(f"Unrecognized landmark schema.")
    
    data = pd.DataFrame(_data, columns=column_list)

    dict = {
        'backend': ext,
        'frame count': data.shape[0],
        'type': 'landmark',
        'format': 'for each frame (rows) [x, y] values of the detected landmarks',
        'schema': schema,
        'dimension': 2,
        'data': data
    }
    
    return dict


def read_canonical_landmarks(file):
    ext = file.split(".")[-1]
    _data = np.loadtxt(file)
    
    num_landmarks = _data.shape[1] // 3
    if num_landmarks == 51:
        schema = 'ibug51'
        column_list = [f'{c}{i}' for i in range(num_landmarks) for c in 'xyz']
    else:
        raise ValueError(f"Unrecognized landmark schema.")
    
    data = pd.DataFrame(_data, columns=column_list)

    dict = {
        'backend': ext,
        'frame count': data.shape[0],
        'type': 'landmark-can',
        'format': 'for each frame (rows) [x, y, z] values of the canonicalized landmarks',
        'schema': schema,
        'dimension': 3,
        'data': data
    }
    
    return dict

    
def read_expression(file):
    ext = file.split(".")[-1]
    _data = np.loadtxt(file)
    num_coeff = _data.shape[1]
    
    if num_coeff == 79:
        schema = '3DI-G79'
        column_list = ['GE' + str(i) for i in range(num_coeff)]
        format = 'for each frame (rows) [GE0, GE1, ..., GE78] values corresponding to global expression coefficients'
    elif num_coeff == 32:
        schema = '3DI-L32'
        le_idx = {'lb': (0,4),
                  'rb': (4,8),
                  'no': (8,12),
                  'le': (12,16),
                  're': (16,20),
                  'ul': (20,25),
                  'll': (25,32)}
        column_list = []
        for key, value in le_idx.items():
            for i, v in enumerate(range(value[0], value[1])):
                column_list.append(f'{key}{i}')
        format = 'for each frame (rows) [lb0, lb1, ...] values corresponding to localized expression coefficients'
    elif num_coeff == 50:
        schema = '3DIlite-L50'
        le_idx = {'lb': (0,5),
                  'rb': (5,10),
                  'no': (10,13),
                  'le': (13,20),
                  're': (20,27),
                  'mo': (27,50)}
        column_list = []
        for key, value in le_idx.items():
            for i, v in enumerate(range(value[0], value[1])):
                column_list.append(f'{key}{i}')
        format = 'for each frame (rows) [lb0, lb1, ...] values corresponding to localized expression coefficients'
    else:
        raise ValueError(f"Unrecognized expression schema.")
      
    data = pd.DataFrame(_data, columns=column_list)  

    dict = {
        'backend': ext,
        'frame count': data.shape[0],
        'type': 'expression',
        'format': format,
        'schema': schema,
        'dimension': 3,
        'data': data
    }
    
    return dict
