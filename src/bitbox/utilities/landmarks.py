import numpy as np

def landmarks_left_right(schema='ibug51'):
    idx = {}
    
    # ordering of right is to mirror left
    if schema == 'ibug51':
        idx = {'lb': np.array([0,1,2,3,4]),
               'rb': np.array([9,8,7,6,5]),
               'lno': np.array([14,15]), # excluding landmarks shared by left/right [10-13,16]
               'rno': np.array([18,17]),
               'le': np.array([19,20,21,22,23,24]),
               're': np.array([28,27,26,25,30,29]),
               'lm': np.array([31,32,33,43,44,50,42,41]), # excluding shared ones [34,45,49,40]   #ul: list(range(31, 37))+list(range(43, 47))
               'rm': np.array([37,36,35,47,46,48,38,39]) #ll: list(range(37, 43))+list(range(47, 51))
              } 
    # elif schema == 'ibug51_mirrored':
    #     idx = {'lb': np.array([9,8,7,6,5]),
    #            'rb': np.array([4,3,2,1,0]),
    #            'no': np.array([10, 11, 12, 13, 18, 17, 16, 15, 14]),
    #            'le': np.array([28, 27, 26, 25, 30, 29]),
    #            're': np.array([22, 21, 20, 19, 24, 23]),
    #            'ul': np.array([37, 36, 35, 34, 33, 32, 47, 46, 45, 44]),
    #            'll': np.array([31, 42, 41, 40, 39, 38, 43, 50, 49, 48])}
    else:
        raise ValueError(f"Landmark schema {schema} not recognized")
    
    return idx