from bitbox.face_backend import FaceProcessor3DI
from bitbox.signal_processing import peak_detection
from bitbox.coordination import intra_person_coordination
from bitbox.expressions import expressivity

import numpy as np

input_file = 'data/elaine.mp4'
output_dir = 'output'

# define a face processor
processor = FaceProcessor3DI()

# set input and output
processor.io(input_file=input_file, output_dir=output_dir)

# run the processor
rect, land, exp_glob, pose, land_can, exp_loc = processor.run_all()

#%% Task 1: Intra-person Coordination
# Quantify coordination across facial expressions (of a single person)


# get local expressions as a numpy array
data = np.array(list(exp_loc['data'].values())).astype('float')

# detect peaks
peaks = peak_detection(data[:,32], num_scales=6, fps=30, smooth=True, visualize=True)

# calculate intra-person coordination
corr_mean, corr_lag, corr_std = intra_person_coordination(exp_loc, width=0.5, fps=30)

# calculate expressivity
expressivity_stats = expressivity(data, axis=0, use_negatives=0, num_scales=6, robust=True, fps=30)