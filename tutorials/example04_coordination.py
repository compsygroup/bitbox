from bitbox.face_backend import FaceProcessor3DI as FP
from bitbox.social import coordination

import numpy as np

input_file = 'data/elaine.mp4'
output_dir = 'output'

# define a face processor
processor = FP(runtime='bitbox:latest')

# set input and output
processor.io(input_file=input_file, output_dir=output_dir)

# run the processor
rect, land, exp_glob, pose, land_can, exp_loc = processor.run_all(normalize=True)

# calculate intra-person coordination
corr_mean, corr_std, corr_lag = coordination(exp_glob, exp_glob, width=1.1, fps=30)
