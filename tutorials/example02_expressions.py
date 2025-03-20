from bitbox.face_backend import FaceProcessor3DI
from bitbox.signal_processing import peak_detection
from bitbox.expressions import expressivity, asymmetry, diversity

input_file = 'data/elaine.mp4'
output_dir = 'output'

# define a face processor
processor = FaceProcessor3DI()

# set input and output
processor.io(input_file=input_file, output_dir=output_dir)

# run the processor
rect, land, exp_global, pose, land_can, exp_local = processor.run_all(normalize=True)

# we will use the localized expressions for the rest of the tutorial

#%% Task 1: Peak Detection
# # Detect peaks and visualize them in one of the expression bases

# # get local expressions as a numpy array
# data = exp_local['data'].values

# # select the expression bases we are intrested in
# expression = data[:, 13]

# # detect peaks
# peaks = peak_detection(expression, num_scales=6, fps=30, smooth=True, visualize=True)

#%% Task 2: Overall expressivity
# Quantify the overall expressivity (and its stats) of the face
expressivity_stats = expressivity(exp_local, use_negatives=0, num_scales=6, robust=True, fps=30)

#%% Task 3: Asymmetry of the facial expressions
asymmetry_scores = asymmetry(land_can)

#%% Task 4: Diversity of the facial expressions
diversity_scores = diversity(exp_local)