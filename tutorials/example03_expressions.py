from bitbox.face_backend import FaceProcessor3DI
from bitbox.signal_processing import peak_detection
from bitbox.expressions import expressivity, asymmetry, diversity

input_file = '/mnt/isilon/schultz_lab/cluster/isilon_usr/compsy/3DI_input/cass_interested1/ACES10290_participant.mp4'
output_dir = '/mnt/isilon/schultz_lab/cluster/isilon_usr/compsy/3DI_output/cass_interested1_2025'

# define a face processor
processor = FaceProcessor3DI(runtime='bitbox:cuda12', camera_model=40)

# set input and output
processor.io(input_file=input_file, output_dir=output_dir)

# run the processor
rect, land, exp_global, pose, land_can, exp_local = processor.run_all(normalize=True)

# we will use the localized expressions for the rest of the tutorial

#%% Task 1: Peak Detection
# # Detect peaks and visualize them in one of the expression bases

# # get local expressions as a numpy array
# data = exp_global['data'].values

# # # select the expression bases we are intrested in
# expression = data[:, 7]

# # # detect peaks
# peaks = peak_detection(expression, scales=2, fps=30, aggregate=True, smooth=True, visualize=True)

#%% Task 2: Overall expressivity
# Quantify the overall expressivity (and its stats) of the face
# expressivity_stats = expressivity(exp_global, axis=0, use_negatives=0, scales=0, robust=True, fps=30)

#%% Task 3: Asymmetry of the facial expressions
asymmetry_scores = asymmetry(land_can)

#%% Task 4: Diversity of the facial expressions
# diversity_scores = diversity(exp_local)