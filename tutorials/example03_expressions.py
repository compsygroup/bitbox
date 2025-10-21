from bitbox.face_backend import FaceProcessor3DI as FP
from bitbox.expressions import expressivity, asymmetry, diversity

input_file = 'data/elaine.mp4'
output_dir = 'output'

# define a face processor
processor = FP(runtime='bitbox:latest')

# set input and output
processor.io(input_file=input_file, output_dir=output_dir)

# run the processor
rect, land, exp_global, pose, land_can, exp_local = processor.run_all(normalize=True)

# we will use the global expressions for the rest of the tutorial

# Overall expressivity
expressivity_stats = expressivity(exp_global, scales=6, aggregate=False, robust=True, fps=30)

# Asymmetry of the facial expressions
asymmetry_scores = asymmetry(land_can)

# Diversity of the facial expressions
diversity_scores = diversity(exp_global, magnitude=True, scales=6, aggregate=False, robust=True, fps=30)