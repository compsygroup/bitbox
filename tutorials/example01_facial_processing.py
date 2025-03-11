from bitbox.face_backend import FaceProcessor3DI
import os

# Please make sure you give the correct full (not relative) path
DIR = '/home/tuncb/Works/code/compsy/bitbox/tutorials'
input_file = os.path.join(DIR, 'data/elaine.mp4') 
output_dir = os.path.join(DIR, 'output')

# define a face processor
processor = FaceProcessor3DI(lite=False)

# set input and output
processor.io(input_file=input_file, output_dir=output_dir)

# detect faces
rects = processor.detect_faces()

# detect landmarks
lands = processor.detect_landmarks()

# compute global expressions
exp_global, pose, lands_can = processor.fit()

# compute localized expressions
exp_local = processor.localized_expressions()