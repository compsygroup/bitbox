from bitbox.face_backend import FaceProcessor3DI as FP
import numpy as np
from bitbox.utilities.visualization import plot

input_file = 'elaine.mp4'
output_dir = 'output'


# define a face processor
processor = FP(runtime='bitbox:latest')

# set input and output
processor.io(input_file=input_file, output_dir=output_dir)


rects = processor.detect_faces() # detect faces
lands = processor.detect_landmarks() # detect landmarks

plot(rects, video_path=input_file, output_dir=output_dir,overlay=lands) # visualize rectangles with optional landmarks
plot(lands, video_path=input_file, output_dir=output_dir) # visualize landmarks 




