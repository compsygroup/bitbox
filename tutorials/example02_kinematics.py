from bitbox.face_backend import FaceProcessor3DI

input_file = 'data/elaine.mp4'
output_dir = 'output'

# define a face processor
processor = FaceProcessor3DI()

# set input and output
processor.io(input_file=input_file, output_dir=output_dir)

# run the processor
rect, land, exp_global, pose, land_can, exp_local = processor.run_all(normalize=True)

