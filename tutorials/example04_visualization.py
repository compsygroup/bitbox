from bitbox.face_backend import FaceProcessor3DI as FP

input_file = 'data/elaine.mp4'
output_dir = 'output'

# define a face processor
processor = FP(runtime='bitbox:latest')

# set input and output
processor.io(input_file=input_file, output_dir=output_dir)

# run the processor
rect, land, exp_global, pose, land_can, exp_local = processor.run_all(normalize=True)

# NOTE: The resulting HTML file will only include the final plot
# comment out the ones you don't want to test

# visualize landmarks at random poses
processor.plot(land, pose=pose)

# visualize landmarks with rectangles overlayed
processor.plot(land, overlay=[rect], video=True) 

# visualize expressions
processor.plot(exp_global, overlay=[rect, land], video=True) 
