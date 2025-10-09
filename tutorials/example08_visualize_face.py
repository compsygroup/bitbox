from bitbox.face_backend import FaceProcessor3DI as FP

input_file = 'data/elaine.mp4'
output_dir = 'output/elaine'


# define a face processor
processor = FP(runtime='bitbox:cuda12')

# set input and output
processor.io(input_file=input_file, output_dir=output_dir)

rect, land, exp_global, pose, land_can, exp_local = processor.run_all(normalize=True)

# processor.plot(land,overlay=rect,pose=pose) # plot landmarks with rectangle overlay and pose
# processor.plot(rect,overlay=land,pose=pose) # plot rectangle with landmarks overlay and pose
processor.plot(land_can,overlay=[rect,land], pose=pose) 




