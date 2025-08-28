from bitbox.face_backend import FaceProcessor3DIlite as FP

input_file = '/home/nairg1/Documents/bitbox/bitbox/tutorials/data/elaine.mp4'
output_dir = '/home/nairg1/Documents/bitbox/bitbox/tutorials/output'


# define a face processor
processor = FP(runtime='bitbox:latest')

# set input and output
processor.io(input_file=input_file, output_dir=output_dir)

rect, land, exp_global, pose, land_can, exp_local = processor.run_all(normalize=True)

processor.plot(exp_global,overlay=[rect,land],video=True) # plot landmarks with rectangle overlay and pose
# processor.plot(rect,overlay=land,video=True) # plot rectangle with landmarks overlay and pose
# processor.plot(pose,overlay=[land,rect],video=True) # plot canonical landmarks with rectangle and landmarks overlay and pose
# processor.plot(exp_local,overlay=[rect,land],video=True)