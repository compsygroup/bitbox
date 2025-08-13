from bitbox.face_backend import FaceProcessor3DI
from bitbox.biomechanics import motion_kinematics, relative_motion, motion_smoothness

input_file = 'data/elaine.mp4'
output_dir = 'output'

# define a face processor
processor = FaceProcessor3DI('bitbox:cuda12')

# set input and output
processor.io(input_file=input_file, output_dir=output_dir)

# run the processor
rects, lands, exp_global, pose, lands_can, exp_local = processor.run_all(normalize=True)

# compute motion kinematics
mrange, path, speed, accelaration = motion_kinematics(pose)
print(mrange, path, speed, accelaration)

# compute motion smoothness
jerk, ldj = motion_smoothness(pose)
print(jerk, ldj)

# compute relative motion stats
# mind, avgd, stdd, maxd = relative_motion(rects, reference=[7])