from bitbox.face_backend import FaceProcessor3DIlite as FP

input_file = 'data/elaine.mp4'
output_dir = 'output'

# define a face processor run on a remote server
print("Don't forget to uncomment file checks in methods!!!")
print("")
server = {
    "host": "reslncarts01.research.chop.edu",
    "port": 1160
}

processor = FP(server=server)

# set input and output
processor.io(input_file=input_file, output_dir=output_dir)

# detect faces
rects = processor.detect_faces()

# detect landmarks
lands = processor.detect_landmarks()

# compute global expressions
exp_global, pose, lands_can = processor.fit()
print(pose)

# # compute localized expressions
# exp_local = processor.localized_expressions()