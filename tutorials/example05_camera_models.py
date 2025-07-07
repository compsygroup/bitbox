from bitbox.face_backend import FaceProcessor3DI as FP

input_file = 'data/IS001_parent.mp4'
output_dir = 'output'

# define a face processor
# processor = FP(camera_model=40) #R2R
# processor = FP(camera_model='models/cameras/TreeCam_logitech.txt') #In-person 30 FPS 
processor = FP(camera_model='models/cameras/TreeCam_1041a.txt') #In-person 60 FPS

# set input and output
processor.io(input_file=input_file, output_dir=output_dir)

# preprocess the video
processor.preprocess(undistort=True)

# detect faces
rects = processor.detect_faces()

# detect landmarks
lands = processor.detect_landmarks()

# compute global expressions
exp_global, pose, lands_can = processor.fit()

# compute localized expressions
#exp_local = processor.localized_expressions()