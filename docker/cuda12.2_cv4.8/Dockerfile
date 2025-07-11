# Base image including 3DI but not the face model file
FROM compsygroup/bitbox:20250507-cuda12.2.2-cudnn8-opencv4.8.1-ubuntu20.04

# Add the model file 
COPY 01_MorphableModel.mat /app/3DI/models/raw/
COPY 01_MorphableModel.mat /app/3DI_lite/data/raw/

# Prepare models to be used by 3DI and 3DI-lite
WORKDIR /app/3DI/models
RUN python3 prepare_BFM.py

WORKDIR /app/3DI_lite
RUN python3 prepare_BFM.py

# make 3DI-lite standalone (not dependent on 3DI)
WORKDIR /app/3DI_lite
RUN cp /app/3DI/video_detect_* .
RUN cp /app/3DI/video_undistort .
RUN cp /app/3DI/models/deploy.prototxt models/
RUN cp /app/3DI/models/res10_300x300_ssd_iter_140000_fp16.caffemodel models/
RUN cp /app/3DI/models/haarcascade_frontalface_default.xml models/
RUN cp /app/3DI/models/mmod_human_face_detector.dat models/
RUN cp /app/3DI/models/opencv_face_detector* models/
RUN cp /app/3DI/models/tf2onnx_onnx_model.onnx models/
RUN cp -r /app/3DI/models/landmark_models models/
RUN cp -r /app/3DI/models/dat_files models/
RUN cp -r /app/3DI/configs /app/3DI_lite/
RUN cp -r /app/3DI/models/MMs/ models/

WORKDIR /app