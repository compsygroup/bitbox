# Base image including 3DI but not the face model file
FROM compsygroup/bitbox:20250507-cuda11.8.0-cudnn8-opencv4.5.5-ubuntu20.04

# Add the model file 
COPY 01_MorphableModel.mat /app/3DI/models/raw/
COPY 01_MorphableModel.mat /app/3DI_lite/data/raw/

# Prepare models to be used by 3DI and 3DI-lite
WORKDIR /app/3DI/models
RUN python3 prepare_BFM.py

WORKDIR /app/3DI_lite
RUN python3 prepare_BFM.py

WORKDIR /app