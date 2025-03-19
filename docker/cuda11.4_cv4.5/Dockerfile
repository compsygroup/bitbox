# Base image including 3DI but not the face model file
FROM compsygroup/3di:0.3.0-cuda11.4.3-cudnn8-opencv4.5.5-ubuntu20.04

# Add the model file 
COPY 01_MorphableModel.mat /app/3DI/models/raw/
COPY 01_MorphableModel.mat /app/3DI_lite/data/raw/

WORKDIR /app/3DI/models
RUN tar -xvzf lmodels.tar.gz
WORKDIR /app/3DI/models/raw
RUN wget --no-check-certificate https://raw.githubusercontent.com/Juyong/3DFace/master/Exp_Pca.bin
RUN cp /app/3DI/models/raw/Exp_Pca.bin /app/3DI_lite/data/raw/.

# Prepare models to be used by 3DI and 3DI-lite
WORKDIR /app/3DI/models
RUN python3 prepare_BFM.py

WORKDIR /app/3DI_lite
RUN python3 prepare_BFM.py

# install insightface
WORKDIR /tmp
RUN git clone https://github.com/deepinsight/insightface.git
RUN cp -r /tmp/insightface/recognition/arcface_torch /app/3DI_lite/models/.

# copy trained models
RUN mkdir /app/3DI_lite/checkpoints
RUN mkdir /app/3DI_lite/models/checkpoints

WORKDIR /app/3DI_lite
RUN wget https://sariyanidi.com/dbox/3DIlite/backbone.pth -P ./checkpoints/
RUN wget https://sariyanidi.com/dbox/3DIlite/medium_model15.00combined_celeb_ytfacesresnet50139979True1e-05-2-BFMmm-23660UNL_STORED.pth -P ./checkpoints/
RUN wget https://sariyanidi.com/dbox/3DIlite/sep_modelv3SP15.00combined_celeb_ytfacesresnet501e-052True139979_V2.pth -P ./checkpoints/
RUN wget https://sariyanidi.com/dbox/3DIlite/resnet50-0676ba61.pth -P ./models/checkpoints/

WORKDIR /app