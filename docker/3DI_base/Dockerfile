# Base image with CUDA 12.0 support
FROM compsygroup/3di:cuda12.0.0-cudnn8-devel-ubuntu22.04-base

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y --fix-broken\
    python3 python3-pip python3-dev python-is-python3 \
    cmake build-essential git libopencv-dev wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    cvxpy scikit-learn==1.3.2 matplotlib opencv-python==4.6.0.66 \
    scipy numpy

# Install packages for openCV
RUN apt-get update && apt-get install -y --fix-broken \
    libjpeg-dev libpng-dev libtiff-dev libwebp-dev libopenexr-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libxine2-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libatlas-base-dev liblapacke-dev gfortran libhdf5-dev \
    libprotobuf-dev protobuf-compiler libgoogle-glog-dev \
    libgflags-dev libeigen3-dev libceres-dev libboost-all-dev \
    libopenblas-dev liblapack-dev liblapacke-dev libopenblas-dev \
    cmake libtbb2 g++ unzip ffmpeg libgtk2.0-dev libavutil-dev libtbb-dev \
    libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev libgtkglext1-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Compile OpenCV with CUDA support
WORKDIR /opencv
RUN git clone --branch 4.7.0 --depth 1 https://github.com/opencv/opencv.git && \
    git clone --branch 4.7.0 --depth 1 https://github.com/opencv/opencv_contrib.git && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_INSTALL_PREFIX=/usr/local \
          -DBUILD_JPEG=ON \
          -DBUILD_WEBP=ON \
          -DBUILD_PNG=ON \
          -DBUILD_TIFF=ON \
          -DWITH_FFMPEG=ON \
          -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
          -D WITH_TBB=ON \
          -D ENABLE_FAST_MATH=1 \
          -D CUDA_FAST_MATH=1 \
          -D WITH_CUBLAS=1  \
          -D WITH_CUDA=ON \
          -D BUILD_opencv_cudacodec=OFF \
          -D WITH_CUDNN=ON \
          -D OPENCV_DNN_CUDA=ON \
          -D WITH_V4L=ON \
          -D WITH_QT=OFF \
          -D WITH_FFMPEG=ON \
          -D WITH_OPENGL=ON \
          -D WITH_GSTREAMER=ON \
          -D OPENCV_GENERATE_PKGCONFIG=ON \
          -D OPENCV_PC_FILE_NAME=opencv.pc \
          -D OPENCV_ENABLE_NONFREE=ON \
          -D INSTALL_PYTHON_EXAMPLES=OFF \
          -D INSTALL_C_EXAMPLES=OFF \
          -D BUILD_EXAMPLES=OFF \
          -D CMAKE_CXX_STANDARD=14 \
          ../opencv && \
    make -j$(nproc) && make install && ldconfig && \
    rm -rf /opencv

# Copy and compile the 3DI app
WORKDIR /app
RUN git clone https://github.com/Computational-Psychiatry/3DI.git .
RUN cd /app/build && chmod +x builder.sh && ./builder.sh
RUN cd /app/build/models && tar -xvzf lmodels.tar.gz
RUN cd /app/build/models/raw && wget https://raw.githubusercontent.com/Juyong/3DFace/master/Exp_Pca.bin
RUN cd /app/build/scripts && chmod +x compute_local_exp_coefficients.py learn_localised_bases.py produce_canonicalized_3Dlandmarks.py
RUN cd /app/build/scripts && chmod +x save_identity_and_shape.py total_variance_rec_pose.py total_variance_rec.py
RUN cd /app/build && chmod +x process_video.py multiframe_from_images.py

# Add the model file and prepare it to be used by 3DI
COPY 01_MorphableModel.mat /app/build/models/raw/
RUN cd /app/build/models && python3 prepare_BFM.py
