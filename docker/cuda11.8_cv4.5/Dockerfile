# Combined Bitbox + OpenFace runtime.

FROM compsygroup/bitbox:20251017-cuda11.8.0-cudnn8-opencv4.5.5-ubuntu20.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# 3DI and 3DI-lite require the BFM face model file
COPY 01_MorphableModel.mat /app/3DI/models/raw/
COPY 01_MorphableModel.mat /app/3DI_lite/data/raw/

RUN set -eux; \
    cd /app/3DI/models; \
    python3 prepare_BFM.py; \
    cd /app/3DI_lite; \
    python3 prepare_BFM.py; \
    cp -rf /app/3DI_lite/data/BFMmm-23660 /app/3DI_lite/models/MMs/.

ENV DEBIAN_FRONTEND=noninteractive \
    OPENFACE_DIR=/app/OpenFace \
    OPENFACE_CONFIG_DIR=/usr/local/etc/OpenFace \
    PATH=/app/OpenFace/build/bin:${PATH}

ARG DLIB_VERSION=19.24.6
ARG OPENFACE_REF=OpenFace_2.2.0

RUN set -eux; \
    cd /tmp; \
    wget -q "https://github.com/davisking/dlib/archive/refs/tags/v${DLIB_VERSION}.tar.gz" -O dlib.tar.gz; \
    tar -xzf dlib.tar.gz; \
    rm dlib.tar.gz; \
    mkdir -p "dlib-${DLIB_VERSION}/build"; \
    cd "dlib-${DLIB_VERSION}/build"; \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DDLIB_USE_CUDA=OFF; \
    make -j"$(nproc)"; \
    make install; \
    ldconfig; \
    rm -rf "/tmp/dlib-${DLIB_VERSION}"

WORKDIR /app

RUN set -eux; \
    git clone --branch "${OPENFACE_REF}" --depth 1 --recursive https://github.com/TadasBaltrusaitis/OpenFace.git "${OPENFACE_DIR}"; \
    cd "${OPENFACE_DIR}"; \
    bash ./download_models.sh; \
    mkdir -p build; \
    cd build; \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=/usr/local/lib/cmake/opencv4; \
    make -j"$(nproc)"; \
    make install; \
    ldconfig

RUN set -eux; \
    mkdir -p "${OPENFACE_CONFIG_DIR}/classifiers" "${OPENFACE_DIR}/classifiers"; \
    cp /usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml "${OPENFACE_CONFIG_DIR}/classifiers/"; \
    ln -sf "${OPENFACE_CONFIG_DIR}/classifiers/haarcascade_frontalface_alt.xml" "${OPENFACE_DIR}/classifiers/haarcascade_frontalface_alt.xml"

WORKDIR /app

CMD ["bash"]
