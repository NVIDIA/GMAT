ARG TAG=latest
FROM nvcr.io/nvidia/pytorch:${TAG}
# FROM busybox:latest
ARG PREFIX=/usr/local

RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libglvnd-dev \
        libgl1-mesa-dev \
        libegl1-mesa-dev \
        libgles2-mesa-dev && \
    echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}' >> /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-gpu-1.8.1.tgz \
    && tar -zxf onnxruntime-linux-x64-gpu-1.8.1.tgz \
    && cd onnxruntime-linux-x64-gpu-1.8.1 \
    && install -m 0755 -d ${PREFIX}/include/onnxruntime \
    && install -m 0644 include/*.h ${PREFIX}/include/onnxruntime \
    && install -m 0644 lib/* ${PREFIX}/lib/ \
    && cd .. \
    && rm -r onnxruntime*

RUN apt-get update \
    && wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip \
    && unzip eigen-3.4.0 \
    && cd eigen-3.4.0 \
    && cp -r Eigen/ /usr/local/include \
    && cd .. \
    && rm -r eigen* \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y libassimp-dev assimp-utils libglm-dev libsdl2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/FFmpeg/nv-codec-headers.git \
    && cd nv-codec-headers \
    && make install \
    && cd .. \
    && rm -r nv-codec-headers

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib