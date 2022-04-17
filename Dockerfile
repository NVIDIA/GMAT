ARG TAG=latest
FROM nvcr.io/nvidia/pytorch:${TAG}
# FROM busybox:latest
ARG PREFIX=/usr/local

RUN apt-get update \
    && wget https://github.com/microsoft/onnxruntime/releases/download/v1.11.0/onnxruntime-linux-x64-gpu-1.11.0.tgz \
    && tar -zxf onnxruntime-linux-x64-gpu-1.11.0.tgz \
    && cd onnxruntime-linux-x64-gpu-1.11.0 \
    && install -m 0755 -d ${PREFIX}/include/onnxruntime \
    && install -m 0644 include/*.h ${PREFIX}/include/onnxruntime \
    && install -m 0644 lib/* ${PREFIX}/lib/ \
    && cd .. \
    && rm -r onnxruntime*

RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip \
    && unzip eigen-3.4.0 \
    && cd eigen-3.4.0 \
    && cp -r Eigen/ /usr/local/include \
    && cd .. \
    && rm -r eigen* \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y libassimp-dev assimp-utils libglm-dev libsdl2-dev libglu1-mesa libglu1-mesa-dev

RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git \
    && cd nv-codec-headers \
    && make install \
    && cd .. \
    && rm -r nv-codec-headers

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib