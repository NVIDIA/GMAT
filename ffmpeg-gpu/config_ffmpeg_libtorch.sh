# !/bin/bash
TORCH_ROOT=${1-"/opt/conda/lib/python3.8/site-packages/torch"}
TORCH_INCPATH="-I$TORCH_ROOT/include/torch/csrc/api/include -I$TORCH_ROOT/include"
TORCH_LIBPATH=$TORCH_ROOT/lib

./configure --enable-nonfree --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include \
--extra-ldflags=-L/usr/local/cuda/lib64 --disable-static --enable-shared --enable-opengl \
--enable-libtensorrt --enable-libopencv \
--extra-ldflags=-L$TORCH_LIBPATH \
--extra-cflags="$TORCH_INCPATH" \
--nvccflags="-gencode arch=compute_75,code=sm_75 -lineinfo -I./" \
# --disable-stripping