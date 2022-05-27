# !/bin/bash
TORCH_ROOT=${1-"/opt/conda/lib/python3.8/site-packages/torch"}
TORCH_INCPATH="-I$TORCH_ROOT/include/torch/csrc/api/include -I$TORCH_ROOT/include"
TORCH_LIBPATH=$TORCH_ROOT/lib
CV_CUDA_INCPATH="-I../cv-cuda/include"
CV_CUDA_LIBPATH="../cv-cuda/build/lib"

./configure --enable-nonfree --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include \
--extra-ldflags=-L/usr/local/cuda/lib64 --disable-static --enable-shared --enable-opengl \
--enable-libtensorrt --enable-libopencv --enable-libcv_cuda \
--extra-ldflags=-L$TORCH_LIBPATH \
--extra-cflags="$TORCH_INCPATH" \
--extra-cflags=$CV_CUDA_INCPATH \
--extra-ldflags=-L$CV_CUDA_LIBPATH \
--nvccflags="-gencode arch=compute_75,code=sm_75 -lineinfo -Xcompiler -fPIC -I./ $TORCH_INCPATH" \
# --disable-stripping