# !/bin/bash
CONF_FLAGS="--enable-nonfree --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64"
# CONF_FLAGS+=" --disable-static --enable-shared --enable-opengl --enable-libtensorrt --enable-libopencv --enable-libcv_cuda"
CONF_FLAGS+=" --disable-stripping"
NVCC_FLAGS="-gencode arch=compute_86,code=sm_86 -lineinfo -Xcompiler -fPIC -I./"

if [ "$1" = "torch" ]
then 
    TORCH_ROOT=${2-"/opt/conda/lib/python3.8/site-packages/torch"}
    TORCH_INCPATH="-I$TORCH_ROOT/include/torch/csrc/api/include -I$TORCH_ROOT/include"
    TORCH_LIBPATH=$TORCH_ROOT/lib
    CONF_FLAGS+=" --enable-libtorch --extra-ldflags=-L$TORCH_LIBPATH --extra-cflags="$TORCH_INCPATH""
    NVCC_FLAGS+=" $TORCH_INCPATH"
fi

# echo "configure options: $CONF_FLAGS"
# echo "nvccflags: $NVCC_FLAGS"

./configure $CONF_FLAGS --nvccflags="$NVCC_FLAGS"

# ./configure --enable-nonfree --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include \
# --extra-ldflags=-L/usr/local/cuda/lib64 --disable-static --enable-shared --enable-opengl \
# --enable-libtensorrt --enable-libopencv --enable-libcv_cuda --enable-libtorch \
# --extra-ldflags=-L$TORCH_LIBPATH \
# --extra-cflags="$TORCH_INCPATH" \
# --nvccflags="-gencode arch=compute_75,code=sm_75 -lineinfo -Xcompiler -fPIC -I./ $TORCH_INCPATH" \
# --disable-stripping