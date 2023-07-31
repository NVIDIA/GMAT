[![License](https://img.shields.io/badge/License-MIT-yellogreen.svg)](https://opensource.org/licenses/Apache-2.0)

![Platform](https://img.shields.io/badge/Platform-linux--64-gray)

GPU toolkit on Multimedia, AI and Transcoding (GMAT)
=============================================
GMAT is a collection of tools for accelerating video/image processing and transcoding using GPUs in datacenter.

### Upgrade ffmpeg-gpu-demo to GMAT
For those who have known this repo as ffmpeg-gpu-demo, you can still find the demo pipelines in the [ffmpeg-gpu-demo branch](https://github.com/NVIDIA/FFmpeg-GPU-Demo/tree/ffmpeg-gpu-demo). GMAT is developed based on ffmpeg-gpu-demo, and features like tensorrt filter are kept in GMAT. We have been listening and gathering feedback from the industry during the past year, and developed tools that users told us useful but missing in GPU video processing. We organized these tools together and hence the new GMAT, hope you can find what you need here : )

## Features
* [ffmpeg-gpu](): GPU-enhanced ffmpeg
    * GPU filters: crop, rotate, flip, smooth, tensorrt
    * libgpuscale: GPU accelerated libswscale, providing rgb<->yuv conversion and scaling on GPU
    * CUDA Runtime API support
* [MeTrans SDK](): GPU transcoding toolkit
    * GPU codec tools: Programs to access and benchmark nvdec/nvenc/nvjpeg
    * Smart decoding: Decode video frames at uniform intervals, or decode frames with scene cut detection.
    * HEIF codec: HEIF image encoding/decoding accelerated by nvenc/nvdec

It should be noted that GMAT does not aim to provide a complete set of APIs for GPU video processing, there are a lot of great libraries/SDKs doing that already. Instead, our target is to solve the missing puzzle pieces. It's intented to use GMAT along with other libraries/SDKs you have been using in your current pipeline or solution. You can carve out whatever you need and integrate it into your project.

### Feedbacks are welcome!
If you want to do something using GPU but can't find it in GMAT or elsewhere, you are welcome to submit an issue or even better, a PR, we would be happy to look into it.

## Getting Started
Currently, ffmpeg-gpu and MeTrans need to be compiled separately.

### Compile ffmpeg-gpu
ffmpeg-gpu has the following dependencies:
- CUDA Toolkit >= 11.0
- TensorRT >= 8.2 (Optional)
- CV-CUDA >= 0.3.0 (Optional)

We strongly recommend you to start with the [NGC containers](https://catalog.ngc.nvidia.com/containers). If you would like to use the tensorrt filter, choose the [tensorrt image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt); If you do not care for the tensorrt filter, the [CUDA image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda) will do. Note that for CUDA images, tags including `devel` are required since we need NVCC and CUDA header files, which are not included in `base` or `runtime` images.

```Bash
# use CUDA image
docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -it --rm -v $(pwd):/gmat nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu20.04 bash

# use TensorRT image
docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -it --rm -v $(pwd):/gmat nvcr.io/nvidia/tensorrt:23.07-py3 bash
```

Once inside the container, you can install CV-CUDA. Please refer to [CV-CUDA's GitHub repo](https://github.com/CVCUDA/CV-CUDA/tree/v0.3.0-beta) for more details on installation.
```Bash
wget https://github.com/CVCUDA/CV-CUDA/releases/download/v0.3.1-beta/nvcv-lib-0.3.1_beta-cuda12-x86_64-linux.deb
wget https://github.com/CVCUDA/CV-CUDA/releases/download/v0.3.1-beta/nvcv-dev-0.3.1_beta-cuda12-x86_64-linux.deb

sudo apt install ./nvcv-lib-0.3.1_beta-cuda12-x86_64-linux.deb ./nvcv-dev-0.3.1_beta-cuda12-x86_64-linux.deb
```

Like compiling the original ffmpeg, we need to configure ffmpeg before running make.
```Bash
cd gmat/ffmpeg-gpu
./configure --disable-ptx-compression --enable-cvcuda --enable-libtensorrt --extra-cflags=-I/opt/nvidia/cvcuda0/include/ --disable-static --enable-shared --enable-nonfree --enable-cuda-nvcc --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --nvccflags='-gencode arch=compute_86,code=sm_86 -I./' --extra-libs=-lstdc++
make -j10
```

The TensorRT filter is enabled by the `--enable-libtensorrt` option. Remember to change the `--nvccflags` option to match your GPU arch. E.g. `arch=compute_86,code=sm_86` is for Ampere GPUs except A100. If you are using L4 (Ada), you should change it to `arch=compute_89,code=sm_89`. FFmpeg does not support PTX JIT at this moment, CUDA will report `no kernel image available` if you don't get the arch correct.

### Compile MeTrans SDK
MeTranshas the following dependencies:
- CUDA Toolkit
- ffmpeg-gpu
- HEIF (Optional)

(Optional) If you need HEIF codec, you need to install [HEIF reader/writer](https://github.com/nokiatech/heif) before compiling MeTrans:
```Bash
git clone https://github.com/nokiatech/heif.git
cd heif/build
cmake ../srcs
make -j10
sudo mkdir /usr/local/include/heif
sudo cp -r ../srcs/api/* /usr/local/include/heif
sudo cp ./lib/*.so /usr/local/lib
```

MeTrans does not require ffmpeg-gpu to be built with cvcuda or tensorrt, so we can use the CUDA image:
```Bash
docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -it --rm -v $(pwd):/gmat nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu20.04 bash

# build and install ffmpeg-gpu
cd gmat/ffmpeg-gpu
./configure --disable-ptx-compression --disable-static --enable-shared --enable-nonfree --enable-cuda-nvcc --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --nvccflags='-gencode arch=compute_86,code=sm_86 -I./' --extra-libs=-lstdc++
make -j10
sudo make install

cd ../metrans
make -j10
# make all_but_gl -j10
```
The compiled binaries are located in `metrans/build`.

If you need HEIF codec, run `make all_but_gl -j10` instead of `make -j10`. If you also need the `AppNvDecGL` sample which demonstrates how to display decoded video frames in OpenGL using CUDA OpenGL interoperation, run `make all -j10`. Note that `AppNvDecGL` requires OpenGL to be setup properly, it is recommended to run the sample on a local machine, as setup display on a remote environment can be tricky.

## License

FFmpeg GPU Demo is under MIT license, check out the [LICENSE](LICENSE.md) for details.