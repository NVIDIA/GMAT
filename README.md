FFmpeg GPU Demo
==========================
This demo shows a ffmpeg-based full-GPU rendering and inference pipeline. The code is based on ffmpeg release 4.4. The project is composed of several new filters in FFmpeg, a clip rendered by these filters is demonstrated in Fig 1.

![Fig 1](doc/images/rio_360_mask_10s.gif)|
|:--:|
| *Fig 1 Clip rendered by FFmpeg GPU Demo* |

## Features
* [Pose filter](doc/Pose_Filter.md) (putting a mask on everyone's face)
* [TensorRT filter](doc/Tensorrt_Filter.md)

We are still actively developing this project, and we will continuously update this list. Please refer to the documents for details of each feature, including how to build and run them.

It should be noted that __the purpose of this project is demonstration__. As the name *FFmpeg GPU __Demo__* indicates, we would like to show you how to build such a pipeline, rather than building a product or turn-key solution.

## Getting started
The project has complex dependencies, we offer a Dockerfile to quickly deploy the environment. We assumed that you have installed the [NVIDIA GPU driver](https://www.nvidia.com/download/index.aspx) and [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).
You can enable all the features following the commands below:
```bash
git clone --recursive https://github.com/NVIDIA/FFmpeg-GPU-Demo.git
docker pull nvcr.io/nvidia/pytorch:22.03-py3
cd ffmpeg-gpu-demo
docker build -t ffmpeg-gpu-demo:22.03-py3 --build-arg TAG=22.03-py3 .
docker run --gpus all -it --rm -e NVIDIA_DRIVER_CAPABILITIES=all -v $(pwd):/workspace/ffmpeg-gpu-demo ffmpeg-gpu-demo:22.03-py3
cd ffmpeg-gpu-demo/ffmpeg-gpu/
bash config_ffmpeg_libtorch.sh
make -j10
make install
```
If you just want a specific feature, please refer to the feature's doc for simplified building. We will provide a complete docker image in the future, so that you can pull & run directly.

Our project provides a AI+graphics pipeline in FFmpeg, as shown in the GIF above. Sample command:
```bash
ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i <input> -vf scale_npp=1280:720,pose="./img2pose_v1_ft_300w_lp_static_nopost.onnx":8,format_cuda=rgbpf32,tensorrt="./onnx_models/ESRGAN_x4_dynamic.onnx",format_cuda=nv12 -c:v h264_nvenc <output>
```
Please refer to the [pose filter doc](doc/Pose_Filter.md) for how to run the pipeline.

## Additional Resources
If you are interested in the tech details of our project, check out our GTC 2022 talk: [AI-based Cloud Rendering: Full-GPU Pipeline in FFmpeg](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41609/)

FFmpeg GPU Demo is first developed by NVIDIA DevTech & SA team, and currently maintained by Xiaowei Wang. Authors include Yiming Liu, Jinzhong(Thor) Wu and Xiaowei Wang.

FFmpeg GPU Demo is under MIT license, check out the [LICENSE](LICENSE.md) for details.
