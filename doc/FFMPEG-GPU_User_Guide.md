# FFMPEG-GPU User Guide

ffmpeg-gpu is built on top of ffmpeg with several enhancements on GPU, so that users can access more GPU acceleration capabilities when using ffmpeg. The enhancements include:
- More GPU filters
- GPU accelerated libswscale
- CUDA Runtime API support

## GPU Filters
FFmpeg provides several GPU filters, such as scale_cuda, overlay_cuda and etc. However, there are still some common filters that are not accelerated by GPU in ffmpeg. In ffmpeg-gpu, we provide five addtional GPU filters:
- crop_nvcv
- rotate_nvcv
- flip_nvcv
- smooth_nvcv
- tensorrt

The first four filters are similar to the corresponding ffmpeg CPU filters, except that these filters only accept RGB inputs. A sample command to call these filters:

```Bash
ffmpeg -hwaccel cuda -i input.mp4 -vf format=rgb24,hwupload_cuda,crop_nvcv=640:480,flip_nvcv=0,smooth_nvcv=gaussian -c:v hevc_nvenc -preset p3 output.mp4
```

You can use the following command to check available options of each filter
```Bash
ffmpeg -h filter=smooth_nvcv # substitute smooth_nvcv with the filter you want to use
```

## GPU accelerated libswscale
libswscale is a powerful library in ffmpeg providing yuv/rgb scaling and yuv<->rgb conversion under different color spaces. libswscale can only run on CPU, we extend libswscale to GPU so that the conversions and scaling can be accelerated by GPU as well. We will refer to it as libgpuscale in the documents.

Calling libswscale on GPU is the same as calling it on CPU, except that you need to pass the `SWS_HWACCEL_CUDA` flag when creating `SwsContext`, and the input and output must be in __GPU memory__.

```C++
uint8_t *src_data[4], *dst_data[4];     // src and dst image pointer
int src_linesize[4], dst_linesize[4];   // linesizes of the src and dst images
int src_w, src_h, dst_w, dst_h;         // src and dst image size

struct SwsContext *sws_ctx_cuda;
// set the SWS_HWACCEL_CUDA flag when creating SwsContext
sws_ctx_cuda = sws_getContext(src_w, src_h, src_fmt,
                              dst_w, dst_h, dst_fmt,
                              SWS_HWACCEL_CUDA, NULL, NULL, NULL);

// src_data and dst_data must be arrays of CUDA memory pointers
sws_scale(sws_ctx_cuda, (const uint8_t * const*)src_data,
          src_linesize, 0, src_h, dst_data, dst_linesize);
```
Currently, libgpuscale supports the following conversions:

yuv <-> rgb:
- nv12/yuv420p/yuv444 <->rgb24/rgba32
- p010/yuv420p10 <-> rgb48/rgba64

yuv <-> yuv:
- Any conversions between nv12/p010/p016/yuv420p/yuv420p10/yuv420p16/yuv444p/yuv444p10/yuv444p16

rgb <-> rgb:
- rgb24/bgr24

Supported pixel formats on scaling:
- yuv420p, nv12,
  yuv420p10, yuv420p16,
  p010, p016,
  yuv444p, 
  rgba, bgra,
  rgb24, bgr24,
  rgba64, bgra64,
  rgb32, bgr32

Supported color spaces:
- BT.709
- BT.601
- NTSC-FCC
- SMPTE 240M
- BT.2020

## CUDA Runtime API support
FFmpeg uses CUDA Driver API rather than CUDA Runtime API, the advantage of using CUDA Driver API is that CUDA Toolkit installation is not required, you can run ffmpeg with GPU enabled once the GPU driver is installed. However, CUDA Driver API is not as friendly as CUDA Runtime API in terms of device management. For instance, programmers need to manage CUDA context manually in CUDA Driver API, while CUDA context is transparent in CUDA Runtime API. Besides, FFmpeg only provides a limited set of CUDA Driver APIs, some useful APIs are not included. Moreover, in datacenters, CUDA Toolkits are usually deployed alongside CUDA drivers, enabling CUDA Runtime API support in FFmpeg should have no extra burden for datacenter use cases.

Enabling CUDA Runtime API is transparent in ffmpeg-gpu. If your system has CUDA Toolkit installed, it will be detected when you run the `configure` script. If `configure` finds CUDA Runtime libraries and headers, `CONFIG_CUDART` will be defined as 1 in `ffmpeg-gpu/config.h`, then you can use all CUDA Runtime APIs when developing new filters, instead of being limited to the partial CUDA Driver APIs provided by FFmpeg. You can launch kernels using the `<<<>>>` symbol, and the kernels can be compiled into `.o` directly by nvcc instead of ptx. You can even let the CUDA Runtime manage the CUDA context for you.

We prepared a CUDA filter development template for you as a reference, please see `ffmpeg-gpu/libavfilter/vf_template_cuda.cpp`.