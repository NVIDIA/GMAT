# How to build a GPU filter in FFmpeg

This document describes how to customize a GPU filter in FFmpeg. Since FFmpeg provides a [tutorial](../ffmpeg-gpu/doc/writing_filters.txt) on writing simple CPU filters, this document will focus on issues that are important in GPU filters.

## Getting started

It is recommended to read the [filter writing tutorial](../ffmpeg-gpu/doc/writing_filters.txt), the tutorial will tell how to setup compilation tools (configure and Makefile) to accomodate your new filter, as well as the meanings of the interfaces.

## Customizing GPU filter

### GPU device & memory management in FFmpeg

FFmpeg uses CUDA Driver API to manage GPU device, which is loaded using `dlopen()`. All API symbols are stored in the `CudaFunctions` struct. You can bypass this mechanism by including CUDA header directly in your filter:

```cpp
#include <cuda.h>
```

We need to use Driver API because FFmpeg manually manages CUDA context. When you add `-hwaccel cuda -hwaccel_output_format cuda` or call `-vf hwupload_cuda` in commandline, libavutil will create a CUDA context on the selected GPU. The CUDA context is contained in the AVCUDADeviceContext struct, please refer to the sample GPU filters for how to access AVCUDADeviceContext. 

When implementing your GPU filter, you should call `cuCtxPushCurrent()` before any explicit/implicit CUDA calls, and call `cuCtxPopCurrent()`. This will make sure that all your CUDA calls happens within a valid CUDA context.

Each filter is responsible for allocating its output and free its input. FFmpeg uses `AVHWFramesContext` to manage frames in non-CPU hardwares. You need to create a `AVHWFramesContext` for your output frame in `config_props` using `av_hwframe_ctx_alloc` and `av_hwframe_ctx_init`. Then in `filter_frame`, you can use `av_frame_alloc` to allocate a `AVFrame` struct and `av_hwframe_get_buffer` to allocate necessary GPU memory. `av_hwframe_get_buffer` will allocate GPU memory from a memory pool, so do not worry about `cudaMalloc` overheads.

### Use C and C++ together

FFmpeg is written in C, while many filter development involves C++. The `extern "C"` keyword is your friend who makes C++ codes linkable to C codes. An illustration of `extern "C"` can be found [here](https://isocpp.org/wiki/faq/mixing-c-and-cpp). You can also refer to other GPU filters which uses C++ codes (e.g. vf_cvtColor)

### HW flag

Make sure to add `.flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE` when defining your `AVFilter` struct. FFmpeg does not allow setting `AVHWFramesContext` in filters without this flag.