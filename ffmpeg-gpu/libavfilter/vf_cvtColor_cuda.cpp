/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

extern "C"
{
} // extern "C"

#include <cv_cuda.h>
#include <opencv2/imgproc.hpp>

#include <unordered_map>

using namespace cv;
using namespace cuda_op;

// #define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, cu, x)

extern "C"
{

#include "avfilter.h"
#include "libavutil/hwcontext.h"
// #undef CUDA_VERSION
#include "libavutil/hwcontext_cuda_internal.h"
#include "formats.h"
#include "internal.h"
#include "libavutil/error.h"
#include "libavutil/macros.h"
// #include "libavutil/cuda_check.h"
#include "libavutil/frame.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixfmt.h"
#include "libavutil/pixdesc.h"
#include "libavformat/avio.h"
#include "libavutil/log.h"
#include "libavutil/buffer.h"

typedef struct CvtParam
{
    enum ColorConversionCodes cvt_code;
    enum AVPixelFormat        out_pix_fmt;
    int                       in_channels;
    int                       out_channels;
}CvtParam;

typedef struct CvtColorCopyParam {
    size_t line_width;
    size_t line_number;
};

typedef struct CvtColorCudaContext {
    const AVClass *av_class;
    enum AVPixelFormat in_fmt, out_fmt;
    const AVPixFmtDescriptor *in_desc, *out_desc;
    CvtParam param;
    enum cuda_op::DataType dtype;
    size_t bytes_per_pixel;
    int cv_in_h, cv_in_w; // exact size
    int frame_h, frame_w; // padded size
    DataShape input_shape, output_shape;
    CvtColorCopyParam in_copy_param[4], out_copy_param[4];
    // enum ColorDepth color_depth;

    // int tighten;

    AVBufferRef *hw_frames_ctx;

    char *outfmt_opt;
    char *cvt_code_opt;
    char *dtype_opt;
    CvtColor *cvt_color_class;
    void *cv_input_buffer, *cv_output_buffer;
    void *workspace;
    size_t workspace_size;

    CUstream cu_stream;
} CvtColorCudaContext;

// Self-defined CUDA check functions as cuda_check.h is not available for cpp due to void* function pointers
static inline bool check_cu(CUresult e, void *ctx, int iLine, const char *szFile) {
    if (e != CUDA_SUCCESS) {
        const char* pStr;
        cuGetErrorName(e, &pStr);
        av_log(ctx, AV_LOG_ERROR, "CUDA driver API error: %s, at line %d in file %s\n",
        pStr, iLine, szFile);
        return false;
    }
    return true;
}

static inline bool check(cudaError_t e, void *ctx, int iLine, const char *szFile) {
    if (e != cudaSuccess) {
        av_log(ctx, AV_LOG_ERROR, "CUDA runtime API error: %s, at line %d in file %s\n",
            cudaGetErrorName(e), iLine, szFile);
        return false;
    }
    return true;
}

#define ck(call) check(call, s, __LINE__, __FILE__)
#define ck_cu(call) check_cu(call, s, __LINE__, __FILE__)

#define OFFSET(x) offsetof(CvtColorCudaContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM

static const AVOption cvtColor_cuda_options[] = {
    { "pix_fmt", "OpenCV color conversion code", OFFSET(outfmt_opt), AV_OPT_TYPE_STRING, .flags = AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM },
    // { "cvt_code", "OpenCV color conversion code", OFFSET(cvt_code_opt), AV_OPT_TYPE_STRING, .flags = AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM },
    { "dtype", "OpenCV data type", OFFSET(dtype_opt), AV_OPT_TYPE_STRING, {.str = "8u"}, .flags = AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM},
    { NULL }
};

AVFILTER_DEFINE_CLASS(cvtColor_cuda);

static const enum AVPixelFormat supported_fmts[] = {
    AV_PIX_FMT_NV12,
    // AV_PIX_FMT_RGBPF32LE,
    // AV_PIX_FMT_RGBPF32CHW,
    // AV_PIX_FMT_NV21,
    AV_PIX_FMT_RGB24,
    AV_PIX_FMT_BGR24,
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_0RGB32,
    AV_PIX_FMT_0BGR32
};

static std::unordered_map<std::string, CvtParam> const cvt_table =
{
    {"bgr2rgb", {COLOR_BGR2RGB, AV_PIX_FMT_RGB24, 3, 3}},
    {"rgb2bgr", {COLOR_RGB2BGR, AV_PIX_FMT_BGR24, 3, 3}},
    {"nv122rgb", {COLOR_YUV2RGB_NV12, AV_PIX_FMT_RGB24, 1, 3}},
    {"nv122rgb0", {COLOR_YUV2RGBA_NV12, AV_PIX_FMT_RGB0, 1, 4}},
    {"yuv420p2rgb0", {COLOR_YUV2RGBA_IYUV, AV_PIX_FMT_RGB0, 1, 4}},
    // {"yuv420p2rgb", {COLOR_YUV2RGB_IYUV, AV_PIX_FMT_RGB24, 1, 3}},
    {"rgb2yuv444p", {COLOR_RGB2YUV, AV_PIX_FMT_YUV444P, 3, 3}},
    // {"rgb2nv12", {COLOR_RGB2YUV, AV_PIX_FMT_NV12, 3, 1}}, // cv-cuda not support rgb2nv12 yet

};

static std::unordered_map<std::string, enum cuda_op::DataType> const dtype_table =
{
    {"8u", kCV_8U},
    {"16u", kCV_16U},
    {"32f", kCV_32F},
};

static inline bool getYUV4xxFlag(int cvt_code)
{
    bool YUV4xx = false;
    switch(cvt_code)
    {
    case COLOR_YUV2RGB_NV12:
    case COLOR_YUV2RGB_NV21:
    case COLOR_YUV2BGR_NV12:
    case COLOR_YUV2BGR_NV21:
    case COLOR_YUV2RGBA_NV12:
    case COLOR_YUV2RGBA_NV21:
    case COLOR_YUV2BGRA_NV12:
    case COLOR_YUV2BGRA_NV21:
    case COLOR_YUV2RGB_IYUV:
    case COLOR_YUV2RGB_YV12:
    case COLOR_YUV2BGR_IYUV:
    case COLOR_YUV2BGR_YV12:
    case COLOR_YUV2RGBA_IYUV:
    case COLOR_YUV2RGBA_YV12:
    case COLOR_YUV2BGRA_IYUV:
    case COLOR_YUV2BGRA_YV12:
    case COLOR_YUV2RGB_YUY2:
    case COLOR_YUV2RGB_YVYU:
    case COLOR_YUV2BGR_YUY2:
    case COLOR_YUV2BGR_YVYU:
    case COLOR_YUV2RGBA_YUY2:
    case COLOR_YUV2RGBA_YVYU:
    case COLOR_YUV2BGRA_YUY2:
    case COLOR_YUV2BGRA_YVYU:
    case COLOR_YUV2RGB_UYVY:
    case COLOR_YUV2BGR_UYVY:
    case COLOR_YUV2RGBA_UYVY:
    case COLOR_YUV2BGRA_UYVY:
    case COLOR_YUV2GRAY_420:
    case COLOR_YUV2GRAY_YUY2:
    case COLOR_YUV2GRAY_UYVY:
        YUV4xx = true;
        break;
    default:
        break;
    }
    return YUV4xx;
}

static inline int getResizeRatio(int cvt_code)
{
    int resize_ratio = 3;
    switch(cvt_code)
    {
    case COLOR_YUV2RGBA_NV12:
    case COLOR_YUV2BGRA_NV12:
    case COLOR_YUV2RGBA_NV21:
    case COLOR_YUV2BGRA_NV21:
    case COLOR_YUV2RGBA_YV12:
    case COLOR_YUV2BGRA_YV12:
    case COLOR_YUV2RGBA_IYUV:
    case COLOR_YUV2BGRA_IYUV:
    case COLOR_YUV2RGB_NV12:
    case COLOR_YUV2BGR_NV12:
    case COLOR_YUV2RGB_NV21:
    case COLOR_YUV2BGR_NV21:
    case COLOR_YUV2RGB_YV12:
    case COLOR_YUV2BGR_YV12:
    case COLOR_YUV2RGB_IYUV:
    case COLOR_YUV2BGR_IYUV:
    case COLOR_YUV2GRAY_420:
    {
        resize_ratio = 2;
    }
    break;
    default:
        break;
    }
    return resize_ratio;
}

static inline size_t getDataSize(DataShape data, size_t bpp)
{
    return data.N * data.C * data.H * data.W * bpp;
}

static int query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_CUDA,
        AV_PIX_FMT_NONE
    };
    AVFilterFormats *fmt_list = ff_make_format_list((const int*)pix_fmts);
    if (!fmt_list)
        return AVERROR(ENOMEM);
    return ff_set_common_formats(ctx, fmt_list);
}

static av_cold int init(AVFilterContext *ctx)
{
    CvtColorCudaContext *s = reinterpret_cast<CvtColorCudaContext*>(ctx->priv);
    int ret;

    if (!s->outfmt_opt)
    {
        av_log(s, AV_LOG_ERROR, "No pixel format specified.\n");
        return AVERROR(EINVAL);
    }

    // if (!s->cvt_code_opt)
    // {
    //     av_log(s, AV_LOG_ERROR, "No color conversion code specified.\n");
    //     return AVERROR(EINVAL);
    // }

    // std::string cvt_code_string{s->cvt_code_opt};
    // auto cvt_code_find = cvt_table.find(cvt_code_string);
    // if (cvt_code_find == cvt_table.end())
    // {
    //     av_log(s, AV_LOG_ERROR, "Color conversion codes unsupported.\n");
    //     return AVERROR(EINVAL);
    // }
    // s->param = cvt_code_find->second;

    // auto dtype_find = cvt_table.find(std::string{s->dtype_opt});
    // if (dtype_find == cvt_table.end())
    // {
    //     av_log(s, AV_LOG_ERROR, "Color conversion codes unsupported.\n");
    //     return AVERROR(EINVAL);
    // }
    // s->dtype = dtype_find->second;

    if (strcmp(s->dtype_opt, "8u") == 0)
    {
        s->dtype = kCV_8U;
        s->bytes_per_pixel = 1;
    }
    else if (strcmp(s->dtype_opt, "16u") == 0)
    {
        s->dtype = kCV_16U;
        s->bytes_per_pixel = 2;
    }
    else if (strcmp(s->dtype_opt, "32f") == 0)
    {
        s->dtype = kCV_32F;
        s->bytes_per_pixel = 4;
    }
    else
    {
        av_log(s, AV_LOG_ERROR, "Data type unsupported.\n");
        return AVERROR(EINVAL);
    }

    return 0;
}

static int config_props(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = ctx->inputs[0];
    CvtColorCudaContext *s = (CvtColorCudaContext*)ctx->priv;

    AVHWFramesContext *in_frame_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVBufferRef *hw_device_ref = in_frame_ctx->device_ref;
    AVCUDADeviceContext *hw_ctx = (AVCUDADeviceContext*)in_frame_ctx->device_ctx->hwctx;
    // CudaFunctions *cu = hw_ctx->internal->cuda_dl;

    AVBufferRef *out_ref = NULL;
    AVHWFramesContext *out_ctx;
    CUcontext dummy;
    int ret;

    // CvtColorCopyParam in_copy_param[4], out_copy_param[4];
    // const AVPixFmtDescriptor *in_desc;
    // const AVPixFmtDescriptor *out_desc;

    int w = inlink->w;
    int h = inlink->h;
    s->cv_in_w = w;
    s->cv_in_h = h;
    int cvt_code = s->param.cvt_code;
    DataShape input_shape;
    DataShape output_shape;

    const char* in_pixfmt_name = av_get_pix_fmt_name(in_frame_ctx->sw_format);
    const char* out_pixfmt_name = s->outfmt_opt;
    std::string cvt_string;
    cvt_string += in_pixfmt_name;
    cvt_string += "2";
    cvt_string += out_pixfmt_name;

    auto cvt_code_find = cvt_table.find(cvt_string);
    if (cvt_code_find == cvt_table.end())
    {
        av_log(s, AV_LOG_ERROR, "Color conversion unsupported.\n");
        return AVERROR(EINVAL);
    }
    s->param = cvt_code_find->second;
    cvt_code = s->param.cvt_code;

    output_shape = DataShape(s->param.out_channels, h, w);
    if (!getYUV4xxFlag(cvt_code))
    {
        input_shape = DataShape(s->param.in_channels, h, w);
    }
    else
    {
        int resize_ratio = getResizeRatio(cvt_code);
        input_shape = DataShape(s->param.in_channels, h * 3 / resize_ratio, w);
    }

    s->input_shape = input_shape;
    s->output_shape = output_shape;
    s->cvt_color_class = new CvtColor(input_shape, output_shape);
    s->workspace_size = s->cvt_color_class->calBufferSize(input_shape, output_shape, s->dtype);

    ck_cu(cuCtxPushCurrent(hw_ctx->cuda_ctx));

    out_ref = av_hwframe_ctx_alloc(hw_device_ref);
    if (!out_ref)
        return AVERROR(ENOMEM);
    out_ctx = (AVHWFramesContext*)out_ref->data;

    s->in_fmt = in_frame_ctx->sw_format;
    s->out_fmt = s->param.out_pix_fmt;

    out_ctx->format = AV_PIX_FMT_CUDA;
    out_ctx->sw_format = s->out_fmt;
    out_ctx->width = in_frame_ctx->width;
    out_ctx->height = in_frame_ctx->height;

    s->frame_w = in_frame_ctx->width;
    s->frame_h = in_frame_ctx->height;

    ret = av_hwframe_ctx_init(out_ref);
    if (ret < 0)
    {
        av_buffer_unref(&out_ref);
        return ret;
    }
    s->hw_frames_ctx = out_ref;
    outlink->hw_frames_ctx = av_buffer_ref(s->hw_frames_ctx);
    if (!outlink->hw_frames_ctx)
        return AVERROR(ENOMEM);

    ck(cudaMalloc(&s->workspace, s->workspace_size));
    // ck(cudaMalloc(&s->cv_input_buffer, h * w * s->param.in_channels * s->bytes_per_pixel));
    // ck(cudaMalloc(&s->cv_output_buffer, h * w * s->param.out_channels * s->bytes_per_pixel));
    ck(cudaMalloc(&s->cv_input_buffer, getDataSize(input_shape, s->bytes_per_pixel)));
    ck(cudaMalloc(&s->cv_output_buffer, getDataSize(output_shape, s->bytes_per_pixel)));

    ck_cu(cuCtxPopCurrent(&dummy));

    s->in_desc = av_pix_fmt_desc_get(s->in_fmt);
    s->out_desc = av_pix_fmt_desc_get(s->out_fmt);

    // memset(s->in_copy_param, 0, sizeof(CvtColorCopyParam));
    // memset(s->out_copy_param, 0, sizeof(CvtColorCopyParam));
    // for (int i = 0; i < 4; i++) {
    //     int in_plane = s->in_desc->comp[i].plane;
    //     int out_plane = s->out_desc->comp[i].plane;

    //     // s->in_copy_param[in_plane].line_width += i==0? w : AV_CEIL_RSHIFT(w, s->in_desc->log2_chroma_w);
    //     // s->in_copy_param[in_plane].line_number += i==0? h : AV_CEIL_RSHIFT(h, s->in_desc->log2_chroma_h);
    //     // s->out_copy_param[in_plane].line_width += i==0? w : AV_CEIL_RSHIFT(w, s->out_desc->log2_chroma_w);
    //     // s->out_copy_param[in_plane].line_number += i==0? h : AV_CEIL_RSHIFT(h, s->out_desc->log2_chroma_h);

    // }

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    CvtColorCudaContext *s = (CvtColorCudaContext*)ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    AVHWFramesContext     *frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVCUDADeviceContext *hw_ctx = (AVCUDADeviceContext*)frames_ctx->device_ctx->hwctx;
    // CudaFunctions *cu = hw_ctx->internal->cuda_dl;
    CUstream stream = hw_ctx->stream;
    CUcontext dummy;

    const AVPixFmtDescriptor *in_desc;
    const AVPixFmtDescriptor *out_desc;
    CvtColor* cvt_color_class = s->cvt_color_class;
    volatile AVFrame* in_frame = in;
    int in_frame_copy_w = in_frame->width * 3;

    int ret;
    AVFrame *out = av_frame_alloc();
    if (!out)
    {
        ret = AVERROR(ENOMEM);
        goto fail;
        return ret;
    }

    ck_cu(cuCtxPushCurrent(hw_ctx->cuda_ctx));
    ret = av_hwframe_get_buffer(s->hw_frames_ctx, out, 0);
    if (ret < 0)
        goto fail;

    // if ((s->in_desc->flags & AV_PIX_FMT_FLAG_RGB) && (s->in_desc->flags & AV_PIX_FMT_FLAG_PLANAR)) // planar RGB
    // {
    //     size_t dst_height = s->cv_in_h * s->in_desc->nb_components;
    //     ck(cudaMemcpy2DAsync(s->cv_input_buffer, s->cv_in_w * s->bytes_per_pixel, in->data[0], in->linesize[0],
    //         in->width, dst_height, cudaMemcpyDeviceToDevice, stream));
    // }
    // else if (s->in_desc->flags & AV_PIX_FMT_FLAG_PLANAR) // planar YUV
    // {
    //     size_t dst_pitch_luma = s->cv_in_w * s->bytes_per_pixel;
    //     ck(cudaMemcpy2DAsync(s->cv_input_buffer, dst_pitch_luma, in->data[0], in->linesize[0],
    //         dst_pitch_luma, s->cv_in_h, cudaMemcpyDeviceToDevice, stream));

    //     int num_planes = av_pix_fmt_count_planes(s->in_fmt);
    //     int dst_pitch_chroma = s->cv_in_w >> s->in_desc->log2_chroma_w;
    //     int dst_height_chroma = s->cv_in_h >> s->in_desc->log2_chroma_h;

    //     if (num_planes == 2)
    //         dst_pitch_chroma *= 2;
    //     else
    //         dst_height_chroma *= 2;
    //     ck(cudaMemcpy2DAsync(s->cv_input_buffer, dst_pitch_chroma, in->data[1], in->linesize[1],
    //         dst_pitch_chroma, dst_height_chroma, cudaMemcpyDeviceToDevice, stream));

    // }
    // else // packed RGB & YUV
    // {
    //     size_t dst_pitch = s->cv_in_w * s->in_desc->nb_components * s->bytes_per_pixel;
    //     ck(cudaMemcpy2DAsync(s->cv_input_buffer, dst_pitch, in->data[0], in->linesize[0],
    //         dst_pitch, s->cv_in_h, cudaMemcpyDeviceToDevice, stream));
    // }

    if (s->in_desc->flags & AV_PIX_FMT_FLAG_RGB) // rgb or bgr
    {
        size_t dst_height = s->cv_in_h;
        size_t dst_width = s->input_shape.W * s->input_shape.C * s->bytes_per_pixel;
        ck(cudaMemcpy2DAsync(s->cv_input_buffer, dst_width, in->data[0], in->linesize[0],
            s->cv_in_w * 3, dst_height, cudaMemcpyDeviceToDevice, stream));
    }
    else if (!strcmp(s->in_desc->name, "nv12"))
    {
        size_t dst_height = s->cv_in_h * 3 / 2;
        // printf("in_data: %lu, %lu; offset: %lu; linesize: %lu\n", in->data[0], in->data[1], s->cv_in_h * s->cv_in_w, in->linesize[0]);
        ck(cudaMemcpy2DAsync(s->cv_input_buffer, s->cv_in_w * s->bytes_per_pixel, in->data[0], in->linesize[0],
            s->cv_in_w * s->bytes_per_pixel, dst_height, cudaMemcpyDeviceToDevice, stream));
    }
    else if (!strcmp(s->in_desc->name, "yuv420p"))
    {
        size_t dst_height = s->cv_in_h;
        ck(cudaMemcpy2DAsync(s->cv_input_buffer, s->cv_in_w * s->bytes_per_pixel, in->data[0], in->linesize[0],
            s->cv_in_w * s->bytes_per_pixel, dst_height, cudaMemcpyDeviceToDevice, stream));

        // copy uv plane
        int h_shift = s->in_desc->log2_chroma_h;
        int w_shift = s->in_desc->log2_chroma_w;
        int u_offset = s->cv_in_w * s->bytes_per_pixel * dst_height;
        int v_offset = s->cv_in_w * s->bytes_per_pixel * dst_height / 4 * 5;
        ck(cudaMemcpy2DAsync(((char*)s->cv_input_buffer) + u_offset, (s->cv_in_w >> w_shift) * s->bytes_per_pixel, in->data[1], in->linesize[1],
            (s->cv_in_w >> w_shift) * s->bytes_per_pixel, (dst_height >> h_shift), cudaMemcpyDeviceToDevice, stream));
        ck(cudaMemcpy2DAsync(((char*)s->cv_input_buffer) + v_offset, (s->cv_in_w >> w_shift) * s->bytes_per_pixel, in->data[2], in->linesize[2],
            (s->cv_in_w >> w_shift) * s->bytes_per_pixel, (dst_height >> h_shift), cudaMemcpyDeviceToDevice, stream));
    }

    // printf("in->linesize[0]: %d\n", in->linesize[0]);
    cvt_color_class->infer(&s->cv_input_buffer, &s->cv_output_buffer, s->workspace, s->param.cvt_code, 0, s->input_shape, cuda_op::DataFormat::kHWC, s->dtype, stream);

    if (s->out_desc->flags & AV_PIX_FMT_FLAG_RGB) // rgb0 or bgr0
    {
        size_t dst_height = s->cv_in_h;
        size_t src_width = s->output_shape.W * s->output_shape.C * s->bytes_per_pixel;
        ck(cudaMemcpy2DAsync(out->data[0], out->linesize[0], s->cv_output_buffer, src_width,
            src_width, dst_height, cudaMemcpyDeviceToDevice, stream));
    }
    else if (!strcmp(s->out_desc->name, "nv12"))
    {
        size_t dst_height = s->output_shape.H * 3 / 2;
        size_t src_width = s->output_shape.W * s->bytes_per_pixel;
        ck(cudaMemcpy2DAsync(out->data[0], out->linesize[0], s->cv_output_buffer, src_width,
            src_width, dst_height, cudaMemcpyDeviceToDevice, stream));
    }
    else if (!strcmp(s->out_desc->name, "yuv444p"))
    {
        size_t dst_height = s->output_shape.H;
        size_t dst_width = s->output_shape.W * s->bytes_per_pixel;
        ck(cudaMemcpy2DAsync(out->data[0], out->linesize[0], s->cv_input_buffer, dst_width,
            dst_width, dst_height, cudaMemcpyDeviceToDevice, stream));

        int h_shift = s->out_desc->log2_chroma_h;
        int w_shift = s->out_desc->log2_chroma_w;
        int u_offset = s->output_shape.W * s->bytes_per_pixel * dst_height;
        ck(cudaMemcpy2DAsync(out->data[1], out->linesize[1], s->cv_input_buffer + u_offset, s->output_shape.W * s->bytes_per_pixel,
            (s->cv_in_w >> w_shift) * s->bytes_per_pixel, (dst_height >> h_shift) * 2, cudaMemcpyDeviceToDevice, stream));
    }
    // if ((s->out_desc->flags & AV_PIX_FMT_FLAG_RGB) && (s->out_desc->flags & AV_PIX_FMT_FLAG_PLANAR)) // planar RGB
    // {
    //     size_t dst_height = s->output_shape.H * s->out_desc->nb_components;
    //     size_t src_width = s->output_shape.W * s->out_desc->nb_components * s->bytes_per_pixel;
    //     ck(cudaMemcpy2DAsync(out->data[0], out->linesize[0], s->cv_output_buffer, src_width,
    //         src_width, dst_height, cudaMemcpyDeviceToDevice, stream));
    // }
    // else if (s->out_desc->flags & AV_PIX_FMT_FLAG_PLANAR) // planar YUV
    // {
    //     size_t dst_pitch_luma = s->cv_in_w * s->bytes_per_pixel;
    //     ck(cudaMemcpy2DAsync(s->cv_input_buffer, dst_pitch_luma, in->data[0], in->linesize[0],
    //         dst_pitch_luma, s->cv_in_h, cudaMemcpyDeviceToDevice, stream));

    //     int num_planes = av_pix_fmt_count_planes(s->in_fmt);
    //     int dst_pitch_chroma = s->cv_in_w >> s->in_desc->log2_chroma_w;
    //     int dst_height_chroma = s->cv_in_h >> s->in_desc->log2_chroma_h;

    //     if (num_planes == 2)
    //         dst_pitch_chroma *= 2;
    //     else
    //         dst_height_chroma *= 2;
    //     ck(cudaMemcpy2DAsync(s->cv_input_buffer, dst_pitch_chroma, in->data[1], in->linesize[1],
    //         dst_pitch_chroma, dst_height_chroma, cudaMemcpyDeviceToDevice, stream));

    // }

    // ck(cudaMemcpy2DAsync(out->data[0], out->linesize[0], s->cv_output_buffer, s->output_shape.W * 3 * s->bytes_per_pixel, out->width, out->height, cudaMemcpyDeviceToDevice, stream));
    ck(cudaStreamSynchronize(stream));

    ck_cu(cuCtxPopCurrent(&dummy));

    ret = av_frame_copy_props(out, in);
    if (ret < 0)
        goto fail;

    av_frame_free(&in);
    return ff_filter_frame(outlink, out);
fail:
    av_frame_free(&in);
    av_frame_free(&out);
    return ret;
}
static av_cold void uninit(AVFilterContext *ctx)
{
    CvtColorCudaContext *s = reinterpret_cast<CvtColorCudaContext*>(ctx->priv);
    if (s->workspace){
        cudaFree(s->workspace);
    }
    if (s->cv_input_buffer){
        cudaFree(s->cv_input_buffer);
    }
    if (s->cv_output_buffer){
        cudaFree(s->cv_output_buffer);
    }

    if (s->cvt_color_class){
        delete s->cvt_color_class;
    }
}

static const AVFilterPad cvtColor_cuda_inputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
    },
    { NULL }
};

static const AVFilterPad cvtColor_cuda_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_props,
    },
    { NULL }
};

AVFilter ff_vf_cvtColor_cuda = {
    .name = "cvtColor_cuda",
    .description = NULL_IF_CONFIG_SMALL("OpenCV cvtColor CUDA implementation."),
    .inputs = cvtColor_cuda_inputs,
    .outputs = cvtColor_cuda_outputs,
    .priv_class = &cvtColor_cuda_class,
    .init = init,
    .uninit = uninit,
    .query_formats =query_formats,
    .priv_size = sizeof(CvtColorCudaContext),
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE
};

} // extern "C"