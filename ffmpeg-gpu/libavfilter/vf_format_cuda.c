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

#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include "libavutil/error.h"
#include "libavutil/macros.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/cuda_check.h"
#include "libavutil/frame.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixfmt.h"
#include "libavformat/avio.h"
#include "libavutil/log.h"
#include "libavutil/buffer.h"

#include "format_cuda.h"

#include <limits.h>

#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, cu, x)

enum ColorDepth {
    DEPTH_8_BIT,
    DEPTH_10_BIT,
    DEPTH_16_BIT,
};

typedef struct FormatCudaContext {
    const AVClass *class;
    enum AVPixelFormat in_fmt, out_fmt;
    enum ColorDepth color_depth;

    int tighten;

    AVBufferRef *hw_frames_ctx;

    char *pix_fmt;

    CUstream cu_stream;
} FormatCudaContext;

#define OFFSET(x) offsetof(FormatCudaContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM

static const AVOption format_cuda_options[] = {
    { "pix_fmt", "Target pixel formats", OFFSET(pix_fmt), AV_OPT_TYPE_STRING, .flags = AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM },
    { NULL }
};

AVFILTER_DEFINE_CLASS(format_cuda);

static const enum AVPixelFormat supported_fmts[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_RGBPF32LE,
};

static av_cold int init(AVFilterContext *ctx)
{
    FormatCudaContext *s = ctx->priv;
    int ret;

    if (!s->pix_fmt)
    {
        av_log(s, AV_LOG_ERROR, "No output pixel format specified.\n");
        return AVERROR(EINVAL);
    }
    ret = ff_parse_pixel_format(&s->out_fmt, s->pix_fmt, s);
    if (ret < 0)
        return ret;

    s->tighten = 0;
    if (s->out_fmt == AV_PIX_FMT_RGBPF32LE) s->tighten = 1;

    for (int i = 0; i < FF_ARRAY_ELEMS(supported_fmts); i++)
    {
        if (s->out_fmt == supported_fmts[i])
        {
            ret = 1;
            break;
        }
    }

    return 0;
}

static int query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_CUDA,
        AV_PIX_FMT_NONE
    };
    AVFilterFormats *fmt_list = ff_make_format_list(pix_fmts);
    if (!fmt_list)
        return AVERROR(ENOMEM);
    return ff_set_common_formats(ctx, fmt_list);
}

static void set_color_depth(FormatCudaContext *s)
{
    if (s->in_fmt == AV_PIX_FMT_P010LE || s->out_fmt == AV_PIX_FMT_P010LE)
        s->color_depth = DEPTH_10_BIT;
    else if (s->in_fmt == AV_PIX_FMT_P016LE || s->out_fmt == AV_PIX_FMT_P016LE)
        s->color_depth = DEPTH_16_BIT;
    else
        s->color_depth = DEPTH_8_BIT;

}

static int config_props(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = ctx->inputs[0];
    FormatCudaContext *s = ctx->priv;

    AVHWFramesContext *in_frame_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVBufferRef *hw_device_ref = in_frame_ctx->device_ref;
    AVCUDADeviceContext *hw_ctx = (AVCUDADeviceContext*)in_frame_ctx->device_ctx->hwctx;
    CudaFunctions *cu = hw_ctx->internal->cuda_dl;

    AVBufferRef *out_ref = NULL;
    AVHWFramesContext *out_ctx;
    CUcontext dummy;

    int ret;

    s->in_fmt = in_frame_ctx->sw_format;
    s->cu_stream = hw_ctx->stream;
    // int channels, model_in_h, model_in_w, model_out_h, model_out_w;
    set_color_depth(s);

    CHECK_CU(cu->cuCtxPushCurrent(hw_ctx->cuda_ctx));

    out_ref = av_hwframe_ctx_alloc(hw_device_ref);
    if (!out_ref)
        return AVERROR(ENOMEM);
    out_ctx = (AVHWFramesContext*)out_ref->data;

    out_ctx->format = AV_PIX_FMT_CUDA;
    out_ctx->sw_format = s->out_fmt;
    out_ctx->width = s->tighten ? inlink->w : in_frame_ctx->width;
    out_ctx->height = s->tighten ? inlink->h : in_frame_ctx->height;

    ret = av_hwframe_ctx_init(out_ref);
    if (ret < 0)
    {
        av_buffer_unref(&out_ref);
        return ret;
    }

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));

    s->hw_frames_ctx = out_ref;
    outlink->hw_frames_ctx = av_buffer_ref(s->hw_frames_ctx);
    if (!outlink->hw_frames_ctx)
        return AVERROR(ENOMEM);

    return 0;
}

static int format_convert(FormatCudaContext *s, AVFrame *in, AVFrame *out)
{
    // int ret;
    switch (s->color_depth)
    {
        case DEPTH_8_BIT:
        switch (s->out_fmt)
        {
            // case AV_PIX_FMT_GRAYF32LE:
            case AV_PIX_FMT_RGBPF32LE:
                if (s->in_fmt != AV_PIX_FMT_NV12) {
                    av_log(s, AV_LOG_ERROR, "Unsupported input/output pixel format combination.\n");
                    return AVERROR(EINVAL);
                }
                nv12_to_rgbpf32(s->cu_stream, in->data, in->linesize, out->data,
                                out->linesize, in->width, in->height, in->colorspace);
                return 0;

            case AV_PIX_FMT_NV12:
                rgbpf32_to_nv12(s->cu_stream, in->data, in->linesize, out->data,
                                out->linesize, in->width, in->height, in->colorspace);
                return 0;

            default:
                av_log(s, AV_LOG_ERROR, "Input frame format not supported\n");
                return AVERROR(EINVAL);
        }

        default:
        av_log(s, AV_LOG_ERROR, "Bit depth not supported yet\n");
        return AVERROR(EINVAL);
    }

}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    FormatCudaContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    AVHWFramesContext     *frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVCUDADeviceContext *hw_ctx = frames_ctx->device_ctx->hwctx;
    CudaFunctions *cu = hw_ctx->internal->cuda_dl;
    CUcontext dummy;

    int ret;
    AVFrame *out = av_frame_alloc();
    if (!out)
    {
        ret = AVERROR(ENOMEM);
        goto fail;
        return ret;
    }

    CHECK_CU(cu->cuCtxPushCurrent(hw_ctx->cuda_ctx));

    ret = av_hwframe_get_buffer(s->hw_frames_ctx, out, 0);

    ret = format_convert(s, in, out);
    // CHECK_CU(cu->cuStreamSynchronize(hw_ctx->stream));

    if (ret < 0)
        goto fail;

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));

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

}

static const AVFilterPad format_cuda_inputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        // .config_props = config_props,
        .filter_frame = filter_frame,
    },
    { NULL }
};

static const AVFilterPad format_cuda_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_props,
    },
    { NULL }
};

AVFilter ff_vf_format_cuda = {
    .name = "format_cuda",
    .description = NULL_IF_CONFIG_SMALL("Convert between YUV and RGB color space."),
    .inputs = format_cuda_inputs,
    .outputs = format_cuda_outputs,
    .priv_class = &format_cuda_class,
    .init = init,
    .uninit = uninit,
    // .query_formats =query_formats,
    FILTER_QUERY_FUNC(query_formats),
    .priv_size = sizeof(FormatCudaContext),
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE
};