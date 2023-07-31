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

/**
 * @file
 * implementing a generic DL inference filter using TensorRT.
 */

#include "tensorrt.h"

#include "avfilter.h"
#include "libavformat/avio.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/avassert.h"
#include "libavutil/imgutils.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/cuda_check.h"
#include "filters.h"
#include "formats.h"
#include "internal.h"
#include "libavutil/time.h"
#include "libavutil/buffer.h"
#include "libavutil/error.h"
#include "libavutil/frame.h"
#include "libavutil/log.h"
#include "libavutil/pixfmt.h"

#include <limits.h>
#include <cuda_runtime.h>

// ==========================Filter Implementation=============================
#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, cu, x)

#define OFFSET(x) offsetof(TensorrtContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM

static const AVOption tensorrt_options[] = {
    {"engine", "path to the TRT engine file",   OFFSET(engine_filename), AV_OPT_TYPE_STRING, {.str = NULL}, 0,  0,       FLAGS},
    {NULL}
};

AVFILTER_DEFINE_CLASS(tensorrt);

static int init(AVFilterContext *ctx)
{
    TensorrtContext *s = (TensorrtContext*)ctx->priv;

    if (!s->engine_filename)
    {
        av_log(s, AV_LOG_ERROR, "No engine file provided.\n");
        return AVERROR(EINVAL);
    }

    init_trt(s);

    av_log(s, AV_LOG_VERBOSE, "Initialize TensorRT filter and load TensorRT filter\n");
    return 0;
}

static int query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE
    };
    AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
    return ff_set_common_formats(ctx, fmts_list);
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = ctx->inputs[0];
    TensorrtContext *s = ctx->priv;

    AVHWFramesContext *in_frame_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVBufferRef *hw_device_ref = in_frame_ctx->device_ref;
    AVCUDADeviceContext *hw_ctx = (AVCUDADeviceContext*)in_frame_ctx->device_ctx->hwctx;
    CudaFunctions *cu = hw_ctx->internal->cuda_dl;

    AVBufferRef *out_ref = NULL;
    AVHWFramesContext *out_ctx;
    CUcontext dummy;

    int ret;

    enum AVPixelFormat fmt = in_frame_ctx->sw_format;

    switch (fmt) {
    case AV_PIX_FMT_RGBPF32LE:
        s->channels = 3;
        break;
    case AV_PIX_FMT_NV12:
        s->channels = 1;
        break;
        default:
        av_log(s, AV_LOG_ERROR, "Pixel format not supported\n");
        return AVERROR(EINVAL);
    }

    CHECK_CU(cu->cuCtxPushCurrent(hw_ctx->cuda_ctx));

    ret = config_props_trt(s, inlink, cu);
    if (ret < 0)
        return ret;

    outlink->w = s->out_w;
    outlink->h = s->out_h;

    out_ref = av_hwframe_ctx_alloc(hw_device_ref);
    if (!out_ref)
        return AVERROR(ENOMEM);
    out_ctx = (AVHWFramesContext*)out_ref->data;

    out_ctx->format = AV_PIX_FMT_CUDA;
    out_ctx->sw_format = in_frame_ctx->sw_format;
    out_ctx->width = s->out_w;
    out_ctx->height = s->out_h;

    ret = av_hwframe_ctx_init(out_ref);
    if (ret < 0)
    {
        av_buffer_unref(&out_ref);
        return ret;
    }

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));

    av_buffer_unref(&s->hw_frames_ctx);

    s->hw_frames_ctx = inlink->hw_frames_ctx;
    outlink->hw_frames_ctx = av_buffer_ref(out_ref);
    if (!outlink->hw_frames_ctx)
        return AVERROR(ENOMEM);

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    AVHWFramesContext     *frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVCUDADeviceContext *hw_ctx = frames_ctx->device_ctx->hwctx;
    CudaFunctions *cu = hw_ctx->internal->cuda_dl;
    CUcontext dummy;

    int ret;

    CHECK_CU(cu->cuCtxPushCurrent(hw_ctx->cuda_ctx));

    ret = filter_frame_trt(inlink, in);
    if (ret < 0)
        return ret;

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));

    return 0;
}

static void uninit(AVFilterContext *ctx)
{
    TensorrtContext *s = ctx->priv;

    free_trt(s);
}

static const AVFilterPad tensorrt_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
    },
    { NULL }
};

static const AVFilterPad tensorrt_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
    },
    { NULL }
};

AVFilter ff_vf_tensorrt = {
    .name          = "tensorrt",
    .description   = NULL_IF_CONFIG_SMALL("Apply TensorRT filter to the input."),
    .priv_size     = sizeof(TensorrtContext),
    .init          = init,
    .uninit        = uninit,
    .inputs        = tensorrt_inputs,
    .outputs       = tensorrt_outputs,
    FILTER_QUERY_FUNC(query_formats),
    .priv_class    = &tensorrt_class,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};