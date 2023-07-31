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
#include "libavutil/hwcontext.h"
// #undef CUDA_VERSION
#include "libavutil/hwcontext_cuda_internal.h"
#include "formats.h"
#include "internal.h"
#include "libavutil/error.h"
#include "libavutil/macros.h"
#include "libavutil/cuda_check.h"
#include "libavutil/frame.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixfmt.h"
#include "libavutil/pixdesc.h"
#include "libavformat/avio.h"
#include "libavutil/log.h"
#include "libavutil/buffer.h"

#include <cuda_runtime.h>

typedef struct TemplateCudaContext {
    int frame_h, frame_w; // padded size
    enum AVPixelFormat in_fmt, out_fmt;
    AVBufferRef *hw_frames_ctx;
} TemplateCudaContext;

static const AVOption template_cuda_options[] = {
    // { "pix_fmt", "OpenCV color conversion code", OFFSET(outfmt_opt), AV_OPT_TYPE_STRING, .flags = AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM },
    // { "cvt_code", "OpenCV color conversion code", OFFSET(cvt_code_opt), AV_OPT_TYPE_STRING, .flags = AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM },
    // { "dtype", "OpenCV data type", OFFSET(dtype_opt), AV_OPT_TYPE_STRING, {.str = "8u"}, .flags = AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM},
    { NULL }
};

AVFILTER_DEFINE_CLASS(template_cuda);

static const enum AVPixelFormat supported_fmts[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_RGB24,
    AV_PIX_FMT_BGR24,
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_0RGB32,
    AV_PIX_FMT_0BGR32
};

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
    TemplateCudaContext *s = reinterpret_cast<TemplateCudaContext*>(ctx->priv);
    int ret;

    return 0;
}

static int config_props(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = ctx->inputs[0];
    TemplateCudaContext *s = (TemplateCudaContext*)ctx->priv;

    AVHWFramesContext *in_frame_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVBufferRef *hw_device_ref = in_frame_ctx->device_ref;
    AVCUDADeviceContext *hw_ctx = (AVCUDADeviceContext*)in_frame_ctx->device_ctx->hwctx;
    // CudaFunctions *cu = hw_ctx->internal->cuda_dl;

    AVBufferRef *out_ref = NULL;
    AVHWFramesContext *out_ctx;
    CUcontext dummy;
    int ret;

    CHECK_CU(cuCtxPushCurrent(hw_ctx->cuda_ctx));

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

    ck_cu(cuCtxPopCurrent(&dummy));

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    TemplateCudaContext *s = (TemplateCudaContext*)ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    AVHWFramesContext     *frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVCUDADeviceContext *hw_ctx = (AVCUDADeviceContext*)frames_ctx->device_ctx->hwctx;
    // CudaFunctions *cu = hw_ctx->internal->cuda_dl;
    CUstream stream = hw_ctx->stream;
    CUcontext dummy;

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
    TemplateCudaContext *s = reinterpret_cast<TemplateCudaContext*>(ctx->priv);
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

static const AVFilterPad template_cuda_inputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
    },
    { NULL }
};

static const AVFilterPad template_cuda_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_props,
    },
    { NULL }
};

AVFilter ff_vf_template_cuda = {
    .name = "template_cuda",
    .description = NULL_IF_CONFIG_SMALL("CUDA filter implementation template"),
    .inputs = template_cuda_inputs,
    .outputs = template_cuda_outputs,
    .priv_class = &template_cuda_class,
    .init = init,
    .uninit = uninit,
    .query_formats =query_formats,
    .priv_size = sizeof(TemplateCudaContext),
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE
};