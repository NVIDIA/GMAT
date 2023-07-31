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

#include <asm-generic/errno-base.h>
#include <float.h>
#include <nvcv/Tensor.h>
#include <cvcuda/OpMedianBlur.h>
#include <cvcuda/OpGaussian.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
#include "libavutil/imgutils.h"


typedef struct SmoothNvcvContext {
    const AVClass *class;
    // int frame_h, frame_w; // padded size
    enum AVPixelFormat in_fmt, out_fmt;
    AVBufferRef *hw_frames_ctx;

    int type;
    int kw, kh;
    double sigmaX, sigmaY;

    NVCVOperatorHandle smooth_handle;
    NVCVTensorLayout   layout;
    NVCVBorderType border_type;
} SmoothNvcvContext;

static inline int check_cu(CUresult e, void *avctx, int iLine, const char *szFile) {
    if (e != CUDA_SUCCESS) {
        const char *szErrName = NULL;
        cuGetErrorName(e, &szErrName);
        av_log(avctx, AV_LOG_ERROR, "CUDA driver API error %s at line %d in file %s", szErrName, iLine, szFile);
        return 0;
    }
    return 1;
}

#define OFFSET(x) offsetof(SmoothNvcvContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM
#define CHECK_CU(x) check_cu(x, ctx, __LINE__, __FILE__)
#define CK_NVCV(ret) if (ret != NVCV_SUCCESS) av_log(ctx, AV_LOG_ERROR, "NVCV error %d at line %d\n", ret, __LINE__);

enum {
    SMOOTH_TYPE_DEFAULT,
    SMOOTH_TYPE_GAUSSIAN,
    SMOOTH_TYPE_MEDIAN
};

static const AVOption smooth_nvcv_options[] = {
    { "type", "Smooth filter type (gaussian, median)", OFFSET(type), AV_OPT_TYPE_INT, {.i64=SMOOTH_TYPE_DEFAULT}, 0, 2, .flags=FLAGS, "type"},
        { "gaussian",  "gaussian blur", 0, AV_OPT_TYPE_CONST, { .i64 = SMOOTH_TYPE_GAUSSIAN }, 0, 0, FLAGS, "type" },
        { "median",  "median blur", 0, AV_OPT_TYPE_CONST, { .i64 = SMOOTH_TYPE_MEDIAN }, 0, 0, FLAGS, "type" },
    { "kw", "Filter kernel width", OFFSET(kw), AV_OPT_TYPE_INT, {.i64=3}, 1, INT_MAX, .flags=FLAGS},
    { "kh", "Filter kernel height", OFFSET(kh), AV_OPT_TYPE_INT, {.i64=3}, 1, INT_MAX, .flags=FLAGS},
    { "border_type", "Border mode to be used when accessing elements outside input image (only for gaussian)", OFFSET(border_type), AV_OPT_TYPE_INT, {.i64=0}, 0, 4, .flags=FLAGS, "border_type"},
        { "constant",  "constant", 0, AV_OPT_TYPE_CONST, { .i64 = 0 }, 0, 0, FLAGS, "border_type" },
        { "replicate",  "replicate", 0, AV_OPT_TYPE_CONST, { .i64 = 1 }, 0, 0, FLAGS, "border_type" },
        { "reflect",  "reflect", 0, AV_OPT_TYPE_CONST, { .i64 = 2 }, 0, 0, FLAGS, "border_type" },
        { "warp",  "warp", 0, AV_OPT_TYPE_CONST, { .i64 = 3 }, 0, 0, FLAGS, "border_type" },
        { "reflect101",  "reflect101", 0, AV_OPT_TYPE_CONST, { .i64 = 4 }, 0, 0, FLAGS, "border_type" },
    { "sigmaX", "Gaussian kernel standard deviation in X direction(only for gaussian)", OFFSET(sigmaX), AV_OPT_TYPE_DOUBLE, {.dbl=0}, 0, DBL_MAX, .flags=FLAGS},
    { "sigmaY", "Gaussian kernel standard deviation in Y direction(only for gaussian)", OFFSET(sigmaY), AV_OPT_TYPE_DOUBLE, {.dbl=0}, 0, DBL_MAX, .flags=FLAGS},
    { NULL }
};

AVFILTER_DEFINE_CLASS(smooth_nvcv);

static const enum AVPixelFormat supported_formats[] = {
    // AV_PIX_FMT_NV12,
    AV_PIX_FMT_RGB24,
    AV_PIX_FMT_BGR24,
    // AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_0RGB32,
    AV_PIX_FMT_0BGR32,
    AV_PIX_FMT_RGBA,
    AV_PIX_FMT_BGRA
};

static int format_is_supported(enum AVPixelFormat fmt)
{
    int i;

    for (i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++)
        if (supported_formats[i] == fmt)
            return 1;
    return 0;
}
static av_cold int init(AVFilterContext *ctx) {
    SmoothNvcvContext *s = ctx->priv;

    switch(s->type) {
        case SMOOTH_TYPE_GAUSSIAN:
        default:
            CK_NVCV(cvcudaGaussianCreate(&s->smooth_handle, s->kw, s->kh, 1));
        case SMOOTH_TYPE_DEFAULT:
        case SMOOTH_TYPE_MEDIAN:
            CK_NVCV(cvcudaMedianBlurCreate(&s->smooth_handle, 1));

    }
    CK_NVCV(nvcvTensorLayoutMake("NHWC", &s->layout));
    
    return 0;
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

static int config_props(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = ctx->inputs[0];
    SmoothNvcvContext *s = (SmoothNvcvContext*)ctx->priv;

    AVHWFramesContext *in_frame_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVBufferRef *hw_device_ref = in_frame_ctx->device_ref;
    AVCUDADeviceContext *hw_ctx = (AVCUDADeviceContext*)in_frame_ctx->device_ctx->hwctx;
    // CudaFunctions *cu = hw_ctx->internal->cuda_dl;

    AVBufferRef *out_ref = NULL;
    AVHWFramesContext *out_ctx;
    CUcontext dummy;
    int ret;

    if (s->kw > in_frame_ctx->width || s->kh > in_frame_ctx->height) {
        av_log(ctx, AV_LOG_ERROR, "The kernel size cannot be bigger than the image\n");
        return EINVAL;
    }

    CHECK_CU(cuCtxPushCurrent(hw_ctx->cuda_ctx));

    out_ref = av_hwframe_ctx_alloc(hw_device_ref);
    if (!out_ref)
        return AVERROR(ENOMEM);
    out_ctx = (AVHWFramesContext*)out_ref->data;

    s->in_fmt = in_frame_ctx->sw_format;
    s->out_fmt = s->in_fmt;

    out_ctx->format = AV_PIX_FMT_CUDA;
    out_ctx->sw_format = s->out_fmt;
    out_ctx->width = in_frame_ctx->width;
    out_ctx->height = in_frame_ctx->height;

    if (!format_is_supported(s->in_fmt)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported input format: %s\n",
               av_get_pix_fmt_name(s->in_fmt));
        return AVERROR(ENOSYS);
    }
    if (!format_is_supported(s->out_fmt)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported output format: %s\n",
               av_get_pix_fmt_name(s->out_fmt));
        return AVERROR(ENOSYS);
    }

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

    CHECK_CU(cuCtxPopCurrent(&dummy));

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    SmoothNvcvContext *s = (SmoothNvcvContext*)ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    AVHWFramesContext     *frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVCUDADeviceContext *hw_ctx = (AVCUDADeviceContext*)frames_ctx->device_ctx->hwctx;
    // CudaFunctions *cu = hw_ctx->internal->cuda_dl;
    CUstream stream = hw_ctx->stream;
    CUcontext dummy;

    const AVPixFmtDescriptor *desc_src = av_pix_fmt_desc_get(s->in_fmt);
    const AVPixFmtDescriptor *desc_dst = av_pix_fmt_desc_get(s->out_fmt);
    NVCVTensorHandle   cv_in_handle, cv_out_handle;
    NVCVOperatorHandle *smooth_handle = &s->smooth_handle;
    NVCVTensorLayout   *cv_layout     = &s->layout;
    NVCVTensorData     cv_in_data = {.dtype = NVCV_DATA_TYPE_U8,
                                     .rank = 4,
                                    //  .shape = {src_h, src_w, desc_src->nb_components},
                                     .bufferType = NVCV_TENSOR_BUFFER_STRIDED_CUDA,
                                    };
    NVCVTensorData     cv_out_data = {.dtype = NVCV_DATA_TYPE_U8,
                                     .rank = 4,
                                    //  .shape = {dst_h, dst_w, desc_dst->nb_components},
                                     .bufferType = NVCV_TENSOR_BUFFER_STRIDED_CUDA,
                                    };

    int ret;
    AVFrame *out = av_frame_alloc();
    if (!out)
    {
        ret = AVERROR(ENOMEM);
        goto fail;
        return ret;
    }

    CHECK_CU(cuCtxPushCurrent(hw_ctx->cuda_ctx));
    ret = av_hwframe_get_buffer(s->hw_frames_ctx, out, 0);
    if (ret < 0)
        goto fail;

    // nvcv operation start
    if (desc_src->flags & AV_PIX_FMT_FLAG_RGB)
    {
        cv_in_data.shape[0] = 1;
        cv_in_data.shape[1] = in->height;
        cv_in_data.shape[2] = in->width;
        cv_in_data.shape[3] = desc_src->comp[0].step;
        cv_in_data.buffer.strided.strides[0] = in->height * in->linesize[0];
        cv_in_data.buffer.strided.strides[1] = in->linesize[0];
        cv_in_data.buffer.strided.strides[2] = desc_src->comp[0].step;
        cv_in_data.buffer.strided.strides[3] = 1;
        cv_in_data.buffer.strided.basePtr       = in->data[0];
        cv_in_data.layout                  = *cv_layout;

        cv_out_data.shape[0] = 1;
        cv_out_data.shape[1] = out->height;
        cv_out_data.shape[2] = out->width;
        cv_out_data.shape[3] = desc_dst->comp[0].step;
        cv_out_data.buffer.strided.strides[0] = out->height * out->linesize[0];
        cv_out_data.buffer.strided.strides[1] = out->linesize[0];
        cv_out_data.buffer.strided.strides[2] = desc_dst->comp[0].step;
        cv_out_data.buffer.strided.strides[3] = 1;
        cv_out_data.buffer.strided.basePtr       = out->data[0];
        cv_out_data.layout                  = *cv_layout;

        CK_NVCV(ret = nvcvTensorWrapDataConstruct(&cv_in_data, NULL, NULL, &cv_in_handle));
        CK_NVCV(ret = nvcvTensorWrapDataConstruct(&cv_out_data, NULL, NULL, &cv_out_handle));

        switch(s->type) {
            case SMOOTH_TYPE_GAUSSIAN:
                CK_NVCV(ret = cvcudaGaussianSubmit(*smooth_handle, stream, cv_in_handle,
                                         cv_out_handle, s->kw, s->kh, s->sigmaX, s->sigmaY, s->border_type));
            case SMOOTH_TYPE_DEFAULT:
            case SMOOTH_TYPE_MEDIAN:
                CK_NVCV(ret = cvcudaMedianBlurSubmit(*smooth_handle, stream, cv_in_handle,
                                         cv_out_handle, s->kw, s->kh));
        }
    }

    CHECK_CU(cuCtxPopCurrent(&dummy));

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
    SmoothNvcvContext *s = (SmoothNvcvContext*)(ctx->priv);
    nvcvOperatorDestroy(s->smooth_handle);
}

static const AVFilterPad smooth_nvcv_inputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
    }
};

static const AVFilterPad smooth_nvcv_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_props,
    }
};

AVFilter ff_vf_smooth_nvcv = {
    .name = "smooth_nvcv",
    .description = NULL_IF_CONFIG_SMALL("CUDA filter implementation template"),

    FILTER_INPUTS(smooth_nvcv_inputs),
    FILTER_OUTPUTS(smooth_nvcv_outputs),

    // .inputs = smooth_nvcv_inputs,
    // .outputs = smooth_nvcv_outputs,
    .priv_size = sizeof(SmoothNvcvContext),
    .priv_class = &smooth_nvcv_class,

    .init = init,
    .uninit = uninit,
    .formats.query_func =query_formats,
    .priv_size = sizeof(SmoothNvcvContext),
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE
};