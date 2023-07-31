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
 * implementing a filter to select which frame passes in the filterchain on GPU
 */
#include "config_components.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "libavutil/avstring.h"
#include "libavutil/eval.h"
#include "libavutil/fifo.h"
#include "libavutil/imgutils.h"
#include "libavutil/internal.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/cuda_check.h"
#include "avfilter.h"
#include "audio.h"
#include "formats.h"
#include "internal.h"
#include "video.h"
#include "scene_sad_cuda.h"

// #define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, s->hwctx->internal->cuda_dl, x)

static const char *const var_names[] = {
    "TB",                ///< timebase

    "pts",               ///< original pts in the file of the frame
    "start_pts",         ///< first PTS in the stream, expressed in TB units
    "prev_pts",          ///< previous frame PTS
    "prev_selected_pts", ///< previous selected frame PTS

    "t",                 ///< timestamp expressed in seconds
    "start_t",           ///< first PTS in the stream, expressed in seconds
    "prev_t",            ///< previous frame time
    "prev_selected_t",   ///< previously selected time

    "pict_type",         ///< the type of picture in the movie
    "I",
    "P",
    "B",
    "S",
    "SI",
    "SP",
    "BI",
    "PICT_TYPE_I",
    "PICT_TYPE_P",
    "PICT_TYPE_B",
    "PICT_TYPE_S",
    "PICT_TYPE_SI",
    "PICT_TYPE_SP",
    "PICT_TYPE_BI",

    "interlace_type",    ///< the frame interlace type
    "PROGRESSIVE",
    "TOPFIRST",
    "BOTTOMFIRST",

    "consumed_samples_n",///< number of samples consumed by the filter (only audio)
    "samples_n",         ///< number of samples in the current frame (only audio)
    "sample_rate",       ///< sample rate (only audio)

    "n",                 ///< frame number (starting from zero)
    "selected_n",        ///< selected frame number (starting from zero)
    "prev_selected_n",   ///< number of the last selected frame

    "key",               ///< tell if the frame is a key frame
    "pos",               ///< original position in the file of the frame

    "scene",

    "concatdec_select",  ///< frame is within the interval set by the concat demuxer

    NULL
};

enum var_name {
    VAR_TB,

    VAR_PTS,
    VAR_START_PTS,
    VAR_PREV_PTS,
    VAR_PREV_SELECTED_PTS,

    VAR_T,
    VAR_START_T,
    VAR_PREV_T,
    VAR_PREV_SELECTED_T,

    VAR_PICT_TYPE,
    VAR_I,
    VAR_P,
    VAR_B,
    VAR_S,
    VAR_SI,
    VAR_SP,
    VAR_BI,
    VAR_PICT_TYPE_I,
    VAR_PICT_TYPE_P,
    VAR_PICT_TYPE_B,
    VAR_PICT_TYPE_S,
    VAR_PICT_TYPE_SI,
    VAR_PICT_TYPE_SP,
    VAR_PICT_TYPE_BI,

    VAR_INTERLACE_TYPE,
    VAR_INTERLACE_TYPE_P,
    VAR_INTERLACE_TYPE_T,
    VAR_INTERLACE_TYPE_B,

    VAR_CONSUMED_SAMPLES_N,
    VAR_SAMPLES_N,
    VAR_SAMPLE_RATE,

    VAR_N,
    VAR_SELECTED_N,
    VAR_PREV_SELECTED_N,

    VAR_KEY,
    VAR_POS,

    VAR_SCENE,

    VAR_CONCATDEC_SELECT,

    VAR_VARS_NB
};

typedef struct SelectGPUContext {
    const AVClass *class;
    char *expr_str;
    AVExpr *expr;
    double var_values[VAR_VARS_NB];
    int bitdepth;
    int nb_planes;
    ptrdiff_t width[4];
    ptrdiff_t height[4];
    int do_scene_detect;            ///< 1 if the expression requires scene detection variables, 0 otherwise
    // ff_scene_sad_fn sad;            ///< Sum of the absolute difference function (scene detect only)
    double prev_mafd;               ///< previous MAFD                           (scene detect only)
    AVFrame *prev_picref;           ///< previous frame                          (scene detect only)
    double select;
    int select_out;                 ///< mark the selected output pad index
    int nb_outputs;
    uint64_t *d_sum;                ///< sad sum on GPU
    uint8_t **d_src1, **d_src2;     ///< GPU frame pointer array
    uint8_t *d_stride1, *d_stride2; ///< GPU frame linesize array
    uint8_t *d_width, *d_height;
} SelectGPUContext;

#define OFFSET(x) offsetof(SelectGPUContext, x)
#define DEFINE_OPTIONS(filt_name, FLAGS)                            \
static const AVOption filt_name##_options[] = {                     \
    { "expr", "set an expression to use for selecting frames", OFFSET(expr_str), AV_OPT_TYPE_STRING, { .str = "1" }, .flags=FLAGS }, \
    { "e",    "set an expression to use for selecting frames", OFFSET(expr_str), AV_OPT_TYPE_STRING, { .str = "1" }, .flags=FLAGS }, \
    { "outputs", "set the number of outputs", OFFSET(nb_outputs), AV_OPT_TYPE_INT, {.i64 = 1}, 1, INT_MAX, .flags=FLAGS }, \
    { "n",       "set the number of outputs", OFFSET(nb_outputs), AV_OPT_TYPE_INT, {.i64 = 1}, 1, INT_MAX, .flags=FLAGS }, \
    { NULL }                                                            \
}

static int request_frame(AVFilterLink *outlink);

static av_cold int init(AVFilterContext *ctx)
{
    SelectGPUContext *select = ctx->priv;
    int i, ret;

    if ((ret = av_expr_parse(&select->expr, select->expr_str,
                             var_names, NULL, NULL, NULL, NULL, 0, ctx)) < 0) {
        av_log(ctx, AV_LOG_ERROR, "Error while parsing expression '%s'\n",
               select->expr_str);
        return ret;
    }
    select->do_scene_detect = !!strstr(select->expr_str, "scene");

    for (i = 0; i < select->nb_outputs; i++) {
        AVFilterPad pad = { 0 };

        pad.name = av_asprintf("output%d", i);
        if (!pad.name)
            return AVERROR(ENOMEM);
        pad.type = ctx->filter->inputs[0].type;
        pad.request_frame = request_frame;
        if ((ret = ff_append_outpad_free_name(ctx, &pad)) < 0) {
            return ret;
        }
    }

    return 0;
}

#define INTERLACE_TYPE_P 0
#define INTERLACE_TYPE_T 1
#define INTERLACE_TYPE_B 2

static int config_input(AVFilterLink *inlink)
{
    AVHWFramesContext   *frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVCUDADeviceContext *device_hwctx = frames_ctx->device_ctx->hwctx;
    CUcontext dummy, cuda_ctx = device_hwctx->cuda_ctx;

    SelectGPUContext *select = inlink->dst->priv;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(frames_ctx->sw_format);
    int is_yuv = !(desc->flags & AV_PIX_FMT_FLAG_RGB) &&
                 (desc->flags & AV_PIX_FMT_FLAG_PLANAR) &&
                 desc->nb_components >= 3;

    // select->hw_device_ctx = device_hwctx;
    select->bitdepth = desc->comp[0].depth;
    select->nb_planes = is_yuv ? 1 : av_pix_fmt_count_planes(frames_ctx->sw_format);

    for (int plane = 0; plane < select->nb_planes; plane++) {
        ptrdiff_t line_size = av_image_get_linesize(frames_ctx->sw_format, inlink->w, plane);
        int vsub = desc->log2_chroma_h;

        select->width[plane] = line_size >> (select->bitdepth > 8);
        select->height[plane] = plane == 1 || plane == 2 ?  AV_CEIL_RSHIFT(inlink->h, vsub) : inlink->h;
    }

    select->var_values[VAR_N]          = 0.0;
    select->var_values[VAR_SELECTED_N] = 0.0;

    select->var_values[VAR_TB] = av_q2d(inlink->time_base);

    select->var_values[VAR_PREV_PTS]          = NAN;
    select->var_values[VAR_PREV_SELECTED_PTS] = NAN;
    select->var_values[VAR_PREV_SELECTED_T]   = NAN;
    select->var_values[VAR_PREV_T]            = NAN;
    select->var_values[VAR_START_PTS]         = NAN;
    select->var_values[VAR_START_T]           = NAN;

    select->var_values[VAR_I]  = AV_PICTURE_TYPE_I;
    select->var_values[VAR_P]  = AV_PICTURE_TYPE_P;
    select->var_values[VAR_B]  = AV_PICTURE_TYPE_B;
    select->var_values[VAR_SI] = AV_PICTURE_TYPE_SI;
    select->var_values[VAR_SP] = AV_PICTURE_TYPE_SP;
    select->var_values[VAR_BI] = AV_PICTURE_TYPE_BI;
    select->var_values[VAR_PICT_TYPE_I]  = AV_PICTURE_TYPE_I;
    select->var_values[VAR_PICT_TYPE_P]  = AV_PICTURE_TYPE_P;
    select->var_values[VAR_PICT_TYPE_B]  = AV_PICTURE_TYPE_B;
    select->var_values[VAR_PICT_TYPE_SI] = AV_PICTURE_TYPE_SI;
    select->var_values[VAR_PICT_TYPE_SP] = AV_PICTURE_TYPE_SP;
    select->var_values[VAR_PICT_TYPE_BI] = AV_PICTURE_TYPE_BI;

    select->var_values[VAR_INTERLACE_TYPE_P] = INTERLACE_TYPE_P;
    select->var_values[VAR_INTERLACE_TYPE_T] = INTERLACE_TYPE_T;
    select->var_values[VAR_INTERLACE_TYPE_B] = INTERLACE_TYPE_B;

    select->var_values[VAR_PICT_TYPE]         = NAN;
    select->var_values[VAR_INTERLACE_TYPE]    = NAN;
    select->var_values[VAR_SCENE]             = NAN;
    select->var_values[VAR_CONSUMED_SAMPLES_N] = NAN;
    select->var_values[VAR_SAMPLES_N]          = NAN;

    select->var_values[VAR_SAMPLE_RATE] =
        inlink->type == AVMEDIA_TYPE_AUDIO ? inlink->sample_rate : NAN;

    if (CONFIG_SELECT_FILTER && select->do_scene_detect) {
        // select->sad = ff_scene_sad_cuda;
        // if (!select->sad)
        //     return AVERROR(EINVAL);
        cuCtxPushCurrent(cuda_ctx);
        printf("Allocate pinned memory\n");
        
        cudaMalloc((void**)&select->d_sum, sizeof(uint64_t));
        cudaMallocHost((void**)&select->d_src1, select->nb_planes * sizeof(uint8_t*));
        cudaMallocHost((void**)&select->d_src2, select->nb_planes * sizeof(uint8_t*));
        cudaMallocHost((void**)&select->d_stride1, select->nb_planes * sizeof(uint8_t*));
        cudaMallocHost((void**)&select->d_stride2, select->nb_planes * sizeof(uint8_t*));
        cudaMallocHost((void**)&select->d_width, select->nb_planes * sizeof(uint8_t*));
        cudaMallocHost((void**)&select->d_height, select->nb_planes * sizeof(uint8_t*));
        
        cudaMemcpy(select->d_width, select->width, sizeof(ptrdiff_t) * select->nb_planes, cudaMemcpyDefault);
        cudaMemcpy(select->d_height, select->height, sizeof(ptrdiff_t) * select->nb_planes, cudaMemcpyDefault);

        cuCtxPopCurrent(&dummy);
    }
    return 0;
}

static double get_scene_score(AVFilterContext *ctx, AVFrame *frame)
{
    double ret = 0;
    SelectGPUContext *select = ctx->priv;
    AVFrame *prev_picref = select->prev_picref;
    AVHWDeviceContext *hw_device_ctx = (AVHWDeviceContext*)ctx->hw_device_ctx->data;
    AVCUDADeviceContext *device_hwctx = hw_device_ctx->hwctx;
    CUcontext dummy, cuda_ctx = device_hwctx->cuda_ctx;

    cuCtxPushCurrent(cuda_ctx);

    if (prev_picref &&
        frame->height == prev_picref->height &&
        frame->width  == prev_picref->width) {
        uint64_t sad = 0;
        double mafd, diff;
        uint64_t count = 0;
        uint64_t plane_sad;

        cudaMemcpy(select->d_src1, prev_picref->data, sizeof(uint8_t*) * select->nb_planes, cudaMemcpyDefault);
        cudaMemcpy(select->d_src2, frame->data, sizeof(uint8_t*) * select->nb_planes, cudaMemcpyDefault);
        cudaMemcpy(select->d_stride1, prev_picref->linesize, sizeof(int) * select->nb_planes, cudaMemcpyDefault);
        cudaMemcpy(select->d_stride2, frame->linesize, sizeof(int) * select->nb_planes, cudaMemcpyDefault);

        ff_scene_sad_cuda(select->d_src1, select->d_stride1,
                        select->d_src2, select->d_stride2,
                        select->d_width, select->d_height,
                        select->nb_planes, select->d_sum);
        cudaMemcpy(&sad, select->d_sum, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        for (int plane = 0; plane < select->nb_planes; plane++) {
            // uint64_t plane_sad;
            // select->sad(prev_picref->data[plane], prev_picref->linesize[plane],
            //         frame->data[plane], frame->linesize[plane],
            //         select->width[plane], select->height[plane], &plane_sad);
            // sad += plane_sad;
            count += select->width[plane] * select->height[plane];
        }

        emms_c();
        mafd = (double)sad / count / (1ULL << (select->bitdepth - 8));
        diff = fabs(mafd - select->prev_mafd);
        ret  = av_clipf(FFMIN(mafd, diff) / 100., 0, 1);
        select->prev_mafd = mafd;
        av_frame_free(&prev_picref);
    }
    cuCtxPopCurrent(&dummy);
    select->prev_picref = av_frame_clone(frame);
    return ret;
}

// static double get_scene_score(AVFilterContext *ctx, AVFrame *frame)
// {
//     double ret = 0;
//     SelectGPUContext *select = ctx->priv;
//     AVFrame *prev_picref = select->prev_picref;

//     AVHWFramesContext     *frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
//     AVCUDADeviceContext *device_hwctx = frames_ctx->device_ctx->hwctx;
//     CUcontext dummy, cuda_ctx = device_hwctx->cuda_ctx;
// }

static double get_concatdec_select(AVFrame *frame, int64_t pts)
{
    AVDictionary *metadata = frame->metadata;
    AVDictionaryEntry *start_time_entry = av_dict_get(metadata, "lavf.concatdec.start_time", NULL, 0);
    AVDictionaryEntry *duration_entry = av_dict_get(metadata, "lavf.concatdec.duration", NULL, 0);
    if (start_time_entry) {
        int64_t start_time = strtoll(start_time_entry->value, NULL, 10);
        if (pts >= start_time) {
            if (duration_entry) {
              int64_t duration = strtoll(duration_entry->value, NULL, 10);
              if (pts < start_time + duration)
                  return -1;
              else
                  return 0;
            }
            return -1;
        }
        return 0;
    }
    return NAN;
}

static void select_frame(AVFilterContext *ctx, AVFrame *frame)
{
    SelectGPUContext *select = ctx->priv;
    AVFilterLink *inlink = ctx->inputs[0];
    double res;

    if (isnan(select->var_values[VAR_START_PTS]))
        select->var_values[VAR_START_PTS] = TS2D(frame->pts);
    if (isnan(select->var_values[VAR_START_T]))
        select->var_values[VAR_START_T] = TS2D(frame->pts) * av_q2d(inlink->time_base);

    select->var_values[VAR_N  ] = inlink->frame_count_out;
    select->var_values[VAR_PTS] = TS2D(frame->pts);
    select->var_values[VAR_T  ] = TS2D(frame->pts) * av_q2d(inlink->time_base);
    select->var_values[VAR_POS] = frame->pkt_pos == -1 ? NAN : frame->pkt_pos;
    select->var_values[VAR_KEY] = frame->key_frame;
    select->var_values[VAR_CONCATDEC_SELECT] = get_concatdec_select(frame, av_rescale_q(frame->pts, inlink->time_base, AV_TIME_BASE_Q));

    switch (inlink->type) {
    case AVMEDIA_TYPE_AUDIO:
        select->var_values[VAR_SAMPLES_N] = frame->nb_samples;
        break;

    case AVMEDIA_TYPE_VIDEO:
        select->var_values[VAR_INTERLACE_TYPE] =
            !frame->interlaced_frame ? INTERLACE_TYPE_P :
        frame->top_field_first ? INTERLACE_TYPE_T : INTERLACE_TYPE_B;
        select->var_values[VAR_PICT_TYPE] = frame->pict_type;
        if (select->do_scene_detect) {
            char buf[32];
            select->var_values[VAR_SCENE] = get_scene_score(ctx, frame);
            // TODO: document metadata
            snprintf(buf, sizeof(buf), "%f", select->var_values[VAR_SCENE]);
            av_dict_set(&frame->metadata, "lavfi.scene_score", buf, 0);
        }
        break;
    }

    select->select = res = av_expr_eval(select->expr, select->var_values, NULL);
    av_log(inlink->dst, AV_LOG_DEBUG,
           "n:%f pts:%f t:%f key:%d",
           select->var_values[VAR_N],
           select->var_values[VAR_PTS],
           select->var_values[VAR_T],
           frame->key_frame);

    switch (inlink->type) {
    case AVMEDIA_TYPE_VIDEO:
        av_log(inlink->dst, AV_LOG_DEBUG, " interlace_type:%c pict_type:%c scene:%f",
               (!frame->interlaced_frame) ? 'P' :
               frame->top_field_first     ? 'T' : 'B',
               av_get_picture_type_char(frame->pict_type),
               select->var_values[VAR_SCENE]);
        break;
    case AVMEDIA_TYPE_AUDIO:
        av_log(inlink->dst, AV_LOG_DEBUG, " samples_n:%d consumed_samples_n:%f",
               frame->nb_samples,
               select->var_values[VAR_CONSUMED_SAMPLES_N]);
        break;
    }

    if (res == 0) {
        select->select_out = -1; /* drop */
    } else if (isnan(res) || res < 0) {
        select->select_out = 0; /* first output */
    } else {
        select->select_out = FFMIN(ceilf(res)-1, select->nb_outputs-1); /* other outputs */
    }

    av_log(inlink->dst, AV_LOG_DEBUG, " -> select:%f select_out:%d\n", res, select->select_out);

    if (res) {
        select->var_values[VAR_PREV_SELECTED_N]   = select->var_values[VAR_N];
        select->var_values[VAR_PREV_SELECTED_PTS] = select->var_values[VAR_PTS];
        select->var_values[VAR_PREV_SELECTED_T]   = select->var_values[VAR_T];
        select->var_values[VAR_SELECTED_N] += 1.0;
        if (inlink->type == AVMEDIA_TYPE_AUDIO)
            select->var_values[VAR_CONSUMED_SAMPLES_N] += frame->nb_samples;
    }

    select->var_values[VAR_PREV_PTS] = select->var_values[VAR_PTS];
    select->var_values[VAR_PREV_T]   = select->var_values[VAR_T];
}

static int filter_frame(AVFilterLink *inlink, AVFrame *frame)
{
    AVFilterContext *ctx = inlink->dst;
    SelectGPUContext *select = ctx->priv;

    select_frame(ctx, frame);
    if (select->select)
        return ff_filter_frame(ctx->outputs[select->select_out], frame);

    av_frame_free(&frame);
    return 0;
}

static int request_frame(AVFilterLink *outlink)
{
    AVFilterLink *inlink = outlink->src->inputs[0];
    int ret = ff_request_frame(inlink);
    return ret;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    SelectGPUContext *select = ctx->priv;
    int i;

    av_expr_free(select->expr);
    select->expr = NULL;

    for (i = 0; i < ctx->nb_outputs; i++)
        av_freep(&ctx->output_pads[i].name);

    if (select->do_scene_detect) {
        av_frame_free(&select->prev_picref);
    }
}

// #if CONFIG_ASELECT_FILTER

// DEFINE_OPTIONS(aselect, AV_OPT_FLAG_AUDIO_PARAM|AV_OPT_FLAG_FILTERING_PARAM);
// AVFILTER_DEFINE_CLASS(aselect);

// static av_cold int aselect_init(AVFilterContext *ctx)
// {
//     SelectGPUContext *select = ctx->priv;
//     int ret;

//     if ((ret = init(ctx)) < 0)
//         return ret;

//     if (select->do_scene_detect) {
//         av_log(ctx, AV_LOG_ERROR, "Scene detection is ignored in aselect filter\n");
//         return AVERROR(EINVAL);
//     }

//     return 0;
// }

// static const AVFilterPad avfilter_af_aselect_inputs[] = {
//     {
//         .name         = "default",
//         .type         = AVMEDIA_TYPE_AUDIO,
//         .config_props = config_input,
//         .filter_frame = filter_frame,
//     },
//     { NULL }
// };

// AVFilter ff_af_aselect = {
//     .name        = "aselect",
//     .description = NULL_IF_CONFIG_SMALL("Select audio frames to pass in output."),
//     .init        = aselect_init,
//     .uninit      = uninit,
//     .priv_size   = sizeof(SelectGPUContext),
//     .inputs      = avfilter_af_aselect_inputs,
//     .priv_class  = &aselect_class,
//     .flags       = AVFILTER_FLAG_DYNAMIC_OUTPUTS,
// };
// #endif /* CONFIG_ASELECT_FILTER */

#if CONFIG_SELECT_FILTER

static int query_formats(AVFilterContext *ctx)
{
    SelectGPUContext *select = ctx->priv;

    if (!select->do_scene_detect) {
        return ff_default_query_formats(ctx);
    } else {
        int ret;
        static const enum AVPixelFormat pix_fmts[] = {
            AV_PIX_FMT_CUDA,
            AV_PIX_FMT_NONE
        };
        AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);

        if (!fmts_list)
            return AVERROR(ENOMEM);
        ret = ff_set_common_formats(ctx, fmts_list);
        if (ret < 0)
            return ret;
    }
    return 0;
}

DEFINE_OPTIONS(select, AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM);
AVFILTER_DEFINE_CLASS(select);

static av_cold int select_cuda_init(AVFilterContext *ctx)
{
    int ret;

    if ((ret = init(ctx)) < 0)
        return ret;

    return 0;
}

static const AVFilterPad avfilter_vf_select_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input,
        .filter_frame = filter_frame,
    },
    { NULL }
};

AVFilter ff_vf_select_cuda = {
    .name          = "select_cuda",
    .description   = NULL_IF_CONFIG_SMALL("Select video frames to pass in output."),
    .init          = select_cuda_init,
    .uninit        = uninit,
    .priv_size     = sizeof(SelectGPUContext),
    .priv_class    = &select_class,
    .inputs        = avfilter_vf_select_inputs,
    FILTER_QUERY_FUNC(query_formats),
    .flags         = AVFILTER_FLAG_DYNAMIC_OUTPUTS,
};
#endif /* CONFIG_SELECT_FILTER */
