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

#ifndef AVFILTER_DNN_DNN_IO_PROC_H
#define AVFILTER_DNN_DNN_IO_PROC_H

#include <libavutil/log.h>
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
#include "avfilter.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"

#define NUM_TRT_IO 2

// typedef struct TensorrtContext TensorrtContext;
typedef struct TensorrtContext {
    const AVClass *av_class;

    char *engine_filename;
    int batch_size;
    void *trt_in, *trt_out;
    void *trt_io[NUM_TRT_IO];
    int trt_in_index, trt_out_index;
    int dynamic_shape;
    int in_w, in_h, out_h, out_w, channels;

    int cached, is_onnx;

    AVBufferRef *hw_frames_ctx;
    AVCUDADeviceContext *hwctx;
    AVFrame *output;

    void *trt_model;
}TensorrtContext;

void init_trt(TensorrtContext *s);
int config_props_trt(TensorrtContext *s, AVFilterLink *inlink, CudaFunctions *cu);
int filter_frame_trt(AVFilterLink *inlink, AVFrame *in);
void free_trt(TensorrtContext *s);

#endif