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

#pragma once

#include <cuda_runtime.h>
#include <utility>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
// #include <torch/torch.h>

// #include <cuda.h>

// using namespace std;

void decode_locations(float* loc, float* priors, int num_prior, std::pair<float, float>variance, float* out_boxes,
    cudaStream_t stream=0);

void cub_dry_run(void* &cub_d_temp_storage, size_t &cub_temp_storage_bytes, size_t num_vertices);

void similar_transform(float* vertices, uint32_t vertices_num, at::Tensor& roi_box,
    void* cub_temp_storage, size_t cub_temp_storage_bytes, float* cub_out,
    cudaStream_t stream=0, uint32_t size=120);

void similar_transform_transpose(float* vertices, float** d_vertices_out, uint32_t vertices_num, at::Tensor& roi_box,
    void* cub_temp_storage, size_t cub_temp_storage_bytes, float* cub_out,
    cudaStream_t stream=0, uint32_t size=120);

template<typename T>
void crop_and_copy(T *d_in, T *d_out, int in_width, int out_width,
    int sx, int sy, int dw, int dh, int dsx, int dsy, cudaStream_t stream=0);