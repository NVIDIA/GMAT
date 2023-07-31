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
 * CUDA Scene SAD functions
 */
#include <cstdint>

extern "C" {
#include "scene_sad_cuda.h"
}

#include <cub/cub.cuh>
#include <cuda_runtime.h>

template<class T>
__global__ static void scene_sad_kernel(GPU_SCENE_SAD_PARAMS) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    // int plane = blockIdx.z;

    uint64_t sad = 0;

    #pragma unroll
    for (int i = 0; i < nb_planes; i++) {

        if (idx >= width[i] || idy >= height[i]) continue;

        int linesize1 = stride1[i];
        int linesize2 = stride2[i];
        sad += abs(reinterpret_cast<T>(src1[i][idx + linesize1 * idy]) -
            reinterpret_cast<T>(src2[i][idx + linesize2 * idy]));
    }

    typedef cub::BlockReduce<int, 32, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 16> BlockReduce;
    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduce::TempStorage temp_storage;
    // Compute the block-wide sum for thread0
    unsigned long long int aggregate = BlockReduce(temp_storage).Sum(sad);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(sum, aggregate);
    }
}

template<uint8_t> __global__ static void scene_sad_kernel(GPU_SCENE_SAD_PARAMS);

extern "C" {
void ff_scene_sad_cuda(GPU_SCENE_SAD_PARAMS) {
    dim3 blockSize(32, 16);
    dim3 gridSize((width[0] - 1) / 32 + 1, (height[0] - 1) / 16 + 1);

    scene_sad_kernel<uint8_t>
    <<<gridSize, blockSize>>>(src1, stride1, src2, stride2, width, height, nb_planes, sum);
}
}