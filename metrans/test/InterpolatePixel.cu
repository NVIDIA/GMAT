/*
* Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#include <cuda_runtime.h>
#include "NvCodec/NvCommon.h"

static __global__ void InterpolatePixel_Kernel(cudaTextureObject_t tex, float xSrc, float ySrc, float *pDst) {
    *pDst = tex2D<float>(tex, xSrc, ySrc);
}

void InterpolatePixel(unsigned char *dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight, float xSrc, float ySrc, float *dpDst) {
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = dpSrc;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uint8_t>();
    resDesc.res.pitch2D.width = nSrcWidth;
    resDesc.res.pitch2D.height = nSrcHeight;
    resDesc.res.pitch2D.pitchInBytes = nSrcPitch;

    cudaTextureDesc texDesc = {};
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;

    cudaTextureObject_t tex=0;
    ck(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
    InterpolatePixel_Kernel<<<dim3(1, 1), dim3(1, 1)>>>(tex, xSrc, ySrc, dpDst);
    ck(cudaDestroyTextureObject(tex));
}
