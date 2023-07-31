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
#include "NvCommon.h"

template<typename YuvUnitx2>
static __global__ void Scale(cudaTextureObject_t texY, cudaTextureObject_t texUv,
        uint8_t *pDst, int nDstPitch, int nDstWidth, int nDstHeight,
        float fxScale, float fyScale) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x,
        iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= nDstWidth / 2 || iy >= nDstHeight / 2) {
        return;
    }

    int x = ix * 2, y = iy * 2;
    float fx = x + 0.5f, fy = y + 0.5f;
    typedef decltype(YuvUnitx2::x) YuvUnit;
    const int MAX = (1 << (sizeof(YuvUnit) * 8)) - 1;
    *(YuvUnitx2 *)(pDst + y * nDstPitch + x * sizeof(YuvUnit)) = YuvUnitx2 {
        (YuvUnit)(tex2D<float>(texY, fx * fxScale, fy * fyScale) * MAX),
        (YuvUnit)(tex2D<float>(texY, (fx + 1) * fxScale, fy * fyScale) * MAX)
    };
    y++;
    fy += 1.0f;
    *(YuvUnitx2 *)(pDst + y * nDstPitch + x * sizeof(YuvUnit)) = YuvUnitx2 {
        (YuvUnit)(tex2D<float>(texY, fx * fxScale, fy * fyScale) * MAX),
        (YuvUnit)(tex2D<float>(texY, (fx + 1) * fxScale, fy * fyScale) * MAX)
    };
    float2 uv = tex2D<float2>(texUv, (ix + 0.5f) * fxScale, (nDstHeight + iy + 0.5f) * fyScale);
    *(YuvUnitx2 *)(pDst + (nDstHeight + iy) * nDstPitch + ix * 2 * sizeof(YuvUnit)) = YuvUnitx2 {(YuvUnit)(uv.x * MAX), (YuvUnit)(uv.y * MAX)};
}

template <typename YuvUnitx2>
static void Scale(unsigned char *dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight, unsigned char *dpDst, int nDstPitch, int nDstWidth, int nDstHeight) {
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = dpSrc;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<decltype(YuvUnitx2::x)>();
    resDesc.res.pitch2D.width = nSrcWidth;
    resDesc.res.pitch2D.height = nSrcHeight;
    resDesc.res.pitch2D.pitchInBytes = nSrcPitch;

    cudaTextureDesc texDesc = {};
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;

    cudaTextureObject_t texY=0;
    ck(cudaCreateTextureObject(&texY, &resDesc, &texDesc, NULL));

    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<YuvUnitx2>();
    resDesc.res.pitch2D.width = nSrcWidth / 2;
    resDesc.res.pitch2D.height = nSrcHeight * 3 / 2;

    cudaTextureObject_t texUv=0;
    ck(cudaCreateTextureObject(&texUv, &resDesc, &texDesc, NULL));

    Scale<YuvUnitx2><<<dim3((nDstWidth + 31)/32, (nDstHeight + 31)/32), dim3(16, 16)>>>(texY, texUv, dpDst, 
        nDstPitch, nDstWidth, nDstHeight, 1.0f * nSrcWidth / nDstWidth, 1.0f * nSrcHeight / nDstHeight);

    ck(cudaDestroyTextureObject(texY));
    ck(cudaDestroyTextureObject(texUv));
}

void ScaleNv12(unsigned char *dpSrcNv12, int nSrcPitch, int nSrcWidth, int nSrcHeight, unsigned char *dpDstNv12, int nDstPitch, int nDstWidth, int nDstHeight) {
    return Scale<uchar2>(dpSrcNv12, nSrcPitch, nSrcWidth, nSrcHeight, dpDstNv12, nDstPitch, nDstWidth, nDstHeight);
}

void ScaleP016(unsigned char *dpSrcP016, int nSrcPitch, int nSrcWidth, int nSrcHeight, unsigned char *dpDstP016, int nDstPitch, int nDstWidth, int nDstHeight) {
    return Scale<ushort2>(dpSrcP016, nSrcPitch, nSrcWidth, nSrcHeight, dpDstP016, nDstPitch, nDstWidth, nDstHeight);
}

static __device__ float BicubicCoefficient(float d) {
    d = abs(d);
    const float a = -0.5f;
    return d > 2.0f ? 0 : (d > 1.0f ? a * d * d * d - 5.0f * a * d * d + 8.0f * a * d - 4.0f * a : (a + 2.0f) * d * d * d - (a + 3.0f) * d * d + 1.0f); 
}

static __device__ uint8_t BicubicLuma(unsigned char *dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight, float fxSrc, float fySrc) {
    int sx0 = (int)fxSrc - 1, sy0 = (int)fySrc - 1;
    int sx[] = {sx0, sx0+1, sx0+2, sx0+3};
    int sy[] = {sy0, sy0+1, sy0+2, sy0+3};
    float cx[4], cy[4];
    for (int i = 0; i < 4; i++) {
        cx[i] = BicubicCoefficient(sx[i] - fxSrc);
        cy[i] = BicubicCoefficient(sy[i] - fySrc);
    }
    float r = 0;
    for (int y = 0; y < 4; y++) {
        float rx = 0;
        for (int x = 0; x < 4; x++) {
            rx += dpSrc[nSrcPitch * sy[y] + sx[x]] * cx[x];
        }
        r += rx * cy[y];
    }
    return (uint8_t)max(min(r, 255.0f), 0.0f);
}

static __device__ uchar2 BicubicChroma(uchar2 *dpSrc, int nSrcPitchWidth, int nSrcWidth, int nSrcHeight, float fxSrc, float fySrc) {
    int sx0 = (int)fxSrc - 1, sy0 = (int)fySrc - 1;
    int sx[] = {sx0, sx0+1, sx0+2, sx0+3};
    int sy[] = {sy0, sy0+1, sy0+2, sy0+3};
    float cx[4], cy[4];
    for (int i = 0; i < 4; i++) {
        cx[i] = BicubicCoefficient(sx[i] - fxSrc);
        cy[i] = BicubicCoefficient(sy[i] - fySrc);
    }
    float ru = 0, rv = 0;
    for (int y = 0; y < 4; y++) {
        float rux = 0, rvx = 0;
        for (int x = 0; x < 4; x++) {
            uchar2 uv = dpSrc[nSrcPitchWidth * sy[y] + sx[x]];
            rux += uv.x * cx[x];
            rvx += uv.y * cx[x];
        }
        ru += rux * cy[y];
        rv += rvx * cy[y];
    }
    return uchar2{(uint8_t)max(min(ru, 255.0f), 0.0f), (uint8_t)max(min(rv, 255.0f), 0.0f)};
}

static __global__ void ScaleNv12_Bicubic_Kernel(unsigned char *dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight, unsigned char *dpDst, int nDstPitch, int nDstWidth, int nDstHeight) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x,
        iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= nDstWidth / 2 || iy >= nDstHeight / 2) {
        return;
    }

    int x = ix * 2, y = iy * 2;
    const float fxScale = (float)nSrcWidth / nDstWidth, fyScale = (float)nSrcHeight / nDstHeight;
    *(uchar2 *)(dpDst + y * nDstPitch + x * sizeof(uchar2::x)) = uchar2 {
        BicubicLuma(dpSrc, nSrcPitch, nSrcWidth, nSrcHeight, min(max(x * fxScale, 2.0f), (float)(nSrcWidth - 2)), min(max(y * fyScale, 2.0f), (float)(nSrcHeight - 2))),
        BicubicLuma(dpSrc, nSrcPitch, nSrcWidth, nSrcHeight, min(max((x + 1) * fxScale, 2.0f), (float)(nSrcWidth - 2)), min(max(y * fyScale, 2.0f), (float)(nSrcHeight - 2))),
    };
    y++;
    *(uchar2 *)(dpDst + y * nDstPitch + x * sizeof(uchar2::x)) = uchar2 {
        BicubicLuma(dpSrc, nSrcPitch, nSrcWidth, nSrcHeight, min(max(x * fxScale, 2.0f), (float)(nSrcWidth - 2)), min(max(y * fyScale, 2.0f), (float)(nSrcHeight - 2))),
        BicubicLuma(dpSrc, nSrcPitch, nSrcWidth, nSrcHeight, min(max((x + 1) * fxScale, 2.0f), (float)(nSrcWidth - 2)), min(max(y * fyScale, 2.0f), (float)(nSrcHeight - 2))),
    };
    *(uchar2 *)(dpDst + (nDstHeight + iy) * nDstPitch + ix * 2 * sizeof(uchar2::x)) = BicubicChroma(
        (uchar2 *)(dpSrc + nSrcHeight * nSrcPitch), nSrcPitch / 2, nSrcWidth / 2, nSrcHeight / 2, 
        min(max(ix * fxScale, 2.0f), (float)(nSrcWidth / 2 - 2)), 
        min(max(iy * fyScale, 2.0f), (float)(nSrcHeight / 2 - 2))
    );
}

void ScaleNv12_Bicubic(unsigned char *dpSrcNv12, int nSrcPitch, int nSrcWidth, int nSrcHeight, unsigned char *dpDstNv12, int nDstPitch, int nDstWidth, int nDstHeight) {
    return ScaleNv12_Bicubic_Kernel<<<dim3((nDstWidth + 31) / 32, (nDstHeight +  31) / 32), dim3(16, 16)>>>(dpSrcNv12, nSrcPitch, nSrcWidth, nSrcHeight, dpDstNv12, nDstPitch, nDstWidth, nDstHeight);
}
