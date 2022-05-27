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

// #include <ATen/ATen.h>
// #include <ATen/AccumulateType.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>
// #include <torch/library.h>
#include <cstddef>
#include <stdint.h>
#include <stdio.h>

#include <cuda.h>
#include "pose_kernel.h"

#include "libavutil/pixfmt.h"

__constant__ float matYuv2Rgb[3][3];
__constant__ float matRgb2Yuv[3][3];

void inline GetConstants(int iMatrix, float &wr, float &wb, int &black, int &white, int &max) {
    black = 16; white = 235;
    max = 255;

    switch (iMatrix)
    {
    case AVCOL_SPC_BT709:
    default:
        wr = 0.2126f; wb = 0.0722f;
        break;

    case AVCOL_SPC_FCC:
        wr = 0.30f; wb = 0.11f;
        break;

    case AVCOL_SPC_BT470BG:
        wr = 0.2990f; wb = 0.1140f;
        break;

    case AVCOL_SPC_SMPTE240M:
        wr = 0.212f; wb = 0.087f;
        break;

    case AVCOL_SPC_BT2020_NCL:
    case AVCOL_SPC_BT2020_CL:
        wr = 0.2627f; wb = 0.0593f;
        // 10-bit only
        black = 64 << 6; white = 940 << 6;
        max = (1 << 16) - 1;
        break;
    }
}

static void SetMatYuv2Rgb(int iMatrix, CUstream stream = 0) {
    float wr, wb;
    int black, white, max;
    GetConstants(iMatrix, wr, wb, black, white, max);
    float mat[3][3] = {
        1.0f, 0.0f, (1.0f - wr) / 0.5f,
        1.0f, -wb * (1.0f - wb) / 0.5f / (1 - wb - wr), -wr * (1 - wr) / 0.5f / (1 - wb - wr),
        1.0f, (1.0f - wb) / 0.5f, 0.0f,
    };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mat[i][j] = (float)(1.0 * max / (white - black) * mat[i][j]);
        }
    }
    cudaMemcpyToSymbolAsync(matYuv2Rgb, mat, sizeof(mat), 0, cudaMemcpyHostToDevice, stream);
}

static void SetMatRgb2Yuv(int iMatrix, CUstream stream = 0) {
    float wr, wb;
    int black, white, max;
    GetConstants(iMatrix, wr, wb, black, white, max);
    float mat[3][3] = {
        wr, 1.0f - wb - wr, wb,
        -0.5f * wr / (1.0f - wb), -0.5f * (1 - wb - wr) / (1.0f - wb), 0.5f,
        0.5f, -0.5f * (1.0f - wb - wr) / (1.0f - wr), -0.5f * wb / (1.0f - wr),
    };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mat[i][j] = (float)(1.0 * (white - black) / max * mat[i][j]);
        }
    }
    cudaMemcpyToSymbolAsync(matRgb2Yuv, mat, sizeof(mat), 0, cudaMemcpyHostToDevice, stream);
}

template<class T>
__device__ static T Clamp(T x, T lower, T upper) {
    return x < lower ? lower : (x > upper ? upper : x);
}

template<class Rgb, class YuvUnit>
__device__ inline Rgb YuvToRgbForPixel(YuvUnit y, YuvUnit u, YuvUnit v) {
    const int
        low = 1 << (sizeof(YuvUnit) * 8 - 4),
        mid = 1 << (sizeof(YuvUnit) * 8 - 1);
    float fy = (int)y - low, fu = (int)u - mid, fv = (int)v - mid;
    const float maxf = (1 << sizeof(YuvUnit) * 8) - 1.0f;
    YuvUnit
        r = (YuvUnit)Clamp(matYuv2Rgb[0][0] * fy + matYuv2Rgb[0][1] * fu + matYuv2Rgb[0][2] * fv, 0.0f, maxf),
        g = (YuvUnit)Clamp(matYuv2Rgb[1][0] * fy + matYuv2Rgb[1][1] * fu + matYuv2Rgb[1][2] * fv, 0.0f, maxf),
        b = (YuvUnit)Clamp(matYuv2Rgb[2][0] * fy + matYuv2Rgb[2][1] * fu + matYuv2Rgb[2][2] * fv, 0.0f, maxf),
        a = (YuvUnit)maxf;

    Rgb rgb{};
    const int nShift = abs((int)sizeof(YuvUnit) - (int)sizeof(rgb.c.r)) * 8;
    if (sizeof(YuvUnit) >= sizeof(rgb.c.r)) {
        rgb.c.r = r >> nShift;
        rgb.c.g = g >> nShift;
        rgb.c.b = b >> nShift;
        rgb.c.a = a >> nShift;
    } else {
        rgb.c.r = r << nShift;
        rgb.c.g = g << nShift;
        rgb.c.b = b << nShift;
        rgb.c.a = a << nShift;
    }
    return rgb;
}

template<>
__device__ inline RGBAF32 YuvToRgbForPixel(uint8_t y, uint8_t u, uint8_t v) {
    const int
        low = 1 << (sizeof(uint8_t) * 8 - 4),
        mid = 1 << (sizeof(uint8_t) * 8 - 1);
    float fy = (int)y - low, fu = (int)u - mid, fv = (int)v - mid;
    const float maxf = (1 << sizeof(uint8_t) * 8) - 1.0f;
    uint8_t
        r = (uint8_t)Clamp(matYuv2Rgb[0][0] * fy + matYuv2Rgb[0][1] * fu + matYuv2Rgb[0][2] * fv, 0.0f, maxf),
        g = (uint8_t)Clamp(matYuv2Rgb[1][0] * fy + matYuv2Rgb[1][1] * fu + matYuv2Rgb[1][2] * fv, 0.0f, maxf),
        b = (uint8_t)Clamp(matYuv2Rgb[2][0] * fy + matYuv2Rgb[2][1] * fu + matYuv2Rgb[2][2] * fv, 0.0f, maxf);

    RGBAF32 rgb{};

    rgb.c.r = r / maxf;
    rgb.c.g = g / maxf;
    rgb.c.b = b / maxf;
    return rgb;
}

template<class YuvUnitx2, class Rgb, class RgbIntx2>
__global__ static void YuvToRgbKernel(uint8_t *pYuv, int nYuvPitch, uint8_t *pRgb, int nRgbPitch, int nWidth, int nHeight) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    uint8_t *pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;
    uint8_t *pDst = pRgb + x * sizeof(Rgb) + y * nRgbPitch;

    YuvUnitx2 l0 = *(YuvUnitx2 *)pSrc;
    YuvUnitx2 l1 = *(YuvUnitx2 *)(pSrc + nYuvPitch);
    YuvUnitx2 ch = *(YuvUnitx2 *)(pSrc + (nHeight - y / 2) * nYuvPitch);

    *(RgbIntx2 *)pDst = RgbIntx2 {
        YuvToRgbForPixel<Rgb>(l0.x, ch.x, ch.y).d,
        YuvToRgbForPixel<Rgb>(l0.y, ch.x, ch.y).d,
    };
    *(RgbIntx2 *)(pDst + nRgbPitch) = RgbIntx2 {
        YuvToRgbForPixel<Rgb>(l1.x, ch.x, ch.y).d,
        YuvToRgbForPixel<Rgb>(l1.y, ch.x, ch.y).d,
    };
}

template<class YuvUnitx2, class Rgb, class RgbIntx2>
__global__ static void YuvToRgbKernel(uint8_t *pYuv, int nYuvPitch, uint8_t *pRgb, int nRgbPitch, int nWidth, int nHeight, int H) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    uint8_t *pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;
    uint8_t *pDst = pRgb + x * sizeof(Rgb) + y * nRgbPitch;

    YuvUnitx2 l0 = *(YuvUnitx2 *)pSrc;
    YuvUnitx2 l1 = *(YuvUnitx2 *)(pSrc + nYuvPitch);
    YuvUnitx2 ch = *(YuvUnitx2 *)(pSrc + (H - y / 2) * nYuvPitch);

    *(RgbIntx2 *)pDst = RgbIntx2 {
        YuvToRgbForPixel<Rgb>(l0.x, ch.x, ch.y).d,
        YuvToRgbForPixel<Rgb>(l0.y, ch.x, ch.y).d,
    };
    *(RgbIntx2 *)(pDst + nRgbPitch) = RgbIntx2 {
        YuvToRgbForPixel<Rgb>(l1.x, ch.x, ch.y).d,
        YuvToRgbForPixel<Rgb>(l1.y, ch.x, ch.y).d,
    };
}

template<class YuvUnitx2, class Rgb, class RgbIntx2>
__global__ static void YuvToRgbKernel(uint8_t *pYuv, int nYuvPitch, uint8_t *pRgb, int nRgbPitch, int nWidth, int nHeight, int srcWidth, int srcHeight) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= srcWidth || y + 1 >= srcHeight) {
        return;
    }

    uint8_t *pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;
    uint8_t *pDst = pRgb + x * sizeof(Rgb) + y * nRgbPitch;

    YuvUnitx2 l0 = *(YuvUnitx2 *)pSrc;
    YuvUnitx2 l1 = *(YuvUnitx2 *)(pSrc + nYuvPitch);
    YuvUnitx2 ch = *(YuvUnitx2 *)(pSrc + (nHeight - y / 2) * nYuvPitch);

    *(RgbIntx2 *)pDst = RgbIntx2 {
        YuvToRgbForPixel<Rgb>(l0.x, ch.x, ch.y).d,
        YuvToRgbForPixel<Rgb>(l0.y, ch.x, ch.y).d,
    };
    *(RgbIntx2 *)(pDst + nRgbPitch) = RgbIntx2 {
        YuvToRgbForPixel<Rgb>(l1.x, ch.x, ch.y).d,
        YuvToRgbForPixel<Rgb>(l1.y, ch.x, ch.y).d,
    };
}

template<class YuvUnitx2, class Rgb, class RgbIntx2>
__global__ static void YuvToRgbKernel(uint8_t *pY, uint8_t *pUv, int nYuvPitch, uint8_t *pRgb, int nRgbPitch, int nWidth, int nHeight) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    uint8_t *pSrc = pY + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;
    uint8_t *pSrcUv = pUv + x * sizeof(YuvUnitx2) / 2 + y / 2 * nYuvPitch;
    uint8_t *pDst = pRgb + x * sizeof(Rgb) + y * nRgbPitch;

    YuvUnitx2 l0 = *(YuvUnitx2 *)pSrc;
    YuvUnitx2 l1 = *(YuvUnitx2 *)(pSrc + nYuvPitch);
    YuvUnitx2 ch = *(YuvUnitx2 *)pSrcUv;

    *(RgbIntx2 *)pDst = RgbIntx2 {
        YuvToRgbForPixel<Rgb>(l0.x, ch.x, ch.y).d,
        YuvToRgbForPixel<Rgb>(l0.y, ch.x, ch.y).d,
    };
    *(RgbIntx2 *)(pDst + nRgbPitch) = RgbIntx2 {
        YuvToRgbForPixel<Rgb>(l1.x, ch.x, ch.y).d,
        YuvToRgbForPixel<Rgb>(l1.y, ch.x, ch.y).d,
    };
}

template<class YuvUnitx2, class Rgb, class RgbIntx2>
__global__ static void Yuv444ToRgbKernel(uint8_t *pYuv, int nYuvPitch, uint8_t *pRgb, int nRgbPitch, int nWidth, int nHeight) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y);
    if (x + 1 >= nWidth || y  >= nHeight) {
        return;
    }

    uint8_t *pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;
    uint8_t *pDst = pRgb + x * sizeof(Rgb) + y * nRgbPitch;

    YuvUnitx2 l0 = *(YuvUnitx2 *)pSrc;
    YuvUnitx2 ch1 = *(YuvUnitx2 *)(pSrc + (nHeight * nYuvPitch));
    YuvUnitx2 ch2 = *(YuvUnitx2 *)(pSrc + (2 * nHeight * nYuvPitch));

    *(RgbIntx2 *)pDst = RgbIntx2{
        YuvToRgbForPixel<Rgb>(l0.x, ch1.x, ch2.x).d,
        YuvToRgbForPixel<Rgb>(l0.y, ch1.y, ch2.y).d,
    };
}

template<class YuvUnitx2, class Rgb, class RgbUnitx2>
__global__ static void YuvToRgbPlanarKernel(uint8_t *pYuv, int nYuvPitch, uint8_t *pRgbp, int nRgbpPitch, int nWidth, int nHeight) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    uint8_t *pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;

    YuvUnitx2 l0 = *(YuvUnitx2 *)pSrc;
    YuvUnitx2 l1 = *(YuvUnitx2 *)(pSrc + nYuvPitch);
    YuvUnitx2 ch = *(YuvUnitx2 *)(pSrc + (nHeight - y / 2) * nYuvPitch);

    Rgb rgb0 = YuvToRgbForPixel<Rgb>(l0.x, ch.x, ch.y),
        rgb1 = YuvToRgbForPixel<Rgb>(l0.y, ch.x, ch.y),
        rgb2 = YuvToRgbForPixel<Rgb>(l1.x, ch.x, ch.y),
        rgb3 = YuvToRgbForPixel<Rgb>(l1.y, ch.x, ch.y);

    uint8_t *pDst = pRgbp + x * sizeof(RgbUnitx2) / 2 + y * nRgbpPitch;
    *(RgbUnitx2 *)pDst = RgbUnitx2 {rgb0.v.x, rgb1.v.x};
    *(RgbUnitx2 *)(pDst + nRgbpPitch) = RgbUnitx2 {rgb2.v.x, rgb3.v.x};
    pDst += nRgbpPitch * nHeight;
    *(RgbUnitx2 *)pDst = RgbUnitx2 {rgb0.v.y, rgb1.v.y};
    *(RgbUnitx2 *)(pDst + nRgbpPitch) = RgbUnitx2 {rgb2.v.y, rgb3.v.y};
    pDst += nRgbpPitch * nHeight;
    *(RgbUnitx2 *)pDst = RgbUnitx2 {rgb0.v.z, rgb1.v.z};
    *(RgbUnitx2 *)(pDst + nRgbpPitch) = RgbUnitx2 {rgb2.v.z, rgb3.v.z};
}

template<class YuvUnitx2, class Rgb, class RgbUnitx2>
__global__ static void YuvToRgbPlanarKernel(uint8_t *pY, uint8_t *pUV, int yLinesize, int uvLinesize,
                                            uint8_t *pR, uint8_t *pG, uint8_t *pB, int rgbpLinesize,
                                            int nWidth, int nHeight) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    uint8_t *pSrcY = pY + x * sizeof(YuvUnitx2) / 2 + y * yLinesize;
    uint8_t *pSrcUV = pUV + x * sizeof(YuvUnitx2) / 2 + y / 2 * uvLinesize;

    YuvUnitx2 l0 = *(YuvUnitx2 *)pSrcY;
    YuvUnitx2 l1 = *(YuvUnitx2 *)(pSrcY + yLinesize);
    YuvUnitx2 ch = *(YuvUnitx2 *)(pSrcUV);

    // if (x == 0 && y == 0) printf("Y[0], U[0], V[0] = %d, %d, %d\n", l0.x, ch.x, ch.y);

    Rgb rgb0 = YuvToRgbForPixel<Rgb>(l0.x, ch.x, ch.y),
        rgb1 = YuvToRgbForPixel<Rgb>(l0.y, ch.x, ch.y),
        rgb2 = YuvToRgbForPixel<Rgb>(l1.x, ch.x, ch.y),
        rgb3 = YuvToRgbForPixel<Rgb>(l1.y, ch.x, ch.y);

    uint8_t *pDstR = pR + x * sizeof(RgbUnitx2) / 2 + y * rgbpLinesize;
    uint8_t *pDstG = pG + x * sizeof(RgbUnitx2) / 2 + y * rgbpLinesize;
    uint8_t *pDstB = pB + x * sizeof(RgbUnitx2) / 2 + y * rgbpLinesize;
    *(RgbUnitx2 *)pDstR = RgbUnitx2 {rgb0.v.x, rgb1.v.x};
    // if (x == 0 && y == 0) printf("R[0] = %f\n", rgb0.v.x);
    *(RgbUnitx2 *)(pDstR + rgbpLinesize) = RgbUnitx2 {rgb2.v.x, rgb3.v.x};
    // if (x == 0 && y == 0) printf("R[%d] = %f\n", rgbpLinesize, rgb2.v.x);
    // pDst += nRgbpPitch * nHeight;
    *(RgbUnitx2 *)pDstG = RgbUnitx2 {rgb0.v.y, rgb1.v.y};
    *(RgbUnitx2 *)(pDstG + rgbpLinesize) = RgbUnitx2 {rgb2.v.y, rgb3.v.y};
    // pDst += nRgbpPitch * nHeight;
    *(RgbUnitx2 *)pDstB = RgbUnitx2 {rgb0.v.z, rgb1.v.z};
    *(RgbUnitx2 *)(pDstB + rgbpLinesize) = RgbUnitx2 {rgb2.v.z, rgb3.v.z};
}

template<class YuvUnitx2, class Rgb, class RgbUnitx2>
__global__ static void Yuv444ToRgbPlanarKernel(uint8_t *pYuv, int nYuvPitch, uint8_t *pRgbp, int nRgbpPitch, int nWidth, int nHeight) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y);
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }

    uint8_t *pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;

    YuvUnitx2 l0 = *(YuvUnitx2 *)pSrc;
    YuvUnitx2 ch1 = *(YuvUnitx2 *)(pSrc + (nHeight * nYuvPitch));
    YuvUnitx2 ch2 = *(YuvUnitx2 *)(pSrc + (2 * nHeight * nYuvPitch));

    Rgb rgb0 = YuvToRgbForPixel<Rgb>(l0.x, ch1.x, ch2.x),
        rgb1 = YuvToRgbForPixel<Rgb>(l0.y, ch1.y, ch2.y);


    uint8_t *pDst = pRgbp + x * sizeof(RgbUnitx2) / 2 + y * nRgbpPitch;
    *(RgbUnitx2 *)pDst = RgbUnitx2{ rgb0.v.x, rgb1.v.x };

    pDst += nRgbpPitch * nHeight;
    *(RgbUnitx2 *)pDst = RgbUnitx2{ rgb0.v.y, rgb1.v.y };

    pDst += nRgbpPitch * nHeight;
    *(RgbUnitx2 *)pDst = RgbUnitx2{ rgb0.v.z, rgb1.v.z };
}

template <class COLOR32>
void Nv12ToColor32(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    YuvToRgbKernel<uchar2, COLOR32, uint2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2)>>>
        (dpNv12, nNv12Pitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

// template <class COLOR32>
// void Nv12ToColor32(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int H, int iMatrix) {
//     SetMatYuv2Rgb(iMatrix);
//     YuvToRgbKernel<uchar2, COLOR32, uint2>
//         <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2)>>>
//         (dpNv12, nNv12Pitch, dpBgra, nBgraPitch, nWidth, nHeight, H);
// }

template <class COLOR32>
void Nv12ToColor32(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int targetW, int targetH, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    YuvToRgbKernel<uchar2, COLOR32, uint2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2)>>>
        (dpNv12, nNv12Pitch, dpBgra, nBgraPitch, nWidth, nHeight, targetW, targetH);
}

template <class COLOR32>
void Nv12ToColor32(uint8_t **dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    YuvToRgbKernel<uchar2, COLOR32, uint2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2)>>>
        (dpNv12[0], dpNv12[1], nNv12Pitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR64>
void Nv12ToColor64(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    YuvToRgbKernel<uchar2, COLOR64, ulonglong2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2)>>>
        (dpNv12, nNv12Pitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR32>
void YUV444ToColor32(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    Yuv444ToRgbKernel<uchar2, COLOR32, uint2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2), dim3(32, 2) >>>
        (dpYUV444, nPitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR64>
void YUV444ToColor64(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    Yuv444ToRgbKernel<uchar2, COLOR64, ulonglong2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2), dim3(32, 2) >>>
        (dpYUV444, nPitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR32>
void P016ToColor32(uint8_t *dpP016, int nP016Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    YuvToRgbKernel<ushort2, COLOR32, uint2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2)>>>
        (dpP016, nP016Pitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR64>
void P016ToColor64(uint8_t *dpP016, int nP016Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    YuvToRgbKernel<ushort2, COLOR64, ulonglong2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2)>>>
        (dpP016, nP016Pitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR32>
void YUV444P16ToColor32(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    Yuv444ToRgbKernel<ushort2, COLOR32, uint2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2), dim3(32, 2) >>>
        (dpYUV444, nPitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR64>
void YUV444P16ToColor64(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    Yuv444ToRgbKernel<ushort2, COLOR64, ulonglong2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2), dim3(32, 2) >>>
        (dpYUV444, nPitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR32>
void Nv12ToColorPlanar(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    YuvToRgbPlanarKernel<uchar2, COLOR32, uchar2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2)>>>
        (dpNv12, nNv12Pitch, dpBgrp, nBgrpPitch, nWidth, nHeight);
}

template <class COLOR32>
void P016ToColorPlanar(uint8_t *dpP016, int nP016Pitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    YuvToRgbPlanarKernel<ushort2, COLOR32, uchar2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2)>>>
        (dpP016, nP016Pitch, dpBgrp, nBgrpPitch, nWidth, nHeight);
}

template <class COLOR32>
void YUV444ToColorPlanar(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    Yuv444ToRgbPlanarKernel<uchar2, COLOR32, uchar2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2), dim3(32, 2) >>>
        (dpYUV444, nPitch, dpBgrp, nBgrpPitch, nWidth, nHeight);
}

template <class COLOR32>
void YUV444P16ToColorPlanar(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    Yuv444ToRgbPlanarKernel<ushort2, COLOR32, uchar2>
        << <dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2), dim3(32, 2) >> >
        (dpYUV444, nPitch, dpBgrp, nBgrpPitch, nWidth, nHeight);
}

// Explicit Instantiation
template void Nv12ToColor32<BGRA32>(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void Nv12ToColor32<RGBA32>(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void Nv12ToColor32<RGBA32>(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int srcWidth, int srcHeight, int iMatrix);
template void Nv12ToColor32<RGBA32>(uint8_t **dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void Nv12ToColor64<BGRA64>(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void Nv12ToColor64<RGBA64>(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444ToColor32<BGRA32>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444ToColor32<RGBA32>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444ToColor64<BGRA64>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444ToColor64<RGBA64>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void P016ToColor32<BGRA32>(uint8_t *dpP016, int nP016Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void P016ToColor32<RGBA32>(uint8_t *dpP016, int nP016Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void P016ToColor64<BGRA64>(uint8_t *dpP016, int nP016Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void P016ToColor64<RGBA64>(uint8_t *dpP016, int nP016Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444P16ToColor32<BGRA32>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444P16ToColor32<RGBA32>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444P16ToColor64<BGRA64>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444P16ToColor64<RGBA64>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void Nv12ToColorPlanar<BGRA32>(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix);
template void Nv12ToColorPlanar<RGBA32>(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix);
template void P016ToColorPlanar<BGRA32>(uint8_t *dpP016, int nP016Pitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix);
template void P016ToColorPlanar<RGBA32>(uint8_t *dpP016, int nP016Pitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444ToColorPlanar<BGRA32>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444ToColorPlanar<RGBA32>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444P16ToColorPlanar<BGRA32>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444P16ToColorPlanar<RGBA32>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix);

template<class YuvUnit, class RgbUnit>
__device__ inline YuvUnit RgbToY(RgbUnit r, RgbUnit g, RgbUnit b) {
    const YuvUnit low = 1 << (sizeof(YuvUnit) * 8 - 4);
    return matRgb2Yuv[0][0] * r + matRgb2Yuv[0][1] * g + matRgb2Yuv[0][2] * b + low;
}

template<class YuvUnit, class RgbUnit>
__device__ inline YuvUnit RgbToU(RgbUnit r, RgbUnit g, RgbUnit b) {
    const YuvUnit mid = 1 << (sizeof(YuvUnit) * 8 - 1);
    return matRgb2Yuv[1][0] * r + matRgb2Yuv[1][1] * g + matRgb2Yuv[1][2] * b + mid;
}

template<class YuvUnit, class RgbUnit>
__device__ inline YuvUnit RgbToV(RgbUnit r, RgbUnit g, RgbUnit b) {
    const YuvUnit mid = 1 << (sizeof(YuvUnit) * 8 - 1);
    return matRgb2Yuv[2][0] * r + matRgb2Yuv[2][1] * g + matRgb2Yuv[2][2] * b + mid;
}

template<class YuvUnitx2, class Rgb, class RgbIntx2>
__global__ static void RgbToYuvKernel(uint8_t *pRgb, int nRgbPitch, uint8_t *pYuv, int nYuvPitch, int nWidth, int nHeight) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    uint8_t *pSrc = pRgb + x * sizeof(Rgb) + y * nRgbPitch;
    RgbIntx2 int2a = *(RgbIntx2 *)pSrc;
    RgbIntx2 int2b = *(RgbIntx2 *)(pSrc + nRgbPitch);

    Rgb rgb[4] = {int2a.x, int2a.y, int2b.x, int2b.y};
    decltype(Rgb::c.r)
        r = (rgb[0].c.r + rgb[1].c.r + rgb[2].c.r + rgb[3].c.r) / 4,
        g = (rgb[0].c.g + rgb[1].c.g + rgb[2].c.g + rgb[3].c.g) / 4,
        b = (rgb[0].c.b + rgb[1].c.b + rgb[2].c.b + rgb[3].c.b) / 4;

    uint8_t *pDst = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;
    *(YuvUnitx2 *)pDst = YuvUnitx2 {
        RgbToY<decltype(YuvUnitx2::x)>(rgb[0].c.r, rgb[0].c.g, rgb[0].c.b),
        RgbToY<decltype(YuvUnitx2::x)>(rgb[1].c.r, rgb[1].c.g, rgb[1].c.b),
    };
    *(YuvUnitx2 *)(pDst + nYuvPitch) = YuvUnitx2 {
        RgbToY<decltype(YuvUnitx2::x)>(rgb[2].c.r, rgb[2].c.g, rgb[2].c.b),
        RgbToY<decltype(YuvUnitx2::x)>(rgb[3].c.r, rgb[3].c.g, rgb[3].c.b),
    };
    *(YuvUnitx2 *)(pDst + (nHeight - y / 2) * nYuvPitch) = YuvUnitx2 {
        RgbToU<decltype(YuvUnitx2::x)>(r, g, b),
        RgbToV<decltype(YuvUnitx2::x)>(r, g, b),
    };
}


static __device__ float2 operator*(float2 x, float y) { return make_float2(x.x * y, x.y * y); }
template<class YuvUnitx2, class Rgb, class RgbIntx2>
__global__ static void RgbpToYuvKernel(uint8_t *pRgb, int nRgbPitch, uint8_t *pYuv, int nYuvPitch, int nWidth, int nHeight) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    float max = (1 << sizeof(YuvUnitx2::x) * 8) - 1.0f;
    // if (x == 0 && y == 0) printf("max = %f\n", max);

    // uint8_t *pSrc = pRgb + x * sizeof(Rgb) + y * nRgbPitch;
    // RgbIntx2 int2a = *(RgbIntx2 *)pSrc;
    // RgbIntx2 int2b = *(RgbIntx2 *)(pSrc + nRgbPitch);

    uint8_t *pSrcR = pRgb + x * sizeof(Rgb::c.r) + y * nRgbPitch;
    uint8_t *pSrcG = pRgb + nHeight * nRgbPitch + x * sizeof(Rgb::c.r) + y * nRgbPitch;
    uint8_t *pSrcB = pRgb + nHeight * nRgbPitch * 2 + x * sizeof(Rgb::c.r) + y * nRgbPitch;

    RgbIntx2 int2aR = *(RgbIntx2 *)pSrcR * max, int2bR = *(RgbIntx2 *)(pSrcR + nRgbPitch) * max;
    RgbIntx2 int2aG = *(RgbIntx2 *)pSrcG * max, int2bG = *(RgbIntx2 *)(pSrcG + nRgbPitch) * max;
    RgbIntx2 int2aB = *(RgbIntx2 *)pSrcB * max, int2bB = *(RgbIntx2 *)(pSrcB + nRgbPitch) * max;

    // if (x == 0 && y == 0) printf("R[0], G[0], B[0] = %f, %f, %f\n", int2aR.x, int2aG.x, int2aB.x);

    // Rgb rgb[4] = {int2a.x, int2a.y, int2b.x, int2b.y};
    // decltype(Rgb::c.r)
    //     r = (rgb[0].c.r + rgb[1].c.r + rgb[2].c.r + rgb[3].c.r) / 4,
    //     g = (rgb[0].c.g + rgb[1].c.g + rgb[2].c.g + rgb[3].c.g) / 4,
    //     b = (rgb[0].c.b + rgb[1].c.b + rgb[2].c.b + rgb[3].c.b) / 4;

    decltype(Rgb::c.r)
        r = (int2aR.x + int2aR.y + int2bR.x + int2bR.y) / 4,
        g = (int2aG.x + int2aG.y + int2bG.x + int2bG.y) / 4,
        b = (int2aB.x + int2aB.y + int2bB.x + int2bB.y) / 4;

    uint8_t *pDst = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;
    *(YuvUnitx2 *)pDst = YuvUnitx2 {
        RgbToY<decltype(YuvUnitx2::x)>(int2aR.x, int2aG.x, int2aB.x),
        RgbToY<decltype(YuvUnitx2::x)>(int2aR.y, int2aG.y, int2aB.y),
    };
    *(YuvUnitx2 *)(pDst + nYuvPitch) = YuvUnitx2 {
        RgbToY<decltype(YuvUnitx2::x)>(int2bR.x, int2bG.x, int2bB.x),
        RgbToY<decltype(YuvUnitx2::x)>(int2bR.y, int2bG.y, int2bG.y),
    };
    *(YuvUnitx2 *)(pDst + (nHeight - y / 2) * nYuvPitch) = YuvUnitx2 {
        RgbToU<decltype(YuvUnitx2::x)>(r, g, b),
        RgbToV<decltype(YuvUnitx2::x)>(r, g, b),
    };
}

// void Bgra64ToP016(uint8_t *dpBgra, int nBgraPitch, uint8_t *dpP016, int nP016Pitch, int nWidth, int nHeight, int iMatrix) {
//     SetMatRgb2Yuv(iMatrix);
//     RgbToYuvKernel<ushort2, BGRA64, ulonglong2>
//         <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2)>>>
//         (dpBgra, nBgraPitch, dpP016, nP016Pitch, nWidth, nHeight);
// }
void nv12_to_rgbpf32(CUstream stream, uint8_t **dp_nv12, int *nv12_pitch, uint8_t **dp_rgbpf32, int rgbpf32_pitch, int width, int height, int matrix)
{
    SetMatYuv2Rgb(matrix, stream);
    YuvToRgbPlanarKernel<uchar2, RGBAF32, float2>
    <<<dim3((width + 63) / 32 / 2, (height + 3) / 2 / 2), dim3(32, 2), 0, stream>>>
    (dp_nv12[0], dp_nv12[1], nv12_pitch[0], nv12_pitch[1], dp_rgbpf32[0], dp_rgbpf32[1],
        dp_rgbpf32[2], rgbpf32_pitch, width, height);
}

// void nv12_to_rgbpf32chw(CUStream stream, uint8_t **dp_nv12, int *nv12_pitch, uint8_t **dp_rgbpf32, int *rgbpf32_pitch, int width, int height, int matrix)
void rgbpf32_to_nv12(CUstream stream, uint8_t **dp_rgbpf32, int *rgbpf32_pitch, uint8_t **dp_nv12, int *nv12_pitch, int width, int height, int matrix)
{
    SetMatRgb2Yuv(matrix, stream);
    RgbpToYuvKernel<uchar2, RGBAF32, float2>
        <<<dim3((width + 63) / 32 / 2, (height + 3) / 2 / 2), dim3(32, 2), 0 , stream>>>
        (dp_rgbpf32[0], rgbpf32_pitch[0], dp_nv12[0], nv12_pitch[0], width, height);
}

// #define CUDA_1D_KERNEL_LOOP(i, n)                                \
// for (int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < (n); \
//         i += (blockDim.x * gridDim.x))

// template <typename integer>
// constexpr __host__ __device__ inline integer ceil_div(integer n, integer m) {
// return (n + m - 1) / m;
// }
// int const threadsPerBlock = sizeof(unsigned long long) * 8;

// template <typename T>
// __device__ inline bool devIoU(
//     T const* const a,
//     T const* const b,
//     const float threshold) {
//   T left = max(a[0], b[0]), right = min(a[2], b[2]);
//   T top = max(a[1], b[1]), bottom = min(a[3], b[3]);
//   T width = max(right - left, (T)0), height = max(bottom - top, (T)0);
//   using acc_T = at::acc_type<T, /*is_cuda=*/true>;
//   acc_T interS = (acc_T)width * height;
//   acc_T Sa = ((acc_T)a[2] - a[0]) * (a[3] - a[1]);
//   acc_T Sb = ((acc_T)b[2] - b[0]) * (b[3] - b[1]);
//   return (interS / (Sa + Sb - interS)) > threshold;
// }

// template <typename T>
// __global__ void nms_kernel_impl(
//     int n_boxes,
//     double iou_threshold,
//     const T* dev_boxes,
//     unsigned long long* dev_mask) {
//   const int row_start = blockIdx.y;
//   const int col_start = blockIdx.x;

//   if (row_start > col_start)
//     return;

//   const int row_size =
//       min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
//   const int col_size =
//       min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

//   __shared__ T block_boxes[threadsPerBlock * 4];
//   if (threadIdx.x < col_size) {
//     block_boxes[threadIdx.x * 4 + 0] =
//         dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 0];
//     block_boxes[threadIdx.x * 4 + 1] =
//         dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 1];
//     block_boxes[threadIdx.x * 4 + 2] =
//         dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 2];
//     block_boxes[threadIdx.x * 4 + 3] =
//         dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 3];
//   }
//   __syncthreads();

//   if (threadIdx.x < row_size) {
//     const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
//     const T* cur_box = dev_boxes + cur_box_idx * 4;
//     int i = 0;
//     unsigned long long t = 0;
//     int start = 0;
//     if (row_start == col_start) {
//       start = threadIdx.x + 1;
//     }
//     for (i = start; i < col_size; i++) {
//       if (devIoU<T>(cur_box, block_boxes + i * 4, iou_threshold)) {
//         t |= 1ULL << i;
//       }
//     }
//     const int col_blocks = ceil_div(n_boxes, threadsPerBlock);
//     dev_mask[cur_box_idx * col_blocks + col_start] = t;
//   }
// }

// at::Tensor nms_kernel_gpu(
//     const at::Tensor& dets,
//     const at::Tensor& scores,
//     double iou_threshold) {
//   TORCH_CHECK(dets.is_cuda(), "dets must be a CUDA tensor");
//   TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");

//   TORCH_CHECK(
//       dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
//   TORCH_CHECK(
//       dets.size(1) == 4,
//       "boxes should have 4 elements in dimension 1, got ",
//       dets.size(1));
//   TORCH_CHECK(
//       scores.dim() == 1,
//       "scores should be a 1d tensor, got ",
//       scores.dim(),
//       "D");
//   TORCH_CHECK(
//       dets.size(0) == scores.size(0),
//       "boxes and scores should have same number of elements in ",
//       "dimension 0, got ",
//       dets.size(0),
//       " and ",
//       scores.size(0))

//   at::cuda::CUDAGuard device_guard(dets.device());

//   if (dets.numel() == 0) {
//     return at::empty({0}, dets.options().dtype(at::kLong));
//   }

//   auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
//   auto dets_sorted = dets.index_select(0, order_t).contiguous();

//   int dets_num = dets.size(0);

//   const int col_blocks = ceil_div(dets_num, threadsPerBlock);

//   at::Tensor mask =
//       at::empty({dets_num * col_blocks}, dets.options().dtype(at::kLong));

//   dim3 blocks(col_blocks, col_blocks);
//   dim3 threads(threadsPerBlock);
//   cudaStream_t stream = at::cuda::getCurrentCUDAStream();

//   AT_DISPATCH_FLOATING_TYPES_AND_HALF(
//       dets_sorted.scalar_type(), "nms_kernel_gpu", [&] {
//         nms_kernel_impl<scalar_t><<<blocks, threads, 0, stream>>>(
//             dets_num,
//             iou_threshold,
//             dets_sorted.data_ptr<scalar_t>(),
//             (unsigned long long*)mask.data_ptr<int64_t>());
//       });

//   at::Tensor mask_cpu = mask.to(at::kCPU);
//   unsigned long long* mask_host =
//       (unsigned long long*)mask_cpu.data_ptr<int64_t>();

//   std::vector<unsigned long long> remv(col_blocks);
//   memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

//   at::Tensor keep =
//       at::empty({dets_num}, dets.options().dtype(at::kLong).device(at::kCPU));
//   int64_t* keep_out = keep.data_ptr<int64_t>();

//   int num_to_keep = 0;
//   for (int i = 0; i < dets_num; i++) {
//     int nblock = i / threadsPerBlock;
//     int inblock = i % threadsPerBlock;

//     if (!(remv[nblock] & (1ULL << inblock))) {
//       keep_out[num_to_keep++] = i;
//       unsigned long long* p = mask_host + i * col_blocks;
//       for (int j = nblock; j < col_blocks; j++) {
//         remv[j] |= p[j];
//       }
//     }
//   }

//   AT_CUDA_CHECK(cudaGetLastError());
//   return order_t.index(
//       {keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep)
//            .to(order_t.device(), keep.scalar_type())});
// }

// TORCH_LIBRARY_IMPL(torchvision, CUDA, m) {
//   m.impl(TORCH_SELECTIVE_NAME("torchvision::nms"), TORCH_FN(nms_kernel_gpu));
// }