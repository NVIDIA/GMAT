/*
* Copyright 2017-2020 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#include <cstddef>
#include <cstdint>
#include <stdint.h>
#include <stdio.h>
#ifdef __cplusplus
extern "C"
{
#endif
#include <cuda.h>

#include "libavutil/pixfmt.h"
#ifdef __cplusplus
}
#endif
#include "format_cuda.h"

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

void SetMatYuv2Rgb(int iMatrix, CUstream stream = 0) {
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

void SetMatRgb2Yuv(int iMatrix, CUstream stream = 0) {
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
__device__ static inline Rgb YuvToRgbForPixel(YuvUnit y, YuvUnit u, YuvUnit v) {
    const int
        low = 1 << (sizeof(YuvUnit) * 8 - 4),
        mid = 1 << (sizeof(YuvUnit) * 8 - 1);
    float fy = (int)y - low, fu = (int)u - mid, fv = (int)v - mid;
    const float maxf = (1 << sizeof(YuvUnit) * 8) - 1.0f;
    YuvUnit
        r = (YuvUnit)Clamp(matYuv2Rgb[0][0] * fy + matYuv2Rgb[0][1] * fu + matYuv2Rgb[0][2] * fv, 0.0f, maxf),
        g = (YuvUnit)Clamp(matYuv2Rgb[1][0] * fy + matYuv2Rgb[1][1] * fu + matYuv2Rgb[1][2] * fv, 0.0f, maxf),
        b = (YuvUnit)Clamp(matYuv2Rgb[2][0] * fy + matYuv2Rgb[2][1] * fu + matYuv2Rgb[2][2] * fv, 0.0f, maxf);

    Rgb rgb{};
    const int nShift = abs((int)sizeof(YuvUnit) - (int)sizeof(rgb.c.r)) * 8;
    if (sizeof(YuvUnit) >= sizeof(rgb.c.r)) {
        rgb.c.r = r >> nShift;
        rgb.c.g = g >> nShift;
        rgb.c.b = b >> nShift;
    } else {
        rgb.c.r = r << nShift;
        rgb.c.g = g << nShift;
        rgb.c.b = b << nShift;
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

    rgb.c.r = r;
    rgb.c.g = g;
    rgb.c.b = b;
    return rgb;
}

template<>
__device__ inline BGRAF32 YuvToRgbForPixel(uint8_t y, uint8_t u, uint8_t v) {
    const int
        low = 1 << (sizeof(uint8_t) * 8 - 4),
        mid = 1 << (sizeof(uint8_t) * 8 - 1);
    float fy = (int)y - low, fu = (int)u - mid, fv = (int)v - mid;
    const float maxf = (1 << sizeof(uint8_t) * 8) - 1.0f;
    uint8_t
        r = (uint8_t)Clamp(matYuv2Rgb[0][0] * fy + matYuv2Rgb[0][1] * fu + matYuv2Rgb[0][2] * fv, 0.0f, maxf),
        g = (uint8_t)Clamp(matYuv2Rgb[1][0] * fy + matYuv2Rgb[1][1] * fu + matYuv2Rgb[1][2] * fv, 0.0f, maxf),
        b = (uint8_t)Clamp(matYuv2Rgb[2][0] * fy + matYuv2Rgb[2][1] * fu + matYuv2Rgb[2][2] * fv, 0.0f, maxf);

    BGRAF32 rgb{};

    rgb.c.r = r;
    rgb.c.g = g;
    rgb.c.b = b;
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

template<class Rgb>
__device__ static inline Rgb normalizePixel(Rgb pixel, float norm, float3 shift){
    Rgb rgb{};
    rgb.c.r = (pixel.c.r - shift.x) / norm;
    rgb.c.g = (pixel.c.g - shift.y) / norm;
    rgb.c.b = (pixel.c.b - shift.z) / norm;

    return rgb;
}

// Support shift and normalization
template<class YuvUnitx2, class Rgb, class RgbUnitx2>
__global__ static void YuvToRgbPlanarKernel(uint8_t *pY, uint8_t *pUV, int yLinesize, int uvLinesize,
                                            uint8_t *p1, uint8_t *p2, uint8_t *p3, int rgbpLinesize,
                                            int nWidth, int nHeight, float norm=255.0f, 
                                            float shift_r=0, float shift_g=0, float shift_b=0) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    float3 shift_rgb{shift_r, shift_g, shift_b};

    uint8_t *pSrcY = pY + x * sizeof(YuvUnitx2) / 2 + y * yLinesize;
    uint8_t *pSrcUV = pUV + x * sizeof(YuvUnitx2) / 2 + y / 2 * uvLinesize;

    YuvUnitx2 l0 = *(YuvUnitx2 *)pSrcY;
    YuvUnitx2 l1 = *(YuvUnitx2 *)(pSrcY + yLinesize);
    YuvUnitx2 ch = *(YuvUnitx2 *)(pSrcUV);

    // if (x == 0 && y == 0) printf("Y[0], U[0], V[0] = %d, %d, %d\n", l0.x, ch.x, ch.y);
    
    Rgb rgb0 = normalizePixel(YuvToRgbForPixel<Rgb>(l0.x, ch.x, ch.y), norm, shift_rgb),
        rgb1 = normalizePixel(YuvToRgbForPixel<Rgb>(l0.y, ch.x, ch.y), norm, shift_rgb),
        rgb2 = normalizePixel(YuvToRgbForPixel<Rgb>(l1.x, ch.x, ch.y), norm, shift_rgb),
        rgb3 = normalizePixel(YuvToRgbForPixel<Rgb>(l1.y, ch.x, ch.y), norm, shift_rgb);

    uint8_t *pDst1 = p1 + x * sizeof(RgbUnitx2) / 2 + y * rgbpLinesize;
    uint8_t *pDst2 = p2 + x * sizeof(RgbUnitx2) / 2 + y * rgbpLinesize;
    uint8_t *pDst3 = p3 + x * sizeof(RgbUnitx2) / 2 + y * rgbpLinesize;
    *(RgbUnitx2 *)pDst1 = RgbUnitx2 {rgb0.v.x, rgb1.v.x};
    // if (x == 0 && y == 0) printf("R[0] = %f\n", rgb0.v.x);
    *(RgbUnitx2 *)(pDst1 + rgbpLinesize) = RgbUnitx2 {rgb2.v.x, rgb3.v.x};
    // if (x == 0 && y == 0) printf("R[%d] = %f\n", rgbpLinesize, rgb2.v.x);
    // pDst += nRgbpPitch * nHeight;
    *(RgbUnitx2 *)pDst2 = RgbUnitx2 {rgb0.v.y, rgb1.v.y};
    *(RgbUnitx2 *)(pDst2 + rgbpLinesize) = RgbUnitx2 {rgb2.v.y, rgb3.v.y};
    // pDst += nRgbpPitch * nHeight;
    *(RgbUnitx2 *)pDst3 = RgbUnitx2 {rgb0.v.z, rgb1.v.z};
    *(RgbUnitx2 *)(pDst3 + rgbpLinesize) = RgbUnitx2 {rgb2.v.z, rgb3.v.z};
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

void Bgra64ToP016(uint8_t *dpBgra, int nBgraPitch, uint8_t *dpP016, int nP016Pitch, int nWidth, int nHeight, int iMatrix) {
    SetMatRgb2Yuv(iMatrix);
    RgbToYuvKernel<ushort2, BGRA64, ulonglong2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2)>>>
        (dpBgra, nBgraPitch, dpP016, nP016Pitch, nWidth, nHeight);
}

#ifdef __cplusplus
extern "C"
{
#endif
void nv12_to_rgbpf32(CUstream stream, uint8_t **dp_nv12, int *nv12_pitch, uint8_t **dp_rgbpf32, int *rgbpf32_pitch, int width, int height, int matrix)
{
    SetMatYuv2Rgb(matrix, stream);
    YuvToRgbPlanarKernel<uchar2, RGBAF32, float2>
    <<<dim3((width + 63) / 32 / 2, (height + 3) / 2 / 2), dim3(32, 2), 0, stream>>>
    (dp_nv12[0], dp_nv12[1], nv12_pitch[0], nv12_pitch[1], dp_rgbpf32[0], dp_rgbpf32[1],
        dp_rgbpf32[2], rgbpf32_pitch[0], width, height);
}
void nv12_to_rgbpf32_shift(CUstream stream, uint8_t **dp_nv12, int *nv12_pitch, uint8_t **dp_rgbpf32, int *rgbpf32_pitch,
    int width, int height, float norm, float* shift, int matrix)
{
    SetMatYuv2Rgb(matrix, stream);
    YuvToRgbPlanarKernel<uchar2, RGBAF32, float2>
    <<<dim3((width + 63) / 32 / 2, (height + 3) / 2 / 2), dim3(32, 2), 0, stream>>>
    (dp_nv12[0], dp_nv12[1], nv12_pitch[0], nv12_pitch[1], dp_rgbpf32[0], dp_rgbpf32[1],
        dp_rgbpf32[2], rgbpf32_pitch[0], width, height, norm, shift[0], shift[1], shift[2]);
}
void nv12_to_bgrpf32_shift(CUstream stream, uint8_t **dp_nv12, int *nv12_pitch, uint8_t **dp_rgbpf32, int *rgbpf32_pitch,
    int width, int height, float norm, float* shift, int matrix)
{
    SetMatYuv2Rgb(matrix, stream);
    YuvToRgbPlanarKernel<uchar2, BGRAF32, float2>
    <<<dim3((width + 63) / 32 / 2, (height + 3) / 2 / 2), dim3(32, 2), 0, stream>>>
    (dp_nv12[0], dp_nv12[1], nv12_pitch[0], nv12_pitch[1], dp_rgbpf32[0], dp_rgbpf32[1],
        dp_rgbpf32[2], rgbpf32_pitch[0], width, height, norm, shift[0], shift[1], shift[2]);
}
// void nv12_to_rgbpf32(CUstream stream, uint8_t **dp_nv12, int *nv12_pitch, uint8_t *dp_rgbpf32, int rgbpf32_pitch, int width, int height, int matrix)
// {
//     uint8_t* dp_rgbpf32_data[3];
//     dp_rgbpf32_data[0] = dp_rgbpf32;
//     dp_rgbpf32_data[1] = dp_rgbpf32 + rgbpf32_pitch * height;
//     dp_rgbpf32_data[2] = dp_rgbpf32 + rgbpf32_pitch * height * 2;

//     SetMatYuv2Rgb(matrix, stream);
//     YuvToRgbPlanarKernel<uchar2, RGBAF32, float2>
//     <<<dim3((width + 63) / 32 / 2, (height + 3) / 2 / 2), dim3(32, 2), 0, stream>>>
//     (dp_nv12[0], dp_nv12[1], nv12_pitch[0], nv12_pitch[1], dp_rgbpf32_data[0], dp_rgbpf32_data[1],
//         dp_rgbpf32_data[2], rgbpf32_pitch, width, height);
// }

// void nv12_to_rgbpf32chw(CUStream stream, uint8_t **dp_nv12, int *nv12_pitch, uint8_t **dp_rgbpf32, int *rgbpf32_pitch, int width, int height, int matrix)
void rgbpf32_to_nv12(CUstream stream, uint8_t **dp_rgbpf32, int *rgbpf32_pitch, uint8_t **dp_nv12, int *nv12_pitch, int width, int height, int matrix)
{
    SetMatRgb2Yuv(matrix, stream);
    RgbpToYuvKernel<uchar2, RGBAF32, float2>
        <<<dim3((width + 63) / 32 / 2, (height + 3) / 2 / 2), dim3(32, 2), 0 , stream>>>
        (dp_rgbpf32[0], rgbpf32_pitch[0], dp_nv12[0], nv12_pitch[0], width, height);
}
#ifdef __cplusplus
}
#endif