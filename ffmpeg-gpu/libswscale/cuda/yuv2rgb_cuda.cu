#include <stdint.h>
#include <cstdio>
#include <type_traits>
#include <cmath>
#include <cuda.h>

#include "datatypes.h"

#define DEFAULT_ALPHA 255

extern "C" {
// __device__ float* matYuv2Rgb;
// __device__ float* matRgb2Yuv;
// __constant__ float matYuv2Rgb[3][3];
// __constant__ float matRgb2Yuv[3][3];
__constant__ float matYuv2Rgb[9];
__constant__ float matRgb2Yuv[9];
}

template<class T>
__device__ static T clamp(T x, T lower, T upper) {
    return x < lower ? lower : (x > upper ? upper : x);
}

__host__ __device__ static uchar3 operator+=(uchar3& a, const uint8_t& b) {
    a.x = a.x + b;
    a.y = a.y + b;
    a.z = a.z + b;
    return a;
}

__host__ __device__ static uchar4 operator+=(uchar4& a, const uint8_t& b) {
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
    return a;
}

__host__ __device__ static ushort3 operator+=(ushort3& a, const short& b) {
    a.x = a.x + b;
    a.y = a.y + b;
    a.z = a.z + b;
    return a;
}

__host__ __device__ static ushort4 operator+=(ushort4& a, const short& b) {
    a.x = a.x + b;
    a.y = a.y + b;
    a.z = a.z + b;
    a.w = a.w + b;
    return a;
}

__host__ __device__ static float3 operator+=(float3& a, const float& b) {
    a.x = a.x + b;
    a.y = a.y + b;
    a.z = a.z + b;
    return a;
}

__host__ __device__ static float4 operator+=(float4& a, const float& b) {
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
    return a;
}

// yuv2rgb conversions

template<class Rgb, class YuvUnit>
__device__ inline Rgb yuv2rgb_for_pixel(YuvUnit y, YuvUnit u, YuvUnit v) {
    const int 
        low = 1 << (sizeof(YuvUnit) * 8 - 4),
        mid = 1 << (sizeof(YuvUnit) * 8 - 1);
    float fy = (int)y - low, fu = (int)u - mid, fv = (int)v - mid;
    const float maxf = (1 << sizeof(YuvUnit) * 8) - 1.0f;

    // DEBUG
    // if (blockIdx.x == 0 && blockIdx.y == 0) printf("[%f %f %f]  ",  fy, fu, fv);

    YuvUnit 
        r = (YuvUnit)clamp(matYuv2Rgb[0] * fy + matYuv2Rgb[1] * fu + matYuv2Rgb[2] * fv, 0.0f, maxf),
        g = (YuvUnit)clamp(matYuv2Rgb[3] * fy + matYuv2Rgb[4] * fu + matYuv2Rgb[5] * fv, 0.0f, maxf),
        b = (YuvUnit)clamp(matYuv2Rgb[6] * fy + matYuv2Rgb[7] * fu + matYuv2Rgb[8] * fv, 0.0f, maxf);
    
    Rgb rgb{};
    rgb.v += DEFAULT_ALPHA; // set default value to 255
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

    // DEBUG
    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) printf("\nOutput RGB: \n");
    // __threadfence();
    // if (blockIdx.x == 0 && blockIdx.y == 0) printf("[%d %d %d]  ",  r, g, b);
    return rgb;
}

template<>
__device__ inline RGBF32 yuv2rgb_for_pixel(uint8_t y, uint8_t u, uint8_t v) {
    const int
        low = 1 << (sizeof(uint8_t) * 8 - 4),
        mid = 1 << (sizeof(uint8_t) * 8 - 1);
    float fy = (int)y - low, fu = (int)u - mid, fv = (int)v - mid;
    const float maxf = (1 << sizeof(uint8_t) * 8) - 1.0f;
    uint8_t
        r = (uint8_t)clamp(matYuv2Rgb[0] * fy + matYuv2Rgb[1] * fu + matYuv2Rgb[2] * fv, 0.0f, maxf),
        g = (uint8_t)clamp(matYuv2Rgb[3] * fy + matYuv2Rgb[4] * fu + matYuv2Rgb[5] * fv, 0.0f, maxf),
        b = (uint8_t)clamp(matYuv2Rgb[6] * fy + matYuv2Rgb[7] * fu + matYuv2Rgb[8] * fv, 0.0f, maxf);

    RGBF32 rgb{};

    rgb.c.r = r;
    rgb.c.g = g;
    rgb.c.b = b;
    return rgb;
}

template<>
__device__ inline RGBAF32 yuv2rgb_for_pixel(uint8_t y, uint8_t u, uint8_t v) {
    const int
        low = 1 << (sizeof(uint8_t) * 8 - 4),
        mid = 1 << (sizeof(uint8_t) * 8 - 1);
    float fy = (int)y - low, fu = (int)u - mid, fv = (int)v - mid;
    const float maxf = (1 << sizeof(uint8_t) * 8) - 1.0f;
    uint8_t
        r = (uint8_t)clamp(matYuv2Rgb[0] * fy + matYuv2Rgb[1] * fu + matYuv2Rgb[2] * fv, 0.0f, maxf),
        g = (uint8_t)clamp(matYuv2Rgb[3] * fy + matYuv2Rgb[4] * fu + matYuv2Rgb[5] * fv, 0.0f, maxf),
        b = (uint8_t)clamp(matYuv2Rgb[6] * fy + matYuv2Rgb[7] * fu + matYuv2Rgb[8] * fv, 0.0f, maxf);

    RGBAF32 rgb{};

    rgb.c.r = r;
    rgb.c.g = g;
    rgb.c.b = b;
    return rgb;
}
// template<>
// __device__ inline RGBA32 yuv2rgb_for_pixel(uint8_t y, uint8_t u, uint8_t v) {
//     const int 
//         low = 1 << (sizeof(uint8_t) * 8 - 4),
//         mid = 1 << (sizeof(uint8_t) * 8 - 1);
//     float fy = (int)y - low, fu = (int)u - mid, fv = (int)v - mid;
//     const float maxf = (1 << sizeof(uint8_t) * 8) - 1.0f;

//     uint8_t 
//         r = (uint8_t)clamp(matYuv2Rgb[0] * fy + matYuv2Rgb[1] * fu + matYuv2Rgb[2] * fv, 0.0f, maxf),
//         g = (uint8_t)clamp(matYuv2Rgb[3] * fy + matYuv2Rgb[4] * fu + matYuv2Rgb[5] * fv, 0.0f, maxf),
//         b = (uint8_t)clamp(matYuv2Rgb[6] * fy + matYuv2Rgb[7] * fu + matYuv2Rgb[8] * fv, 0.0f, maxf);
    
//     RGBA32 rgb{};
//     // DEBUG
//     // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) printf("\nOutput RGB: \n");
//     // __threadfence();
//     // if (blockIdx.x == 0 && blockIdx.y == 0) printf("[%d %d %d %d]  ",  rgb.c.r, rgb.c.g, rgb.c.b, rgb.c.a);
//     const int nShift = abs((int)sizeof(uint8_t) - (int)sizeof(rgb.c.r)) * 8;
//     if (sizeof(uint8_t) >= sizeof(rgb.c.r)) {
//         rgb.c.r = r >> nShift;
//         rgb.c.g = g >> nShift;
//         rgb.c.b = b >> nShift;
//         rgb.c.a = DEFAULT_ALPHA;
//     } else {
//         rgb.c.r = r << nShift;
//         rgb.c.g = g << nShift;
//         rgb.c.b = b << nShift;
//         rgb.c.a = DEFAULT_ALPHA;
//     }
//     return rgb;
// }

template<class YuvUnitx2, class Rgb, class RgbIntx2>
__global__ static void yuv2rgb_kernel(const uint8_t *pYuv, int nYuvPitch, 
                                      uint8_t *pRgb, int nRgbPitch, 
                                      int nWidth, int nHeight) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    const uint8_t *pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;
    uint8_t *pDst = pRgb + x * sizeof(Rgb) + y * nRgbPitch;

    YuvUnitx2 l0 = *(YuvUnitx2 *)pSrc;
    YuvUnitx2 l1 = *(YuvUnitx2 *)(pSrc + nYuvPitch);
    YuvUnitx2 ch = *(YuvUnitx2 *)(pSrc + (nHeight - y / 2) * nYuvPitch);

    *(RgbIntx2 *)pDst = RgbIntx2 {
        yuv2rgb_for_pixel<Rgb>(l0.x, ch.x, ch.y),
        yuv2rgb_for_pixel<Rgb>(l0.y, ch.x, ch.y),
    };
    *(RgbIntx2 *)(pDst + nRgbPitch) = RgbIntx2 {
        yuv2rgb_for_pixel<Rgb>(l1.x, ch.x, ch.y), 
        yuv2rgb_for_pixel<Rgb>(l1.y, ch.x, ch.y),
    };
}

template<class YuvUnitx2, class Rgb, class RgbIntx2>
__global__ static void yuv2rgb_odd_kernel(const uint8_t *pYuv, int nYuvPitch, 
                                      uint8_t *pRgb, int nRgbPitch, 
                                      int nWidth, int nHeight) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    int nUvPitch = (nYuvPitch + 1 ) / 2 * 2;
    bool oddWidth = (nWidth & 0x1) == 0x1;
    bool oddHeight = (nHeight & 0x1) == 0x1;
    if (x >= nWidth || y >= nHeight) {
        return;
    }

    YuvUnitx2 l0;
    YuvUnitx2 l1;
    YuvUnitx2 ch;
    bool rightEdge = (x == nWidth - 1) && oddWidth;
    bool bottomEdge = (y == nHeight - 1) && oddHeight;
    const uint8_t *pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;
    const uint8_t *pSrcUv = pYuv + nHeight * nYuvPitch;
    uint8_t *pDst = pRgb + x * sizeof(Rgb) + y * nRgbPitch;

    using YuvUnit = decltype(l0.x);
    l0 = YuvUnitx2{*(YuvUnit *)pSrc, rightEdge ? uint8_t(0) : *(YuvUnit *)(pSrc + sizeof(YuvUnit))};
    l1 = bottomEdge ? YuvUnitx2{0, 0} : 
        YuvUnitx2{*(YuvUnit *)(pSrc + nYuvPitch), 
                  rightEdge ? uint8_t(0) : *(YuvUnit *)(pSrc + nYuvPitch + sizeof(YuvUnit))};
    // ch = YuvUnitx2{*(YuvUnit *)(pSrc + (nHeight - y / 2) * (nYuvPitch + 1) / 2 * 2), 
    //                *(YuvUnit *)(pSrc + (nHeight - y / 2) * (nYuvPitch + 1) / 2 * 2 + sizeof(YuvUnit))};
    ch = YuvUnitx2{*(YuvUnit *)(pSrcUv + y / 2 * nUvPitch + x * sizeof(YuvUnit)), 
                   *(YuvUnit *)(pSrcUv + y / 2 * nUvPitch + x * sizeof(YuvUnit) + 1)};

    (*(RgbIntx2 *)pDst).x = yuv2rgb_for_pixel<Rgb>(l0.x, ch.x, ch.y);
    if (!rightEdge) (*(RgbIntx2 *)pDst).y = yuv2rgb_for_pixel<Rgb>(l0.y, ch.x, ch.y);
    if (!bottomEdge) {
        (*(RgbIntx2 *)(pDst + nRgbPitch)).x = yuv2rgb_for_pixel<Rgb>(l1.x, ch.x, ch.y);
        if (!rightEdge) (*(RgbIntx2 *)(pDst + nRgbPitch)).y = yuv2rgb_for_pixel<Rgb>(l1.y, ch.x, ch.y);
    }

    if (!rightEdge) {
        *(RgbIntx2 *)pDst = RgbIntx2 {
            yuv2rgb_for_pixel<Rgb>(l0.x, ch.x, ch.y),
            yuv2rgb_for_pixel<Rgb>(l0.y, ch.x, ch.y),
        };
    } else {
        *(Rgb *)pDst = yuv2rgb_for_pixel<Rgb>(l0.x, ch.x, ch.y);
    }
    if (!bottomEdge) {
        if (!rightEdge) {
            *(RgbIntx2 *)(pDst + nRgbPitch) = RgbIntx2 {
                yuv2rgb_for_pixel<Rgb>(l1.x, ch.x, ch.y), 
                yuv2rgb_for_pixel<Rgb>(l1.y, ch.x, ch.y),
            };
        } else {
             *(Rgb *)(pDst + nRgbPitch) = yuv2rgb_for_pixel<Rgb>(l1.x, ch.x, ch.y);
        }
    }
}

template<class YuvUnit, class Rgb, class RgbIntx2>
__global__ static void yuv02rgb_kernel(const uint8_t *pY, const uint8_t *pUv, int nYPitch, 
                                      uint8_t *pRgb, int nRgbPitch, 
                                      int nWidth, int nHeight) {
    using YuvUnitx2 = RGBClass2<YuvUnit>;
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    int nUvPitch = (nYPitch) / 2;
    const uint8_t *pSrcY = pY + x * sizeof(YuvUnitx2) / 2 + y * nYPitch;
    const uint8_t *pSrcUv = pUv + x * sizeof(YuvUnitx2) / 2 / 2 + y / 2 * nUvPitch;
    uint8_t *pDst = pRgb + x * sizeof(Rgb) + y * nRgbPitch;

    YuvUnitx2 l0 = *(YuvUnitx2 *)pSrcY;
    YuvUnitx2 l1 = *(YuvUnitx2 *)(pSrcY + nYPitch);
    YuvUnitx2 ch = YuvUnitx2 {
            *(YuvUnit *)(pSrcUv + y / 2 * nUvPitch),
            *(YuvUnit *)(pSrcUv + y / 2 * nUvPitch + nUvPitch * nHeight / 2),

    };

    *(RgbIntx2 *)pDst = RgbIntx2 {
        yuv2rgb_for_pixel<Rgb>(l0.x, ch.x, ch.y),
        yuv2rgb_for_pixel<Rgb>(l0.y, ch.x, ch.y),
    };
    *(RgbIntx2 *)(pDst + nRgbPitch) = RgbIntx2 {
        yuv2rgb_for_pixel<Rgb>(l1.x, ch.x, ch.y), 
        yuv2rgb_for_pixel<Rgb>(l1.y, ch.x, ch.y),
    };
}

template<class YuvUnit, class Rgb, class RgbIntx2>
__global__ static void yuv02rgb_odd_kernel(const uint8_t *pY, const uint8_t *pUv, int nYPitch, 
                                      uint8_t *pRgb, int nRgbPitch, 
                                      int nWidth, int nHeight) {
    using YuvUnitx2 = RGBClass2<YuvUnit>;
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    bool oddWidth = (nWidth & 0x1) == 0x1;
    bool oddHeight = (nHeight & 0x1) == 0x1;
    if (x >= nWidth || y >= nHeight) {
        return;
    }

    YuvUnitx2 l0;
    YuvUnitx2 l1;
    YuvUnitx2 ch;
    bool rightEdge = (x == nWidth - 1) && oddWidth;
    bool bottomEdge = (y == nHeight - 1) && oddHeight;

    int nUvPitch = (nYPitch + 1) / 2;
    const uint8_t *pSrcY = pY + x * sizeof(YuvUnitx2) / 2 + y * nYPitch;
    const uint8_t *pSrcUv = pUv + x * sizeof(YuvUnitx2) / 2 / 2 + y / 2 * nUvPitch;
    uint8_t *pDst = pRgb + x * sizeof(Rgb) + y * nRgbPitch;

    l0 = YuvUnitx2{*(YuvUnit *)pSrcY, rightEdge ? uint8_t(0) : *(YuvUnit *)(pSrcY + sizeof(YuvUnit))};
    l1 = bottomEdge ? 
        YuvUnitx2{0, 0} : YuvUnitx2{*(YuvUnit *)(pSrcY + nYPitch), 
                                    rightEdge ? uint8_t(0) : *(YuvUnit *)(pSrcY + nYPitch + sizeof(YuvUnit))};
    ch = YuvUnitx2 {
            *(YuvUnit *)(pSrcUv),
            *(YuvUnit *)(pSrcUv + nUvPitch * (nHeight + 1) / 2),
    };

    *(Rgb*)pDst = yuv2rgb_for_pixel<Rgb>(l0.x, ch.x, ch.y);
    if (!rightEdge) *(Rgb*)(pDst + sizeof(Rgb)) = yuv2rgb_for_pixel<Rgb>(l0.y, ch.x, ch.y);
    if (!bottomEdge) {
        *(Rgb*)(pDst + nRgbPitch) = yuv2rgb_for_pixel<Rgb>(l1.x, ch.x, ch.y);
        if (!rightEdge) *(Rgb*)(pDst + nRgbPitch + sizeof(Rgb)) = yuv2rgb_for_pixel<Rgb>(l1.y, ch.x, ch.y);
    }
    
}

template<class YuvUnitx2, class Rgb, class RgbIntx2>
__global__ static void yuv4442rgb_kernel(uint8_t *pYuv, int nYuvPitch, uint8_t *pRgb, int nRgbPitch, int nWidth, int nHeight) {
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
        yuv2rgb_for_pixel<Rgb>(l0.x, ch1.x, ch2.x).d,
        yuv2rgb_for_pixel<Rgb>(l0.y, ch1.y, ch2.y).d,
    };
}

template<class YuvUnit, class Rgb, class RgbIntx2>
__global__ static void yuv4442rgb_odd_kernel(uint8_t *pYuv, int nYuvPitch, uint8_t *pRgb, int nRgbPitch, int nWidth, int nHeight) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x);
    int y = (threadIdx.y + blockIdx.y * blockDim.y);
    if (x >= nWidth || y >= nHeight) {
        return;
    }

    uint8_t *pSrc = pYuv + x * sizeof(YuvUnit) / 2 + y * nYuvPitch;
    uint8_t *pDst = pRgb + x * sizeof(Rgb) + y * nRgbPitch;

    YuvUnit l0 = *(YuvUnit *)pSrc;
    YuvUnit ch1 = *(YuvUnit *)(pSrc + (nHeight * nYuvPitch));
    YuvUnit ch2 = *(YuvUnit *)(pSrc + (2 * nHeight * nYuvPitch));

    *(Rgb *)pDst = yuv2rgb_for_pixel<Rgb>(l0, ch1, ch2).d;
}

template<class Rgb>
__device__ static inline Rgb normalize_pixel(Rgb pixel, float norm, float3 shift){
    Rgb rgb{};
    rgb.c.r = (pixel.c.r - shift.x) / norm;
    rgb.c.g = (pixel.c.g - shift.y) / norm;
    rgb.c.b = (pixel.c.b - shift.z) / norm;

    return rgb;
}

// Support shift and normalization
template<class YuvUnitx2, class Rgb, class RgbUnitx2>
__global__ static void yuv2rgb_planar_kernel(const uint8_t *pY, int nv12Linesize,
                                            uint8_t *pRgb, int rgbpLinesize,
                                            int nWidth, int nHeight, float norm=255.0f, 
                                            float shift_r=0, float shift_g=0, float shift_b=0) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    float3 shift_rgb{shift_r, shift_g, shift_b};

    const uint8_t *pUV = pY + nv12Linesize * nHeight;
    const uint8_t *pSrcY = pY + x * sizeof(YuvUnitx2) / 2 + y * nv12Linesize;
    const uint8_t *pSrcUV = pUV + x * sizeof(YuvUnitx2) / 2 + y / 2 * nv12Linesize;

    YuvUnitx2 l0 = *(YuvUnitx2 *)pSrcY;
    YuvUnitx2 l1 = *(YuvUnitx2 *)(pSrcY + nv12Linesize);
    YuvUnitx2 ch = *(YuvUnitx2 *)(pSrcUV);

    // if (x == 0 && y == 0) printf("Y[0], U[0], V[0] = %d, %d, %d\n", l0.x, ch.x, ch.y);
    
    Rgb rgb0 = normalize_pixel(yuv2rgb_for_pixel<Rgb>(l0.x, ch.x, ch.y), norm, shift_rgb),
        rgb1 = normalize_pixel(yuv2rgb_for_pixel<Rgb>(l0.y, ch.x, ch.y), norm, shift_rgb),
        rgb2 = normalize_pixel(yuv2rgb_for_pixel<Rgb>(l1.x, ch.x, ch.y), norm, shift_rgb),
        rgb3 = normalize_pixel(yuv2rgb_for_pixel<Rgb>(l1.y, ch.x, ch.y), norm, shift_rgb);

    uint8_t *pDst1 = pRgb + x * sizeof(RgbUnitx2) / 2 + y * rgbpLinesize;
    uint8_t *pDst2 = pRgb + nHeight * rgbpLinesize + x * sizeof(RgbUnitx2) / 2 + y * rgbpLinesize;
    uint8_t *pDst3 = pRgb + nHeight * rgbpLinesize * 2 + x * sizeof(RgbUnitx2) / 2 + y * rgbpLinesize;
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
__global__ static void yuv2rgb_odd_planar_kernel(const uint8_t *pY, int nv12Linesize,
                                            uint8_t *pRgb, int rgbpLinesize,
                                            int nWidth, int nHeight, float norm=255.0f, 
                                            float shift_r=0, float shift_g=0, float shift_b=0) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    bool oddWidth = (nWidth & 0x1) == 0x1;
    bool oddHeight = (nHeight & 0x1) == 0x1;
    if (x >= nWidth || y >= nHeight) {
        return;
    }

    bool rightEdge = (x == nWidth - 1) && oddWidth;
    bool bottomEdge = (y == nHeight - 1) && oddHeight;
    float3 shift_rgb{shift_r, shift_g, shift_b};

    int nUvPitch = (nv12Linesize + 1) / 2 * 2;
    const uint8_t *pUV = pY + nv12Linesize * nHeight;
    const uint8_t *pSrcY = pY + x * sizeof(YuvUnitx2) / 2 + y * nv12Linesize;
    // const uint8_t *pSrcUV = pUV + x * sizeof(YuvUnitx2) / 2 + y / 2 * nv12Linesize;

    YuvUnitx2 l0;
    YuvUnitx2 l1;
    YuvUnitx2 ch;
    using YuvUnit = decltype(l0.x);

    l0 = YuvUnitx2{*(YuvUnit *)pSrcY, rightEdge ? uint8_t(0) : *(YuvUnit *)(pSrcY + sizeof(YuvUnit))};
    l1 = bottomEdge ? YuvUnitx2{0, 0} : 
        YuvUnitx2{*(YuvUnit *)(pSrcY + nv12Linesize), 
                  rightEdge ? uint8_t(0) : *(YuvUnit *)(pSrcY + nv12Linesize + sizeof(YuvUnit))};
    // ch = YuvUnitx2{*(YuvUnit *)(pSrc + (nHeight - y / 2) * (nYuvPitch + 1) / 2 * 2), 
    //                *(YuvUnit *)(pSrc + (nHeight - y / 2) * (nYuvPitch + 1) / 2 * 2 + sizeof(YuvUnit))};
    ch = YuvUnitx2{*(YuvUnit *)(pUV + y / 2 * nUvPitch + x * sizeof(YuvUnit)), 
                   *(YuvUnit *)(pUV + y / 2 * nUvPitch + x * sizeof(YuvUnit) + 1)};

    // if (x == 0 && y == 0) printf("Y[0], U[0], V[0] = %d, %d, %d\n", l0.x, ch.x, ch.y);
    
    Rgb rgb0 = normalize_pixel(yuv2rgb_for_pixel<Rgb>(l0.x, ch.x, ch.y), norm, shift_rgb),
        rgb1 = normalize_pixel(yuv2rgb_for_pixel<Rgb>(l0.y, ch.x, ch.y), norm, shift_rgb),
        rgb2 = normalize_pixel(yuv2rgb_for_pixel<Rgb>(l1.x, ch.x, ch.y), norm, shift_rgb),
        rgb3 = normalize_pixel(yuv2rgb_for_pixel<Rgb>(l1.y, ch.x, ch.y), norm, shift_rgb);

    uint8_t *pDst1 = pRgb + x * sizeof(RgbUnitx2) / 2 + y * rgbpLinesize;
    uint8_t *pDst2 = pRgb + nHeight * rgbpLinesize + x * sizeof(RgbUnitx2) / 2 + y * rgbpLinesize;
    uint8_t *pDst3 = pRgb + nHeight * rgbpLinesize * 2 + x * sizeof(RgbUnitx2) / 2 + y * rgbpLinesize;

    if (!rightEdge) {
        *(RgbUnitx2 *)pDst1 = RgbUnitx2 {rgb0.v.x, rgb1.v.x};
        *(RgbUnitx2 *)pDst2 = RgbUnitx2 {rgb0.v.y, rgb1.v.y};
        *(RgbUnitx2 *)pDst3 = RgbUnitx2 {rgb0.v.z, rgb1.v.z};
    } else {
        *(Rgb *)pDst1 = rgb0.v.x;
        *(Rgb *)pDst2 = rgb0.v.y;
        *(Rgb *)pDst3 = rgb0.v.z;
    }
    if (!bottomEdge) {
        if (!rightEdge) {
            *(RgbUnitx2 *)(pDst1 + rgbpLinesize) = RgbUnitx2 {rgb2.v.x, rgb3.v.x};
            *(RgbUnitx2 *)(pDst2 + rgbpLinesize) = RgbUnitx2 {rgb2.v.y, rgb3.v.y};
            *(RgbUnitx2 *)(pDst3 + rgbpLinesize) = RgbUnitx2 {rgb2.v.z, rgb3.v.z};
        }
        else {
            *(Rgb *)(pDst1 + rgbpLinesize) = rgb2.v.x;
            *(Rgb *)(pDst2 + rgbpLinesize) = rgb2.v.y;
            *(Rgb *)(pDst3 + rgbpLinesize) = rgb2.v.z;
        }
    }
}

template<class YuvUnitx2, class Rgb, class RgbUnitx2>
__global__ static void yuv2rgb_planar_kernel(const uint8_t *pY, const uint8_t *pUV, int yLinesize, int uvLinesize,
                                            uint8_t *p1, uint8_t *p2, uint8_t *p3, int rgbpLinesize,
                                            int nWidth, int nHeight, float norm=255.0f, 
                                            float shift_r=0, float shift_g=0, float shift_b=0) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    float3 shift_rgb{shift_r, shift_g, shift_b};

    const uint8_t *pSrcY = pY + x * sizeof(YuvUnitx2) / 2 + y * yLinesize;
    const uint8_t *pSrcUV = pUV + x * sizeof(YuvUnitx2) / 2 + y / 2 * uvLinesize;

    YuvUnitx2 l0 = *(YuvUnitx2 *)pSrcY;
    YuvUnitx2 l1 = *(YuvUnitx2 *)(pSrcY + yLinesize);
    YuvUnitx2 ch = *(YuvUnitx2 *)(pSrcUV);

    // if (x == 0 && y == 0) printf("Y[0], U[0], V[0] = %d, %d, %d\n", l0.x, ch.x, ch.y);
    
    Rgb rgb0 = normalize_pixel(yuv2rgb_for_pixel<Rgb>(l0.x, ch.x, ch.y), norm, shift_rgb),
        rgb1 = normalize_pixel(yuv2rgb_for_pixel<Rgb>(l0.y, ch.x, ch.y), norm, shift_rgb),
        rgb2 = normalize_pixel(yuv2rgb_for_pixel<Rgb>(l1.x, ch.x, ch.y), norm, shift_rgb),
        rgb3 = normalize_pixel(yuv2rgb_for_pixel<Rgb>(l1.y, ch.x, ch.y), norm, shift_rgb);

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

// yuv420 to rgb, need to test
template <class COLOR>
void yuv4202color(const uint8_t *dpY, const uint8_t *dpUv, int nYPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, CUstream stream) {
    // set_mat_yuv2rgb(iMatrix);
    yuv02rgb_odd_kernel<uint8_t, COLOR, RGBClass2<COLOR>>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2), 0, stream>>>
        (dpY, dpUv, nYPitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR>
void nv122color(const uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, CUstream stream) {
    // set_mat_yuv2rgb(iMatrix);
    yuv2rgb_odd_kernel<uchar2, COLOR, RGBClass2<COLOR>>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2), 0, stream>>>
        (dpNv12, nNv12Pitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR, class ColorUnitx2>
void nv122color_planar(const uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, CUstream stream) {
    // SetMatYuv2Rgb(iMatrix);
    yuv2rgb_planar_kernel<uchar2, COLOR, ColorUnitx2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2), 0, stream>>>
        (dpNv12, nNv12Pitch, dpBgrp, nBgrpPitch, nWidth, nHeight);
}

template <class COLOR32>
void nv122color32(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, CUstream stream) {
    // set_mat_yuv2rgb(iMatrix);
    yuv2rgb_kernel<uchar2, COLOR32, RGBClass2<COLOR32>>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2), 0, stream>>>
        (dpNv12, nNv12Pitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR64>
void nv12tocolor64(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, CUstream stream) {
    // set_mat_yuv2rgb(iMatrix);
    yuv2rgb_kernel<uchar2, COLOR64, RGBClass2<COLOR64>>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2), 0, stream>>>
        (dpNv12, nNv12Pitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR32>
void yuv4442color32(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, CUstream stream) {
    // set_mat_yuv2rgb(iMatrix);
    yuv4442rgb_kernel<uchar2, COLOR32, RGBClass2<COLOR32>>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2), dim3(32, 2) , 0, stream>>>
        (dpYUV444, nPitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR64>
void yuv4442color64(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, CUstream stream) {
    // set_mat_yuv2rgb(iMatrix);
    yuv4442rgb_kernel<uchar2, COLOR64, RGBClass2<COLOR64>>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2), dim3(32, 2) , 0, stream>>>
        (dpYUV444, nPitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR32>
void p0162color32(uint8_t *dpP016, int nP016Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, CUstream stream) {
    // set_mat_yuv2rgb(iMatrix);
    yuv2rgb_kernel<ushort2, COLOR32, RGBClass2<COLOR32>>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2), 0, stream>>>
        (dpP016, nP016Pitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR64>
void p0162color64(uint8_t *dpP016, int nP016Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, CUstream stream) {
    // set_mat_yuv2rgb(iMatrix);
    yuv2rgb_kernel<ushort2, COLOR64, RGBClass2<COLOR64>>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2), 0, stream>>>
        (dpP016, nP016Pitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR32>
void yuv444P162color32(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, CUstream stream) {
    // set_mat_yuv2rgb(iMatrix);
    yuv4442rgb_kernel<ushort2, COLOR32, RGBClass2<COLOR32>>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2), dim3(32, 2) , 0, stream>>>
        (dpYUV444, nPitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR64>
void yuv444P162color64(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, CUstream stream) {
    // set_mat_yuv2rgb(iMatrix);
    yuv4442rgb_kernel<ushort2, COLOR64, RGBClass2<COLOR64>>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2), dim3(32, 2) , 0, stream>>>
        (dpYUV444, nPitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

// Explicit Instantiation
template void yuv4202color<RGB24>(const uint8_t *dpY, const uint8_t *dpUv, int nYPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, CUstream stream);
template void yuv4202color<BGR24>(const uint8_t *dpY, const uint8_t *dpUv, int nYPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, CUstream stream);
template void yuv4202color<RGBA32>(const uint8_t *dpY, const uint8_t *dpUv, int nYPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, CUstream stream);
template void yuv4202color<BGRA32>(const uint8_t *dpY, const uint8_t *dpUv, int nYPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, CUstream stream);
template void yuv4202color<RGBA64>(const uint8_t *dpY, const uint8_t *dpUv, int nYPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, CUstream stream);
template void yuv4202color<BGRA64>(const uint8_t *dpY, const uint8_t *dpUv, int nYPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, CUstream stream);
template void nv122color<RGB24>(const uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, CUstream stream);
template void nv122color<BGR24>(const uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, CUstream stream);
template void nv122color<BGRA32>(const uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, CUstream stream);
template void nv122color<RGBA32>(const uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, CUstream stream);
template void nv122color<BGRA64>(const uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, CUstream stream);
template void nv122color<RGBA64>(const uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, CUstream stream);
template void nv122color_planar<RGBF32, float2>(const uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, CUstream stream);
template void nv122color_planar<BGRF32, float2>(const uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, CUstream stream);

// rgb2yuv conversions
template<class YuvUnit, class RgbUnit>
__device__ inline YuvUnit RgbToY(RgbUnit r, RgbUnit g, RgbUnit b) {
    const YuvUnit low = 1 << (sizeof(YuvUnit) * 8 - 4);
    return matRgb2Yuv[0] * r + matRgb2Yuv[1] * g + matRgb2Yuv[2] * b + low;
}

template<class YuvUnit, class RgbUnit>
__device__ inline YuvUnit RgbToU(RgbUnit r, RgbUnit g, RgbUnit b) {
    const YuvUnit mid = 1 << (sizeof(YuvUnit) * 8 - 1);
    return matRgb2Yuv[3] * r + matRgb2Yuv[4] * g + matRgb2Yuv[5] * b + mid;
}

template<class YuvUnit, class RgbUnit>
__device__ inline YuvUnit RgbToV(RgbUnit r, RgbUnit g, RgbUnit b) {
    const YuvUnit mid = 1 << (sizeof(YuvUnit) * 8 - 1);
    return matRgb2Yuv[6] * r + matRgb2Yuv[7] * g + matRgb2Yuv[8] * b + mid;
}

template<class YuvUnitx2, class Rgb, class RgbIntx2>
__global__ static void RgbToYuvKernel(const uint8_t *pRgb, int nRgbPitch, uint8_t *pYuv, int nYuvPitch, int nWidth, int nHeight) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    const uint8_t *pSrc = pRgb + x * sizeof(Rgb) + y * nRgbPitch;
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

template<class YuvUnit, class Rgb, class RgbIntx2>
__global__ static void RgbToYuvpKernel(const uint8_t *pRgb, int nRgbPitch,
                                       uint8_t *pY, int nYPitch,
                                       uint8_t *pUv, int nUvPitch,
                                       int nWidth, int nHeight) {
    using YuvUnitx2 = RGBClass2<YuvUnit>;
    
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    const uint8_t *pSrc = pRgb + x * sizeof(Rgb) + y * nRgbPitch;
    RgbIntx2 int2a = *(RgbIntx2 *)pSrc;
    RgbIntx2 int2b = *(RgbIntx2 *)(pSrc + nRgbPitch);

    Rgb rgb[4] = {int2a.x, int2a.y, int2b.x, int2b.y};
    decltype(Rgb::c.r)
        r = (rgb[0].c.r + rgb[1].c.r + rgb[2].c.r + rgb[3].c.r) / 4,
        g = (rgb[0].c.g + rgb[1].c.g + rgb[2].c.g + rgb[3].c.g) / 4,
        b = (rgb[0].c.b + rgb[1].c.b + rgb[2].c.b + rgb[3].c.b) / 4;

    uint8_t *pDstY = pY + x * sizeof(YuvUnitx2) / 2 + y * nYPitch;
    uint8_t *pDstUv = pUv + x * sizeof(YuvUnit) / 2 + y * nUvPitch / 2;
    *(YuvUnitx2 *)pDstY = YuvUnitx2 {
        RgbToY<YuvUnit>(rgb[0].c.r, rgb[0].c.g, rgb[0].c.b),
        RgbToY<YuvUnit>(rgb[1].c.r, rgb[1].c.g, rgb[1].c.b),
    };
    *(YuvUnitx2 *)(pDstY + nUvPitch) = YuvUnitx2 {
        RgbToY<YuvUnit>(rgb[2].c.r, rgb[2].c.g, rgb[2].c.b),
        RgbToY<YuvUnit>(rgb[3].c.r, rgb[3].c.g, rgb[3].c.b),
    };
    *(YuvUnit *)(pDstUv)                           = RgbToU<YuvUnit>(r, g, b);
    *(YuvUnit *)(pDstUv + nHeight / 2 * nUvPitch) = RgbToV<YuvUnit>(r, g, b);
}

void Bgra64ToP016(const uint8_t *dpBgra, int nBgraPitch, uint8_t *dpP016, int nP016Pitch, int nWidth, int nHeight, CUstream stream) {
    // set_mat_rgb2yuv(iMatrix);
    RgbToYuvKernel<ushort2, BGRA64, ulonglong2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2), 0, stream>>>
        (dpBgra, nBgraPitch, dpP016, nP016Pitch, nWidth, nHeight);
}

template <class COLOR>
void color2nv12(const uint8_t *dpColor, int nColorPitch, uint8_t *dpYuv, int nYuvPitch, int nWidth, int nHeight, CUstream stream) {
    // set_mat_rgb2yuv(iMatrix);
    RgbToYuvKernel<RGBClass2<uint8_t>, RGB24, RGBClass2<RGB24>>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2), 0, stream>>>
        (dpColor, nColorPitch, dpYuv, nYuvPitch, nWidth, nHeight);
}

template <class COLOR>
void color2yuv420(const uint8_t *dpColor, int nColorPitch, uint8_t *dpY, int nYPitch, uint8_t *dpUv, int nUvPitch, int nWidth, int nHeight, CUstream stream) {
    // set_mat_rgb2yuv(iMatrix);
    RgbToYuvpKernel<uint8_t, RGB24, RGBClass2<RGB24>>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2), 0, stream>>>
        (dpColor, nColorPitch, dpY, nYPitch, dpUv, nUvPitch, nWidth, nHeight);
}

template void color2nv12<RGB24>(const uint8_t *dpColor, int nColorPitch, uint8_t *dpYuv, int nYuvPitch, int nWidth, int nHeight, CUstream stream);
template void color2nv12<BGR24>(const uint8_t *dpColor, int nColorPitch, uint8_t *dpYuv, int nYuvPitch, int nWidth, int nHeight, CUstream stream);
template void color2nv12<RGBA32>(const uint8_t *dpColor, int nColorPitch, uint8_t *dpYuv, int nYuvPitch, int nWidth, int nHeight, CUstream stream);
template void color2nv12<BGRA32>(const uint8_t *dpColor, int nColorPitch, uint8_t *dpYuv, int nYuvPitch, int nWidth, int nHeight, CUstream stream);
template void color2nv12<RGBA64>(const uint8_t *dpColor, int nColorPitch, uint8_t *dpYuv, int nYuvPitch, int nWidth, int nHeight, CUstream stream);
template void color2nv12<BGRA64>(const uint8_t *dpColor, int nColorPitch, uint8_t *dpYuv, int nYuvPitch, int nWidth, int nHeight, CUstream stream);
template void color2yuv420<RGB24>(const uint8_t *dpColor, int nColorPitch, uint8_t *dpY, int nYPitch, uint8_t *dpUv, int nUvPitch, int nWidth, int nHeight, CUstream stream);
template void color2yuv420<BGR24>(const uint8_t *dpColor, int nColorPitch, uint8_t *dpY, int nYPitch, uint8_t *dpUv, int nUvPitch, int nWidth, int nHeight, CUstream stream);
template void color2yuv420<RGBA32>(const uint8_t *dpColor, int nColorPitch, uint8_t *dpY, int nYPitch, uint8_t *dpUv, int nUvPitch, int nWidth, int nHeight, CUstream stream);
template void color2yuv420<BGRA32>(const uint8_t *dpColor, int nColorPitch, uint8_t *dpY, int nYPitch, uint8_t *dpUv, int nUvPitch, int nWidth, int nHeight, CUstream stream);
template void color2yuv420<RGBA64>(const uint8_t *dpColor, int nColorPitch, uint8_t *dpY, int nYPitch, uint8_t *dpUv, int nUvPitch, int nWidth, int nHeight, CUstream stream);
template void color2yuv420<BGRA64>(const uint8_t *dpColor, int nColorPitch, uint8_t *dpY, int nYPitch, uint8_t *dpUv, int nUvPitch, int nWidth, int nHeight, CUstream stream);

extern "C" {
// include swscale_internal.h will cause many compiling errors
// #include "libswscale/swscale_internal.h"
#include "libavutil/pixfmt.h"

static void inline get_constants(enum AVColorSpace cspace, float* wr, float* wb, int* black, int* white, int* max) {
    *black = 16; *white = 235;
    *max = 255;

    switch (cspace)
    {
    case AVCOL_SPC_BT709:
        *wr = 0.2126f; *wb = 0.0722f;
        break;

    case AVCOL_SPC_FCC:
        *wr = 0.30f; *wb = 0.11f;
        break;

    case AVCOL_SPC_BT470BG:
    case AVCOL_SPC_SMPTE170M:
    default:
        *wr = 0.2990f; *wb = 0.1140f;
        break;

    case AVCOL_SPC_SMPTE240M:
        *wr = 0.212f; *wb = 0.087f;
        break;

    case AVCOL_SPC_BT2020_NCL:
    case AVCOL_SPC_BT2020_CL:
        *wr = 0.2627f; *wb = 0.0593f;
        // 10-bit only
        *black = 64 << 6; *white = 940 << 6;
        *max = (1 << 16) - 1;
        break;
    }
}

void set_mat_yuv2rgb_cuda(enum AVColorSpace cspace) {
    float wr, wb;
    int black, white, max;
    get_constants(cspace, &wr, &wb, &black, &white, &max);
    float mat[3][3] = {
        {1.0f, 0.0f, (1.0f - wr) / 0.5f},
        {1.0f, -wb * (1.0f - wb) / 0.5f / (1 - wb - wr), -wr * (1 - wr) / 0.5f / (1 - wb - wr)},
        {1.0f, (1.0f - wb) / 0.5f, 0.0f},
    };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mat[i][j] = (float)(1.0 * max / (white - black) * mat[i][j]);
        }
    }
    cudaMemcpyToSymbolAsync(matYuv2Rgb, mat, sizeof(mat), 0, cudaMemcpyHostToDevice, 0);
}

void set_mat_rgb2yuv_cuda(enum AVColorSpace cspace) {
    float wr, wb;
    int black, white, max;
    get_constants(cspace, &wr, &wb, &black, &white, &max);
    float mat[3][3] = {
        {wr, 1.0f - wb - wr, wb},
        {-0.5f * wr / (1.0f - wb), -0.5f * (1 - wb - wr) / (1.0f - wb), 0.5f},
        {0.5f, -0.5f * (1.0f - wb - wr) / (1.0f - wr), -0.5f * wb / (1.0f - wr)},
    };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mat[i][j] = (float)(1.0 * (white - black) / max * mat[i][j]);
        }
    }
    cudaMemcpyToSymbolAsync(matRgb2Yuv, mat, sizeof(mat), 0, cudaMemcpyHostToDevice, 0);
}

// void ff_yuv2rgb_init_tables_cuda(SwsContext *c) {
//     enum AVColorSpace cspace = c->cspace;
//     if (isYUV(c->srcFormat) && isRGB(c->dstFormat)) {
//         cudaMalloc(&c->d_mat_yuv2rgbf, sizeof(float) * 9);
//         set_mat_yuv2rgb(cspace, c->d_mat_yuv2rgbf);
//     }
//     if (isRGB(c->srcFormat) && isYUV(c->dstFormat)) {
//         cudaMalloc(&c->d_mat_rgb2yuvf, sizeof(float) * 9);
//         set_mat_rgb2yuv(cspace, c->d_mat_rgb2yuvf);
//     }
// }

int yuv2rgb_cuda(const uint8_t *src[], int srcStride[],
                uint8_t *dst[], int dstStride[],
                int nWidth, int nHeight,
                int srcFormat, int dstFormat, CUstream stream) {
    if (srcFormat == AV_PIX_FMT_NV12) {
        switch (dstFormat) {
            case AV_PIX_FMT_RGB24: 
                nv122color<RGB24>(src[0], srcStride[0], dst[0], dstStride[0], nWidth, nHeight, stream); return 0;
            case AV_PIX_FMT_BGR24:
                nv122color<BGR24>(src[0], srcStride[0], dst[0], dstStride[0], nWidth, nHeight, stream); return 0;
            case AV_PIX_FMT_RGBA:
                nv122color<RGBA32>(src[0], srcStride[0], dst[0], dstStride[0], nWidth, nHeight, stream);
                cudaGetLastError();
                return 0;
            case AV_PIX_FMT_BGRA:
                nv122color<BGRA32>(src[0], srcStride[0], dst[0], dstStride[0], nWidth, nHeight, stream); return 0;
            case AV_PIX_FMT_RGBA64:
                nv122color<RGBA64>(src[0], srcStride[0], dst[0], dstStride[0], nWidth, nHeight, stream); return 0;
            case AV_PIX_FMT_BGRA64:
                nv122color<BGRA64>(src[0], srcStride[0], dst[0], dstStride[0], nWidth, nHeight, stream); return 0;
            case AV_PIX_FMT_RGBPF32LE:
                nv122color_planar<RGBF32, float2>(src[0], srcStride[0], dst[0], dstStride[0], nWidth, nHeight, stream);
                cudaGetLastError();
                return 0;
        }
    }

    if (srcFormat == AV_PIX_FMT_YUV420P) {
        switch (dstFormat) {
            case AV_PIX_FMT_RGB24:
                yuv4202color<RGB24>(src[0], src[1], srcStride[0], dst[0], dstStride[0], nWidth, nHeight, stream); return 0;
            case AV_PIX_FMT_BGR24:
                yuv4202color<BGR24>(src[0], src[1], srcStride[0], dst[0], dstStride[0], nWidth, nHeight, stream); return 0;
            case AV_PIX_FMT_RGBA:
                yuv4202color<RGBA32>(src[0], src[1], srcStride[0], dst[0], dstStride[0], nWidth, nHeight, stream); return 0;
            case AV_PIX_FMT_BGRA:
                yuv4202color<BGRA32>(src[0], src[1], srcStride[0], dst[0], dstStride[0], nWidth, nHeight, stream); return 0;
            case AV_PIX_FMT_RGBA64:
                yuv4202color<RGBA64>(src[0], src[1], srcStride[0], dst[0], dstStride[0], nWidth, nHeight, stream); return 0;
            case AV_PIX_FMT_BGRA64:
                yuv4202color<BGRA64>(src[0], src[1], srcStride[0], dst[0], dstStride[0], nWidth, nHeight, stream); return 0;
        }
    }

    return -1;
}

int rgb2yuv_cuda(const uint8_t *src[], int srcStride[],
                uint8_t *dst[], int dstStride[],
                int nWidth, int nHeight,
                int srcFormat, int dstFormat, CUstream stream) {
    if (dstFormat == AV_PIX_FMT_NV12) {
        switch (srcFormat) {
            case AV_PIX_FMT_RGB24: 
                color2nv12<RGB24>(src[0], srcStride[0], dst[0], dstStride[0], nWidth, nHeight, stream); return 0;
            case AV_PIX_FMT_BGR24:
                color2nv12<BGR24>(src[0], srcStride[0], dst[0], dstStride[0], nWidth, nHeight, stream); return 0;
            case AV_PIX_FMT_RGBA:
                color2nv12<RGBA32>(src[0], srcStride[0], dst[0], dstStride[0], nWidth, nHeight, stream); return 0;
            case AV_PIX_FMT_BGRA:
                color2nv12<BGRA32>(src[0], srcStride[0], dst[0], dstStride[0], nWidth, nHeight, stream); return 0;
            case AV_PIX_FMT_RGBA64:
                color2nv12<RGBA64>(src[0], srcStride[0], dst[0], dstStride[0], nWidth, nHeight, stream); return 0;
            case AV_PIX_FMT_BGRA64:
                color2nv12<BGRA64>(src[0], srcStride[0], dst[0], dstStride[0], nWidth, nHeight, stream); return 0;
        }
    }
    if (dstFormat == AV_PIX_FMT_YUV420P) {
        switch (srcFormat) {
            case AV_PIX_FMT_RGB24: 
                color2yuv420<RGB24>(src[0], srcStride[0], dst[0], dstStride[0], dst[1], dstStride[1], nWidth, nHeight, stream); return 0;
            case AV_PIX_FMT_BGR24:
                color2yuv420<BGR24>(src[0], srcStride[0], dst[0], dstStride[0], dst[1], dstStride[1], nWidth, nHeight, stream); return 0;
            case AV_PIX_FMT_RGBA:
                color2yuv420<RGBA32>(src[0], srcStride[0], dst[0], dstStride[0], dst[1], dstStride[1], nWidth, nHeight, stream); return 0;
            case AV_PIX_FMT_BGRA:
                color2yuv420<BGRA32>(src[0], srcStride[0], dst[0], dstStride[0], dst[1], dstStride[1], nWidth, nHeight, stream); return 0;
            case AV_PIX_FMT_RGBA64:
                color2yuv420<RGBA64>(src[0], srcStride[0], dst[0], dstStride[0], dst[1], dstStride[1], nWidth, nHeight, stream); return 0;
            case AV_PIX_FMT_BGRA64:
                color2yuv420<BGRA64>(src[0], srcStride[0], dst[0], dstStride[0], dst[1], dstStride[1], nWidth, nHeight, stream); return 0;
        }
    }

    return -1;
}
} // extern C