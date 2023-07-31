
#ifndef SWSCALE_RGB2RGB_CUDA_H
#define SWSCALE_RGB2RGB_CUDA_H

// #ifdef __cplusplus
// extern "C" {
// #endif
#include <stdint.h>

union BGRA32 {
    uint32_t d;
    uchar4 v;
    struct {
        uint8_t b, g, r, a;
    } c;
};

union RGBA32 {
    uint32_t d;
    uchar4 v;
    struct {
        uint8_t r, g, b, a;
    } c;
};

union ARGB32 {
    uint32_t d;
    uchar4 v;
    struct {
        uint8_t a, r, g, b;
    } c;
};

union BGRA64 {
    uint64_t d;
    ushort4 v;
    struct {
        uint16_t b, g, r, a;
    } c;
};

union RGBA64 {
    uint64_t d;
    ushort4 v;
    struct {
        uint16_t r, g, b, a;
    } c;
};

union RGB24 {
    uchar3 v;
    struct {
        uint8_t r, g, b;
    } c;
};

union BGR24 {
    uchar3 v;
    struct {
        uint8_t b, g, r;
    } c;
};

union RGBAF32 {
    float4 v;
    struct {
        float r, g, b, a;
    } c;
};

union BGRAF32 {
    float4 v;
    struct {
        float b, g, r, a;
    } c;
};

union RGBF32 {
    float3 v;
    struct {
        float r, g, b;
    } c;
};

union BGRF32 {
    float3 v;
    struct {
        float b, g, r;
    } c;
};

template<typename T>
struct RGBClass2 {
    T x;
    T y;
};

typedef enum ColorSpaceStandard {
    ColorSpaceStandard_BT709 = 1,
    ColorSpaceStandard_Unspecified = 2,
    ColorSpaceStandard_Reserved = 3,
    ColorSpaceStandard_FCC = 4,
    ColorSpaceStandard_BT470 = 5,
    ColorSpaceStandard_BT601 = 6,
    ColorSpaceStandard_SMPTE240M = 7,
    ColorSpaceStandard_YCgCo = 8,
    ColorSpaceStandard_BT2020 = 9,
    ColorSpaceStandard_BT2020C = 10
} ColorSpaceStandard;


// #ifdef __cplusplus
// }
// #endif
#endif // SWSCALE_RGB2RGB_CUDA_H