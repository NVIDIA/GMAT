#pragma once
#include <stdint.h>
// #include <cuda.h>
#include <cuda_runtime.h>

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

#ifdef __cplusplus
template <class COLOR32>
void Nv12ToColor32(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template <class COLOR32>
void Nv12ToColor32(uint8_t **dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template<class COLOR32>
void Color32ToNv12(uint8_t *dpBgra, int nBgraPitch, uint8_t *dpNv12, int nNv12Pitch, int nWidth, int nHeight, int iMatrix);
#endif

#ifdef __cplusplus
extern "C"
{
#endif
void nv12_to_rgbpf32(cudaStream_t stream, uint8_t **dp_nv12, int *nv12_pitch, uint8_t **dp_rgbpf32, int *rgbpf32_pitch,
    int width, int height, int matrix);

void nv12_to_rgbpf32_shift(cudaStream_t stream, uint8_t **dp_nv12, int *nv12_pitch, uint8_t **dp_rgbpf32, int *rgbpf32_pitch,
    int width, int height, float norm, float* shift, int matrix);

void nv12_to_bgrpf32_shift(cudaStream_t stream, uint8_t **dp_nv12, int *nv12_pitch, uint8_t **dp_rgbpf32, int *rgbpf32_pitch,
    int width, int height, float norm, float* shift, int matrix);

// void nv12_to_rgbpf32(cudaStream_t stream, uint8_t **dp_nv12, int *nv12_pitch, uint8_t *dp_rgbpf32, int rgbpf32_pitch, int width, int height, int matrix);
void rgbpf32_to_nv12(cudaStream_t stream, uint8_t **dp_rgbpf32, int *rgbpf32_pitch, uint8_t **dp_nv12, int *nv12_pitch, int width, int height, int matrix);
#ifdef __cplusplus
}
#endif