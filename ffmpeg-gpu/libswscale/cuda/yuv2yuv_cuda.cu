// nvcc -gencode arch=compute_86,code=sm_86 -I./ -std=c++11 -m64  -c libswscale/cuda/yuv2yuv_cuda.cu -o libswscale/cuda/yuv2yuv_cuda.o
#include <stdint.h>
#include <cstdio>
#include <type_traits>
#include <cmath>
#include <cuda.h>

static const ushort mask_10bit = 0xFFC0;
static const ushort mask_16bit = 0xFFFF;

#define C8 \
    operator uint8_t() const { return x; }

struct Unit10b;
struct Unit16b;

struct Unit8b {
    typedef uint8_t val_t;
    uint8_t x;
    __host__ __device__ Unit8b(){}
    __host__ __device__ Unit8b(uint8_t x): x(x){}
    __host__ __device__ operator uint8_t() const { return x; }
    __host__ __device__ operator Unit10b() const;
    __host__ __device__ operator Unit16b() const;
    // operator uint16_t() const { return static_cast<uint16_t>(x); }
};

struct Unit10b {
    typedef uint16_t val_t;
    uint16_t x;
    __host__ __device__ Unit10b(){}
    __host__ __device__ Unit10b(uint16_t x): x(x){}
    __host__ __device__ operator uint16_t() const { return x; }
    __host__ __device__ operator Unit8b()  const;
    __host__ __device__ operator Unit16b() const;
};

struct Unit16b {
    typedef uint16_t val_t;
    uint16_t x;
    __host__ __device__ Unit16b(){}
    __host__ __device__ Unit16b(uint16_t x): x(x){}
    __host__ __device__ operator uint16_t() const { return x; }
    __host__ __device__ operator Unit8b()  const;
    __host__ __device__ operator Unit10b() const;
};

template<typename Unit>
struct Unitx2 {
    Unit x;
    Unit y;

    template<class T> __host__ __device__ operator Unitx2<T>() {return Unitx2<T>{T(x), T(y)};}
};

__host__ __device__ Unit8b::operator Unit10b() const { return ((uint16_t)x | ((uint16_t)x << 8)) & mask_10bit; }
__host__ __device__ Unit8b::operator Unit16b() const { return ((uint16_t)x | ((uint16_t)x << 8)) & mask_16bit; }

__host__ __device__ Unit10b::operator Unit8b() const { return x >> 8; }
__host__ __device__ Unit10b::operator Unit16b() const { return x | (x >> 10); }

__host__ __device__ Unit16b::operator Unit8b() const { return x >> 8; }
__host__ __device__ Unit16b::operator Unit10b() const { return x & mask_10bit; }

template<class SrcUnit, class DstUnit>
static __device__ DstUnit convert_bit_depth(SrcUnit in);

template<> __device__ Unit8b convert_bit_depth(Unit8b in) { return in; }
template<> __device__ Unit10b convert_bit_depth(Unit8b in) { return ((uint16_t)in | ((uint16_t)in << 8)) & mask_10bit; }
template<> __device__ Unit16b convert_bit_depth(Unit8b in) {
    Unit16b out;
    out.x = ((uint16_t)in.x | ((uint16_t)in.x << 8)) & mask_16bit;
    return out;

}
template<> __device__ Unitx2<Unit8b> convert_bit_depth(Unitx2<Unit8b> in) { return in; }
template<> __device__ Unitx2<Unit10b> convert_bit_depth(Unitx2<Unit8b> in) {
    Unitx2<Unit10b> out;
    out.x = ((uint16_t)in.x | ((uint16_t)in.x << 8)) & mask_10bit;
    out.y = ((uint16_t)in.y | ((uint16_t)in.y << 8)) & mask_10bit;
    return out;
}
template<> __device__ Unitx2<Unit16b> convert_bit_depth(Unitx2<Unit8b> in) {
    Unitx2<Unit16b> out;
    out.x = ((uint16_t)in.x | ((uint16_t)in.x << 8)) & mask_16bit;
    out.y = ((uint16_t)in.y | ((uint16_t)in.y << 8)) & mask_16bit;
    return out;
}

template<class YuvUnit>
struct Pixel_NV12 {
    YuvUnit lum;
    using YuvUnitx2 = Unitx2<YuvUnit>;
    YuvUnitx2 chr;
    __host__ __device__ void read(const uint8_t* src[], int x, int y, int* linesize)
    {
        int offset_y = x * sizeof(YuvUnit) + y * linesize[0];
        int offset_uv = ((x >> 1) * sizeof(YuvUnitx2)) + ((y >> 1) * linesize[1]);
        lum = ((YuvUnit*)(src[0] + offset_y))[0];
        chr = ((YuvUnitx2*)(src[1] + offset_uv))[0];
    }
    __host__ __device__ void write(uint8_t* dst[], int x, int y, int* linesize)
    {
        int offset_y = x * sizeof(YuvUnit) + y * linesize[0];
        int offset_uv = ((x >> 1) * sizeof(YuvUnitx2)) + ((y >> 1) * linesize[1]);
        ((YuvUnit*)(dst[0] + offset_y))[0] = lum;
        ((YuvUnitx2*)(dst[1] + offset_uv))[0] = chr;
    }
};

// |y|y| |uv|
// |y|y|

// template<class YuvUnit>
// struct Pixel_NV12x4 {
//     YuvUnitx2 lum12; // |y|y|
//     YuvUnitx2 lum22;
//     using YuvUnitx2 = Unitx2<YuvUnit>;
//     YuvUnitx2 chr1; // |u|u|
//     YuvUnitx2 chr2;
//     YuvUnitx2 chr3;
//     YuvUnitx2 chr4;
//     __host__ __device__ void read(const uint8_t* src[], int x, int y, int* linesize)
//     {
//         int offset_y1 = x * sizeof(YuvUnitx2) + y * linesize[0];
//         int offset_y2 = x * sizeof(YuvUnitx2) + (y + 1) * linesize[0];
//         int offset_uv = ((x >> 1) * sizeof(YuvUnitx2)) + ((y >> 1) * linesize[1]);
//         lum12 = ((YuvUnitx2*)(src[0] + offset_y1))[0];
//         lum22 = ((YuvUnitx2*)(src[0] + offset_y2))[0];
//         chr1 = ((YuvUnitx2*)(src[1] + offset_uv))[0];
//         chr2 = chr1;
//         chr3 = chr1;
//         chr4 = chr1;
//     }
//     __host__ __device__ void write(uint8_t* dst[], int x, int y, int* linesize)
//     {
//         int offset_y = x * sizeof(YuvUnit) + y * linesize[0];
//         int offset_uv = ((x >> 1) * sizeof(YuvUnitx2)) + ((y >> 1) * linesize[1]);
//         ((YuvUnit*)(dst[0] + offset_y))[0] = lum;
//         ((YuvUnitx2*)(dst[1] + offset_uv))[0] = chr;
//     }
// };

template<class YuvUnit>
struct Pixel_I420 {
    using YuvUnitx2 = Unitx2<YuvUnit>;
    YuvUnit lum;
    YuvUnitx2 chr;
    __host__ __device__ void read(const uint8_t* src[], int x, int y, int* linesize)
    {
        int offset_y = x * sizeof(YuvUnit) + y * linesize[0];
        int offset_u = ((x >> 1) * sizeof(YuvUnit)) + ((y >> 1) * linesize[1]);
        int offset_v = ((x >> 1) * sizeof(YuvUnit)) + ((y >> 1) * linesize[2]);
        lum = ((YuvUnit*)(src[0] + offset_y))[0];
        chr.x = ((YuvUnit*)(src[1] + offset_u))[0];
        chr.y = ((YuvUnit*)(src[2] + offset_v))[0];
    }
    __host__ __device__ void write(uint8_t* dst[], int x, int y, int* linesize)
    {
        int offset_y = x * sizeof(YuvUnit) + y * linesize[0];
        int offset_u = ((x >> 1) * sizeof(YuvUnit)) + ((y >> 1) * linesize[1]);
        int offset_v = ((x >> 1) * sizeof(YuvUnit)) + ((y >> 1) * linesize[2]);
        ((YuvUnit*)(dst[0] + offset_y))[0] = lum;
        ((YuvUnit*)(dst[1] + offset_u))[0] = chr.x;
        ((YuvUnit*)(dst[2] + offset_v))[0] = chr.y;
    }
};

template<class YuvUnit>
struct Pixel_444 {
    using YuvUnitx2 = Unitx2<YuvUnit>;
    YuvUnit lum;
    YuvUnitx2 chr;
    __host__ __device__ void read(const uint8_t* src[], int x, int y, int* linesize)
    {
        int offset_y = x * sizeof(YuvUnit) + y * linesize[0];
        int offset_u = (x * sizeof(YuvUnit)) + (y * linesize[1]);
        int offset_v = (x * sizeof(YuvUnit)) + (y * linesize[2]);
        lum = ((YuvUnit*)(src[0] + offset_y))[0];
        chr.x = ((YuvUnit*)(src[1] + offset_u))[0];
        chr.y = ((YuvUnit*)(src[2] + offset_v))[0];
    }
    __host__ __device__ void write(uint8_t* dst[], int x, int y, int* linesize)
    {
        int offset_y = x * sizeof(YuvUnit) + y * linesize[0];
        int offset_u = (x * sizeof(YuvUnit)) + (y * linesize[1]);
        int offset_v = (x * sizeof(YuvUnit)) + (y * linesize[2]);
        ((YuvUnit*)(dst[0] + offset_y))[0] = lum;
        ((YuvUnit*)(dst[1] + offset_u))[0] = chr.x;
        ((YuvUnit*)(dst[2] + offset_v))[0] = chr.y;
    }
};

template<class SrcUnit, class DstUnit>
__global__ static void nv122yuv420p_kernel(const uint8_t *src1, const uint8_t *src2, 
                                      const uint8_t *src3, const uint8_t *src4,
                                      int src_pitch1, int src_pitch2, int src_pitch3, int src_pitch4,
                                      uint8_t *dst1, uint8_t *dst2, 
                                      uint8_t *dst3, uint8_t *dst4,
                                      int dst_pitch1, int dst_pitch2, int dst_pitch3, int dst_pitch4,
                                      int width, int height) {
    using SrcUnitx2 = Unitx2<SrcUnit>;
    using DstUnitx2 = Unitx2<DstUnit>;

    SrcUnitx2 lum_src01;
    SrcUnitx2 lum_src23;
    SrcUnitx2 chr_src;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width / 2 || y >= height / 2) return;

    lum_src01 = *(SrcUnitx2*)(src1 + x * sizeof(SrcUnitx2) + 2 * y * src_pitch1);
    lum_src23 = *(SrcUnitx2*)(src1 + x * sizeof(SrcUnitx2) + (2 * y + 1) * src_pitch1);
    chr_src = *(SrcUnitx2*)(src2 + x * sizeof(SrcUnitx2) + y * src_pitch2);

    *(DstUnitx2*)(dst1 + x * sizeof(DstUnitx2) + 2 * y * dst_pitch1) = convert_bit_depth<SrcUnitx2, DstUnitx2>(lum_src01);
    *(DstUnitx2*)(dst1 + x * sizeof(DstUnitx2) + (2 * y + 1) * dst_pitch1) = convert_bit_depth<SrcUnitx2, DstUnitx2>(lum_src23);
    *(DstUnit*)(dst2 + x * sizeof(DstUnit) + y * dst_pitch2) = convert_bit_depth<SrcUnit, DstUnit>(chr_src.x);
    *(DstUnit*)(dst3 + x * sizeof(DstUnit) + y * dst_pitch3) = convert_bit_depth<SrcUnit, DstUnit>(chr_src.y);
}


template<class SrcUnit, class DstUnit>
__global__ static void yuv420p2nv12_kernel(const uint8_t *src1, const uint8_t *src2, 
                                      const uint8_t *src3, const uint8_t *src4,
                                      int src_pitch1, int src_pitch2, int src_pitch3, int src_pitch4,
                                      uint8_t *dst1, uint8_t *dst2, 
                                      uint8_t *dst3, uint8_t *dst4,
                                      int dst_pitch1, int dst_pitch2, int dst_pitch3, int dst_pitch4,
                                      int width, int height) {
    using SrcUnitx2 = Unitx2<SrcUnit>;
    using DstUnitx2 = Unitx2<DstUnit>;

    SrcUnitx2 lum_src01;
    SrcUnitx2 lum_src23;
    SrcUnitx2 chr_src;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width / 2 || y >= height / 2) return;

    lum_src01 = *(SrcUnitx2*)(src1 + x * sizeof(SrcUnitx2) + 2 * y * src_pitch1);
    lum_src23 = *(SrcUnitx2*)(src1 + x * sizeof(SrcUnitx2) + (2 * y + 1) * src_pitch1);
    chr_src.x = *(SrcUnit*)(src2 + x * sizeof(SrcUnit) + y * src_pitch2);
    chr_src.y = *(SrcUnit*)(src3 + x * sizeof(SrcUnit) + y * src_pitch3);

    *(DstUnitx2*)(dst1 + x * sizeof(DstUnitx2) + 2 * y * dst_pitch1) = convert_bit_depth<SrcUnitx2, DstUnitx2>(lum_src01);
    *(DstUnitx2*)(dst1 + x * sizeof(DstUnitx2) + (2 * y + 1) * dst_pitch1) = convert_bit_depth<SrcUnitx2, DstUnitx2>(lum_src23);
    *(DstUnitx2*)(dst2 + x * sizeof(DstUnitx2) + y * dst_pitch2) = convert_bit_depth<SrcUnitx2, DstUnitx2>(chr_src);
}

template<class SrcPix, class DstPix>
__global__ static void yuv2yuv_kernel(const uint8_t *src1, const uint8_t *src2, 
                                      const uint8_t *src3, const uint8_t *src4,
                                      int src_pitch1, int src_pitch2, int src_pitch3, int src_pitch4,
                                      uint8_t *dst1, uint8_t *dst2, 
                                      uint8_t *dst3, uint8_t *dst4,
                                      int dst_pitch1, int dst_pitch2, int dst_pitch3, int dst_pitch4,
                                      int width, int height) {
    SrcPix src_pix;
    DstPix dst_pix;
    using SrcUnit = decltype(src_pix.lum);
    using SrcUnitx2 = decltype(src_pix.chr);
    using DstUnit = decltype(dst_pix.lum);
    using DstUnitx2 = decltype(dst_pix.chr);

    const uint8_t* src[4] = {src1, src2, src3, src4};
    uint8_t* dst[4] = {dst1, dst2, dst3, dst4};
    int src_pitch[4] = {src_pitch1, src_pitch2, src_pitch3, src_pitch4};
    int dst_pitch[4] = {dst_pitch1, dst_pitch2, dst_pitch3, dst_pitch4};

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;

    // int src_offset = y * src_pitch + x * sizeof(SrcPix);
    // int dst_offset = y * dst_pitch + x * sizeof(DstPix);
    src_pix.read(src, x, y, src_pitch);
    dst_pix.lum = convert_bit_depth<SrcUnit, DstUnit>(src_pix.lum);
    dst_pix.chr = convert_bit_depth<SrcUnitx2, DstUnitx2>(src_pix.chr);

    dst_pix.write(dst, x, y, dst_pitch);

    return;
}

template<class SrcUnit, class DstUnit>
static void nv122yuv420p_kernel_launch(const uint8_t *src[], int* src_pitch, 
                                            uint8_t *dst[], int* dst_pitch, 
                                            int width, int height, CUstream stream) {
    nv122yuv420p_kernel<SrcUnit, DstUnit>
        <<<dim3((width + 31) / 32 / 2, (height + 3) / 4 / 2), dim3(32, 4), 0, stream>>>
        (src[0], src[1], src[2], src[3], src_pitch[0], src_pitch[1], src_pitch[2], src_pitch[3], 
         dst[0], dst[1], dst[2], dst[3], dst_pitch[0], dst_pitch[1], dst_pitch[2], dst_pitch[3], width, height);
}

template<class SrcUnit, class DstUnit>
static void yuv420p2nv12_kernel_launch(const uint8_t *src[], int* src_pitch, 
                                            uint8_t *dst[], int* dst_pitch, 
                                            int width, int height, CUstream stream) {
    yuv420p2nv12_kernel<SrcUnit, DstUnit>
        <<<dim3((width + 31) / 32 / 2, (height + 3) / 4 / 2), dim3(32, 4), 0, stream>>>
        (src[0], src[1], src[2], src[3], src_pitch[0], src_pitch[1], src_pitch[2], src_pitch[3], 
         dst[0], dst[1], dst[2], dst[3], dst_pitch[0], dst_pitch[1], dst_pitch[2], dst_pitch[3], width, height);
}
template<class SrcPix, class DstPix>
static void yuv2yuv_kernel_launch(const uint8_t *src[], int* src_pitch, 
                                            uint8_t *dst[], int* dst_pitch, 
                                            int width, int height, CUstream stream) {
    yuv2yuv_kernel<SrcPix, DstPix>
        <<<dim3((width + 31) / 32, (height + 3) / 4), dim3(32, 4), 0, stream>>>
        (src[0], src[1], src[2], src[3], src_pitch[0], src_pitch[1], src_pitch[2], src_pitch[3], 
         dst[0], dst[1], dst[2], dst[3], dst_pitch[0], dst_pitch[1], dst_pitch[2], dst_pitch[3], width, height);
}

extern "C" {
#include "libavutil/pixfmt.h"

int yuv2yuv_cuda(const uint8_t *src[], int src_stride[],
                uint8_t *dst[], int dst_stride[],
                int width, int height,
                int srcFormat, int dstFormat, CUstream stream) {
    if (srcFormat == dstFormat) {
        // int img_size = av_image_get_buffer_size(srcFormat, width, height, int align);
        for (int i = 0; i < 4; i++) {
            if (src[i] && dst[i])
            cudaMemcpy2DAsync(dst[i], dst_stride[i], src[i], src_stride[i], width, height, cudaMemcpyDefault, stream);
        }
    }
    else if (srcFormat == AV_PIX_FMT_NV12) {
        switch (dstFormat) {
            case AV_PIX_FMT_YUV420P:
                nv122yuv420p_kernel_launch<Unit8b, Unit8b>(src, src_stride, dst, dst_stride, width, height, stream);
                break;
            case AV_PIX_FMT_P010:
                yuv2yuv_kernel_launch<Pixel_NV12<Unit8b>, Pixel_NV12<Unit10b>>(src, src_stride, dst, dst_stride, width, height, stream);
                break;
            case AV_PIX_FMT_P016:
                yuv2yuv_kernel_launch<Pixel_NV12<Unit8b>, Pixel_NV12<Unit16b>>(src, src_stride, dst, dst_stride, width, height, stream);
                break;
            case AV_PIX_FMT_YUV420P10:
                yuv2yuv_kernel_launch<Pixel_NV12<Unit8b>, Pixel_I420<Unit10b>>(src, src_stride, dst, dst_stride, width, height, stream);
                break;
            case AV_PIX_FMT_YUV420P16:
                yuv2yuv_kernel_launch<Pixel_NV12<Unit8b>, Pixel_I420<Unit16b>>(src, src_stride, dst, dst_stride, width, height, stream);
                break;
        }
    }
    else if (srcFormat == AV_PIX_FMT_YUV420P) {
        switch (dstFormat) {
            case AV_PIX_FMT_NV12:
                yuv420p2nv12_kernel_launch<Unit8b, Unit8b>(src, src_stride, dst, dst_stride, width, height, stream);
                break;
            case AV_PIX_FMT_P010:
                yuv2yuv_kernel_launch<Pixel_I420<Unit8b>, Pixel_NV12<Unit10b>>(src, src_stride, dst, dst_stride, width, height, stream);
                break;
            case AV_PIX_FMT_P016:
                yuv2yuv_kernel_launch<Pixel_I420<Unit8b>, Pixel_NV12<Unit16b>>(src, src_stride, dst, dst_stride, width, height, stream);
                break;
            case AV_PIX_FMT_YUV420P10:
                yuv2yuv_kernel_launch<Pixel_I420<Unit8b>, Pixel_I420<Unit10b>>(src, src_stride, dst, dst_stride, width, height, stream);
                break;
            case AV_PIX_FMT_YUV420P16:
                yuv2yuv_kernel_launch<Pixel_I420<Unit8b>, Pixel_I420<Unit16b>>(src, src_stride, dst, dst_stride, width, height, stream);
                break;
        }
    }

    return -1;
}
}