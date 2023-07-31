#include <cstdint>
#include <cuda.h>

#include "datatypes.h"

template <class Rgb>
__global__ static void rgb2bgr_kernel(const uint8_t *src, uint8_t *dst, int srcStride, int dstStride, int width, int height) {
    int tidX = threadIdx.x + blockIdx.x * blockDim.x;
    int tidY = threadIdx.y + blockIdx.y * blockDim.y;

    if (tidX >= width || tidY >= height) return;

    // int rgbSrcStride = srcStride / sizeof(Rgb);
    // int rgbDstStride = dstStride / sizeof(Rgb);
    Rgb *rgb_src = (Rgb*)src;
    Rgb* rgb_dst = (Rgb*)dst;

    Rgb pixel = *(Rgb*)(src + tidX * sizeof(Rgb) + tidY * srcStride);
    (*(Rgb*)(dst + tidX * sizeof(Rgb) + tidY * dstStride)).c.r = pixel.c.b;
    (*(Rgb*)(dst + tidX * sizeof(Rgb) + tidY * dstStride)).c.g = pixel.c.g;
    (*(Rgb*)(dst + tidX * sizeof(Rgb) + tidY * dstStride)).c.b = pixel.c.r;
}

extern "C"
{
#include "libswscale/rgb2rgb.h"

// void rgb24tobgr24_cuda(SwsContext *c, const uint8_t *src[],
//                         int srcStride[], int srcSliceY,
//                         int srcSliceH, uint8_t *dstParam[],
//                         int dstStride[]) {
//     rgb2bgr_kernel<RGB24>
//         <<<dim3(c->srcW / 32 + 1, c->srcH / 16 + 1), dim3(32, 16)>>>
//         (src[0], dst[0], srcStride[0], dstStride[0], c->srcW, c->srcH);
// }

void rgb24tobgr24_cuda(const uint8_t *src[], uint8_t *dst[], int srcStride[], int dstStride[], int width, int height, CUstream stream)
{
    rgb2bgr_kernel<RGB24>
        <<<dim3((width + 31) / 32, (height + 15) / 16), dim3(32, 16), 0, stream>>>
        (src[0], dst[0], srcStride[0], dstStride[0], width, height);
}

void rgb2rgb_init_cuda(void)
{
    // rgb24tobgr24 = rgb24tobgr24_cuda;
}

} // extern C