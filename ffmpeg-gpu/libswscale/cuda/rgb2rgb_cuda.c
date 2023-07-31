#include "libavutil/hwcontext_cuda.h"

// #include "rgb2rgb_cuda.h"
#include "libswscale/rgb2rgb.h"

// av_cold void ff_sws_rgb2rgb_init_cuda(void)
// {
//     // rgb2rgb_init_c();
// }

// SwsFunc ff_getSwsCudaFunc(SwsContext *c)
// {

// }

// void ff_get_unscaled_swscale_cuda(SwsContext *c)
// {
//     const enum AVPixelFormat srcFormat = c->srcFormat;
//     const enum AVPixelFormat dstFormat = c->dstFormat;
//     const int flags = c->flags;
//     const int dstH = c->dstH;
//     const int dstW = c->dstW;
//     if (srcFormat == dstFormat) {

//     }
//     if ((srcFormat == AV_PIX_FMT_RGB24 && dstFormat == AV_PIX_FMT_BGR24) || 
//         (srcFormat == AV_PIX_FMT_RGB24 && dstFormat == AV_PIX_FMT_BGR24)) {
//         c->swscale = rgb24tobgr24_cuda;
//     }
// }