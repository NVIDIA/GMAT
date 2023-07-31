#include <libavutil/imgutils.h>
#include <libavutil/pixfmt.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
#include <libavutil/log.h>

#include <cuda_runtime.h>

struct SwsContext* SwscaleCuda_Nv12ToRgbpf32_Init(int w, int h){
    return sws_getContext(w, h, AV_PIX_FMT_NV12,
                        w, h, AV_PIX_FMT_RGBPF32LE,
                        SWS_HWACCEL_CUDA, NULL, NULL, NULL);
}

int SwscaleCuda_Nv12ToRgbpf32_Convert(struct SwsContext* swsCtx, uint8_t* src, int srcStride, 
                                        uint8_t* dst, int dstStride, int w, int h, cudaStream_t stream) {
    int ret;
    // int srcH = swsCtx->srcH;
    // int srcW = swsCtx->srcW;
    // int dstH = swsCtx->dstH;
    // int dstW = swsCtx->dstW;
    int sliceY = 0, sliceH = h;
    int srcLinesizes[4], dstLinesizes[4];
    uint8_t* srcData[4], *dstData[4];

    ret = av_image_fill_linesizes(srcLinesizes, AV_PIX_FMT_NV12, w);
    ret = av_image_fill_pointers(srcData, AV_PIX_FMT_NV12, h, src, srcLinesizes);
    ret = av_image_fill_linesizes(dstLinesizes, AV_PIX_FMT_RGBPF32LE, w);
    ret = av_image_fill_pointers(dstData, AV_PIX_FMT_RGBPF32LE, h, dst, dstLinesizes);

    if (ret < 0) {
        av_log(swsCtx, AV_LOG_ERROR, "Error filling memory pointers\n");
    }
    sws_setCudaStream(swsCtx, stream);
    return sws_scale(swsCtx, srcData, srcLinesizes, sliceY, sliceH, dstData, dstLinesizes);
}

void SwscaleCuda_Nv12ToRgbpf32_Delete(struct SwsContext* swsCtx) {
    sws_freeContext_cuda(swsCtx);
}
