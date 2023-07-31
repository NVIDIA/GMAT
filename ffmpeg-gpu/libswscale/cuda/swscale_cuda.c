#include "libswscale/swscale.h"
#include "libswscale/swscale_internal.h"

#include <libavutil/imgutils.h>
#include "libavutil/pixfmt.h"
#include "libavutil/pixdesc.h"

#include <libavutil/log.h>
#include <libavutil/mem.h>
// #include <nvcv/Image.h>
#include <nvcv/Tensor.h>
#include <cvcuda/OpResize.h>

#include <cuda_runtime.h>

static int check_nvcv(int err, void* c, int line) {
     if (err != NVCV_SUCCESS) {
        av_log(c, AV_LOG_ERROR, "NVCV error %d at line %d\n", err, line);
    }
    return err;
}

#define CK_NVCV(r) \
    ret = check_nvcv(r, c, __LINE__); \
    // if (ret != NVCV_SUCCESS) return ret;

// #define CK_NVCV(r) \
//     ret = r; \
//     if (ret != NVCV_SUCCESS) { \
//         av_log(c, AV_LOG_ERROR, "NVCV error %d at line %d\n", ret, __LINE__); \
//         return ret; \
//     }

static const enum AVPixelFormat supported_fmts[] = {
    AV_PIX_FMT_YUV420P, AV_PIX_FMT_NV12,
    AV_PIX_FMT_YUV420P10, AV_PIX_FMT_YUV420P16,
    AV_PIX_FMT_P010, AV_PIX_FMT_P016,
    AV_PIX_FMT_YUV444P, 
    // AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_YUVA420P,
    AV_PIX_FMT_RGBA, AV_PIX_FMT_BGRA,
    AV_PIX_FMT_RGB24, AV_PIX_FMT_BGR24,
    AV_PIX_FMT_RGBA64, AV_PIX_FMT_BGRA64,
    AV_PIX_FMT_RGB32, AV_PIX_FMT_BGR32,
};

extern int yuv2rgb_cuda(const uint8_t *dpSrc[], int srcStride[],
                        uint8_t *dpDst[], int dstStride[],
                        int nWidth, int nHeight,
                        int srcFormat, int dstFormat, CUstream stream);
extern int rgb2yuv_cuda(const uint8_t *src[], int srcStride[],
                        uint8_t *dst[], int dstStride[],
                        int nWidth, int nHeight,
                        int srcFormat, int dstFormat, CUstream stream);
extern int yuv2yuv_cuda(const uint8_t *src[], int src_stride[],
                        uint8_t *dst[], int dst_stride[],
                        int width, int height,
                        int srcFormat, int dstFormat, CUstream stream);
extern void set_mat_yuv2rgb_cuda(enum AVColorSpace cspace);
extern void set_mat_rgb2yuv_cuda(enum AVColorSpace cspace);

static int check_formats(enum AVPixelFormat fmt) {
    int size = sizeof(supported_fmts) / sizeof(supported_fmts[0]);
    for (int i = 0; i < size; i++) {
        if (fmt == supported_fmts[i]) return 0;
    }
    return -1;
}

static int map_resize_algo(int flags) {
    if (flags & SWS_BILINEAR) return NVCV_INTERP_LINEAR;
    else if (flags & SWS_BICUBIC) return NVCV_INTERP_CUBIC;
    else if (flags & SWS_AREA) return NVCV_INTERP_AREA;
    else return NVCV_INTERP_LINEAR;
}

void ff_yuv2rgb_init_tables_cuda(SwsContext *c) {
    enum AVColorSpace cspace = c->cspace;
    if (isYUV(c->srcFormat) && isRGB(c->dstFormat)) {
        set_mat_yuv2rgb_cuda(cspace);
    }
    if (isRGB(c->srcFormat) && isYUV(c->dstFormat)) {
        set_mat_rgb2yuv_cuda(cspace);
    }
}

int ff_sws_free_swscale_cuda(SwsContext *c) {
    NVCVTensorHandle cv_resize_tensor = c->cv_resize_tensor;
    NVCVImageBatchHandle cv_in_batch_handle = c->cv_in_batch_handle;
    NVCVImageBatchHandle cv_out_batch_handle = c->cv_out_batch_handle;
    NVCVTensorLayout   *cv_layout      = c->cv_layout;
    int ret = 0;

    if (c->convert_unscaled) return 0;

    if (cv_resize_tensor) nvcvTensorDecRef(cv_resize_tensor, NULL);
    CK_NVCV(nvcvImageBatchVarShapeClear(cv_in_batch_handle));
    CK_NVCV(nvcvImageBatchVarShapeClear(cv_out_batch_handle));
    for (int i = 0; i < 4; i++) {
        nvcvImageDecRef(c->cv_images[i], NULL);
        nvcvImageDecRef(c->cv_out_images[i], NULL);
    }
    nvcvImageBatchDecRef(cv_in_batch_handle, NULL);
    nvcvImageBatchDecRef(cv_out_batch_handle, NULL);
    av_free(cv_layout);
    // av_free(resize_handle);
    // av_free(cv_resize_tensor);

    return 0;
}

// TODO: add SwsContext.conversion_type (rgb2rgb, rgb2yuv, yuv2yuv, yuv2rgb)
int ff_sws_init_swscale_cuda(SwsContext *c) {
    int src_w = c->srcW;
    int src_h = c->srcH;
    int dst_w = c->dstW;
    int dst_h = c->dstH;
    enum AVPixelFormat srcFormat = c->srcFormat;
    enum AVPixelFormat dstFormat = c->dstFormat;
    const AVPixFmtDescriptor *desc_src = av_pix_fmt_desc_get(srcFormat);
    const AVPixFmtDescriptor *desc_dst = av_pix_fmt_desc_get(dstFormat);
    int ret;

    // NVCVOperatorHandle *resize_handle = av_malloc(sizeof(NVCVOperatorHandle));
    // NVCVTensorHandle   cv_in_handle, cv_out_handle;
    NVCVOperatorHandle resize_handle;
    NVCVTensorLayout   *cv_layout     = av_malloc(sizeof(NVCVTensorLayout));
    // NVCVTensorLayout   cv_layout;
    NVCVTensorRequirements cv_req;
    NVCVImageRequirements  cv_img_req;
    NVCVImageBatchVarShapeRequirements cv_var_req;
    // NVCVTensorHandle   *cv_resize_tensor = av_malloc(sizeof(NVCVTensorHandle));
    NVCVTensorHandle   cv_resize_tensor;
    NVCVImageBatchHandle cv_resize_img_batch;

    NVCVImageHandle cv_y;
    NVCVImageHandle cv_u;
    NVCVImageHandle cv_v;
    NVCVImageHandle cv_a;
    // NVCVImageHandle cv_images[4];

    int64_t tshape[4];
    int yuv_scale = isYUV(srcFormat) && isYUV(dstFormat);

    if ((ret = cvcudaResizeCreate(&resize_handle)) != NVCV_SUCCESS) {
        printf("Error creating resize operator: %d\n", ret);
    }

    if ((ret = nvcvTensorLayoutMake("NHWC", cv_layout)) != NVCV_SUCCESS) {
        printf("Error creating resize input layout: %d\n", ret);
    }
    c->cv_resize_handle = resize_handle;
    c->cv_layout        = cv_layout;

    if (yuv_scale) {

        if (dstFormat == AV_PIX_FMT_NV12) {
            CK_NVCV(nvcvImageBatchVarShapeCalcRequirements(4, &cv_var_req));
            CK_NVCV(nvcvImageBatchVarShapeConstruct(&cv_var_req, NULL, (NVCVImageBatchHandle*)&c->cv_in_batch_handle));
            CK_NVCV(nvcvImageBatchVarShapeConstruct(&cv_var_req, NULL, (NVCVImageBatchHandle*)&c->cv_out_batch_handle));

            if (srcFormat == AV_PIX_FMT_NV12) {
                // image for input y
                CK_NVCV(nvcvImageCalcRequirements(src_w, src_h, NVCV_IMAGE_FORMAT_U8, 1, 1, &cv_img_req));
                CK_NVCV(nvcvImageConstruct(&cv_img_req, NULL, &cv_y));
                // image for input u
                CK_NVCV(nvcvImageCalcRequirements(src_w >> 1, 
                    src_h >> 1, NVCV_IMAGE_FORMAT_U8, 1, 1, &cv_img_req));
                CK_NVCV(nvcvImageConstruct(&cv_img_req, NULL, &cv_u));
                // image for input v
                CK_NVCV(nvcvImageCalcRequirements(src_w >> 1, 
                    src_h >> 1, NVCV_IMAGE_FORMAT_U8, 1, 1, &cv_img_req));
                CK_NVCV(nvcvImageConstruct(&cv_img_req, NULL, &cv_v));

                c->cv_images[0] = cv_y;
                c->cv_images[1] = cv_u;
                c->cv_images[2] = cv_v;
                CK_NVCV(nvcvImageBatchVarShapePushImages(c->cv_in_batch_handle, (NVCVImageHandle*)(c->cv_images), 3));
            }

            // image for output y
            CK_NVCV(nvcvImageCalcRequirements(dst_w, dst_h, NVCV_IMAGE_FORMAT_U8, 1, 1, &cv_img_req));
            CK_NVCV(nvcvImageConstruct(&cv_img_req, NULL, &cv_y));
            // image for output u
            CK_NVCV(nvcvImageCalcRequirements(dst_w >> 1, 
                dst_h >> 1, NVCV_IMAGE_FORMAT_U8, 1, 1, &cv_img_req));
            CK_NVCV(nvcvImageConstruct(&cv_img_req, NULL, &cv_u));
            // image for output v
            CK_NVCV(nvcvImageCalcRequirements(dst_w >> 1, 
                dst_h >> 1, NVCV_IMAGE_FORMAT_U8, 1, 1, &cv_img_req));
            CK_NVCV(nvcvImageConstruct(&cv_img_req, NULL, &cv_v));

            c->cv_out_images[0] = cv_y;
            c->cv_out_images[1] = cv_u;
            c->cv_out_images[2] = cv_v;
            CK_NVCV(nvcvImageBatchVarShapePushImages(c->cv_out_batch_handle, (NVCVImageHandle*)(c->cv_out_images), 3));
            return 0;
            // c->cv_in_batch_handle = cv_resize_img_batch;
        }
        // av_log(c, AV_LOG_ERROR, "yuv scaling is not supported yet\n");
        // return -1;
        // cv_resize_img_batch = av_malloc(sizeof(NVCVImageBatchHandle));
        // c->cv_out_batch_handle = av_malloc(sizeof(NVCVImageBatchHandle));
        CK_NVCV(nvcvImageBatchVarShapeCalcRequirements(4, &cv_var_req));
        CK_NVCV(nvcvImageBatchVarShapeConstruct(&cv_var_req, NULL, &cv_resize_img_batch));
        // CK_NVCV(nvcvImageBatchVarShapeConstruct(&cv_var_req, NULL, (NVCVImageBatchHandle*)&c->cv_out_batch_handle));

        // image for y
        CK_NVCV(nvcvImageCalcRequirements(src_w, src_h, NVCV_IMAGE_FORMAT_U8, 1, 1, &cv_img_req));
        CK_NVCV(nvcvImageConstruct(&cv_img_req, NULL, &cv_y));
        // image for u
        CK_NVCV(nvcvImageCalcRequirements(src_w >> desc_src->log2_chroma_w, 
            src_h >> desc_src->log2_chroma_h, NVCV_IMAGE_FORMAT_U8, 1, 1, &cv_img_req));
        CK_NVCV(nvcvImageConstruct(&cv_img_req, NULL, &cv_u));
        // image for v
        CK_NVCV(nvcvImageCalcRequirements(src_w >> desc_src->log2_chroma_w, 
            src_h >> desc_src->log2_chroma_h, NVCV_IMAGE_FORMAT_U8, 1, 1, &cv_img_req));
        CK_NVCV(nvcvImageConstruct(&cv_img_req, NULL, &cv_v));
        // image for alpha
        if (isALPHA(dstFormat)) {
            CK_NVCV(nvcvImageCalcRequirements(src_w, src_h, NVCV_IMAGE_FORMAT_U8, 1, 1, &cv_img_req));
            CK_NVCV(nvcvImageConstruct(&cv_img_req, NULL, &cv_a));
        }
        if (dstFormat == AV_PIX_FMT_NV12) {
            c->cv_out_images[0] = cv_y;
            c->cv_out_images[1] = cv_u;
            c->cv_out_images[2] = cv_v;
            c->cv_out_images[3] = cv_a;
            
            CK_NVCV(nvcvImageBatchVarShapePushImages(cv_resize_img_batch, (NVCVImageHandle*)(c->cv_out_images),
                                                    isALPHA(srcFormat)? 4 : 3));
            c->cv_out_batch_handle = cv_resize_img_batch;
            CK_NVCV(nvcvImageBatchVarShapeConstruct(&cv_var_req, NULL, (NVCVImageBatchHandle*)&c->cv_in_batch_handle));
        }
        else {
            c->cv_images[0] = cv_y;
            c->cv_images[1] = cv_u;
            c->cv_images[2] = cv_v;
            c->cv_images[3] = cv_a;
            CK_NVCV(nvcvImageBatchVarShapePushImages(cv_resize_img_batch, (NVCVImageHandle*)(c->cv_images),
                                                        isALPHA(srcFormat)? 4 : 3));
            c->cv_in_batch_handle = cv_resize_img_batch;
            CK_NVCV(nvcvImageBatchVarShapeConstruct(&cv_var_req, NULL, (NVCVImageBatchHandle*)&c->cv_out_batch_handle));
        }

        return 0;
    }

    if (isRGB(srcFormat)) {
        tshape[0] = 1;
        tshape[1] = dst_h;
        tshape[2] = dst_w;
        tshape[3] = desc_src->nb_components;
    // } else if (isRGB(dstFormat)) {
    } else if (isRGB(dstFormat)) {
        tshape[0] = 1;
        tshape[1] = src_h;
        tshape[2] = src_w;
        tshape[3] = desc_dst->nb_components;
    }
    if ((ret = nvcvTensorCalcRequirements(4, tshape, NVCV_DATA_TYPE_U8, *cv_layout, 1, 1, &cv_req)) != NVCV_SUCCESS) {
        printf("Error calculating output tensor: %d\n", ret);
    }
    if ((ret = nvcvTensorConstruct(&cv_req, NULL, &cv_resize_tensor)) != NVCV_SUCCESS) {
        printf("Error contructing output tensor: %d\n", ret);
    }
    c->cv_resize_tensor = cv_resize_tensor;

    ff_yuv2rgb_init_tables_cuda(c);

    return ret;
}

int ff_swscale_cuda(SwsContext *c, const uint8_t *src[],
                   int srcStride[], int srcSliceY, int srcSliceH,
                   uint8_t *dst[], int dstStride[],
                   int dstSliceY, int dstSliceH)
{
    int src_w = c->srcW;
    int src_h = c->srcH;
    int dst_w = c->dstW;
    int dst_h = c->dstH;
    enum AVPixelFormat srcFormat = c->srcFormat;
    enum AVPixelFormat dstFormat = c->dstFormat;
    const AVPixFmtDescriptor *desc_src = av_pix_fmt_desc_get(srcFormat);
    const AVPixFmtDescriptor *desc_dst = av_pix_fmt_desc_get(dstFormat);
    int ret = 0;
    uint8_t *cv_ptr[4] = {0};
    int cv_linesizes[4] = {0};

    NVCVTensorHandle   cv_in_handle, cv_out_handle;
    NVCVOperatorHandle resize_handle = (NVCVOperatorHandle)c->cv_resize_handle;
    NVCVTensorLayout   *cv_layout     = (NVCVTensorLayout*)c->cv_layout;
    NVCVTensorHandle   cv_resize_tensor = (NVCVTensorHandle)c->cv_resize_tensor;
    NVCVTensorData     cv_in_data = {.dtype = NVCV_DATA_TYPE_U8,
                                     .rank = 4,
                                    //  .shape = {src_h, src_w, desc_src->nb_components},
                                     .bufferType = NVCV_TENSOR_BUFFER_STRIDED_CUDA,
                                    };
    NVCVTensorData     cv_out_data = {.dtype = NVCV_DATA_TYPE_U8,
                                     .rank = 4,
                                    //  .shape = {dst_h, dst_w, desc_dst->nb_components},
                                     .bufferType = NVCV_TENSOR_BUFFER_STRIDED_CUDA,
                                    };
    NVCVTensorData     cv_resize_data;
    NVCVInterpolationType interp_type = map_resize_algo(c->cspace);

    if ((check_formats(srcFormat) < 0) || (check_formats(dstFormat) < 0)) {
        av_log(c, AV_LOG_ERROR, "Unsupported pixel format\n");
            return -1;
    }

    if (isRGB(srcFormat) && isYUV(dstFormat)) {
        // cv_in_data.shape = {src_h, src_w, desc_src->nb_components};
        cv_in_data.shape[0] = 1;
        cv_in_data.shape[1] = src_h;
        cv_in_data.shape[2] = src_w;
        cv_in_data.shape[3] = desc_src->nb_components;
        cv_in_data.buffer.strided.strides[0] = src_h * srcStride[0];
        cv_in_data.buffer.strided.strides[1] = srcStride[0];
        cv_in_data.buffer.strided.strides[2] = desc_src->nb_components;
        cv_in_data.buffer.strided.strides[3] = 1;
        cv_in_data.buffer.strided.basePtr    = (NVCVByte*)src[0];
        cv_in_data.layout                    = *cv_layout;

        CK_NVCV(ret = nvcvTensorWrapDataConstruct(&cv_in_data, NULL, NULL, &cv_in_handle));
        CK_NVCV(ret = cvcudaResizeSubmit(resize_handle, c->cuda_stream, cv_in_handle,
                                        cv_resize_tensor, interp_type));
        CK_NVCV(ret = nvcvTensorExportData(cv_resize_tensor, &cv_resize_data));
        if (cv_resize_data.shape[1] != dst_h || cv_resize_data.shape[2] != dst_w) {
            av_log(c, AV_LOG_ERROR, "Size error after resize: %ldx%ld vs %dx%d\n", 
                cv_resize_data.shape[1], cv_resize_data.shape[2], dst_h, dst_w);
            return -1;
        }

        cv_ptr[0] = (uint8_t*)cv_resize_data.buffer.strided.basePtr;
        cv_linesizes[0] = cv_resize_data.buffer.strided.strides[1];
        ret = rgb2yuv_cuda((const uint8_t**)cv_ptr, cv_linesizes,
                        dst, dstStride,
                        dst_w, dst_h,
                        srcFormat, dstFormat, c->cuda_stream);
        nvcvTensorDecRef(cv_in_handle, NULL);
    } else if (isRGB(dstFormat) && isYUV(srcFormat)) {
        CK_NVCV(ret = nvcvTensorExportData(cv_resize_tensor, &cv_resize_data));
        if (cv_resize_data.shape[1] != src_h || cv_resize_data.shape[2] != src_w) {
            av_log(c, AV_LOG_ERROR, "Size error before resize: %ldx%ld vs %dx%d\n", 
                cv_resize_data.shape[1], cv_resize_data.shape[2], src_h, src_w);
            return -1;
        }
        cv_ptr[0] = (uint8_t*)cv_resize_data.buffer.strided.basePtr;
        cv_linesizes[0] = cv_resize_data.buffer.strided.strides[1];

        ret = yuv2rgb_cuda(src, srcStride,
                        cv_ptr, cv_linesizes,
                        src_w, src_h,
                        srcFormat, dstFormat, c->cuda_stream);

        cv_out_data.shape[0] = 1;
        cv_out_data.shape[1] = dst_h;
        cv_out_data.shape[2] = dst_w;
        cv_out_data.shape[3] = desc_dst->nb_components;
        cv_out_data.buffer.strided.strides[0] = dst_h * dstStride[0];
        cv_out_data.buffer.strided.strides[1] = dstStride[0];
        cv_out_data.buffer.strided.strides[2] = desc_dst->nb_components;
        cv_out_data.buffer.strided.strides[3] = 1;
        cv_out_data.buffer.strided.basePtr    = (NVCVByte*)dst[0];
        cv_out_data.layout                    = *cv_layout;

        CK_NVCV(ret = nvcvTensorWrapDataConstruct(&cv_out_data, NULL, NULL, &cv_out_handle));
        CK_NVCV(ret = cvcudaResizeSubmit(resize_handle, c->cuda_stream, cv_resize_tensor,
                                        cv_out_handle, interp_type));
        nvcvTensorDecRef(cv_out_handle, NULL);
    } else if (isYUV(srcFormat) && isYUV(dstFormat)) {
        // export resize input images
        // used as output of yuv2yuv conversion
        uint8_t *cv_img_ptr[8];
        int cv_img_pitch[8];
        NVCVImageData image_in_data[4];
        NVCVImageData image_out_data[4];
        NVCVImageHandle cv_resize_in_img[4];
        NVCVImageHandle cv_resize_out_img[4];

        // special path: nv12 scaling
        if (srcFormat == AV_PIX_FMT_NV12 && dstFormat == AV_PIX_FMT_NV12) {
            for (int i = 0; i < desc_dst->nb_components; i++) {
                CK_NVCV(ret = nvcvImageExportData(((NVCVImageHandle*)c->cv_images)[i], &image_in_data[i]));
                cv_img_ptr[i] = (uint8_t*)image_in_data[i].buffer.strided.planes[0].basePtr;
                cv_img_pitch[i] = image_in_data[i].buffer.strided.planes[0].rowStride;
            }

            ret = yuv2yuv_cuda(src, srcStride, cv_img_ptr, cv_img_pitch,
                                src_w, src_h, AV_PIX_FMT_NV12, AV_PIX_FMT_YUV420P, c->cuda_stream);
        }
        // special path: dstFormat == nv12
        // scale first, then convert
        else if (dstFormat == AV_PIX_FMT_NV12) {
            for (int i = 0; i < desc_src->nb_components; i++) {
                image_in_data[i].format = desc_src->comp[i].depth == 8 ? NVCV_IMAGE_FORMAT_U8 : NVCV_IMAGE_FORMAT_U16;
                image_in_data[i].bufferType = NVCV_TENSOR_BUFFER_STRIDED_CUDA;
                image_in_data[i].buffer.strided.numPlanes = 1;
                image_in_data[i].buffer.strided.planes[0].rowStride = srcStride[i];
                image_in_data[i].buffer.strided.planes[0].width  = (i == 0 || i == 3) ? src_w : src_w >> desc_src->log2_chroma_w;
                image_in_data[i].buffer.strided.planes[0].height = (i == 0 || i == 3) ? src_h : src_h >> desc_src->log2_chroma_h;
                image_in_data[i].buffer.strided.planes[0].basePtr = (NVCVByte*)src[i];
                CK_NVCV(nvcvImageWrapDataConstruct(&image_in_data[i], NULL, NULL, &cv_resize_in_img[i]));
            }
            CK_NVCV(nvcvImageBatchVarShapePushImages((NVCVImageBatchHandle)c->cv_in_batch_handle, cv_resize_in_img,
                                                    desc_src->nb_components));
        }
        else {
            for (int i = 0; i < desc_dst->nb_components; i++) {
                CK_NVCV(ret = nvcvImageExportData(((NVCVImageHandle*)c->cv_images)[i], &image_in_data[i]));
                cv_img_ptr[i] = (uint8_t*)image_in_data[i].buffer.strided.planes[0].basePtr;
                cv_img_pitch[i] = image_in_data[i].buffer.strided.planes[0].rowStride;
            }

            // wrap dst ptr as images
            // used as output of resize images
            for (int i = 0; i < desc_dst->nb_components; i++) {
                image_out_data[i].format = desc_dst->comp[i].depth == 8 ? NVCV_IMAGE_FORMAT_U8 : NVCV_IMAGE_FORMAT_U16;
                image_out_data[i].bufferType = NVCV_TENSOR_BUFFER_STRIDED_CUDA;
                image_out_data[i].buffer.strided.numPlanes = 1;
                image_out_data[i].buffer.strided.planes[0].rowStride = dstStride[i];
                image_out_data[i].buffer.strided.planes[0].width  = (i == 0 || i == 3) ? dst_w : dst_w >> desc_dst->log2_chroma_w;
                image_out_data[i].buffer.strided.planes[0].height = (i == 0 || i == 3) ? dst_h : dst_h >> desc_dst->log2_chroma_h;
                image_out_data[i].buffer.strided.planes[0].basePtr    = (NVCVByte*)dst[i];
                CK_NVCV(nvcvImageWrapDataConstruct(&image_out_data[i], NULL, NULL, &cv_resize_out_img[i]));
            }
            CK_NVCV(nvcvImageBatchVarShapePushImages((NVCVImageBatchHandle)c->cv_out_batch_handle, cv_resize_out_img,
                                                    desc_dst->nb_components));

            // yuv2yuv conversion
            if (srcFormat != dstFormat) {
                ret = yuv2yuv_cuda(src, srcStride, cv_img_ptr, cv_img_pitch,
                                src_w, src_h, srcFormat, dstFormat, c->cuda_stream);
            }
        }

        CK_NVCV(ret = cvcudaResizeVarShapeSubmit(resize_handle, c->cuda_stream,
                                                (NVCVImageBatchHandle)c->cv_in_batch_handle,
                                                (NVCVImageBatchHandle)c->cv_out_batch_handle,
                                                interp_type));

        if (srcFormat == AV_PIX_FMT_NV12 && dstFormat == AV_PIX_FMT_NV12) {
            for (int i = 0; i < desc_src->nb_components; i++) {
                CK_NVCV(ret = nvcvImageExportData(((NVCVImageHandle*)c->cv_out_images)[i], &image_out_data[i]));
                cv_img_ptr[i] = (uint8_t*)image_out_data[i].buffer.strided.planes[0].basePtr;
                cv_img_pitch[i] = image_out_data[i].buffer.strided.planes[0].rowStride;
            }

            ret = yuv2yuv_cuda((const uint8_t**)cv_img_ptr, cv_img_pitch, dst, dstStride,
                                dst_w, dst_h, AV_PIX_FMT_YUV420P, AV_PIX_FMT_NV12, c->cuda_stream);
        } else {
            if (dstFormat == AV_PIX_FMT_NV12) {
                for (int i = 0; i < desc_src->nb_components; i++) {
                    CK_NVCV(ret = nvcvImageExportData(((NVCVImageHandle*)c->cv_out_images)[i], &image_out_data[i]));
                    cv_img_ptr[i] = (uint8_t*)image_out_data[i].buffer.strided.planes[0].basePtr;
                    cv_img_pitch[i] = image_out_data[i].buffer.strided.planes[0].rowStride;
                }
                // uint8_t* dbg_buf = malloc(dst_w * dst_h * 3 / 2);
                // cudaMemcpy(dbg_buf, cv_img_ptr[0], dst_w * dst_h, cudaMemcpyDefault);
                // cudaMemcpy(dbg_buf + dst_w * dst_h, cv_img_ptr[1], dst_w * dst_h / 4, cudaMemcpyDefault);
                // cudaMemcpy(dbg_buf + dst_w * dst_h * 5/4, cv_img_ptr[2], dst_w * dst_h / 4, cudaMemcpyDefault);
                // free(dbg_buf);
                ret = yuv2yuv_cuda((const uint8_t**)cv_img_ptr, cv_img_pitch, dst, dstStride,
                                dst_w, dst_h, srcFormat, AV_PIX_FMT_NV12, c->cuda_stream);
            }
            
            CK_NVCV(ret = nvcvImageBatchVarShapePopImages((NVCVImageBatchHandle)c->cv_out_batch_handle, desc_dst->nb_components));
            CK_NVCV(nvcvImageBatchVarShapeClear((NVCVImageBatchHandle)c->cv_out_batch_handle));
            for (int i = 0; i < desc_dst->nb_components; i++) {
                nvcvImageDecRef(cv_resize_in_img[i], NULL);
                nvcvImageDecRef(cv_resize_out_img[i], NULL);
            }
        }

    }

    return ret;
}