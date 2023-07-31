#pragma once

#include "AvFilt.h"
extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/hwcontext_cuda.h>
}

class VidFilt : public AvFilt {
public:
    VidFilt(AVPixelFormat eInputFormat, int nWidth, int nHeight, AVRational timebase, AVRational sar, const char *szFilterDesc, AVBufferRef *ffFrameCtx=NULL)
        : VidFilt(eInputFormat, nWidth, nHeight, timebase, sar, szFilterDesc, false, ffFrameCtx) {}
    bool Filter(uint8_t *pFrame, int nPitch, int64_t pts, std::vector<AVFrame *> &vFrm) {
        // AVPixelFormat pixFmt = (AVPixelFormat)(m_ffFrameCtx ? ((AVHWFramesContext *)m_ffFrameCtx->data)->sw_format : m_frm->format);
        AVPixelFormat pixFmt;
        if (m_ffFrameCtx) {
            AVHWFramesContext* ctx = (AVHWFramesContext*)m_ffFrameCtx->data;
            m_frm->buf[0] = av_buffer_create(pFrame, GetFrameSize(), cudaBufferFree, ctx, 0);

            pixFmt = (AVPixelFormat)(ctx->sw_format);
        }
        ckav(av_image_fill_arrays(m_frm->data, m_frm->linesize, pFrame, 
            pixFmt, m_frm->width, m_frm->height, nPitch ? PitchToAlignment(nPitch, m_frm->width) : 1));
        m_frm->pts = pts == AV_NOPTS_VALUE ? m_pts++ : pts;
        return Filter(m_frm, vFrm);
    }
    using AvFilt::Filter;
    int GetFrameSize() {
        return av_image_get_buffer_size((AVPixelFormat)m_frm->format, m_frm->width, m_frm->height, 1);
    }

protected:
    VidFilt(AVPixelFormat eInputFormat, int nWidth, int nHeight, AVRational timebase, AVRational sar, const char *szFilterDesc, bool bOutputNv12, AVBufferRef *ffFrameCtx)
        : AvFilt(ffFrameCtx) {
        char args[512];
        sprintf(args, "pix_fmt=%d:video_size=%dx%d:time_base=%d/%d:pixel_aspect=%d/%d",
            eInputFormat, nWidth, nHeight, timebase.num, timebase.den, sar.num, sar.den);
        ckav(avfilter_graph_create_filter(&m_filterIn, avfilter_get_by_name("buffer"), "in", args, NULL, m_filterGraph));
        ckav(avfilter_graph_create_filter(&m_filterOut, avfilter_get_by_name("buffersink"), "out", NULL, NULL, m_filterGraph));
        if (bOutputNv12) {
            enum AVPixelFormat pix_fmts[] = { m_ffFrameCtx ? AV_PIX_FMT_CUDA : AV_PIX_FMT_NV12, AV_PIX_FMT_NONE };
            ckav(av_opt_set_int_list(m_filterOut, "pix_fmts", pix_fmts, AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN));
        }

        if (m_ffFrameCtx) {
            int err = 0;
            AVBufferSrcParameters *par = av_buffersrc_parameters_alloc();
            if (!par) {
                LOG(ERROR) <<"Failed to create filter params\n";
                return;
            }
            memset(par, 0, sizeof(*par));
            par->format = AV_PIX_FMT_CUDA;

            par->hw_frames_ctx = m_ffFrameCtx;
            err = av_buffersrc_parameters_set(m_filterIn, par);
            // err = av_buffersrc_parameters_set(m_filterOut, par);
            if (err < 0) {
                LOG(ERROR) <<"Failed to set filter params\n";
                return;
            }
        }
        
        ParseFilter(m_filterGraph, szFilterDesc);
        m_frm->format = eInputFormat;
        m_frm->width = nWidth;
        m_frm->height = nHeight;
    }
    static void cudaBufferFree(void *opaque, uint8_t *data)
    {
        // AVHWFramesContext        *ctx = (AVHWFramesContext*)opaque;
        // AVHWDeviceContext *device_ctx = (AVCUDADeviceContext*)ctx->device_ctx;
        // AVCUDADeviceContext    *hwctx = device_ctx->hwctx;

        // CUcontext dummy;

        // cuCtxPushCurrent(hwctx->cuda_ctx);

        cuMemFree((CUdeviceptr)data);

        // cuCtxPopCurrent(&dummy);
    }

};
