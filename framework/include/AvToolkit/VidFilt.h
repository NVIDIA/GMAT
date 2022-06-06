#pragma once

#include "AvFilt.h"
extern "C" {
#include <libavutil/imgutils.h>
}

class VidFilt : public AvFilt {
public:
    VidFilt(AVPixelFormat eInputFormat, int nWidth, int nHeight, AVRational timebase, AVRational sar, const char *szFilterDesc)
        : VidFilt(eInputFormat, nWidth, nHeight, timebase, sar, szFilterDesc, false) {}
    bool Filter(uint8_t *pFrame, int nPitch, int64_t pts, std::vector<AVFrame *> &vFrm) {
        ckav(av_image_fill_arrays(m_frm->data, m_frm->linesize, pFrame, 
            (AVPixelFormat)m_frm->format, m_frm->width, m_frm->height, nPitch ? PitchToAlignment(nPitch, m_frm->width) : 1));
        m_frm->pts = pts == AV_NOPTS_VALUE ? m_pts++ : pts;
        return Filter(m_frm, vFrm);
    }
    using AvFilt::Filter;
    int GetFrameSize() {
        return av_image_get_buffer_size((AVPixelFormat)m_frm->format, m_frm->width, m_frm->height, 1);
    }

protected:
    VidFilt(AVPixelFormat eInputFormat, int nWidth, int nHeight, AVRational timebase, AVRational sar, const char *szFilterDesc, bool bOutputNv12) {
        char args[512];
        sprintf(args, "pix_fmt=%d:video_size=%dx%d:time_base=%d/%d:pixel_aspect=%d/%d",
            eInputFormat, nWidth, nHeight, timebase.num, timebase.den, sar.num, sar.den);
        ckav(avfilter_graph_create_filter(&m_filterIn, avfilter_get_by_name("buffer"), "in", args, NULL, m_filterGraph));
        ckav(avfilter_graph_create_filter(&m_filterOut, avfilter_get_by_name("buffersink"), "out", NULL, NULL, m_filterGraph));
        if (bOutputNv12) {
            enum AVPixelFormat pix_fmts[] = { AV_PIX_FMT_NV12, AV_PIX_FMT_NONE };
            ckav(av_opt_set_int_list(m_filterOut, "pix_fmts", pix_fmts, AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN));
        }

        ParseFilter(m_filterGraph, szFilterDesc);
        m_frm->format = eInputFormat;
        m_frm->width = nWidth;
        m_frm->height = nHeight;
    }
};
