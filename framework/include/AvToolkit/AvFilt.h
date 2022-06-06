#pragma once

#include <vector>
#include <map>
#include <assert.h>
extern "C" {
#include <libavformat/avformat.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/opt.h>
}
#include "Logger.h"
#include "AvCommon.h"

extern simplelogger::Logger *logger;

class AvFilt {
public:
    AvFilt() {
        m_filterGraph = cknn(avfilter_graph_alloc());
        m_frm = cknn(av_frame_alloc());
    }
    virtual ~AvFilt() {
        av_frame_free(&m_frm);
        avfilter_graph_free(&m_filterGraph);
    }
    bool Filter(std::vector<AVFrame *> vFrmSrc, std::vector<AVFrame *> &vFrmDst) {
        vFrmDst.clear();
        for (AVFrame *frmIn : vFrmSrc) {
            ckav(av_buffersrc_add_frame_flags(m_filterIn, frmIn, AV_BUFFERSRC_FLAG_KEEP_REF));
            while (true) {
                if (m_vFrm.size() <= vFrmDst.size()) {
                    AVFrame *frm = av_frame_alloc();
                    m_vFrm.push_back(frm);
                }
                AVFrame *frm = m_vFrm[vFrmDst.size()];
                av_frame_unref(frm);
                int e = av_buffersink_get_frame(m_filterOut, frm);
                if (e == AVERROR(EAGAIN) || e == AVERROR_EOF) {
                    break;
                } else if (!ckav(e)) {
                    return false;
                }
                vFrmDst.push_back(frm);
            }
        }
        return true;
    }
    bool Filter(AVFrame *frmSrc, std::vector<AVFrame *> &vFrm) {
        return Filter(std::vector<AVFrame *>{frmSrc}, vFrm);
    }
    AVRational GetOutputTimebase() {
        return av_buffersink_get_time_base(m_filterOut);
    }

protected:
    void ParseFilter(AVFilterGraph *filterGraph, const char *szFilterDesc) {
        AVFilterInOut *outputs = cknn(avfilter_inout_alloc());
        outputs->name = av_strdup("in");
        outputs->filter_ctx = m_filterIn;

        AVFilterInOut *inputs = cknn(avfilter_inout_alloc());
        inputs->name = av_strdup("out");
        inputs->filter_ctx = m_filterOut;

        ckav(avfilter_graph_parse_ptr(filterGraph, szFilterDesc, &inputs, &outputs, NULL));
        ckav(avfilter_graph_config(filterGraph, NULL));

        avfilter_inout_free(&inputs);
        avfilter_inout_free(&outputs);
    }

protected:
    AVFilterGraph *m_filterGraph;
    AVFilterContext *m_filterOut = NULL;
    AVFilterContext *m_filterIn = NULL;
    AVFrame *m_frm;
    int64_t m_pts = 0;

private:
    std::vector<AVFrame *> m_vFrm;
};
