#pragma once

#include "AvFilt.h"

class AudFilt : public AvFilt {
public:
    AudFilt(AVSampleFormat eSampleFormat, int nSampleRate, uint64_t uChannelLayout, AVRational timebase, const char *szFilterDesc)  {
        char args[512];
        sprintf(args, "sample_fmt=%s:sample_rate=%d:channel_layout=0x%lx:time_base=%d/%d",
            av_get_sample_fmt_name(eSampleFormat), nSampleRate, uChannelLayout, timebase.num, timebase.den);
        ckav(avfilter_graph_create_filter(&m_filterIn, avfilter_get_by_name("abuffer"), "in", args, NULL, m_filterGraph));
        ckav(avfilter_graph_create_filter(&m_filterOut, avfilter_get_by_name("abuffersink"), "out", NULL, NULL, m_filterGraph));
        ParseFilter(m_filterGraph, szFilterDesc);

        m_frm->format = eSampleFormat;
        m_frm->sample_rate = nSampleRate;
        m_frm->channel_layout = uChannelLayout;
        m_frm->channels = av_get_channel_layout_nb_channels(uChannelLayout);
    }
    bool Filter(uint8_t **apSample, int nSample, int64_t pts, std::vector<AVFrame *> &vFrm) {
        if (m_frm->format <= AV_SAMPLE_FMT_DBL || m_frm->format == AV_SAMPLE_FMT_S64) {
            m_frm->data[0] = apSample[0];
        } else {
            for (int i = 0; i < m_frm->channels; i++) {
                m_frm->data[i] = apSample[i];
            }
        }
        m_frm->nb_samples = nSample;
        m_frm->pts = pts;
        return Filter(m_frm, vFrm);
    }
    using AvFilt::Filter;
};
