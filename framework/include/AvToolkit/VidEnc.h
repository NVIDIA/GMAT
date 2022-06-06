#pragma once

#include "AvEnc.h"
extern "C" {
#include <libavutil/imgutils.h>
}

extern simplelogger::Logger *logger;

class VidEnc : public AvEnc {
public:
    VidEnc(AVPixelFormat eInputFormat, int nWidth, int nHeight, const char *szCodecName, 
        AVCodecID eCodecId = AV_CODEC_ID_MPEG4, AVRational timebase = {}, int nFps = 25, const char *szCodecParam = NULL) 
    {
        AVCodec const *codec = cknn(szCodecName ? avcodec_find_encoder_by_name(szCodecName) : avcodec_find_encoder(eCodecId));
        m_enc = cknn(avcodec_alloc_context3(codec));
        m_enc->codec_type = AVMEDIA_TYPE_VIDEO;
        m_enc->codec_id = codec->id;
        m_enc->pix_fmt = eInputFormat;
        m_enc->width = nWidth;
        m_enc->height = nHeight;
        m_enc->framerate = AVRational{nFps, 1};
        m_enc->time_base = timebase.den * timebase.num == 0 ? av_inv_q(m_enc->framerate) : timebase;
        m_enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

        AVDictionary *dict = NULL;
        if (szCodecParam) ckav(av_dict_parse_string(&dict, szCodecParam, "=", ",", 0));
        ckav(avcodec_open2(m_enc, codec, &dict));
        if (dict) av_dict_free(&dict);

        m_frm = cknn(av_frame_alloc());
        m_frm->format = m_enc->pix_fmt;
        m_frm->width  = m_enc->width;
        m_frm->height = m_enc->height;
    }
    ~VidEnc() {
        for (AVPacket *pkt : m_vPkt) {
            av_packet_free(&pkt);
        }
        av_frame_free(&m_frm);
        avcodec_free_context(&m_enc);
    }

    bool Encode(AVFrame *frm, std::vector<AVPacket *> &vPkt) {
        vPkt.clear();
        if (!ckav(avcodec_send_frame(m_enc, frm))) {
            return false;
        }
        while (true) {
            if (m_vPkt.size() <= vPkt.size()) {
                m_vPkt.push_back(cknn(av_packet_alloc()));
            }
            AVPacket *pkt = m_vPkt[vPkt.size()];
            av_packet_unref(pkt);
            int e = avcodec_receive_packet(m_enc, pkt);
            if (e == AVERROR(EAGAIN) || e == AVERROR_EOF) {
                break;
            } else if (!ckav(e)) {
                return false;
            }
            vPkt.push_back(pkt);
        }
        return true;
    }

    bool Encode(uint8_t *pFrame, int nPitch, int64_t pts, std::vector<AVPacket *> &vPkt) {
        if (!pFrame) return Encode(NULL, vPkt);

        m_frm->pts = pts == AV_NOPTS_VALUE ? m_pts++ : pts;
        ckav(av_image_fill_arrays(m_frm->data, m_frm->linesize, pFrame, 
            (AVPixelFormat)m_frm->format, m_frm->width, m_frm->height, nPitch ? PitchToAlignment(nPitch, m_frm->width) : 1));
        return Encode(m_frm, vPkt);
    }

    int GetFrameSize() {
        return av_image_get_buffer_size((AVPixelFormat)m_frm->format, m_frm->width, m_frm->height, 1);
    }

private:
    AVFrame *m_frm;
    std::vector<AVPacket *> m_vPkt;
    int64_t m_pts = 0;
};
