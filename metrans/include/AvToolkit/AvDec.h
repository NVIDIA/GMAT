#pragma once

#include <vector>
#include <map>
#include <assert.h>
extern "C" {
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
}
#include "Logger.h"
#include "AvCommon.h"

extern simplelogger::Logger *logger;

class AvDec {
public:
    AvDec(AVCodecParameters *par, const char *szCodecName = NULL) {
        AVCodec const *codec = cknn(szCodecName ? avcodec_find_decoder_by_name(szCodecName) : avcodec_find_decoder(par->codec_id));
        m_dec = cknn(avcodec_alloc_context3(codec));
        ckav(avcodec_parameters_to_context(m_dec, par));
        ckav(avcodec_open2(m_dec, codec, NULL));
    }
    virtual ~AvDec() {
        for (AVFrame *frm : m_vFrm) {
            av_frame_free(&frm);
        }
        avcodec_free_context(&m_dec);
    }
    AVCodecContext *GetCodecContext() {
        return m_dec;
    }
    bool Decode(AVPacket *pkt, std::vector<AVFrame *> &vFrm) {
        vFrm.clear();
        ckav(avcodec_send_packet(m_dec, pkt));
        while (true) {
            if (m_vFrm.size() <= vFrm.size()) {
                AVFrame *frm = av_frame_alloc();
                m_vFrm.push_back(frm);
            }
            int e = avcodec_receive_frame(m_dec, m_vFrm[vFrm.size()]);
            if (e == AVERROR(EAGAIN) || e == AVERROR_EOF) {
                break;
            } else if (!ckav(e)) {
                return false;
            }
            vFrm.push_back(m_vFrm[vFrm.size()]);
        }
        return true;
    }

private:
    AVCodecContext *m_dec;
    std::vector<AVFrame *> m_vFrm;
};
