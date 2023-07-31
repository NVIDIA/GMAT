#pragma once

#include <queue>
#include <cuda.h>
#include "AvToolkit/VidDec.h"
#include "AvToolkit/VidFilt.h"
#include "TransDataConverter.h"

class VidDecEx : public VidDec {
public:
    VidDecEx(CUcontext cuContext, AVRational frameRate, AVRational timebase, AVCodecParameters *par, const char *szCodecName = NULL, bool bOriginalPts = false) 
        : VidDec(par, szCodecName), m_frameRate(frameRate), m_timebase(timebase), 
        m_filt(AV_PIX_FMT_YUV420P, par->width, par->height, timebase, AVRational{1,1}, "format=nv12"),
        m_converter(cuContext), bOriginalPts(bOriginalPts) {}
    bool Decode(AVPacket *pkt, std::vector<AVFrame *> &vFrm) {
        if (!bOriginalPts && pkt && pkt->size && pkt->pts != AV_NOPTS_VALUE) {
            qPts.push(pkt->pts); 
        }
        std::vector<AVFrame *> vFrmI420;
        if (!VidDec::Decode(pkt, vFrmI420)) {
            return false;
        }
        for (AVFrame *frm : vFrmI420) {
            if (!bOriginalPts && !qPts.empty()) {
                frm->pts = qPts.top();
                qPts.pop();
            }
            if (frm->pts == AV_NOPTS_VALUE) {
                frm->pts = av_rescale_q_rnd(m_iPacket++, av_inv_q(m_frameRate), m_timebase, (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
            }
        }
        return m_filt.Filter(vFrmI420, vFrm);
    }
    bool DecodeFrmToD(AVPacket *pkt, std::vector<TransData> &vTransData) {
        std::vector<AVFrame *> vFrm;
        return Decode(pkt, vFrm) && m_converter.FrmToD(vFrm, vTransData);
    }

private:
    VidFilt m_filt;
    TransDataConverter m_converter;
    int m_iPacket = 0;
    AVRational m_frameRate;
    AVRational m_timebase;
    bool bOriginalPts;
    std::priority_queue<int64_t, std::vector<int64_t>, std::greater<int64_t>> qPts;
};
