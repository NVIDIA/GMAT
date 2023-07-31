#pragma once

#include "FrameExtractor.h"
#include "AvToolkit/VidFilt.h"
#include <libavutil/hwcontext.h>

#include <string>

class FrameSelect : public FrameExtractor {
public:
    FrameSelect(const char *szFilePath, const char* args, CUcontext cuContext)
    : FrameExtractor{szFilePath, cuContext}, m_argString{"select_gpu='"} {
        int err = 0;
        AVHWFramesContext *frameCtx;
        cudaSetDevice(0);
        if ((err = av_hwdevice_ctx_create(&m_ffDeviceCtx, AV_HWDEVICE_TYPE_CUDA,
                                      NULL, NULL, 1)) < 0) {
            LOG(ERROR) << "Failed to create specified HW device.\n";
            return;
        }
        m_FFframeCtx = av_hwframe_ctx_alloc(m_ffDeviceCtx);
        if (!m_FFframeCtx) {
            LOG(ERROR) << "Failed to create frame ctx.\n";
            return;
        }
        frameCtx = (AVHWFramesContext*)m_FFframeCtx->data;

        frameCtx->format    = AV_PIX_FMT_CUDA;
        frameCtx->sw_format = AV_PIX_FMT_NV12;
        frameCtx->width     = GetWidth();
        frameCtx->height    = GetHeight();

        err = av_hwframe_ctx_init(m_FFframeCtx);

        m_argString += args;
        m_argString += "'";
        m_selectFilter = new VidFilt(AV_PIX_FMT_CUDA, GetWidth(), GetHeight(), AVRational{1, 1}, AVRational{1, 1}, m_argString.c_str(), m_FFframeCtx);

        m_timeInterval = 0;
        m_frameInterval = 0;


        // AVBufferSrcParameters *par = av_buffersrc_parameters_alloc();
        // if (!par) {
        //     LOG(ERROR) <<"Failed to create filter params\n";
        //     return;
        // }
        // memset(par, 0, sizeof(*par));
        // par->format = AV_PIX_FMT_CUDA;

        // par->hw_frames_ctx = m_FFframeCtx;
        // err = av_buffersrc_parameters_set(m_selectFilter->GetInFilterCtx(), par);
        // if (err < 0) {
        //     LOG(ERROR) <<"Failed to set filter params\n";
        //     return;
        // }
    }

    ~FrameSelect() {
        if (m_selectFilter) delete m_selectFilter;
    }

    uint8_t *Extract(CUstream stream = 0) {
        if (m_lpFrame.size()) {
            auto p = m_lpFrame.front();
            dec.UnlockFrame(&p, 1);
            m_lpFrame.pop_front();
            // m_lInfo.pop_front();
            if (m_lpFrame.size()) {
                return m_lpFrame.front();
            }
        }

        while (m_lpFrame.empty()) {
            AVPacket *pkt;
            bool bRef;
            int iFrame;
            double time;
            demuxer.Demux(&pkt, &time, &iFrame, &bRef);
            // if (m_timeTarget == DBL_MIN) {
            //     m_timeTarget = time;
            // }
            if (pkt->size) nPacketDemuxed++;
            // bool bReached = (typeid(T) == typeid(int) ? iFrame - m_frameTarget: time - m_timeTarget) >= 0;

            uint8_t **ppFrame = NULL;
            NvFrameInfo *pInfo = NULL;
            int nFrame = dec.DecodeLockFrame(pkt->data, pkt->size, &ppFrame, &pInfo, CUVID_PKT_ENDOFPICTURE, 0, stream);
            nFrameDecoded += nFrame;

            for (int i = 0; i < nFrame; i++) {
                vector<AVFrame *> vFrm;
                m_selectFilter->Filter(ppFrame[i], pInfo->nFramePitch, 0, vFrm);
                if (vFrm.size() == 0) {
                    dec.UnlockFrame(&ppFrame[i], 1);
                    continue;
                }
                for (int k = 0; k < vFrm.size(); k++) {
                    m_lpFrame.push_back(vFrm[i]->data[0]);
                }
            }
            if (!pkt->size) break;
        }

        if (m_lpFrame.size()) {
            return m_lpFrame.front();
        }
        return nullptr;
    }

private:
    VidFilt *m_selectFilter;
    std::string m_argString;
    AVBufferRef *m_ffDeviceCtx;
    AVBufferRef *m_FFframeCtx;
};