#pragma once

#include "AvToolkit/Demuxer.h"
#include "NvCodec/NvDecLite.h"
#include "NvCodec/NvCommon.h"
#include <list>
#include <float.h>

extern simplelogger::Logger *logger;

using namespace std;

class VideoDemuxer {
public:
    VideoDemuxer(const char *szFilePath) : m_dm(szFilePath, false, false), m_dmSeek(szFilePath, false, false) {
        m_dmSeek.Demux(&m_pkt);
        m_ret = m_dm.Demux(&m_pkt);
        if (m_ret) m_iNextFrame++;
        m_bCached = true;
    }
    VideoDemuxer(uint8_t * const pBuffer, size_t nBufferSize) : m_dm(pBuffer, nBufferSize, false, false), m_dmSeek(pBuffer, nBufferSize, false, false) {
        m_dmSeek.Demux(&m_pkt);
        m_ret = m_dm.Demux(&m_pkt);
        if (m_ret) m_iNextFrame++;
        m_bCached = true;
    }
    ~VideoDemuxer() {}

    bool Demux(AVPacket **pPkt, double *pTime = nullptr, int *piFrame = nullptr, bool *pbRef = nullptr) {
        if (m_bCached) {
            m_bCached = false;
            *pPkt = m_pkt;
        } else {
            m_dmSeek.Demux(&m_pkt);
            m_ret = m_dm.Demux(&m_pkt);
            if (m_ret) m_iNextFrame++;
            *pPkt = m_pkt;
        }

        if (pTime) *pTime = m_ret ? GetCurrentTime() : 0;
        if (piFrame) *piFrame = m_ret ? m_iNextFrame - 1 : 0;

        if (pbRef) *pbRef = true;
        if (pbRef && m_pkt->size && m_dm.GetVideoStream()->codecpar->codec_id == AV_CODEC_ID_H264) {
            uint8_t b = m_pkt->data[2] == 1 ? m_pkt->data[3] : m_pkt->data[4];
            int nal_ref_idc = b >> 5;
            int nal_unit_type = b & 0x1f;
            if (!nal_ref_idc && nal_unit_type == 1) {
            	*pbRef = false;
            }
        }

        return m_ret;
    }

    int SeekKeyFrame(double dTimeInterval) {
        if (!m_pkt || !m_ret) {
            LOG(TRACE) << "Don't seek when m_pkt == NULL or m_ret == false";
            return 0;
        }
        if (!m_dmSeek.m_fmt->iformat->read_seek) {
            LOG(TRACE) << "seeking isn't supported by the container";
            return 0;
        }
        if (av_seek_frame(m_dmSeek.m_fmt, m_dmSeek.m_iVideo, sec2ts(GetCurrentTime() + dTimeInterval), AVSEEK_FLAG_BACKWARD) < 0) {
            LOG(TRACE) << "av_seek_frame() failed";
            return 0;
        }

        AVPacket *pkt = nullptr;
        if (!m_dmSeek.Demux(&pkt) || pkt->pos == m_pkt->pos) {
            return 0;
        }
        if (pkt->pos > m_pkt->pos) {
            int i = m_iNextFrame;
            do {
                m_ret = m_dm.Demux(&m_pkt);
                if (m_ret) m_iNextFrame++;
            } while (pkt->pos != m_pkt->pos);
            m_bCached = true;
            return m_iNextFrame - i - 1;
        }
        do {
            m_dmSeek.Demux(&pkt);
        } while (pkt->pos != m_pkt->pos);
        return 0;
    }

    int SeekKeyFrame(int nFrameInterval) {
        if (!m_pkt || !m_ret) {
            LOG(TRACE) << "Don't seek when m_pkt == NULL or m_ret == false";
            return 0;
        }

        int nSkipped = -1;
        AVPacket *pkt = nullptr;
        for (int i = 0; i < nFrameInterval; i++) {
            m_dmSeek.Demux(&pkt);
            if (pkt->flags & AV_PKT_FLAG_KEY) {
                nSkipped = i;
            }
        }
        if (nSkipped == -1) {
            if (av_seek_frame(m_dmSeek.m_fmt, m_dmSeek.m_iVideo, sec2ts(ts2sec(m_pkt->pts)), AVSEEK_FLAG_BACKWARD) < 0) {
                LOG(ERROR) << "av_seek_frame() failed: m_dmSeek out of sync";
                return 0;
            }
            do {
                m_dmSeek.Demux(&pkt);
            } while (pkt->pos != m_pkt->pos);
            return 0;
        }
        for (int i = 0; i <= nSkipped; i++) {
            m_ret = m_dm.Demux(&m_pkt);
            if (m_ret) m_iNextFrame++;
        }
        m_bCached = true;
        if (av_seek_frame(m_dmSeek.m_fmt, m_dmSeek.m_iVideo, sec2ts(ts2sec(m_pkt->pts)), AVSEEK_FLAG_BACKWARD) < 0) {
            LOG(ERROR) << "av_seek_frame() failed: m_dmSeek out of sync";
            return nSkipped;
        }
        do {
            m_dmSeek.Demux(&pkt);
        } while (pkt->pos != m_pkt->pos);
        return nSkipped;
    }

    AVStream *GetVideoStream() {
        return m_dm.GetVideoStream();
    }

private:
    int64_t sec2ts(double sec) {
        return round(sec / av_q2d(m_dm.m_fmt->streams[m_dm.m_iVideo]->time_base));
    }
    double ts2sec(int64_t ts) {
        return ts * av_q2d(m_dm.m_fmt->streams[m_dm.m_iVideo]->time_base);
    }
    double GetCurrentTime() {
        if (m_pkt->size && m_pkt->pts != AV_NOPTS_VALUE) {
            return ts2sec(m_pkt->pts);
        }
        return (m_iNextFrame - 1) / av_q2d(m_dm.m_fmt->streams[m_dm.m_iVideo]->avg_frame_rate);
    }

    Demuxer m_dm, m_dmSeek;
    AVPacket *m_pkt = nullptr;
    bool m_ret = false;
    bool m_bCached = false;
    int64_t m_iNextFrame = 0;
};

class FrameExtractor {
protected:
    VideoDemuxer demuxer;
    NvDecLite dec;
    list<uint8_t *> m_lpFrame;
    list<NvFrameInfo> m_lInfo;

    int m_frameTarget = 0, m_frameInterval = 0;
    double m_timeTarget = DBL_MIN, m_timeInterval = 0;

    int nPacketDemuxed = 0, nSkipped = 0, nFrameDecoded = 0, nFrameExtracted = 0;

public:
    FrameExtractor(const char *szFilePath, CUcontext cuContext) 
        : demuxer(szFilePath), dec(cuContext, true, 
            FFmpeg2NvCodecId(demuxer.GetVideoStream()->codecpar->codec_id), false, false, nullptr, 
            make_shared<NvDecLite::Dim>(demuxer.GetVideoStream()->codecpar->width, demuxer.GetVideoStream()->codecpar->height).get()) {}
    FrameExtractor(uint8_t * const pBuffer, size_t nBufferSize, CUcontext cuContext) 
        : demuxer(pBuffer, nBufferSize), dec(cuContext, true, 
            FFmpeg2NvCodecId(demuxer.GetVideoStream()->codecpar->codec_id), false, false, nullptr, 
            make_shared<NvDecLite::Dim>(demuxer.GetVideoStream()->codecpar->width, demuxer.GetVideoStream()->codecpar->height).get()) {}
    ~FrameExtractor() {
        if (m_lpFrame.size()) {
            for (auto p : m_lpFrame) {
                dec.UnlockFrame(&p, 1);
            }
        }
        LOG(INFO) << "Total=" << (nPacketDemuxed + nSkipped) << ", nPacketDemuxed=" << nPacketDemuxed 
            << ", nFrameDecoded=" << nFrameDecoded << ", nFrameExtracted=" << nFrameExtracted;
    }
    void SetInterval(int nFrameInterval) {
        m_frameInterval = nFrameInterval;
        m_timeInterval = 0;
    }
    void SetInterval(double timeInterval) {
        m_timeInterval = timeInterval;
        m_frameInterval = 0;
    }
    int GetWidth() {
        if (m_lInfo.empty()) {
            return demuxer.GetVideoStream()->codecpar->width;
        }
        else {
            return m_lInfo.front().nWidth;
        }
    }
    int GetHeight() {
        if (m_lInfo.empty()) {
            return demuxer.GetVideoStream()->codecpar->height;
        }
        else {
            return m_lInfo.front().nHeight;
        }
    }
    int GetFrameSize() {
        return GetWidth() * GetHeight() * demuxer.GetVideoStream()->codecpar->bits_per_raw_sample / 8 * 3 / 2;
    }
    
    uint8_t *Extract(CUstream stream = 0) {
        if (m_frameInterval) {
            return Extract(m_frameInterval, stream);
        }
        return Extract(m_timeInterval, stream);
    }
    bool ExtractToBuffer(uint8_t* pframe, CUstream stream) {
        auto p = Extract(stream);
        if (!p) return false;
        return ck(cudaMemcpyAsync(pframe, p, GetFrameSize(), cudaMemcpyDefault, stream));
    }
    bool ExtractToDeviceBuffer(float *dpBgrp, CUstream stream) {
        auto p = Extract(stream);
        if (!p) return false;
        Nv12ToBgrFloatPlanar(p, GetWidth(), dpBgrp, GetWidth() * sizeof(*dpBgrp), GetWidth(), GetHeight(), 0, stream);
        return true;
    }

private:
    template<class T>
    uint8_t *Extract(T interval, CUstream stream) {
        if (m_lpFrame.size()) {
            auto p = m_lpFrame.front();
            dec.UnlockFrame(&p, 1);
            m_lpFrame.pop_front();
            m_lInfo.pop_front();
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
            if (m_timeTarget == DBL_MIN) {
                m_timeTarget = time;
            }
            if (pkt->size) nPacketDemuxed++;
            bool bReached = (typeid(T) == typeid(int) ? iFrame - m_frameTarget: time - m_timeTarget) >= 0;

            ostringstream oss;
            oss << nPacketDemuxed << "\t" << pkt->size << "\t" << pkt->pos << "\t time=" << time << ",target=";
            if (typeid(T) == typeid(int)) {
                oss << m_frameTarget;
            } else {
                oss << m_timeTarget;
            }
            if (!bReached && !bRef) {
                oss << " skipped";
            }
            LOG(TRACE) << oss.str();

            if (!bReached && !bRef) {
                continue;
            }

            uint8_t **ppFrame = NULL;
            NvFrameInfo *pInfo = NULL;
            int nFrame = dec.DecodeLockFrame(pkt->data, pkt->size, &ppFrame, &pInfo, CUVID_PKT_ENDOFPICTURE, -bReached, stream);
            nFrameDecoded += nFrame;
            
            for (int i = 0; i < nFrame; i++) {
                if (!pInfo[i].dispInfo.timestamp) {
                    dec.UnlockFrame(&ppFrame[i], 1);
                    continue;
                }
                m_lpFrame.push_back(ppFrame[i]);
                m_lInfo.push_back(pInfo[i]);
                nFrameExtracted++;
            }

            if (!pkt->size) break;
            if (bReached && (m_frameInterval || m_timeInterval)) {
                if (typeid(T) == typeid(int)) {
                    nSkipped += demuxer.SeekKeyFrame(m_frameInterval);
                    m_frameTarget += m_frameInterval;
                    m_timeTarget = time;
                } else {
                    nSkipped += demuxer.SeekKeyFrame(m_timeInterval);
                    m_timeTarget += m_timeInterval;
                    m_frameTarget = iFrame;
                }
                if (nSkipped) LOG(TRACE) << "seek " << nSkipped;
            }
        }

        if (m_lpFrame.size()) {
            return m_lpFrame.front();
        }
        return nullptr;
    }
};
