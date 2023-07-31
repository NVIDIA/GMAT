#pragma once

#include <vector>
#include <memory>
extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
};
#include "Logger.h"
#include "AvCommon.h"

extern simplelogger::Logger *logger;

class MuxerBase {
public:
    virtual ~MuxerBase() {}
    virtual bool MuxVideo(AVPacket *pkt, bool bInterleaved = false) = 0;
    virtual bool MuxVideo(uint8_t *pPacketData, int nPacketSize, int64_t pts, int64_t dts, bool bInterleaved = false) {
        if (!pPacketData) {
            return MuxVideo(NULL, bInterleaved);
        }

        auto d = [](AVPacket *pkt){if (pkt) av_packet_free(&pkt);};
        auto pkt = std::unique_ptr<AVPacket, decltype(d)>(av_packet_alloc(), d);
        pkt->data = pPacketData;
        pkt->size = nPacketSize;
        pkt->pts = pts;
        pkt->dts = dts;
        if(!memcmp(pPacketData, "\x00\x00\x00\x01\x67", 5) || !memcmp(pPacketData, "\x00\x00\x01\x67", 4) || !memcmp(pPacketData, "\x00\x00\x00\x01\x40", 5)) {
            pkt->flags |= AV_PKT_FLAG_KEY;
        }
        return MuxVideo(pkt.get(), bInterleaved);
    }
    virtual bool MuxAudio(AVPacket *pkt, bool bInterleaved = false) = 0;
    virtual bool MuxAudio(uint8_t *pPacketData, int nPacketSize, int64_t pts, int64_t dts, bool bInterleaved = false) {
        if (!pPacketData) {
            return MuxAudio(NULL, bInterleaved);
        }

        auto d = [](AVPacket *pkt){if (pkt) av_packet_free(&pkt);};
        auto pkt = std::unique_ptr<AVPacket, decltype(d)>(av_packet_alloc(), d);
        pkt->data = pPacketData;
        pkt->size = nPacketSize;
        pkt->pts = pts;
        pkt->dts = dts;
        return MuxAudio(pkt.get(), bInterleaved);
    }
};

class Muxer : public MuxerBase {
public:
    Muxer(AVCodecParameters *vpar, AVRational vtimebase, AVCodecParameters *apar, AVRational atimebase, 
        const char *szMediaPath, const char *szFormat = NULL) : m_vtimebase(vtimebase), m_atimebase(atimebase) {
        avformat_network_init();

        ckav(avformat_alloc_output_context2(&m_fmt, NULL, szFormat, szMediaPath));
        if (vpar) {
            m_vs = cknn(avformat_new_stream(m_fmt, NULL));
            avcodec_parameters_copy(m_vs->codecpar, vpar);
            m_vs->codecpar->codec_tag = 0;
        }
        if (apar) {
            m_as = cknn(avformat_new_stream(m_fmt, NULL));
            avcodec_parameters_copy(m_as->codecpar, apar);
            m_as->codecpar->codec_tag = 0;
        }
        av_dump_format(m_fmt, 0, szMediaPath, 1);

        ckav(avio_open(&m_fmt->pb, m_fmt->url, AVIO_FLAG_WRITE));
        ckav(avformat_write_header(m_fmt, NULL));
    }
    virtual ~Muxer() {
        if (!m_fmt) {
            return;
        }
        av_write_trailer(m_fmt);
        avio_closep(&m_fmt->pb);
        avformat_free_context(m_fmt);
    }

    virtual bool MuxVideo(AVPacket *pkt, bool bInterleaved = false) {
        if (!m_vs) {
            return false;
        }
        if (pkt && pkt->size == 0) {
            pkt = nullptr;
        }
        if (pkt) {
            pkt->stream_index = m_vs->index;
            pkt->pts = av_rescale_q_rnd(pkt->pts, m_vtimebase, m_vs->time_base, (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
            pkt->dts = av_rescale_q_rnd(pkt->dts, m_vtimebase, m_vs->time_base, (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
        }
        return bInterleaved ? ckav(av_interleaved_write_frame(m_fmt, pkt)) : ckav(av_write_frame(m_fmt, pkt));
    }

    virtual bool MuxAudio(AVPacket *pkt, bool bInterleaved = false) {
        if (!m_as) {
            return false;
        }
        if (pkt && pkt->size == 0) {
            pkt = nullptr;
        }
        if (pkt) {
            pkt->stream_index = m_as->index;
            pkt->pts = av_rescale_q_rnd(pkt->pts, m_atimebase, m_as->time_base, (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
            pkt->dts = av_rescale_q_rnd(pkt->dts, m_atimebase, m_as->time_base, (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
        }
        return bInterleaved ? ckav(av_interleaved_write_frame(m_fmt, pkt)) : ckav(av_write_frame(m_fmt, pkt));
    }

    using MuxerBase::MuxVideo;
    using MuxerBase::MuxAudio;

private:
    AVFormatContext *m_fmt = NULL;
    AVStream *m_vs = NULL, *m_as = NULL;
    AVRational m_vtimebase, m_atimebase;
};

class LazyMuxer : MuxerBase {
public:
    LazyMuxer(const char *szMediaPath, const char *szFormat = NULL) {
        if (szMediaPath) {
            m_szMediaPath = new char[strlen(szMediaPath) + 1];
            strcpy(m_szMediaPath, szMediaPath);
        }
        if (szFormat) {
            m_szFormat = new char[strlen(szFormat) + 1];
            strcpy(m_szFormat, szFormat);
        }
    }
    virtual ~LazyMuxer() {
        for (AVPacket *pkt : m_vPkt) {
            av_packet_free(&pkt);
        }
        if (m_vpar) avcodec_parameters_free(&m_vpar);
        if (m_apar) avcodec_parameters_free(&m_apar);
        if (m_pMuxer) delete m_pMuxer;
        if (m_szFormat) delete[] m_szFormat;
        if (m_szMediaPath) delete[] m_szMediaPath;
    }
    void SetVideoStream(AVCodecParameters *vpar, AVRational vtimebase) {
        if (m_pMuxer) {
            return;
        }
        if (m_apar != NULL) {
            m_pMuxer = new Muxer(vpar, vtimebase, m_apar, m_atimebase, m_szMediaPath, m_szFormat);
            for (AVPacket *pkt : m_vPkt) {
                m_pMuxer->MuxAudio(pkt, false);
            }
            return;
        }
        m_vpar = cknn(avcodec_parameters_alloc());
        ckav(avcodec_parameters_copy(m_vpar, vpar));
        m_vtimebase = vtimebase;
    }
    void SetAudioStream(AVCodecParameters *apar, AVRational atimebase) {
        if (m_pMuxer) {
            return;
        }
        if (m_vpar != NULL) {
            m_pMuxer = new Muxer(m_vpar, m_vtimebase, apar, atimebase, m_szMediaPath, m_szFormat);
            for (AVPacket *pkt : m_vPkt) {
                m_pMuxer->MuxVideo(pkt, false);
            }
            return;
        }
        m_apar = cknn(avcodec_parameters_alloc());
        ckav(avcodec_parameters_copy(m_apar, apar));
        m_atimebase = atimebase;
    }
    
    bool IsVideoStreamSet() {
        return m_vpar != NULL;
    }
    bool IsAudioStreamSet() {
        return m_apar != NULL;
    }

    virtual bool MuxVideo(AVPacket *pkt, bool bInterleaved = false) {
        if (m_pMuxer) return m_pMuxer->MuxVideo(pkt, bInterleaved);

        if (!IsVideoStreamSet()) {
            LOG(ERROR) << "Video stream is not set";
            return false;
        }

        if (!pkt) {
            m_vPkt.push_back(NULL);
            return true;
        }

        AVPacket *pktCopy = cknn(av_packet_alloc());
        ckav(av_packet_ref(pktCopy, pkt));
        m_vPkt.push_back(pktCopy);
        return true;
    }

    virtual bool MuxAudio(AVPacket *pkt, bool bInterleaved = false) {
        if (m_pMuxer) return m_pMuxer->MuxAudio(pkt, bInterleaved);

        if (!IsAudioStreamSet()) {
            LOG(ERROR) << "Audio stream is not set";
            return false;
        }

        if (!pkt) {
            m_vPkt.push_back(NULL);
            return true;
        }

        AVPacket *pktCopy = cknn(av_packet_alloc());
        ckav(av_packet_ref(pktCopy, pkt));
        m_vPkt.push_back(pktCopy);
        return true;
    }

    using MuxerBase::MuxVideo;
    using MuxerBase::MuxAudio;

private:
    Muxer *m_pMuxer = NULL;
    AVCodecParameters *m_vpar = NULL, *m_apar = NULL;
    AVRational m_vtimebase, m_atimebase;
    char *m_szFormat = NULL;
    char *m_szMediaPath = NULL;
    std::vector<AVPacket *> m_vPkt;
};
