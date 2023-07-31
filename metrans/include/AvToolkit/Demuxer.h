#pragma once

#include <memory>
#include <functional>
extern "C" {
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
}
#include "Logger.h"
#include "AvCommon.h"

extern simplelogger::Logger *logger;

class Demuxer {
private:
    class BufferDataProvider {
    public:
        BufferDataProvider(uint8_t * const pBuffer, size_t nBufferSize) : m_pBuffer(pBuffer), m_nBufferSize(nBufferSize) {}
        int Read(uint8_t *pBuf, int nBuf) {
            if (m_iPos >= m_nBufferSize) {
                return AVERROR_EOF;
            }
            int nRead = std::min(nBuf, (int)(m_nBufferSize - m_iPos));
            memcpy(pBuf, m_pBuffer + m_iPos, nRead);
            m_iPos += nRead;
            return nRead;
        }
        int64_t Seek(int64_t nOffset, int eWhence) {
            int iNewPos;
            switch (eWhence) {
            case SEEK_SET: iNewPos = nOffset;
                break;
            case SEEK_CUR: iNewPos = m_iPos + nOffset;
                break;
            case SEEK_END: iNewPos = m_nBufferSize + nOffset;
                break;
            default:
                if (eWhence & AVSEEK_SIZE) {
                    return (int64_t)m_nBufferSize;
                }
                return -1;
            }
            if (iNewPos < 0 || iNewPos > m_nBufferSize) {
                return -1;
            }
            return m_iPos = iNewPos;
        }

    private:
        uint8_t const * const m_pBuffer;
        const size_t m_nBufferSize;
        size_t m_iPos = 0;
    };

public:
    Demuxer(const char *szFilePath, bool bKeepAvcc = false, bool bKeepAudio = false) 
        : Demuxer(CreateFormatContext(szFilePath), bKeepAvcc, bKeepAudio) {}
    Demuxer(uint8_t * const pBuffer, size_t nBufferSize, bool bKeepAvcc = false, bool bKeepAudio = false) 
        : Demuxer(CreateFormatContext(pBuffer, nBufferSize), bKeepAvcc, bKeepAudio) {}
    ~Demuxer() {
        if (m_pkt->data) {
            av_packet_unref(m_pkt.get());
        }
        if (m_pktFiltered->data) {
            av_packet_unref(m_pktFiltered.get());
        }

        if (m_bsf) {
            av_bsf_free(&m_bsf);
        }

        avformat_close_input(&m_fmt);
        if (m_io) {
            av_freep(&m_io->buffer);
            av_freep(&m_io);
        }

        if (m_pDataProvider) {
            delete m_pDataProvider;
        }
    }

    AVStream *GetVideoStream() {
        return m_iVideo >= 0 ? m_fmt->streams[m_iVideo] : NULL;
    }
    AVStream *GetAudioStream() {
        return m_iAudio >= 0 ? m_fmt->streams[m_iAudio] : NULL;
    }

    /* 当出错或EOF时返回false
       这里为了方便与decoder的连接，EOF及以后的packet的size为0，但packet始终不是NULL。
       Muxer为此做了调整（把这种packet转换成了NULL）。*/
    virtual bool Demux(AVPacket **pPkt, bool *pbAudio = nullptr) {
        *pPkt = m_pktEmpty.get();
        if (pbAudio) *pbAudio = false;
        if (!m_fmt) {
            return false;
        }

        if (m_pkt->data) {
            av_packet_unref(m_pkt.get());
        }
        while (true) {
            int e = 0;
            if ((e = av_read_frame(m_fmt, m_pkt.get())) < 0) {
                if (e == AVERROR_EOF) {
                    return false;
                }
                ckav(e);
                return false;
            }
            if ((m_iVideo >= 0 && m_pkt->stream_index == m_iVideo) 
                || (m_bKeepAudio && m_iAudio >= 0 && m_pkt->stream_index == m_iAudio)) {
                break;
            }
            av_packet_unref(m_pkt.get());
        }

        if (m_pkt->stream_index == m_iAudio) {
            *pPkt = m_pkt.get();
            if (pbAudio) {
                *pbAudio = true;
            } else if (m_iAudio != 0) {
                LOG(WARNING) << "Couldn't return bAudio flag";
            }
            return true;
        }

        if (!m_bMp4ToAnnexb) {
            *pPkt = m_pkt.get();
            return true;
        }

        if (!m_bsf) {
            LOG(ERROR) << "No BSF mp4toannexb";
            return false;
        }

        if (m_pktFiltered->data) {
            av_packet_unref(m_pktFiltered.get());
        }
        ckav(av_bsf_send_packet(m_bsf, m_pkt.get()));
        bool ret = ckav(av_bsf_receive_packet(m_bsf, m_pktFiltered.get()));
        
        *pPkt = m_pktFiltered.get();
        return ret;
    }

private:
    Demuxer(AVFormatContext *fmt, bool bKeepAvcc, bool bKeepAudio) : m_bKeepAudio(bKeepAudio), m_fmt(fmt), 
        m_pkt(av_packet_alloc(), pktDeleter), m_pktFiltered(av_packet_alloc(), pktDeleter), 
        m_pktEmpty(av_packet_alloc(), pktDeleter)
    {
        if (!fmt) {
            LOG(ERROR) << "No AVFormatContext provided.";
            return;
        }
        
        ckav(avformat_find_stream_info(fmt, NULL));
        av_dump_format(fmt, 0, NULL, 0);
        m_iVideo = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
        m_iAudio = av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);
        
        if (!bKeepAvcc && m_iVideo >= 0) {
            AVCodecID eCodecId = m_fmt->streams[m_iVideo]->codecpar->codec_id;
            if (eCodecId == AV_CODEC_ID_H264 || eCodecId == AV_CODEC_ID_HEVC) {
                m_bMp4ToAnnexb = true;
                const AVBitStreamFilter *bsfClass = cknn(av_bsf_get_by_name(
                        eCodecId == AV_CODEC_ID_H264 ? "h264_mp4toannexb" : "hevc_mp4toannexb"));
                ckav(av_bsf_alloc(bsfClass, &m_bsf));
                ckav(avcodec_parameters_copy(m_bsf->par_in, fmt->streams[m_iVideo]->codecpar));
                ckav(av_bsf_init(m_bsf));
            }
        }
    }

    AVFormatContext *CreateFormatContext(uint8_t * const pBuffer, size_t nBufferSize) {
        m_pDataProvider = new BufferDataProvider(pBuffer, nBufferSize);
        int avioc_buffer_size = 1 * 1024 * 1024;
        uint8_t *avioc_buffer = (uint8_t *)cknn(av_malloc(avioc_buffer_size));
        m_io = cknn(avio_alloc_context(avioc_buffer, avioc_buffer_size,
            0, m_pDataProvider, BufferRead, NULL, BufferSeek));

        AVFormatContext *fmt = cknn(avformat_alloc_context());
        fmt->pb = m_io;
        ckav(avformat_open_input(&fmt, NULL, NULL, NULL));
        return fmt;
    }

    AVFormatContext *CreateFormatContext(const char *szFilePath) {
        AVFormatContext *fmt = NULL;
        ckav(avformat_open_input(&fmt, szFilePath, NULL, NULL));
        m_io = NULL;
        m_pDataProvider = NULL;
        return fmt;
    }

    static int BufferRead(void *opaque, uint8_t *pBuf, int nBuf) {
        return ((BufferDataProvider *)opaque)->Read(pBuf, nBuf);
    }

    static int64_t BufferSeek(void *opaque, int64_t nOffset, int eWhence) {
        return ((BufferDataProvider *)opaque)->Seek(nOffset, eWhence);
    }

protected:
    AVFormatContext *m_fmt = NULL;
    int m_iVideo = 0, m_iAudio = 0;

private:
    AVIOContext *m_io;
    std::function<void(AVPacket *)> pktDeleter = [](AVPacket *pkt){if (pkt) av_packet_free(&pkt);};
    std::unique_ptr<AVPacket, decltype(pktDeleter)> m_pkt, m_pktFiltered, m_pktEmpty;
    AVBSFContext *m_bsf = NULL;
    bool m_bMp4ToAnnexb = false, m_bKeepAudio;
    BufferDataProvider *m_pDataProvider;

    friend class VideoDemuxer;
};
