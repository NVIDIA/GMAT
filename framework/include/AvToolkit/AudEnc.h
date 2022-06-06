#pragma once

#include "AvToolkit/AvEnc.h"
extern "C" {
#include <libavutil/audio_fifo.h>
#include <libswresample/swresample.h>
}
#include <set>

class AudEnc : public AvEnc {
public:
    AudEnc(AVSampleFormat eInputFormat, int nInputSampleRate, uint64_t uInputChannelLayout, AVCodecID eCodecId, 
        const char *szCodecName = NULL, AVRational timebase = {}, AVCodecParameters *par = NULL) 
    {
        AVCodec const *codec = cknn(szCodecName ? avcodec_find_encoder_by_name(szCodecName) : avcodec_find_encoder(eCodecId));        
        m_enc = cknn(avcodec_alloc_context3(codec));
        SetEncodeParameters(m_enc, codec, par, eInputFormat, nInputSampleRate, uInputChannelLayout);
        m_enc->time_base = timebase;
        ckav(avcodec_open2(m_enc, codec, NULL));

        m_resampler = cknn(swr_alloc_set_opts(NULL, m_enc->channel_layout, m_enc->sample_fmt, m_enc->sample_rate, 
            uInputChannelLayout, eInputFormat, nInputSampleRate, 0, NULL));
        ckav(swr_init(m_resampler));
        m_fifo = cknn(av_audio_fifo_alloc(m_enc->sample_fmt, m_enc->channels, 1));
    }
    ~AudEnc() {
        for (AVPacket *pkt : m_vPkt) {
            av_packet_free(&pkt);
        }
        if (m_apNewSample) {
            av_freep(&m_apNewSample[0]);
            av_freep(&m_apNewSample);
        }
        av_frame_free(&m_frm);

        av_audio_fifo_free(m_fifo);
        swr_free(&m_resampler);
        avcodec_free_context(&m_enc);
    }
    AVCodecContext *GetCodecContext() {
        return m_enc;    
    }

    bool Encode(AVFrame *frm, std::vector<AVPacket *> &vPkt) {
        return frm ? Encode(frm->data, frm->nb_samples, frm->pts, vPkt) : Encode(NULL, 0, 0, vPkt);
    }
    bool Encode(uint8_t **apSample, int nInputSample, int64_t pts, std::vector<AVPacket *> &vPkt) {
        if (!m_frm) {
            m_frm = cknn(av_frame_alloc());
            m_frm->format = m_enc->sample_fmt;
            m_frm->sample_rate = m_enc->sample_rate;
            m_frm->channel_layout = m_enc->channel_layout;
            m_frm->channels = av_get_channel_layout_nb_channels(m_enc->channel_layout);
            m_frm->nb_samples = m_enc->frame_size ? m_enc->frame_size : swr_get_out_samples(m_resampler, nInputSample);
            ckav(av_frame_get_buffer(m_frm, 0));
            ckav(av_frame_make_writable(m_frm));
        }

        vPkt.clear();

        if (!apSample || !nInputSample) {
            if (av_audio_fifo_size(m_fifo) > 0) {
                m_frm->nb_samples = av_audio_fifo_size(m_fifo);
                ckav(av_audio_fifo_read(m_fifo, (void **)m_frm->data, av_audio_fifo_size(m_fifo)));
                m_frm->pts = GetPts(pts, 0, 0);
                if (!EncodeOneFrame(m_frm, vPkt)) {
                    return false;
                }
            }
            return EncodeOneFrame(NULL, vPkt);
        }

        if (m_nInputSample != nInputSample) {
            m_nInputSample = nInputSample;
            m_nNewSample = swr_get_out_samples(m_resampler, nInputSample);
            // reallocate resampler's buffer
            if (m_apNewSample) {
                av_freep(&m_apNewSample[0]);
                av_freep(&m_apNewSample);
            }
            m_apNewSample = (uint8_t **)cknn(av_malloc(m_enc->channels * sizeof(*m_apNewSample)));
            ckav(av_samples_alloc(m_apNewSample, NULL, m_enc->channels, m_nNewSample, m_enc->sample_fmt, 0));
        }

        int nNewSample = 0;
        ckav(nNewSample = swr_convert(m_resampler, m_apNewSample, m_nNewSample, (const uint8_t **)apSample, nInputSample));
        ckav(av_audio_fifo_write(m_fifo, (void **)m_apNewSample, nNewSample));
        while (av_audio_fifo_size(m_fifo) >= m_enc->frame_size) {
            ckav(av_audio_fifo_read(m_fifo, (void **)m_frm->data, m_enc->frame_size));
            m_frm->pts = GetPts(pts, nNewSample, av_audio_fifo_size(m_fifo));
            EncodeOneFrame(m_frm, vPkt);
        }
        m_nTotalSample += nNewSample;
        return true;
    }

private:
    void SetEncodeParameters(AVCodecContext *enc, AVCodec const *codec, AVCodecParameters *par, AVSampleFormat eInputFormat, int nInputSampleRate, uint64_t uInputChannelLayout) {
        if (par) {
            par->codec_type = AVMEDIA_TYPE_AUDIO;
            ckav(avcodec_parameters_to_context(enc, par));
        }
        
        enc->codec_type = AVMEDIA_TYPE_AUDIO;
        enc->codec_id = codec->id;
        if (enc->sample_fmt == -1) {
            enc->sample_fmt = codec->sample_fmts ? codec->sample_fmts[0] : eInputFormat;
            for (int i = 0; codec->sample_fmts && codec->sample_fmts[i] != -1; i++) {
                if (codec->sample_fmts[i] == eInputFormat) {
                    enc->sample_fmt = eInputFormat;
                    break;
                }
            }
        }
        if (enc->bit_rate == 0) {
            enc->bit_rate = 128000;
        }
        if (enc->sample_rate == 0) {
            std::set<int> sSampleRate;
            for (int i = 0; codec->supported_samplerates && codec->supported_samplerates[i]; i++) {
                sSampleRate.insert(codec->supported_samplerates[i]);
            }
            enc->sample_rate = sSampleRate.size() ? *sSampleRate.begin() : nInputSampleRate;
            for (int sampleRate : sSampleRate) {
                if (enc->sample_rate <= sampleRate) {
                    enc->sample_rate = sampleRate;
                }
                if (enc->sample_rate >= nInputSampleRate) {
                    break;
                }
            }
        }
        if (enc->channel_layout == 0) {
            enc->channel_layout = codec->channel_layouts ? codec->channel_layouts[0] : uInputChannelLayout;
            for (int i = 0; codec->channel_layouts && codec->channel_layouts[i]; i++) {
                if (codec->channel_layouts[i] == uInputChannelLayout) {
                    enc->channel_layout = uInputChannelLayout;
                    break;
                }
            }
            enc->channels = av_get_channel_layout_nb_channels(enc->channel_layout);
        }
        if (enc->channels == 0) {
            enc->channels = av_get_channel_layout_nb_channels(enc->channel_layout);
        }
    }

    bool EncodeOneFrame(AVFrame *frm, std::vector<AVPacket *> &vPkt) {
        if (!ckav(avcodec_send_frame(m_enc, frm))) {
            return false;
        }
        while (true) {
            if (m_vPkt.size() <= vPkt.size()) {
                m_vPkt.push_back(cknn(av_packet_alloc()));
            }
            int e = avcodec_receive_packet(m_enc, m_vPkt[vPkt.size()]);
            if (e == AVERROR(EAGAIN) || e == AVERROR_EOF) {
                break;
            } else if (!ckav(e)) {
                return false;
            }
            vPkt.push_back(m_vPkt[vPkt.size()]);
        }
        return true;
    }

    int64_t GetPts(int64_t ptsNow, int nConvertedSample, int nSampleLeft) {
        if (ptsNow == 0 || ptsNow == AV_NOPTS_VALUE) {
            ptsNow = m_nTotalSample * m_enc->time_base.den / m_enc->sample_rate / m_enc->time_base.num;
        } else {
        // adjust the sample number according to the given pts
            m_nTotalSample = ptsNow * m_enc->sample_rate * m_enc->time_base.num / m_enc->time_base.den;
        }
        return ptsNow + (nConvertedSample - nSampleLeft) * m_enc->time_base.den / m_enc->sample_rate / m_enc->time_base.num;
    }

private:
    AVFrame *m_frm = NULL;
    std::vector<AVPacket *> m_vPkt;

    uint8_t **m_apNewSample = NULL;
    int m_nInputSample = 0, m_nNewSample = 0;
    SwrContext *m_resampler = NULL;
    AVAudioFifo *m_fifo = NULL;
    int64_t m_nTotalSample = 0;
};
