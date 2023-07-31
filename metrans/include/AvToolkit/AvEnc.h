#pragma once

#include <vector>
#include <map>
#include <assert.h>
extern "C" {
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavcodec/avcodec.h>
}
#include "Logger.h"
#include "AvCommon.h"

extern simplelogger::Logger *logger;

class AvEnc {
public:
    virtual ~AvEnc() {
        if (m_par) {
            avcodec_parameters_free(&m_par);
        }
    }
    AVCodecParameters *GetCodecParameters() {
        if (!m_par) {
            m_par = avcodec_parameters_alloc();
            ckav(avcodec_parameters_from_context(m_par, m_enc));
        }
        return m_par;
    }
    AVCodecContext *GetCodecContext() {
        return m_enc;
    }

protected:
    AVCodecContext *m_enc;

private:
    AVCodecParameters *m_par = NULL;
};
