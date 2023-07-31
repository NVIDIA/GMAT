#include <iostream>
#include <stdint.h>
#include "AvToolkit/VidDec.h"
#include "AvToolkit/VidEnc.h"
#include "AvToolkit/VidFilt.h"
#include "AvToolkit/AudDec.h"
#include "AvToolkit/AudEnc.h"
#include "AvToolkit/AudFilt.h"
#include "AvToolkit/Demuxer.h"
#include "AvToolkit/Muxer.h"

using namespace std;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

void FilterAndEncodeFrame(VidFilt *pFilt, VidEnc **ppEnc, int nFps, double fScale, AVFrame *frmIn, LazyMuxer *pMuxer) {
    vector<AVFrame *> vFrm;
    pFilt->Filter(frmIn, vFrm);
    for (AVFrame *frm : vFrm) {
        if (!*ppEnc) {
            AVRational timebase = pFilt->GetOutputTimebase();
            timebase.den = (int)(timebase.den / fScale);
            *ppEnc = new VidEnc((AVPixelFormat)frm->format, frm->width, frm->height, "h264_nvenc", AV_CODEC_ID_NONE, timebase, nFps);
        }
        vector<AVPacket *> vPkt;
        (*ppEnc)->Encode(frm, vPkt);
        if (!pMuxer->IsVideoStreamSet()) {
            pMuxer->SetVideoStream((*ppEnc)->GetCodecParameters(), (*ppEnc)->GetCodecContext()->time_base);
        }
        for (AVPacket *pkt : vPkt) {
            pMuxer->MuxVideo(pkt);
        }
    }
}

void FilterAndEncodeFrame(AudFilt *pFilt, AudEnc **ppEnc, AVFrame *frmIn, LazyMuxer *pMuxer) {
    vector<AVFrame *> vFrm;
    pFilt->Filter(frmIn, vFrm);
    for (AVFrame *frm : vFrm) {
        if (!*ppEnc) {
            AVCodecParameters *par = avcodec_parameters_alloc();
            par->sample_rate = 44100;
            *ppEnc = new AudEnc((AVSampleFormat)frm->format, frm->sample_rate, frm->channel_layout, AV_CODEC_ID_MP2, NULL, pFilt->GetOutputTimebase(), par);
            avcodec_parameters_free(&par);
        }
        vector<AVPacket *> vPkt;
        (*ppEnc)->Encode(frm, vPkt);
        if (!pMuxer->IsAudioStreamSet()) {
            pMuxer->SetAudioStream((*ppEnc)->GetCodecParameters(), (*ppEnc)->GetCodecContext()->time_base);
        }
        for (AVPacket *pkt : vPkt) {
            pMuxer->MuxAudio(pkt);
        }
    }
}

void DecodeVideo(const char *szInFilePath, const char *szOutFilePath) {
    Demuxer demuxer(szInFilePath, false, true);
    if (!demuxer.GetVideoStream()) {
        cout << "No video stream in file " << szInFilePath << endl;
        return;
    }

    AVStream *vs = demuxer.GetVideoStream();
    AVCodecParameters *vpar = vs->codecpar;
    VidDec vdec(vpar);
    char szFilterDesc[1024];
    const double fScale = 1.4;
    sprintf(szFilterDesc, "scale=1280x720,minterpolate=fps=%d/%d:mi_mode=dup", (int)(vs->avg_frame_rate.num * fScale * fScale), vs->avg_frame_rate.den);
    VidFilt vfilt((AVPixelFormat)vpar->format, vpar->width, vpar->height, vs->time_base, vpar->sample_aspect_ratio, szFilterDesc);
    VidEnc *pVenc = NULL;

    AVCodecParameters *apar = demuxer.GetAudioStream()->codecpar;
    AudDec adec(apar);
    sprintf(szFilterDesc, "atempo=%f", 1.0 / fScale);
    AudFilt afilt((AVSampleFormat)apar->format, apar->sample_rate, apar->channel_layout, demuxer.GetAudioStream()->time_base, szFilterDesc);
    AudEnc *pAenc = NULL;
    
    LazyMuxer muxer(szOutFilePath);

    AVPacket *pkt = NULL;
    vector<AVFrame *> vFrm;
    do {
        bool bAudio = false;
        demuxer.Demux(&pkt, &bAudio);
        vector<AVFrame *> vFrm;
        if (bAudio) {
            adec.Decode(pkt, vFrm);
            for (AVFrame *frm : vFrm) {
                FilterAndEncodeFrame(&afilt, &pAenc, frm, &muxer);
            }
        } else {
            vdec.Decode(pkt, vFrm);
            for (AVFrame *frm : vFrm) {
                FilterAndEncodeFrame(&vfilt, &pVenc, (int)(av_q2d(vs->avg_frame_rate) * fScale), fScale, frm, &muxer);
            }
        }
    } while (pkt->size);
}

int main(int argc, char *argv[]) {
    DecodeVideo("bunny.mp4", "out.mkv");
    return 0;
}
