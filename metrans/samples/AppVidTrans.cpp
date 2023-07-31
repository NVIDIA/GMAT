#include <iostream>
#include <stdint.h>
#include "AvToolkit/VidDec.h"
#include "AvToolkit/VidEnc.h"
#include "AvToolkit/VidFilt.h"
#include "AvToolkit/Demuxer.h"
#include "AvToolkit/Muxer.h"

using namespace std;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

void TranscodeVideo(const char *szInFilePath, const char *szOutFilePath) {
    Demuxer demuxer(szInFilePath);
    if (!demuxer.GetVideoStream()) {
        cout << "No video stream in file " << szInFilePath << endl;
        return;
    }

    VidDec dec(demuxer.GetVideoStream()->codecpar);
    VidEnc *pEnc = NULL;
    VidFilt *pFilt = NULL;
    Muxer *pMuxer = NULL;
    int n = 0;
    AVPacket *pkt = NULL;
    vector<AVFrame *> vFrm;
    double fScale = 1.5;
    do {
        demuxer.Demux(&pkt);
        dec.Decode(pkt, vFrm);
        for (AVFrame *srcFrm : vFrm) {
            //cout << "src pts = " << srcFrm->pts << endl;
            if (!pEnc) {
                cout << "demuxer timebase=" << demuxer.GetVideoStream()->time_base << endl;
                pFilt = new VidFilt((AVPixelFormat)srcFrm->format, srcFrm->width, srcFrm->height, demuxer.GetVideoStream()->time_base, AVRational{1,1}, "scale=1280x720,hflip");
                cout << "filter timebase=" << pFilt->GetOutputTimebase() << endl;
                pEnc = new VidEnc((AVPixelFormat)srcFrm->format, 1280, 720, "h264_nvenc", AV_CODEC_ID_NONE, pFilt->GetOutputTimebase());
                pMuxer = new Muxer(pEnc->GetCodecParameters(), pEnc->GetCodecContext()->time_base, NULL, AVRational{}, szOutFilePath);
            }
            vector<AVFrame *> vFrm;
            pFilt->Filter(srcFrm, vFrm);
            for (AVFrame *frm : vFrm) {
                //cout << "filtered pts = " << frm->pts << endl;
                vector<AVPacket *> vPkt;
                pEnc->Encode(frm, vPkt);
                for (AVPacket *pkt : vPkt) {
                    pkt->pts = (int64_t)(pkt->pts * fScale);
                    pkt->dts = (int64_t)(pkt->dts * fScale);
                    pMuxer->MuxVideo(pkt);
                    n++;
                }
            }
        }
    } while (pkt->size);
    pFilt->Filter(NULL, vFrm);
    for (AVFrame *frm : vFrm) {
        //cout << "filtered pts = " << frm->pts << endl;
        vector<AVPacket *> vPkt;
        pEnc->Encode(frm, vPkt);
        for (AVPacket *pkt : vPkt) {
            pkt->pts = (int64_t)(pkt->pts * fScale);
            pkt->dts = (int64_t)(pkt->dts * fScale);
            pMuxer->MuxVideo(pkt);
            n++;
        }
    }
    vector<AVPacket *> vPkt;
    pEnc->Encode(NULL, vPkt);
    for (AVPacket *pkt : vPkt) {
        pkt->pts = (int64_t)(pkt->pts * fScale);
        pkt->dts = (int64_t)(pkt->dts * fScale);
        pMuxer->MuxVideo(pkt);
        n++;
    }
    delete pMuxer;
    delete pFilt;
    delete pEnc;
    cout << "Transcoded samples: " << n << endl;
}

int main(int argc, char *argv[]) {
    TranscodeVideo("bunny.mp4", "out.mp4");
    return 0;
}
