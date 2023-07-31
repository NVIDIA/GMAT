#include <iostream>
#include <stdint.h>
#include "AvToolkit/Demuxer.h"
#include "AvToolkit/AudEnc.h"
#include "AvToolkit/AudDec.h"
#include "AvToolkit/AudFilt.h"

using namespace std;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

void FilterAndEncodeFrame(AudFilt *pFilt, AudEnc *pEnc, AVFrame *frmIn, FILE *fpOut) {
    vector<AVFrame *> vFrm;
    pFilt->Filter(frmIn, vFrm);
    for (AVFrame *frm : vFrm) {
        vector<AVPacket *> vPkt;
        pEnc->Encode(frm, vPkt);
        for (AVPacket *pkt : vPkt) {
            fwrite(pkt->data, 1, pkt->size, fpOut);
        }
    }
}

void TranscodeAudio(const char *szInFilePath, const char *szOutFilePath) {
    Demuxer demuxer(szInFilePath, true, true);
    if (!demuxer.GetAudioStream()) {
        cout << "No audio stream in file " << szInFilePath << endl;
        return;
    }

    AVCodecParameters *par = demuxer.GetAudioStream()->codecpar;
    cout << "Audio info: " << avcodec_get_name(par->codec_id) << ", " << par->sample_rate << " Hz, " 
        << par->channels << " channels, " << av_get_sample_fmt_name((AVSampleFormat)par->format) << endl;

    AudDec dec(demuxer.GetAudioStream()->codecpar);
    AudEnc enc(AV_SAMPLE_FMT_S16, 48000, AV_CH_LAYOUT_STEREO, AV_CODEC_ID_MP2);
    AudFilt *pFilt = NULL;
    int n = 0;
    AVPacket *pkt = NULL;
    vector<AVFrame *> vFrm;
    FILE *fpOut = cknn(fopen(szOutFilePath, "wb"));    
    do {
        bool bAudio = false;
        demuxer.Demux(&pkt, &bAudio);
        if (!bAudio) {
            continue;
        }
        dec.Decode(pkt, vFrm);
        for (AVFrame *frm : vFrm) {
            if (!pFilt) {
                pFilt = new AudFilt((AVSampleFormat)frm->format, frm->sample_rate, frm->channel_layout, dec.GetCodecContext()->time_base, "aresample=osf=s16");
            }
            FilterAndEncodeFrame(pFilt, &enc, frm, fpOut);
            n += frm->nb_samples;
        }
    } while (pkt->size);
    fclose(fpOut);
    if (pFilt) delete pFilt;
    cout << "Transcoded samples: " << n << " (" << 1.0 * n / par->sample_rate << " sec)" << endl;
}

int main(int argc, char *argv[]) {
    TranscodeAudio("bunny.aac", "out.mp3");
    return 0;
}
