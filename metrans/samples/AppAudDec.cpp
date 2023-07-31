#include <iostream>
#include <stdint.h>
#include "AvToolkit/AudDec.h"
#include "AvToolkit/Demuxer.h"

using namespace std;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

void SaveFrame(AVFrame *frm, FILE *fpOut) {
    int nBytePerSample = av_get_bytes_per_sample((AVSampleFormat)frm->format);
    if (frm->format <= AV_SAMPLE_FMT_DBL || frm->format == AV_SAMPLE_FMT_S64) {
        fwrite(frm->data[0], 1, nBytePerSample * frm->channels * frm->nb_samples, fpOut);
    } else {
        for (int i = 0; i < frm->nb_samples; i++) {
            for (int ch = 0; ch < frm->channels; ch++) {
                fwrite(frm->data[ch] + nBytePerSample * i, 1, nBytePerSample, fpOut);
            }
        }
    }
}

void DecodeAudio(const char *szInFilePath, const char *szOutFilePath) {
    Demuxer demuxer(szInFilePath, true, true);
    if (!demuxer.GetAudioStream()) {
        cout << "No audio stream in file " << szInFilePath << endl;
        return;
    }

    AVCodecParameters *par = demuxer.GetAudioStream()->codecpar;
    cout << "Audio info: " << avcodec_get_name(par->codec_id) << ", " << par->sample_rate << " Hz, " 
        << par->channels << " channels, " << av_get_sample_fmt_name((AVSampleFormat)par->format) << endl;

    AudDec dec(demuxer.GetAudioStream()->codecpar);
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
            SaveFrame(frm, fpOut);
            n += frm->nb_samples;
        }
    } while (pkt->size);
    fclose(fpOut);
    cout << "Decoded samples: " << n << " (" << 1.0 * n / par->sample_rate << " sec)" << endl;
}

int main(int argc, char *argv[]) {
    DecodeAudio("bunny.aac", "bunny.f32");
    return 0;
}
