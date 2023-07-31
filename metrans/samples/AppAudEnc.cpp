#include <iostream>
#include <stdint.h>
#include "AvToolkit/AudEnc.h"

using namespace std;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

void EncodeAudio(const char *szInFilePath, const char *szOutFilePath, AVCodecParameters *par) 
{
    FILE *fpIn = cknn(fopen(szInFilePath, "rb"));
    FILE *fpOut = cknn(fopen(szOutFilePath, "wb"));

    const int nBytePerSample = sizeof(short) * 2;
    int nSample = 100;
    int nSampleDataSize = nSample * nBytePerSample;
    AudEnc enc(AV_SAMPLE_FMT_S16, 48000, AV_CH_LAYOUT_STEREO, AV_CODEC_ID_MP2, NULL, AVRational{}, par);
    uint8_t *pSamples = new uint8_t[nSampleDataSize];
    int nRead;
    int pts = 0;
    do {
        nRead = fread(pSamples, 1, nSampleDataSize, fpIn);
        int nSampleRead = nRead / nBytePerSample;
        vector<AVPacket *> vPkt;
        if (!enc.Encode(nSampleRead ? &pSamples : NULL, nRead / nBytePerSample, pts, vPkt)) {
            break;
        }
        pts += nSampleRead;
        for (unsigned i = 0; i < vPkt.size(); i++) {
            cout << vPkt[i]->size << "\r";
            fwrite(vPkt[i]->data, 1, vPkt[i]->size, fpOut);
        }
    } while (nRead);
    delete[] pSamples;

    fclose(fpOut);
    fclose(fpIn);
}

int main(int argc, char *argv[]) {
    AVCodecParameters *par = avcodec_parameters_alloc();
    par->bit_rate = 128000;
    EncodeAudio("bunny.s16", "bunny.mp3", par);
    avcodec_parameters_free(&par);
    return 0;
}
