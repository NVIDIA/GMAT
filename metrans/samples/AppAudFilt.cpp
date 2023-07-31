#include <iostream>
#include <stdint.h>
#include "AvToolkit/AudFilt.h"

using namespace std;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

void FilterAndSaveFrame(AudFilt &filt, uint8_t **apSample, int nSample, FILE *fpOut) {
    vector<AVFrame *> vFrm;
    filt.Filter(apSample, nSample, 0, vFrm);
    for (AVFrame * frm : vFrm) {
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
}

void FilterAudio(const char *szInFilePath, const char *szOutFilePath) 
{
    FILE *fpIn = cknn(fopen(szInFilePath, "rb"));
    FILE *fpOut = cknn(fopen(szOutFilePath, "wb"));

    const int nBytePerSample = sizeof(float) * 2;
    int nSample = 100;
    int nSampleDataSize = nSample * nBytePerSample;
    uint8_t *pSamples = new uint8_t[nSampleDataSize];
    AudFilt filt(AV_SAMPLE_FMT_FLT, 48000, AV_CH_LAYOUT_STEREO, AVRational{1, 1}, "aresample=osf=s16");
    int nRead;
	while (nRead = fread(pSamples, 1, nSampleDataSize, fpIn)) {
        FilterAndSaveFrame(filt, &pSamples, nRead / nBytePerSample, fpOut);
    };
	delete[] pSamples;

    fclose(fpOut);
    fclose(fpIn);
}

int main(int argc, char *argv[]) {
	FilterAudio("bunny.f32", "bunny.s16");
    return 0;
}
