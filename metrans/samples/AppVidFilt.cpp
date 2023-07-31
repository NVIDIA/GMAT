#include <iostream>
#include <stdint.h>
#include "AvToolkit/VidFilt.h"

using namespace std;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

void FilterAndSaveFrame(VidFilt &filt, uint8_t *pFrame, FILE *fpOut) {
    vector<AVFrame *> vFrm;
    filt.Filter(pFrame, 0, 0, vFrm);
    for (AVFrame * frm : vFrm) {
	    for (int i = 0; i < frm->height; i++) {
		    fwrite(frm->data[0] + i * frm->linesize[0], 1, frm->width, fpOut);
	    }
	    for (int i = 0; i < frm->height / 2; i++) {
		    fwrite(frm->data[1] + i * frm->linesize[1], 1, frm->width / 2, fpOut);
	    }
	    for (int i = 0; i < frm->height / 2; i++) {
		    fwrite(frm->data[2] + i * frm->linesize[2], 1, frm->width / 2, fpOut);
	    }
    }
}

void FilterFrame(const char *szInFilePath, AVPixelFormat eFormat, int nWidth, int nHeight, 
    const char *szOutFilePath) 
{
    FILE *fpIn = cknn(fopen(szInFilePath, "rb"));
    FILE *fpOut = cknn(fopen(szOutFilePath, "wb"));

    VidFilt filt(eFormat, nWidth, nHeight, AVRational{1, 1}, AVRational{1, 1}, "scale=1280:720,hflip");
    int nFrameSize = filt.GetFrameSize();
    uint8_t *pFrame = new uint8_t[nFrameSize];
    vector<AVPacket *> vPkt;
    while (fread(pFrame, 1, nFrameSize, fpIn) == nFrameSize) {
        FilterAndSaveFrame(filt, pFrame, fpOut);
    }
    delete[] pFrame;

    fclose(fpOut);
    fclose(fpIn);
}

int main(int argc, char *argv[]) {
    int nWidth = 1920, nHeight = 1080;
    FilterFrame("bunny.iyuv", AV_PIX_FMT_YUV420P, nWidth, nHeight, "out.iyuv");
    return 0;
}
