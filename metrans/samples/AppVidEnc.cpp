#include <iostream>
#include <stdint.h>
#include "AvToolkit/VidEnc.h"

using namespace std;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

void EncodeFrame(const char *szInFilePath, AVPixelFormat eFormat, int nWidth, int nHeight, 
    const char *szOutFilePath) 
{
    FILE *fpIn = cknn(fopen(szInFilePath, "rb"));
    FILE *fpOut = cknn(fopen(szOutFilePath, "wb"));

    VidEnc enc(eFormat, nWidth, nHeight, "h264_nvenc");    
    int nFrameSize = enc.GetFrameSize();
    uint8_t *pHostFrame = new uint8_t[nFrameSize];
	vector<AVPacket *> vPkt;
	while (fread(pHostFrame, 1, nFrameSize, fpIn) == nFrameSize) {
        enc.Encode(pHostFrame, 0, 0, vPkt);
        for (AVPacket *pkt : vPkt) {
            cout << pkt->size << "\t\r";
			fwrite(pkt->data, 1, pkt->size, fpOut);
        }
    }
	delete[] pHostFrame;

    fclose(fpOut);
    fclose(fpIn);
}

int main(int argc, char *argv[]) {
    int nWidth = 1920, nHeight = 1080;
    EncodeFrame("bunny.iyuv", AV_PIX_FMT_YUV420P, nWidth, nHeight, "out.h264");
    return 0;
}
