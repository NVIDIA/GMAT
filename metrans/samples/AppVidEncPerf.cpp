#include <iostream>
#include <thread>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "AvToolkit/VidEnc.h"
#include "NvCodec/NvCommon.h"

using namespace std;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

void EncodeFrameCuCtx(const char *szInFilePath, AVPixelFormat eFormat, int nWidth, int nHeight, 
    const char *szOutFilePath) 
{
    FILE *fpIn = cknn(fopen(szInFilePath, "rb"));
    FILE *fpOut = cknn(fopen(szOutFilePath, "wb"));

    VidEnc enc(eFormat, nWidth, nHeight, "h264_nvenc",
                AV_CODEC_ID_MPEG4, {}, 25, NULL, true);
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
void EncodeFrame(const char *szInFilePath, AVPixelFormat eFormat, int nWidth, int nHeight, 
    const char *szOutFilePath) 
{
    FILE *fpIn = cknn(fopen(szInFilePath, "rb"));
    // FILE *fpOut = cknn(fopen(szOutFilePath, "wb"));

    VidEnc enc(eFormat, nWidth, nHeight, "h264_nvenc");    
    int nFrameSize = enc.GetFrameSize();
    uint8_t *pHostFrame = new uint8_t[nFrameSize];
	vector<AVPacket *> vPkt;
	while (fread(pHostFrame, 1, nFrameSize, fpIn) == nFrameSize) {
        enc.Encode(pHostFrame, 0, 0, vPkt);
        for (AVPacket *pkt : vPkt) {
            cout << pkt->size << "\t\r";
			// fwrite(pkt->data, 1, pkt->size, fpOut);
        }
    }
	delete[] pHostFrame;

    // fclose(fpOut);
    fclose(fpIn);
}

int main(int argc, char *argv[]) {
    int nWidth = 1920, nHeight = 1080;
    cudaSetDevice(0);
    // CUcontext dummy;
    // cuCtxGetCurrent(&dummy);
    // std::cout << dummy << std::endl;
    // EncodeFrameCuCtx("bunny.iyuv", AV_PIX_FMT_YUV420P, nWidth, nHeight, "out.h264");
    
    int nThread = 80;
    // std::string outPath{"out.h264"};
    vector<thread *> vThread1, vThread2;
    StopWatch w;
    w.Start();
    for (int i = 0; i < nThread; i++) {
        std::string outPath = std::to_string(i) + "out.h264";
        // std::cout << "Output path: " << outPath << "\n";
        vThread1.push_back(new thread(EncodeFrameCuCtx, "bunny.iyuv", AV_PIX_FMT_YUV420P, nWidth, nHeight, outPath.c_str()));
    }
    for (int i = 0; i < nThread; i++) {
        vThread1[i]->join();
    }
    double t1 = w.Stop();
    cudaDeviceReset();

    w.Start();
    for (int i = 0; i < nThread; i++) {
        std::string outPath = std::to_string(i) + "out.h264";
        // std::cout << "Output path: " << outPath << "\n";
        vThread2.push_back(new thread(EncodeFrame, "bunny.iyuv", AV_PIX_FMT_YUV420P, nWidth, nHeight, outPath.c_str()));
    }
    for (int i = 0; i < nThread; i++) {
        vThread2[i]->join();
    }
    double t2 = w.Stop();

    if (t2) {
        std::cout << "time=" << t2 << " seconds\n";
    }
    return 0;
}
