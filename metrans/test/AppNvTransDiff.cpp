/*
* Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#include <iostream>
#include <algorithm>
#include <thread>
#include <cuda_runtime.h>
#include "AvToolkit/Muxer.h"
#include "NvCodec/NvEncLite.h"
#include "NvCodec/NvCommon.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

using namespace std;

void InterpolatePixel(unsigned char *dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight, float xSrc, float ySrc, float *dpDst);

void TestInterpolation() {
    int nSrcWidth = 256, nSrcHeight = 256, nSrcSize = nSrcWidth * nSrcHeight;
    uint8_t *pSrc = new uint8_t[nSrcSize];
    memset(pSrc, 0, nSrcSize);
    int n = 0;
    for (int i = 0; i < nSrcHeight; i++) {
        for (int j = 0; j <nSrcWidth; j++) {
            pSrc[i * nSrcWidth + j] = n++ % 256;
        }
    }
    uint8_t *dpSrc = NULL;
    size_t nSrcPitch = 0;
    ck(cudaMallocPitch(&dpSrc, &nSrcPitch, nSrcWidth, nSrcHeight));
    ck(cudaMemcpy2D(dpSrc, nSrcPitch, pSrc, nSrcWidth, nSrcWidth, nSrcHeight, cudaMemcpyHostToDevice));

    int nDstSize = 1 * sizeof(float);
    float *pDst = new float[1];
    float *dpDst = NULL;
    ck(cudaMalloc(&dpDst, nDstSize * sizeof(float)));

    for (int y = 0; y < 256; y++) {
        for (int x = 0; x < 256; x++) {
            float xSrc = (float)x + 0.9f, ySrc = (float)y + 0.9f;
            InterpolatePixel(dpSrc, nSrcPitch, nSrcWidth, nSrcHeight, xSrc, ySrc, dpDst);
            ck(cudaMemcpy(pDst, dpDst, nDstSize, cudaMemcpyDeviceToHost));
            cout << (int)(*pDst * 255.0f) << " ";
        }
        cout << endl;
    }

    ck(cudaFree(dpDst));
    ck(cudaFree(dpSrc));
    delete[] pSrc;
    delete[] pDst;
}

void ScaleGeneratedImage() {
    int nSrcWidth = 192, nSrcHeight = 108, nSrcSize = nSrcWidth * nSrcHeight * 3 / 2;
    int nDstWidth = 192, nDstHeight = 108, nDstSize = nDstWidth * nDstHeight * 3 / 2;

    uint8_t *pSrc = new uint8_t[nSrcSize], *pDst = new uint8_t[nDstSize];
    uint8_t *dpSrc = NULL, *dpDst = NULL;
    size_t nSrcPitch = 0;
    ck(cudaMallocPitch(&dpSrc, &nSrcPitch, nSrcWidth, nSrcHeight * 3 / 2));
    ck(cudaMalloc(&dpDst, nDstSize));
    
    int n = 0;
    for (int i = 0; i < nSrcHeight; i++) {
        int x = i % 510;
        n = x <= 255 ? x : 510 - x;
        for (int j = 0; j <nSrcWidth; j++) {
            int y = n++;
            pSrc[i * nSrcWidth + j] = y <= 255 ? y :510 - y;
        }
    }

    FILE *fp = fopen("src.nv12", "wb");
    ck(fp ? 0 : -1);
    fwrite(pSrc, 1, nSrcSize, fp);
    fclose(fp);

    ck(cudaMemcpy2D(dpSrc, nSrcPitch, pSrc, nSrcWidth, nSrcWidth, nSrcHeight * 3 / 2, cudaMemcpyHostToDevice));
    ScaleNv12(dpSrc, nSrcPitch, nSrcWidth, nSrcHeight, dpDst, nDstWidth, nDstWidth, nDstHeight);
    ck(cudaMemcpy(pDst, dpDst, nDstSize, cudaMemcpyDeviceToHost));

    fp = fopen("out.nv12", "wb");
    ck(fp ? 0 : -1);
    fwrite(pDst, 1, nDstSize, fp);
    fclose(fp);

    float value = 0.0f, *dpValue = NULL;
    ck(cudaMalloc(&dpValue, sizeof(float)));
    float xScale = 1.0f * nSrcWidth / nDstWidth, yScale = 1.0f * nSrcHeight / nDstHeight;
    for (int y = 0; y < nDstHeight; y++) {
        for (int x = 0; x < nDstWidth; x++) {
            float xSrc = (x + 0.5f) * xScale, ySrc = (y + 0.5f) * yScale;
            InterpolatePixel(dpSrc, nSrcPitch, nSrcWidth, nSrcHeight, xSrc, ySrc, dpValue);
            ck(cudaMemcpy(&value, dpValue, sizeof(float), cudaMemcpyDeviceToHost));
            //cout << (int)(value * 255.0f) << " ";
            pDst[y * nDstWidth + x] = (uint8_t)(value * 255.0f);
        }
        //cout << endl;
    }
    ck(cudaFree(dpValue));

    fp = fopen("out2.nv12", "wb");
    ck(fp ? 0 : -1);
    fwrite(pDst, 1, nDstSize, fp);
    fclose(fp);

    ck(cudaFree(dpDst));
    ck(cudaFree(dpSrc));
    delete[] pSrc;
    delete[] pDst;
}

void ScaleImage() {
    const char *szSrcFilePath = "src.nv12", *szDstFilePath = "out.nv12";
    int nSrcWidth = 1920, nSrcHeight = 1080, nSrcSize = nSrcWidth * nSrcHeight * 3 / 2;
    int nDstWidth = 1280, nDstHeight = 720, nDstSize = nDstWidth * nDstHeight * 3 / 2;

    uint8_t *pSrc = new uint8_t[nSrcSize], *pDst = new uint8_t[nDstSize];
    uint8_t *dpSrc = NULL, *dpDst = NULL;
    size_t nSrcPitch = 0;
    ck(cudaMallocPitch(&dpSrc, &nSrcPitch, nSrcWidth, nSrcHeight * 3 / 2));
    ck(cudaMalloc(&dpDst, nDstSize));

    FILE *fp = fopen(szSrcFilePath, "rb");
    ck(fp ? 0 : -1);
    fread(pSrc, 1, nSrcSize, fp);
    fclose(fp);

    ck(cudaMemcpy2D(dpSrc, nSrcPitch, pSrc, nSrcWidth, nSrcWidth, nSrcHeight * 3 / 2, cudaMemcpyHostToDevice));
    ScaleNv12(dpSrc, nSrcPitch, nSrcWidth, nSrcHeight, dpDst, nDstWidth, nDstWidth, nDstHeight);
    ck(cudaMemcpy(pDst, dpDst, nDstSize, cudaMemcpyDeviceToHost));

    fp = fopen(szDstFilePath, "wb");
    ck(fp ? 0 : -1);
    fwrite(pDst, 1, nDstSize, fp);
    fclose(fp);

    ck(cudaFree(dpDst));
    ck(cudaFree(dpSrc));
    delete[] pSrc;
    delete[] pDst;
}

void EncodeWidthMux(CUcontext cuContext, char *szInFilePath, int nWidth, int nHeight, NV_ENC_BUFFER_FORMAT eFormat, 
    char *szOutFilePath, NvEncoderInitParam *pInitParam, bool bVerbose) {
    FILE *fpIn = fopen(szInFilePath, "rb");
    if (fpIn == NULL) {
        cout << "Unable to open input file: " << szInFilePath << endl;
        return;
    }

    NvEncLite enc(cuContext, nWidth, nHeight, eFormat, pInitParam);
    if (!enc.ReadyForEncode()) {
        cout << "NvEncLite fails to initialize." << endl;
        return;
    }

    AVCodecParameters *par = ExtractAVCodecParameters(&enc);
    Muxer muxer(par, AVRational{1, 25}, NULL, AVRational{0, 1}, szOutFilePath);

    int nFrameSize = enc.GetFrameSize();
    uint8_t *pHostFrame = new uint8_t[nFrameSize];
    int nFrame = 0;
    AVPacket pkt;
    av_init_packet(&pkt);
    int64_t pts = 0;
    while (true) {
        // Load the next frame from disk
        int nRead = (int)fread(pHostFrame, 1, nFrameSize, fpIn);
        // For receiving encoded packets
        vector<vector<uint8_t>> vPacket;
        // NULL frame means EndEncode()
        enc.EncodeHostFrame(nRead == nFrameSize ? pHostFrame : NULL, 0, vPacket);
        nFrame += (int)vPacket.size();
        for (vector<uint8_t> &packet : vPacket) {
        // For each encoded packet
            if (bVerbose) cout << packet.size() << "\t\r";
            pkt.data = packet.data();
            pkt.size = packet.size();
            pkt.pts = pts++;
            pkt.dts = pkt.pts;
            muxer.MuxVideo(&pkt);
        }
        if (nRead != nFrameSize) break;
    }
    if (bVerbose) cout << endl;
    delete[] pHostFrame;
    fclose(fpIn);

    avcodec_parameters_free(&par);
    cout << "Total frames encoded: " << nFrame << endl << "Saved in file " << szOutFilePath << endl;
}

static int main_EncodeWidthMux(int argc, char **argv) {
    char szInFilePath[256] = "bunny.iyuv",
        szOutFilePath[256] = "";
    int nWidth = 1920, nHeight = 1080;
    NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_IYUV;
    int iGpu = 0;
    bool bVerbose = true;
    int nFrame = 0;
    NvEncoderInitParam initParam;
    CheckDefaultFileExists(szInFilePath);
    if (!*szOutFilePath) {
        sprintf(szOutFilePath, initParam.IsCodecH264() ? "out.mp4" : "out.hevc");
    }

    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    if (iGpu < 0 || iGpu >= nGpu) {
        cout << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << endl;
        return 1;
    }
    CUdevice cuDevice = 0;
    ck(cuDeviceGet(&cuDevice, iGpu));
    char szDeviceName[80];
    ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    cout << "GPU in use: " << szDeviceName << endl;
    CUcontext cuContext = NULL;
    ck(cuCtxCreate(&cuContext, 0, cuDevice));
    return 0;
}

int main(int argc, char **argv) {
    //TestInterpolation();
    ScaleGeneratedImage();
    return 0;
}
