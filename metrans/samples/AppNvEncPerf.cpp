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

#include <stdio.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <cuda.h>
#include "NvCodec/NvEncLite.h"
#include "NvCodec/NvCommon.h"

using namespace std;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

void EncProc(NvEncLite *pEnc, uint8_t *pBuf, uint32_t nBufSize, int nFrameTotal) {
    vector<vector<uint8_t>> vPacket;
    int nFrameSize = pEnc->GetFrameSize();
    int n = (int)(nBufSize / nFrameSize);
    ck(cuCtxSetCurrent((CUcontext)pEnc->GetDevice()));
    for (int i = 0; i < nFrameTotal; i++) {
        int iFrame = i / n % 2 ? (n - i % n - 1) : i % n;
        pEnc->EncodeDeviceFrame(pBuf + iFrame * nFrameSize, 0, vPacket);
    }
    pEnc->EndEncode(vPacket);
}

void ShowHelpAndExit(const char *szBadOption = NULL) {
    if (szBadOption) {
        cout << "Error parsing \"" << szBadOption << "\"" << endl;
    }
    cout << "Options:" << endl
        << "-i           Input file path" << endl
        << "-s           Input resolution in this form: WxH" << endl
        << "-if          Input format: iyuv nv12 yuv444 p010 yuv444p16 bgra" << endl
        << "-gpu         Ordinal of GPU to use" << endl
        << "-frame       Number of frames to encode per thread (default is 1000)" << endl
        << "-thread      Number of encoding thread (default is 2)" << endl
        << "-single      (No value) Use single context (this may result in suboptimal performance; default is multiple contexts)" << endl
        << "-pause       (No value) Pause before exit" << endl
        ;
    cout << NvEncoderInitParam().GetHelpMessage();
    exit(1);
}

void ParseCommandLine(int argc, char *argv[], char *szInputFileName, int &nWidth, int &nHeight, 
    NV_ENC_BUFFER_FORMAT &eFormat, int &iGpu, int &nFrame, int &nThread, 
    bool &bSingle, bool &bPause, NvEncoderInitParam &initParam) 
{
    ostringstream oss;
    for (int i = 1; i < argc; i++) {
        if (!_stricmp(argv[i], "-h")) {
            ShowHelpAndExit();
        }
        if (!_stricmp(argv[i], "-i")) {
            if (++i == argc) {
                ShowHelpAndExit("-i");
            }
            sprintf(szInputFileName, "%s", argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-s")) {
            if (++i == argc || 2 != sscanf(argv[i], "%dx%d", &nWidth, &nHeight)) {
                ShowHelpAndExit("-s");
            }
            continue;
        }
        vector<string> vszFileFormatName = {
            "iyuv", "nv12", "yv12", "yuv444", "p010", "yuv444p16", "bgra", "argb10", "ayuv", "abgr", "abgr10"
        };
        NV_ENC_BUFFER_FORMAT aFormat[] = {
            NV_ENC_BUFFER_FORMAT_IYUV,
            NV_ENC_BUFFER_FORMAT_NV12,
            NV_ENC_BUFFER_FORMAT_YV12,
            NV_ENC_BUFFER_FORMAT_YUV444,
            NV_ENC_BUFFER_FORMAT_YUV420_10BIT,
            NV_ENC_BUFFER_FORMAT_YUV444_10BIT,
            NV_ENC_BUFFER_FORMAT_ARGB,
            NV_ENC_BUFFER_FORMAT_ARGB10,
            NV_ENC_BUFFER_FORMAT_AYUV,
            NV_ENC_BUFFER_FORMAT_ABGR,
            NV_ENC_BUFFER_FORMAT_ABGR10,
        };
        if (!_stricmp(argv[i], "-if")) {
            if (++i == argc) {
                ShowHelpAndExit("-if");
            }
            auto it = find(vszFileFormatName.begin(), vszFileFormatName.end(), argv[i]);
            if (it == vszFileFormatName.end()) {
                ShowHelpAndExit("-if");
            }
            eFormat = aFormat[it - vszFileFormatName.begin()];
            continue;
        }
        if (!_stricmp(argv[i], "-gpu")) {
            if (++i == argc) {
                ShowHelpAndExit("-gpu");
            }
            iGpu = atoi(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-frame")) {
            if (++i == argc) {
                ShowHelpAndExit("-frame");
            }
            nFrame = atoi(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-thread")) {
            if (++i == argc) {
                ShowHelpAndExit("-thread");
            }
            nThread = atoi(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-single")) {
            bSingle = true;
            continue;
        }
        if (!_stricmp(argv[i], "-pause")) {
            bPause = true;
            continue;
        }
        // Regard as encoder parameter
        if (argv[i][0] != '-') {
            ShowHelpAndExit(argv[i]);
        }
        oss << argv[i] << " ";
        while (i + 1 < argc && argv[i + 1][0] != '-') {
            oss << argv[++i] << " ";
        }
    }
    initParam = NvEncoderInitParam(oss.str().c_str());
}

int main(int argc, char **argv) {
    char szInFilePath[256] = "bunny.iyuv";
    int nWidth = 1920, nHeight = 1080;
    NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_IYUV;
    int iGpu = 0;
    int nFrame = 5000;
    int nThread = 2;
    bool bSingle = false;
    bool bPause = false;
    NvEncoderInitParam initParam;
    CheckDefaultFileExists(szInFilePath);
    ParseCommandLine(argc, argv, szInFilePath, nWidth, nHeight, eFormat, 
        iGpu, nFrame, nThread, bSingle, bPause, initParam);

    FILE *fp = fopen(szInFilePath, "r");
    if (!fp) {
        cout << "Unable to open YUV file " << szInFilePath << endl;
        return 1;
    }
    fclose(fp);
    
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

    uint8_t *pBuf = NULL;
    size_t nBufSize = 0;
    BufferedFileReader bufferedFileReader(szInFilePath, true);
    if (!bufferedFileReader.GetBuffer(&pBuf, &nBufSize)) {
        cout << "Failed to read file " << szInFilePath << endl;
        return 1;
    }

    CUcontext cuContext = NULL;
    ck(cuCtxCreate(&cuContext, CU_CTX_SCHED_BLOCKING_SYNC, cuDevice));

    vector<CUdeviceptr> vdpBuf;
    vector<NvEncLite *> vEnc;
    CUdeviceptr dpBuf;
    ck(cuMemAlloc(&dpBuf, nBufSize));
    vdpBuf.push_back(dpBuf);
    ck(cuMemcpyHtoD(dpBuf, pBuf, nBufSize));
    vEnc.push_back(new NvEncLite(cuContext, nWidth, nHeight, NV_ENC_BUFFER_FORMAT_IYUV_PL, &initParam));

    for (int i = 1; i < nThread; i++) {
        if (!bSingle) {
            ck(cuCtxCreate(&cuContext, CU_CTX_SCHED_BLOCKING_SYNC, cuDevice));
            CUdeviceptr dpBuf;
            ck(cuMemAlloc(&dpBuf, nBufSize));
            vdpBuf.push_back(dpBuf);
            ck(cuMemcpyHtoD(vdpBuf[i], pBuf, nBufSize));
        }
        vEnc.push_back(new NvEncLite(cuContext, nWidth, nHeight, NV_ENC_BUFFER_FORMAT_IYUV_PL, &initParam));
    }

    vector<thread *> vThread;
    StopWatch w;
    w.Start();
    for (int i = 0; i < nThread; i++) {
        vThread.push_back(new thread(EncProc, vEnc[i], (uint8_t *)(bSingle ? dpBuf : vdpBuf[i]), nBufSize, nFrame));
    }
    for (int i = 0; i < nThread; i++) {
        vThread[i]->join();
    }
    double t = w.Stop();

    for (int i = 0; i < nThread; i++) {
        ck(cuCtxSetCurrent((CUcontext)vEnc[i]->GetDevice()));
        if (!bSingle && i > 0) {
            ck(cuMemFree(vdpBuf[i]));
        }
        delete vThread[i];
        delete vEnc[i];
    }

    if (t) {
        int nTotal = nFrame * nThread;
        cout << "nTotal=" << nTotal << ", time=" << t << " seconds, FPS=" << nTotal / t << endl;
    }

    if (bPause) {
        cout << "Press any key to quit..." << endl;
        _getch();
    }
    return 0;
}
