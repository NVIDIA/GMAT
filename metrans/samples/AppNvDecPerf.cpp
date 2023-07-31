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

#include <cuda.h>
#include <cudaProfiler.h>
#include <stdio.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <string.h>
#include "AvToolkit/Demuxer.h"
#include "NvCodec/NvDecLite.h"
#include "NvCodec/NvCommon.h"

using namespace std;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

void DecProc(NvDecLite *pDec, vector<uint8_t *> *pvpPacketData, vector<int> *pvnPacketDataSize, int *pnFrameTotal) {
    int nFrameTotal = 0;
    for (unsigned i = 0; i < pvpPacketData->size(); i++) {
        nFrameTotal += pDec->Decode((*pvpPacketData)[i], (*pvnPacketDataSize)[i], nullptr, nullptr);
    }
    nFrameTotal += pDec->Decode(nullptr, 0, nullptr, nullptr);
    *pnFrameTotal = nFrameTotal;
}

void ShowHelpAndExit(const char *szBadOption = NULL) {
    if (szBadOption) {
        cout << "Error parsing \"" << szBadOption << "\"" << endl;
    }
    cout << "Options:" << endl
        << "-i           Input file path" << endl
        << "-gpu         Ordinal of GPU to use" << endl
        << "-thread      Number of encoding thread" << endl
        << "-single      (No value) Use single context (this may result in suboptimal performance; default is multiple contexts)" << endl
        << "-host        (No value) Copy frame to host memory (this may result in suboptimal performance; default is device memory)" << endl
        << "-pause       (No value) Pause before exit" << endl
        ;
    exit(1);
}

void ParseCommandLine(int argc, char *argv[], char *szInputFileName, int &iGpu, int &nThread, bool &bSingle, bool &bHost, bool &bPause) 
{
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
        if (!_stricmp(argv[i], "-gpu")) {
            if (++i == argc) {
                ShowHelpAndExit("-gpu");
            }
            iGpu = atoi(argv[i]);
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
        if (!_stricmp(argv[i], "-host")) {
            bHost = true;
            continue;
        }
        if (!_stricmp(argv[i], "-pause")) {
            bPause = true;
            continue;
        }
        ShowHelpAndExit(argv[i]);
    }
}

int main(int argc, char **argv) {
    char szInFilePath[256] = "perf.h264";
    int iGpu = 0;
    int nThread = 2; 
    bool bSingle = false;
    bool bHost = false;
    bool bPause = false;
    CheckDefaultFileExists(szInFilePath);
    ParseCommandLine(argc, argv, szInFilePath, iGpu, nThread, bSingle, bHost, bPause);
    
    FILE *fp = fopen(szInFilePath, "r");
    if (!fp) {
       cout << "Unable to open file: " << szInFilePath << endl;
       return 1;
    }
    fclose(fp);

    struct stat st;
    if (stat(szInFilePath, &st) != 0) {
        return 1;
    }
    int nBufSize = st.st_size;
    vector<uint8_t> vBuf(nBufSize);    
    uint8_t *pBuf = vBuf.data();

    vector<uint8_t *> vpPacketData;
    vector<int> vnPacketData;

    Demuxer demuxer(szInFilePath);
    AVPacket *pkt = nullptr;
    size_t offset = 0;
    while(demuxer.Demux(&pkt)) {
        memcpy(pBuf + offset, pkt->data, pkt->size);
        vpPacketData.push_back(pBuf + offset);
        vnPacketData.push_back(pkt->size);
        offset += pkt->size;
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

    vector<NvDecLite *> vDec;
    CUcontext cuContext = NULL;
    ck(cuCtxCreate(&cuContext, 0, cuDevice));
    for (int i = 0; i < nThread; i++) {
        if (!bSingle) {
            ck(cuCtxCreate(&cuContext, 0, cuDevice));
        }
        vDec.push_back(new NvDecLite(cuContext, !bHost, FFmpeg2NvCodecId(demuxer.GetVideoStream()->codecpar->codec_id)));
    }

    vector<thread *> vThread;
    vector<int> vnFrame;
    vnFrame.resize(nThread, 0);

    StopWatch watch;
    watch.Start();
    for (int i = 0; i < nThread; i++) {
        vThread.push_back(new thread(DecProc, vDec[i], &vpPacketData, &vnPacketData, &vnFrame[i]));
    }
    for (int i = 0; i < nThread; i++) {
        vThread[i]->join();
    }
    double sec = watch.Stop();

    int nTotal = 0;
    for (int i = 0; i < nThread; i++) {
        nTotal += vnFrame[i];
        delete vThread[i];
        delete vDec[i];
    }
    cout << "Total Frames Decoded=" << nTotal << ", time=" << sec << " seconds, FPS=" << (nTotal / sec) << endl;

    ck(cuProfilerStop());
    if (bPause) {
        cout << "Press any key to quit..." << endl;
        _getch();
    }
    return 0;
}
