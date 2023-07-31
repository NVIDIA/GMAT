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
#include <iostream>
#include "NvCodec/NvDecoderImageProvider.h"
#include "NvCodec/NvCommon.h"
#include "FramePresenterGL.h"

using namespace std;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

int DecodeWithNvDecoderImageProvider(CUcontext cuContext, char *szMediaPath) {
    if (!strstr(szMediaPath, "://")) {
        FILE *fpIn = fopen(szMediaPath, "rb");
        if (fpIn == NULL) {
            cout << "Unable to open file: " << szMediaPath << endl;
            return 1;
        }
        fclose(fpIn);
    }

    Demuxer demuxer(szMediaPath);
    NvDecLiteImageProvider dec(cuContext, &demuxer);
    if (!dec.IsReady()) {
        cout << "Decoder not ready" << endl;
        return 1;
    }
    FramePresenterGL presenter(cuContext, dec.GetWidth(), dec.GetHeight());
    uint8_t *dpFrame = 0;
    int nPitch = 0;
    int nFrame = 0;
    while (presenter.GetDeviceFrameBuffer(&dpFrame, &nPitch) 
            && dec.GetNextImageAsBgra((uint8_t *)dpFrame, nPitch, true)) {
        nFrame++;
    }
    cout << "Total frame decoded: " << nFrame << endl;
    return 0;
}

void ShowHelpAndExit(const char *szBadOption) {
    if (szBadOption) {
        cout << "Error parsing \"" << szBadOption << "\"" << endl;
    }
    cout << "Options:" << endl
        << "-i           Input file path" << endl
        << "-gpu         Ordinal of GPU to use" << endl
        ;
    exit(1);
}

void ParseCommandLine(int argc, char *argv[], char *szInputFileName, int &iGpu) 
{
    ostringstream oss;
    int i;
    for (i = 1; i < argc; i++) {
        if (!_stricmp(argv[i], "-h")) {
            ShowHelpAndExit(NULL);
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
        ShowHelpAndExit(argv[i]);
    }
}

int main(int argc, char **argv) {
    char szInFilePath[256] = "bunny.mp4";
    int iGpu = 0;
    CheckDefaultFileExists(szInFilePath);
    ParseCommandLine(argc, argv, szInFilePath, iGpu);

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
    ck(cuCtxCreate(&cuContext, CU_CTX_SCHED_BLOCKING_SYNC, cuDevice));

    return DecodeWithNvDecoderImageProvider(cuContext, szInFilePath);
}
