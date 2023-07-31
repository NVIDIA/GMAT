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
#include <algorithm>
#include "NvCodec/NvDecoderImageProvider.h"
#include "NvCodec/NvCommon.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

using namespace std;

void ShowHelpAndExit(const char *szBadOption = NULL) {
    if (szBadOption) {
        cout << "Error parsing \"" << szBadOption << "\"" << endl;
    }
    cout << "Options:" << endl
        << "-i           Input file path" << endl
        << "-o           Output file path" << endl
        << "-of          Output format: native bgrp bgra bgra64" << endl
        << "-gpu         Ordinal of GPU to use" << endl
        ;
    exit(1);
}

enum OutputFormat {
    native = 0, bgrp, bgra, bgra64
};
vector<string> vstrOutputFormatName = {
    "native", "bgrp", "bgra", "bgra64"
};

void ParseCommandLine(int argc, char *argv[], char *szInputFileName,
    OutputFormat &eOutputFormat, char *szOutputFileName, int &iGpu) 
{
    ostringstream oss;
    int i;
    for (i = 1; i < argc; i++) {
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
        if (!_stricmp(argv[i], "-o")) {
            if (++i == argc) {
                ShowHelpAndExit("-o");
            }
            sprintf(szOutputFileName, "%s", argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-of")) {
            if (++i == argc) {
                ShowHelpAndExit("-of");
            }
            auto it = find(vstrOutputFormatName.begin(), vstrOutputFormatName.end(), argv[i]);
            if (it == vstrOutputFormatName.end()) {
                ShowHelpAndExit("-of");
            }
            eOutputFormat = (OutputFormat)(it - vstrOutputFormatName.begin());
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
    char szMediaPath[256] = "bunny.h264",
        szOutFilePath[256] = "";
    OutputFormat eOutputFormat = native;
    int iGpu = 0;
    CheckDefaultFileExists(szMediaPath);
    ParseCommandLine(argc, argv, szMediaPath, eOutputFormat, szOutFilePath, iGpu);
    if (!*szOutFilePath) {
        sprintf(szOutFilePath, "out.%s", vstrOutputFormatName[eOutputFormat].c_str());
    }

    if (!strstr(szMediaPath, "://")) {
        FILE *fpIn = fopen(szMediaPath, "rb");
        if (fpIn == NULL) {
            cout << "Unable to open file: " << szMediaPath << endl;
            return 1;
        }
        fclose(fpIn);
    }
    
    FILE *fpOut = fopen(szOutFilePath, "wb");
    if (fpOut == NULL) {
        cout << "Unable to open file: " << szOutFilePath << endl;
        return 1;
    }
    
    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    if (iGpu < 0 || iGpu >= nGpu) {
        LOG(ERROR) << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]";
        return 1;
    }
    CUdevice cuDevice = 0;
    ck(cuDeviceGet(&cuDevice, iGpu));
    char szDeviceName[80];
    ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    LOG(INFO) << "GPU in use: " << szDeviceName;
    CUcontext cuContext = NULL;
    ck(cuCtxCreate(&cuContext, 0, cuDevice));

    BufferedFileReader reader(szMediaPath);
    uint8_t *pBuf;
    size_t nBufSize;
    reader.GetBuffer(&pBuf, &nBufSize);
    Demuxer demuxer(pBuf, nBufSize, false);
    NvDecLiteImageProvider dec(cuContext, &demuxer);
    if (!dec.IsReady()) {
        cout << "Decoder not ready" << endl;
        return 1;
    }
    int nWidth = dec.GetWidth(), nHeight = dec.GetHeight();
    int anSize[] = {0, 3, 4, 8};
    int nFrameSize = eOutputFormat == native ? dec.GetFrameSize() : nWidth * nHeight * anSize[eOutputFormat];
    uint8_t *pImage = new uint8_t[nFrameSize];
    int nFrame = 0;
    /* The supported image formats in this sample is limited. You can add your own GetNextImageAsXXX method (CUDA programming is required).
       Note: color space conversion is done by CUDA kernels, which run slowly in Debug build of Microsoft Visual Studio,
       therefore you should use Release build for production or performance evaluation.*/
    while (true) {
        if (!(eOutputFormat == native && dec.GetNextFrame(pImage)) &&
            !(eOutputFormat == bgrp && dec.GetNextImageAsBgrPlanar(pImage)) && 
            !(eOutputFormat == bgra && dec.GetNextImageAsBgra(pImage)) && 
            !(eOutputFormat == bgra64 && dec.GetNextImageAsBgra64(pImage))) {
            break;
        }
        nFrame++;
        fwrite(pImage, nFrameSize, 1, fpOut);
    }

    fclose(fpOut);

    cout << "Total frame decoded: " << nFrame << endl << "Saved in file " << szOutFilePath << endl;
    return 0;
}
