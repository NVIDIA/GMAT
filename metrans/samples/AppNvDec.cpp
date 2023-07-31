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
#include <cuda.h>
#include "AvToolkit/Demuxer.h"
#include "NvCodec/NvDecLite.h"
#include "NvCodec/NvCommon.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

using namespace std;

void ConvertToPlanar(uint8_t *pHostFrame, int nWidth, int nHeight, int nBitDepth) {
    if (nBitDepth == 8) {
        // nv12->iyuv
        YuvConverter<uint8_t> converter8(nWidth, nHeight);
        converter8.UVInterleavedToPlanar(pHostFrame);
    } else {
        // p010->yuv420p16
        YuvConverter<uint16_t> converter16(nWidth, nHeight);
        converter16.UVInterleavedToPlanar((uint16_t *)pHostFrame);
    }
}

void DecodeFile(CUcontext cuContext, const char *szInFilePath, const char *szOutFilePath, bool bOutPlanar, 
    const NvDecLite::Rect &cropRect, const NvDecLite::Dim &resizeDim) 
{
    Demuxer demuxer(szInFilePath);
    AVCodecParameters *codecpar = demuxer.GetVideoStream()->codecpar;
    NvDecLite dec(cuContext, false, FFmpeg2NvCodecId(codecpar->codec_id), false, false, &cropRect, &resizeDim);
    AVPacket *pkt;
    int nFrame = 0;
    ofstream fOut(szOutFilePath, ios::out | ios::binary);
    do {
        demuxer.Demux(&pkt);
        uint8_t **ppFrame = NULL;
        NvFrameInfo *pInfo = NULL;
        int nFrameReturned = dec.Decode(pkt->data, pkt->size, &ppFrame, &pInfo);
        for (int i = 0; i < nFrameReturned; i++) {
            if (bOutPlanar) {
                ConvertToPlanar(ppFrame[i], pInfo[i].nWidth, pInfo[i].nHeight, pInfo[i].nBitDepth);
            }
            fOut.write(reinterpret_cast<char *>(ppFrame[i]), pInfo[i].nFrameSize);
        }
        nFrame += nFrameReturned;
    } while (pkt->size);

    cout << "Total frame decoded: " << nFrame << endl
            << "Saved in file " << szOutFilePath << " in " 
            << (dec.GetBitDepth() == 8 ? (bOutPlanar ? "iyuv" : "nv12") : (bOutPlanar ? "yuv420p16" : "p010")) 
            << " format" << endl;
}

void ShowDecoderCapability() {
    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    printf("Decoder Capability\n");
    printf("#  %-20.20s", "GPU");
    const char *aszCodecName[] = {"  H264", "  HEVC", "HEVC_10B", "HEVC_12B", "   VP9", " VP9_10B", " VP9_12B", "   VP8", "  MPEG4", "  MPEG2", "   VC1"};
    for (int i = 0; i < sizeof(aszCodecName) / sizeof(aszCodecName[0]); i++) {
        printf(" %-9s", aszCodecName[i]);
    }
    printf("\n");
    cudaVideoCodec aeCodec[] = {cudaVideoCodec_H264, cudaVideoCodec_HEVC, cudaVideoCodec_HEVC, cudaVideoCodec_HEVC, cudaVideoCodec_VP9, 
        cudaVideoCodec_VP9, cudaVideoCodec_VP9, cudaVideoCodec_VP8, cudaVideoCodec_MPEG4, cudaVideoCodec_MPEG2, cudaVideoCodec_VC1};
    int anBitDepthMinus8[] = {0, 0, 2, 4, 0, 2, 4, 0, 0, 0, 0};
    for (int iGpu = 0; iGpu < nGpu; iGpu++) {
        CUdevice cuDevice = 0;
        ck(cuDeviceGet(&cuDevice, iGpu));
        char szDeviceName[80];
        ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
        CUcontext cuContext = NULL;
        ck(cuCtxCreate(&cuContext, 0, cuDevice));

        printf("%-2d %-20.20s", iGpu, szDeviceName);
        for (int i = 0; i < sizeof(aeCodec) / sizeof(aeCodec[0]); i++) {
            CUVIDDECODECAPS decodeCaps = {};
            decodeCaps.eCodecType = aeCodec[i];
            decodeCaps.eChromaFormat = cudaVideoChromaFormat_420;
            decodeCaps.nBitDepthMinus8 = anBitDepthMinus8[i];

            cuvidGetDecoderCaps(&decodeCaps);
            unsigned h = decodeCaps.nMaxMBCount ? decodeCaps.nMaxMBCount * 256 / decodeCaps.nMaxWidth : 0;
            if (!h) {
                printf("     -    ");
            } else if (h == decodeCaps.nMaxHeight) {
                printf(" %4dx%-4d", decodeCaps.nMaxWidth, h);
            } else {
                printf(" %4d_%-4d", decodeCaps.nMaxWidth, h);
            }
        }
        printf("\n");

        ck(cuCtxDestroy(cuContext));
    }
}

void ShowHelpAndExit(const char *szBadOption = NULL) {
    if (szBadOption) {
        cout << "Error parsing \"" << szBadOption << "\"" << endl;
    }
    cout << "Options:" << endl
        << "-i             Input file path" << endl
        << "-o             Output file path" << endl
        << "-outplanar     Convert output to planar format" << endl
        << "-gpu           Ordinal of GPU to use" << endl
        << "-crop l,t,r,b  Crop rectangle in left,top,right,bottom (ignored for case 0)" << endl
        << "-resize WxH    Resize to dimension W times H (ignored for case 0)" << endl
        ;
    cout << endl;
    ShowDecoderCapability();
    exit(1);
}

void ParseCommandLine(int argc, char *argv[], char *szInputFileName, char *szOutputFileName, 
    bool &bOutPlanar, int &iGpu, NvDecLite::Rect &cropRect, NvDecLite::Dim &resizeDim)
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
        if (!_stricmp(argv[i], "-outplanar")) {
            bOutPlanar = true;
            continue;
        }
        if (!_stricmp(argv[i], "-gpu")) {
            if (++i == argc) {
                ShowHelpAndExit("-gpu");
            }
            iGpu = atoi(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-crop")) {
            if (++i == argc || 4 != sscanf(
                    argv[i], "%d,%d,%d,%d", 
                    &cropRect.l, &cropRect.t, &cropRect.r, &cropRect.b)) {
                ShowHelpAndExit("-crop");
            }
            if ((cropRect.r - cropRect.l) % 2 == 1 || (cropRect.b - cropRect.t) % 2 == 1) {
                cout << "Cropping rect must have width and height of even numbers" << endl;
                exit(1);
            }
            continue;
        }
        if (!_stricmp(argv[i], "-resize")) {
            if (++i == argc || 2 != sscanf(argv[i], "%dx%d", &resizeDim.w, &resizeDim.h)) {
                ShowHelpAndExit("-resize");
            }
            if (resizeDim.w % 2 == 1 || resizeDim.h % 2 == 1) {
                cout << "Resizing rect must have width and height of even numbers" << endl;
                exit(1);
            }
            continue;
        }
        ShowHelpAndExit(argv[i]);
    }
}

int main(int argc, char **argv) {
    char szInFilePath[256] = "bunny.h264",
        szOutFilePath[256] = "";
    bool bOutPlanar = false;
    int iGpu = 0;
    NvDecLite::Rect cropRect = {};
    NvDecLite::Dim resizeDim = {};
    CheckDefaultFileExists(szInFilePath);
    ParseCommandLine(argc, argv, szInFilePath, szOutFilePath, bOutPlanar, iGpu, cropRect, resizeDim);
    if (!*szOutFilePath) {
        sprintf(szOutFilePath, bOutPlanar ? "out.planar" : "out.native");
    }

    FILE *fpIn = fopen(szInFilePath, "rb");
    if (fpIn == NULL) {
        cout << "Unable to open file: " << szInFilePath << endl;
        return 1;
    }
    fclose(fpIn);
    FILE *fpOut = fopen(szOutFilePath, "wb");
    if (fpOut == NULL) {
        cout << "Unable to open file: " << szInFilePath << endl;
        return 1;
    }
    fclose(fpOut);

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

    DecodeFile(cuContext, szInFilePath, szOutFilePath, bOutPlanar, cropRect, resizeDim);
    return 0;
}
