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
#include "AvToolkit/Demuxer.h"
#include "NvCodec/NvDecLite.h"
#include "NvCodec/NvEncLite.h"
#include "NvCodec/NvCommon.h"

using namespace std;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

void ShowHelpAndExit(const char *szBadOption = NULL) {
    if (szBadOption) {
        cout << "Error parsing \"" << szBadOption << "\"" << endl;
    }
    cout << "Options:" << endl
        << "-i           input_file" << endl
        << "-o           output_file" << endl
        << "-ob          Bit depth of the output: 8 10" << endl
        << "-gpu         Ordinal of GPU to use" << endl
        ;
    cout << NvEncoderInitParam().GetHelpMessage(false, false, true);
    exit(1);
}

void ParseCommandLine(int argc, char *argv[], char *szInputFileName, char *szOutputFileName, int &nOutBitDepth, int &iGpu, NvEncoderInitParam &initParam) 
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
        if (!_stricmp(argv[i], "-ob")) {
            if (++i == argc) {
                ShowHelpAndExit("-ob");
            }
            nOutBitDepth = atoi(argv[i]);
            if (nOutBitDepth != 8 && nOutBitDepth != 10) {
                ShowHelpAndExit("-ob");
            }
            continue;
        }
        if (!_stricmp(argv[i], "-gpu")) {
            if (++i == argc) {
                ShowHelpAndExit("-gpu");
            }
            iGpu = atoi(argv[i]);
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
    char szInFilePath[260] = "bunny.mp4";
    char szOutFilePath[260] = "";
    int nOutBitDepth = 0;
    int iGpu = 0;
    NvEncoderInitParam initParam;
    CheckDefaultFileExists(szInFilePath);
    ParseCommandLine(argc, argv, szInFilePath, szOutFilePath, nOutBitDepth, iGpu, initParam);
    if (!*szOutFilePath) {
        sprintf(szOutFilePath, initParam.IsCodecH264() ? "out.h264" : "out.hevc");
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

    Demuxer demuxer(szInFilePath);
    AVCodecParameters *codecpar = demuxer.GetVideoStream()->codecpar;
    NvDecLite dec(cuContext, true, FFmpeg2NvCodecId(codecpar->codec_id));

    NvEncLite *pEnc = NULL;
    int nFrame = 0;
    bool bOut10 = false;
    CUdeviceptr dpFrameConverted = 0;
    AVPacket *pkt = NULL;
    do {
        demuxer.Demux(&pkt);        
        uint8_t **ppFrame;
        NvFrameInfo *pInfo;
        int nFrameReturned = dec.Decode(pkt->data, pkt->size, &ppFrame, &pInfo);
        for (int i = 0; i < nFrameReturned; i++) {
            uint8_t *dpFrame = ppFrame[i];
            if (!pEnc) {
                bOut10 = nOutBitDepth ? nOutBitDepth > 8 : dec.GetBitDepth() > 8;
                pEnc = new NvEncLite(cuContext, dec.GetWidth(), dec.GetHeight(), 
                    bOut10 ? NV_ENC_BUFFER_FORMAT_YUV420_10BIT : NV_ENC_BUFFER_FORMAT_NV12, &initParam);
                ck(cuMemAlloc(&dpFrameConverted, pEnc->GetFrameSize()));
            }
            vector<vector<uint8_t>> vPacket;
            if ((bOut10 && dec.GetBitDepth() > 8) || (!bOut10 && dec.GetBitDepth() == 8)) {
            // Input and output have the same bit depth
                pEnc->EncodeDeviceFrame(dpFrame, 0, vPacket);
            } else {
            // Bit depth conversion is needed
                if (bOut10) {
                    ConvertUInt8ToUInt16((uint8_t *)dpFrame, (uint16_t *)dpFrameConverted, dec.GetFrameSize());
                } else {
                    ConvertUInt16ToUInt8((uint16_t *)dpFrame, (uint8_t *)dpFrameConverted, dec.GetFrameSize() / 2);
                }
                pEnc->EncodeDeviceFrame((uint8_t *)dpFrameConverted, 0, vPacket);
            }
            nFrame += (int)vPacket.size();
            for (vector<uint8_t> &packet : vPacket) {
                cout << packet.size() << "\t\r";
                fwrite(packet.data(), 1, packet.size(), fpOut);
            }
        }

        if (!pkt->size && pEnc) {
            vector<vector<uint8_t>> vPacket;
            pEnc->EndEncode(vPacket);
            nFrame += (int)vPacket.size();
            for (vector<uint8_t> &packet : vPacket) {
                cout << packet.size() << "\t\r";
                fwrite(packet.data(), 1, packet.size(), fpOut);
            }
            cout << endl;
            delete pEnc;
        }
    } while (pkt->size);
    
    fclose(fpOut);

    std::cout << "Total frame transcoded: " << nFrame << std::endl << "Saved in file " << szOutFilePath << " of " << (bOut10 ? 10 : 8) << " bit depth" << std::endl;

    return 0;
}
