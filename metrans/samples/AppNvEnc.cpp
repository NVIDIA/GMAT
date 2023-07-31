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
#include <cuda.h>
#include "Logger.h"
#include "NvCodec/NvEncLite.h"
#include "NvCodec/NvCommon.h"

using namespace std;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

void EncodeHostFrame(CUcontext cuContext, char *szInFilePath, int nWidth, int nHeight, NV_ENC_BUFFER_FORMAT eFormat, 
    char *szOutFilePath, NvEncoderInitParam *pInitParam, bool bVerbose) 
{
    FILE *fpIn = fopen(szInFilePath, "rb");
    if (fpIn == NULL) {
        cout << "Unable to open input file: " << szInFilePath << endl;
        return;
    }
    FILE *fpOut = fopen(szOutFilePath, "wb");
    if (!fpOut) {
        cout << "Unable to open output file: " << szOutFilePath << endl;
        return;
    }

    NvEncLite enc(cuContext, nWidth, nHeight, eFormat, pInitParam);
    if (!enc.ReadyForEncode()) {
        cout << "NvEncLite fails to initialize." << endl;
        return;
    }

    int nFrameSize = enc.GetFrameSize();
    uint8_t *pHostFrame = new uint8_t[nFrameSize];
    int nFrame = 0;
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
            fwrite(packet.data(), 1, packet.size(), fpOut);
        }
        if (nRead != nFrameSize) break;
    }
    if (bVerbose) cout << endl;
    delete[] pHostFrame;
    fclose(fpOut);
    fclose(fpIn);

    cout << "Total frames encoded: " << nFrame << endl << "Saved in file " << szOutFilePath << endl;
}

void EncodeDeviceFrameWithPicParam(CUcontext cuContext, char *szInFilePath, int nWidth, int nHeight, NV_ENC_BUFFER_FORMAT eFormat,
    char *szOutFilePath, NvEncoderInitParam *pInitParam, bool bVerbose)
{
    FILE *fpIn = fopen(szInFilePath, "rb");
    if (fpIn == NULL) {
        cout << "Unable to open input file: " << szInFilePath << endl;
        return;
    }
    FILE *fpOut = fopen(szOutFilePath, "wb");
    if (!fpOut) {
        cout << "Unable to open output file: " << szOutFilePath << endl;
        return;
    }

    NvEncLite enc(cuContext, nWidth, nHeight, eFormat, pInitParam);
    if (!enc.ReadyForEncode()) {
        cout << "NvEncLite fails to initialize." << endl;
        return;
    }

    // Prepare a QP delta map
    const int nQpDelta = 20;
    const int nMbSize = pInitParam->IsCodecH264() ? 16 : 32;
    int cxMap = (nWidth + nMbSize - 1) / nMbSize, cyMap = (nHeight + nMbSize - 1) / nMbSize;
    int8_t *qpDeltaMap = new int8_t[cxMap * cyMap];
    for (int y = 0; y < cyMap; y++) {
        for (int x = 0; x < cxMap; x++) {
            qpDeltaMap[y * cxMap + x] = (x - cxMap / 2) * (y - cyMap / 2) > 0 ? nQpDelta : -nQpDelta;
        }
    }

    // Params for one frame
    NV_ENC_PIC_PARAMS picParams = {NV_ENC_PIC_PARAMS_VER};
    picParams.qpDeltaMap = qpDeltaMap;
    picParams.qpDeltaMapSize = cxMap * cyMap;

    int nRead = 0;
    int nFrameSize = enc.GetFrameSize();
    uint8_t *pHostFrame = new uint8_t[nFrameSize];
    CUdeviceptr pDeviceFrame;
    ck(cuMemAlloc(&pDeviceFrame, nFrameSize));
    int nFrame = 0, i = 0;
    do {
        vector<vector<uint8_t>> vPacket;
        nRead = (int)fread(pHostFrame, 1, nFrameSize, fpIn);
        if (nRead == nFrameSize) {
            ck(cuMemcpyHtoD(pDeviceFrame, pHostFrame, nFrameSize));
            if (i && i % 10 == 0) {
            // i == 10, 20, 30, 40 ...
                NV_ENC_RECONFIGURE_PARAMS reconfigureParams = {NV_ENC_RECONFIGURE_PARAMS_VER};
                NV_ENC_CONFIG encodeConfig;
                reconfigureParams.reInitEncodeParams.encodeConfig = &encodeConfig;
                enc.GetInitializeParams(&reconfigureParams.reInitEncodeParams);
                if (i % 20 != 0) {
                // i == 10, 30, ...
                    // Set RC mode to CBR and get frames of even sizes
                    encodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
                    encodeConfig.rcParams.averageBitRate = 5000000;
                    encodeConfig.rcParams.vbvBufferSize = encodeConfig.rcParams.averageBitRate * 
                        reconfigureParams.reInitEncodeParams.frameRateDen / reconfigureParams.reInitEncodeParams.frameRateNum;
                    encodeConfig.rcParams.vbvInitialDelay = encodeConfig.rcParams.vbvBufferSize;
                } // else i = 20, 40, ... Restore the original param setting 
                enc.Reconfigure(&reconfigureParams);
            }
            picParams.encodePicFlags = (i % 5 == 0) ? (NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS) : 0;
            enc.EncodeDeviceFrame((uint8_t *)pDeviceFrame, 0, vPacket, NULL, &picParams);
        } else {
            // EndEncode() can also be called explicitly
            enc.EndEncode(vPacket);
        }
        nFrame += (int)vPacket.size();
        for (vector<uint8_t> &packet : vPacket) {
            if (bVerbose) cout << packet.size() << "\t\r";
            fwrite(packet.data(), 1, packet.size(), fpOut);
        }
        i++;
    } while (nRead == nFrameSize);
    if (bVerbose) cout << endl;
    ck(cuMemFree(pDeviceFrame));
    delete[] pHostFrame;
    delete[] qpDeltaMap;
    fclose(fpOut);
    fclose(fpIn);

    cout << "Total frames encoded: " << nFrame << endl << "Saved in file " << szOutFilePath << endl;
}

void EncodeDeviceFrameWithBufferedFile(CUcontext cuContext, char *szInFilePath, int nWidth, int nHeight, NV_ENC_BUFFER_FORMAT eFormat,
    char *szOutFilePath, NvEncoderInitParam *pInitParam, int nFrameWanted, bool bVerbose)
{
    FILE *fpOut = fopen(szOutFilePath, "wb");
    if (fpOut == NULL) {
        cout << "Unable to open file: " << szOutFilePath << endl;
        return;
    }

    uint8_t *pBuf = NULL;
    size_t nBufSize = 0;
    BufferedFileReader bufferedFileReader(szInFilePath);
    if (!bufferedFileReader.GetBuffer(&pBuf, &nBufSize)) {
        cout << "Failed to load file into memory (maybe file size is too big)" << endl;
        return;
    }

    NvEncLite enc(cuContext, nWidth, nHeight, eFormat, pInitParam);
    int nFrameSize = enc.GetFrameSize();
    int n = (int)(nBufSize / nFrameSize);
    if (!nFrameWanted) {
        nFrameWanted = n;
    }
    int nFrame = 0;
    CUdeviceptr pDeviceFrame;
    ck(cuMemAlloc(&pDeviceFrame, nFrameSize));
    for (int i = 0; i <= nFrameWanted; i++) {
        vector<vector<uint8_t>> vPacket;
        int iFrame = i / n % 2 ? (n - i % n - 1) : i % n;
        ck(cuMemcpyHtoD(pDeviceFrame, pBuf + iFrame * nFrameSize, nFrameSize));
        enc.EncodeDeviceFrame(i == nFrameWanted ? NULL : (uint8_t *)pDeviceFrame, 0, vPacket);
        nFrame += (int)vPacket.size();
        for (vector<uint8_t> &packet : vPacket) {
            if (bVerbose) cout << packet.size() << "\t\r";
            fwrite(packet.data(), 1, packet.size(), fpOut);
        }
    }
    if (bVerbose) cout << endl;
    ck(cuMemFree(pDeviceFrame));
    fclose(fpOut);

    cout << "Total frames encoded: " << nFrame << endl << "Saved in file " << szOutFilePath << endl;
}

int GetCapabilityValue(NV_ENCODE_API_FUNCTION_LIST nvenc, void *hEncoder, GUID guidCodec, NV_ENC_CAPS capsToQuery) {
    NV_ENC_CAPS_PARAM capsParam = {NV_ENC_CAPS_PARAM_VER};
    capsParam.capsToQuery = capsToQuery;
    int v;
    ck(nvenc.nvEncGetEncodeCaps(hEncoder, guidCodec, &capsParam, &v));
    return v;
}

void ShowEncoderCapability() {
    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    printf("Encoder Capability\n");
    printf("#  %-20.20s H264 H264_444 H264_ME H264_WxH  HEVC HEVC_Main10 HEVC_Lossless HEVC_SAO HEVC_444 HEVC_ME HEVC_WxH\n", "GPU");
    NV_ENCODE_API_FUNCTION_LIST nvenc = NvEncLiteUnbuffered::GetNvEncApi();
    for (int iGpu = 0; iGpu < nGpu; iGpu++) {
        CUdevice cuDevice = 0;
        ck(cuDeviceGet(&cuDevice, iGpu));
        char szDeviceName[80];
        ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
        CUcontext cuContext = NULL;
        ck(cuCtxCreate(&cuContext, 0, cuDevice));

        NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS encodeSessionExParams = { NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER };
        encodeSessionExParams.device = cuContext;
        encodeSessionExParams.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
        encodeSessionExParams.apiVersion = NVENCAPI_VERSION;
        void *hEncoder = NULL;
        if (!nvenc.nvEncOpenEncodeSessionEx || !ck(nvenc.nvEncOpenEncodeSessionEx(&encodeSessionExParams, &hEncoder))) {
        //Original    #  %-20.20s H264 H264_444 H264_ME H264_WxH  HEVC HEVC_Main10 HEVC_Lossless HEVC_SAO HEVC_444 HEVC_ME HEVC_WxH
            printf("%-2d %-20.20s   -      -       -        -       -       -            -           -        -       -        -\n", iGpu, szDeviceName);
            continue;
        }
        //Adjusted # %-20.20s H264  H264_444  H264_ME  H264_WxH HEVC  HEVC_Main10  HEVC_Lossless  HEVC_SAO  HEVC_444  HEVC_ME  HEVC_WxH
        printf("%-2d %-20.20s   %s      %s       %s    %4dx%-4d   %s       %s            %s           %s        %s       %s    %4dx%-4d\n", 
            iGpu, szDeviceName,
            GetCapabilityValue(nvenc, hEncoder, NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES) ? "+" : "-",
            GetCapabilityValue(nvenc, hEncoder, NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_SUPPORT_YUV444_ENCODE)       ? "+" : "-",
            GetCapabilityValue(nvenc, hEncoder, NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_SUPPORT_MEONLY_MODE)         ? "+" : "-",
            GetCapabilityValue(nvenc, hEncoder, NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_WIDTH_MAX)                   ,
            GetCapabilityValue(nvenc, hEncoder, NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_HEIGHT_MAX)                  ,
            GetCapabilityValue(nvenc, hEncoder, NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES) ? "+" : "-",
            GetCapabilityValue(nvenc, hEncoder, NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_SUPPORT_10BIT_ENCODE)        ? "+" : "-",
            GetCapabilityValue(nvenc, hEncoder, NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE)     ? "+" : "-",
            GetCapabilityValue(nvenc, hEncoder, NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_SUPPORT_SAO)                 ? "+" : "-",
            GetCapabilityValue(nvenc, hEncoder, NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_SUPPORT_YUV444_ENCODE)       ? "+" : "-",
            GetCapabilityValue(nvenc, hEncoder, NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_SUPPORT_MEONLY_MODE)         ? "+" : "-",
            GetCapabilityValue(nvenc, hEncoder, NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_WIDTH_MAX)                   ,
            GetCapabilityValue(nvenc, hEncoder, NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_HEIGHT_MAX)                 
        );

        ck(cuCtxDestroy(cuContext));
    }
}

void ShowHelpAndExit(const char *szBadOption = NULL) {
    if (szBadOption) {
        cout << "Error parsing \"" << szBadOption << "\"" << endl;
    }
    cout << "Options:" << endl
        << "-i           Input file path" << endl
        << "-o           Output file path" << endl
        << "-s           Input resolution in this form: WxH" << endl
        << "-if          Input format: iyuv nv12 yuv444 p010 yuv444p16 bgra" << endl
        << "-gpu         Ordinal of GPU to use" << endl
        << "-v           Verbose message" << endl
        << "-case        0: Encode frames from host memory" << endl
        << "             1: Encode frames from device memory and set params for each frame." << endl 
        << "                In this case, I-frame appears periodically by per-frame params," << endl 
        << "                and quadrants have different encoding quality by a QP delta map" << endl 
        << "             2: Encode frames from device memory with a buffer" << endl 
        << "-frame       (Only for -case 2) Number of frames to encode" << endl
        ;
    cout << NvEncoderInitParam().GetHelpMessage() << endl;
    ShowEncoderCapability();
    exit(1);
}

void ParseCommandLine(int argc, char *argv[], char *szInputFileName, int &nWidth, int &nHeight, 
    NV_ENC_BUFFER_FORMAT &eFormat, char *szOutputFileName, NvEncoderInitParam &initParam, 
    int &iGpu, bool &bVerbose, int &iCase, int &nFrame) 
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
        if (!_stricmp(argv[i], "-v")) {
            bVerbose = true;
            continue;
        }
        if (!_stricmp(argv[i], "-case")) {
            if (++i == argc) {
                ShowHelpAndExit("-case");
            }
            iCase = atoi(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-frame")) {
            if (++i == argc) {
                ShowHelpAndExit("-frame");
            }
            nFrame = atoi(argv[i]);
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
    char szInFilePath[256] = "bunny.iyuv",
        szOutFilePath[256] = "";
    int nWidth = 1920, nHeight = 1080;
    NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_IYUV;
    int iGpu = 0;
    bool bVerbose = false;
    int iCase = 0;
    int nFrame = 0;
    NvEncoderInitParam initParam;
    CheckDefaultFileExists(szInFilePath);
    ParseCommandLine(argc, argv, szInFilePath, nWidth, nHeight, eFormat, szOutFilePath, initParam, iGpu, bVerbose, iCase, nFrame);
    if (!*szOutFilePath) {
        sprintf(szOutFilePath, initParam.IsCodecH264() ? "out.h264" : "out.hevc");
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

    switch (iCase) {
    default:
    case 0:
        cout << "Encode host frames" << endl;
        EncodeHostFrame(cuContext, szInFilePath, nWidth, nHeight, eFormat, szOutFilePath, &initParam, bVerbose);
        break;
    case 1:
        cout << "Encode device frames with parameters for each picture" << endl;
        EncodeDeviceFrameWithPicParam(cuContext, szInFilePath, nWidth, nHeight, eFormat, szOutFilePath, &initParam, bVerbose);
        break;
    case 2:
        cout << "Encode device frames with a buffer, nFrame=" << (nFrame ? to_string(nFrame) : "(default)") << endl;
        EncodeDeviceFrameWithBufferedFile(cuContext, szInFilePath, nWidth, nHeight, eFormat, szOutFilePath, &initParam, nFrame, bVerbose);
        break;
    }
    return 0;
}
