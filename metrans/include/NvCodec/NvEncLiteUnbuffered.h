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

#pragma once

#include <vector>
#include <list>
#include "nvEncodeAPI.h"
#include <stdint.h>
#include <mutex>
#include "NvEncoderParam.h"
#include "Logger.h"

extern simplelogger::Logger *logger;

struct NvPacketInfo {
    NV_ENC_LOCK_BITSTREAM info;
    int64_t dts;
};

class NvEncLiteUnbuffered {
public:
    NvEncLiteUnbuffered(NV_ENC_DEVICE_TYPE eDeviceType, void *pDevice, int nWidth, int nHeight, NV_ENC_BUFFER_FORMAT eBufferFormat, NvEncoderInitParam *pInitParam = NULL) 
        : NvEncLiteUnbuffered(eDeviceType, pDevice, nWidth, nHeight, eBufferFormat, pInitParam, 0) {}
    virtual ~NvEncLiteUnbuffered();
    bool ReadyForEncode() {return hEncoder != NULL;}
    bool GetSequenceParams(uint8_t **ppSequenceParams, int *pnSize);
    bool EncodeDeviceFrameUnbuffered(void *pDeviceFrame, int nFramePitch, std::vector<std::vector<uint8_t>> &vPacket, std::vector<NvPacketInfo> *pvPacketInfo = NULL, NV_ENC_PIC_PARAMS *pPicParams = NULL);
    bool EndEncode(std::vector<std::vector<uint8_t>> &vPacket, std::vector<NvPacketInfo> *pvPacketInfo = NULL);
    bool Reconfigure(NV_ENC_RECONFIGURE_PARAMS *pReconfigureParams);
    void *GetDevice() {return pDevice;}
    int GetWidth() {return nWidth;}
    int GetHeight() {return nHeight;}
    int GetFrameSize() {
        int w = GetPlaneWidthInBytes();
        int h = 0, hh = 0;
        GetPlaneHeights(h, hh);
        return w * h + w / 2 * hh;
    }
    void GetInitializeParams(NV_ENC_INITIALIZE_PARAMS *pInitializeParams) {
        if (!pInitializeParams || !pInitializeParams->encodeConfig) {
            LOG(ERROR) << "Both pInitializeParams and pInitializeParams->encodeConfig can't be NULL";
            return;
        }
        NV_ENC_CONFIG *pEncodeConfig = pInitializeParams->encodeConfig;
        *pEncodeConfig = encodeConfig;
        *pInitializeParams = initializeParams;
        pInitializeParams->encodeConfig = pEncodeConfig;
    }
    static NV_ENCODE_API_FUNCTION_LIST GetNvEncApi() {return nvenc;}

protected:
    NvEncLiteUnbuffered(NV_ENC_DEVICE_TYPE eDeviceType, void *pDevice, int nWidth, int nHeight, NV_ENC_BUFFER_FORMAT eBufferFormat, 
        NvEncoderInitParam *pInitParam, int nExtraOutputDelay, bool stillImage=false);
    bool DoEncode(NV_ENC_INPUT_PTR inputBuffer, std::vector<std::vector<uint8_t>> &vPacket, std::vector<NvPacketInfo> *pvPacketInfo, NV_ENC_PIC_PARAMS *pPicParams);
    void GetEncodedPacket(std::vector<NV_ENC_OUTPUT_PTR> &vOutputBuffer, std::vector<std::vector<uint8_t>> &vPacket, std::vector<NvPacketInfo> *pvPacketInfo, bool bOutputDelay);
    int GetPlaneWidthInBytes() {
        switch (eBufferFormat) {
        case NV_ENC_BUFFER_FORMAT_NV12:
        case NV_ENC_BUFFER_FORMAT_YV12:
        case NV_ENC_BUFFER_FORMAT_IYUV:
        case NV_ENC_BUFFER_FORMAT_YUV444:
            return nWidth;
        case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
        case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
            return nWidth * 2;
        case NV_ENC_BUFFER_FORMAT_ARGB:
        case NV_ENC_BUFFER_FORMAT_ARGB10:
        case NV_ENC_BUFFER_FORMAT_AYUV:
        case NV_ENC_BUFFER_FORMAT_ABGR:
        case NV_ENC_BUFFER_FORMAT_ABGR10:
            return nWidth * 4;
        default:
            LOG(ERROR) << "Unknown format: " << eBufferFormat << " in " << __FUNCTION__;
            return 0;
        }
    }
    void GetPlaneHeights(int &h, int &hh) {
        switch (eBufferFormat) {
        case NV_ENC_BUFFER_FORMAT_NV12:
        case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
            h = nHeight * 3 / 2;
            return;
        case NV_ENC_BUFFER_FORMAT_YV12:
        case NV_ENC_BUFFER_FORMAT_IYUV:
            h = nHeight;
            hh = nHeight;
            return;
        case NV_ENC_BUFFER_FORMAT_YUV444:
        case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
            h = nHeight * 3;
            return;
        case NV_ENC_BUFFER_FORMAT_ARGB:
        case NV_ENC_BUFFER_FORMAT_ARGB10:
        case NV_ENC_BUFFER_FORMAT_AYUV:
        case NV_ENC_BUFFER_FORMAT_ABGR:
        case NV_ENC_BUFFER_FORMAT_ABGR10:
            h = nHeight;
            return;
        default:
            LOG(ERROR) << "Unknown format: " << eBufferFormat << " in " << __FUNCTION__;
            return;
        }
    }
    static NV_ENCODE_API_FUNCTION_LIST LoadNvEncApi();

protected:
    void *pDevice;
    NV_ENC_DEVICE_TYPE eDeviceType;
    int nWidth, nHeight;
    NV_ENC_BUFFER_FORMAT eBufferFormat;
    uint8_t aSequenceParams[256];
    int nOutputDelay = 0;
    
    void *hEncoder = NULL;
    int nEncoderBuffer = 0;
    std::vector<NV_ENC_OUTPUT_PTR> vBitstreamOutputBuffer;
    std::vector<void *> vpCompletionEvent;
    int iToSend = 0, iGot = 0;

    static NV_ENCODE_API_FUNCTION_LIST nvenc;

private:
    NV_ENC_INITIALIZE_PARAMS initializeParams = {};
    NV_ENC_CONFIG encodeConfig = {};
    std::list<int64_t> lDts;
    int64_t dtsOffset = 0;
};

#if defined AVCODEC_AVCODEC_H
inline AVCodecParameters *ExtractAVCodecParameters(NvEncLiteUnbuffered *pEnc) {
    AVCodecParameters *par = avcodec_parameters_alloc();

    uint8_t *spspps = NULL;
    int nSize = 0;
    pEnc->GetSequenceParams(&spspps, &nSize);
    par->extradata = (uint8_t *)av_malloc(nSize + AV_INPUT_BUFFER_PADDING_SIZE);
    par->extradata_size = nSize;
    memcpy(par->extradata, spspps, nSize);

    NV_ENC_INITIALIZE_PARAMS param;
    NV_ENC_CONFIG encodeConfig;
    param.encodeConfig = &encodeConfig;
    pEnc->GetInitializeParams(&param);

    par->codec_type = AVMEDIA_TYPE_VIDEO;
    par->codec_id = param.encodeGUID == NV_ENC_CODEC_H264_GUID ? AV_CODEC_ID_H264 : AV_CODEC_ID_HEVC;
    par->bit_rate = param.encodeConfig->rcParams.averageBitRate;
    par->bits_per_coded_sample = 24;
    par->bits_per_raw_sample = 8;
    par->profile = 100;
    par->level = 40;
    par->width = param.encodeWidth;
    par->height = param.encodeHeight;
    par->chroma_location = AVCHROMA_LOC_LEFT;
    par->video_delay = 2;

    return par;
}
#endif