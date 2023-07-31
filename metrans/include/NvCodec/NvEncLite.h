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
#include <stdint.h>
#include <mutex>
#include <cuda.h>
#include "NvEncLiteUnbuffered.h"

extern simplelogger::Logger *logger;

class NvEncLite : public NvEncLiteUnbuffered {
public:
    NvEncLite(CUcontext cuContext, int nWidth, int nHeight, NV_ENC_BUFFER_FORMAT eBufferFormat, NvEncoderInitParam *pInitParam = NULL, int nExtraOutputDelay = 0, bool stillImage = false);
    virtual ~NvEncLite();
    bool EncodeDeviceFrame(uint8_t *pDeviceFrame, int nFramePitch, std::vector<std::vector<uint8_t>> &vPacket, std::vector<NvPacketInfo> *pvPacketInfo = NULL, NV_ENC_PIC_PARAMS *pPicParams = NULL);
    bool EncodeHostFrame(uint8_t *pHostFrame, int nFramePitch, std::vector<std::vector<uint8_t>> &vPacket, std::vector<NvPacketInfo> *pvPacketInfo = NULL, NV_ENC_PIC_PARAMS *pPicParams = NULL);
 
protected:
    bool EncodeFrame(uint8_t *pFrame, bool bDeviceFrame, int nFramePitch, std::vector<std::vector<uint8_t>> &vPacket, std::vector<NvPacketInfo> *pvPacketInfo, NV_ENC_PIC_PARAMS *pPicParams);
    void CopyFrame(uint8_t *pSrcFrame, bool bDeviceFrame, int nSrcPitch, CUdeviceptr pDstFrame, int nDstPitch);
    int GetDeviceFrameBufferPitch() {
        return (GetPlaneWidthInBytes() + 15) / 16 * 16;
    }
    int GetDeviceFrameBufferSize() {
        int w = GetDeviceFrameBufferPitch();
        int h = 0, hh = 0;
        GetPlaneHeights(h, hh);
        return w * h + w / 2 * hh;
    }

protected:
    std::vector<CUdeviceptr> vpDeviceFrame;
    std::vector<NV_ENC_REGISTERED_PTR> vRegisteredResource;
    std::vector<NV_ENC_INPUT_PTR> vDeviceInputBuffer;
};
