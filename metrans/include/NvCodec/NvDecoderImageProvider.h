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

#include <deque>
#include "AvToolkit/Demuxer.h"
#include "NvDecLite.h"
#include "NvCodec/NvCommon.h"

extern simplelogger::Logger *logger;

class NvDecLiteImageProvider {
public:
    NvDecLiteImageProvider(CUcontext cuContext, Demuxer *pDemuxer) : 
        cuContext(cuContext), pDemuxer(pDemuxer) 
    {
        pDec = new NvDecLite(cuContext, true, FFmpeg2NvCodecId(pDemuxer->GetVideoStream()->codecpar->codec_id));
        bReady = DecodeNextFrame();
        ck(cuCtxPushCurrent(cuContext));
        ck(cuMemAlloc(&dpImage, pDec->GetWidth() * pDec->GetHeight() * 8));
        ck(cuCtxPopCurrent(NULL));
    }
    ~NvDecLiteImageProvider() {
        if (dpImage) {
            ck(cuCtxPushCurrent(cuContext));
            ck(cuMemFree(dpImage));
            ck(cuCtxPopCurrent(NULL));
        }
        if (pDec) {
            delete pDec;
        }
    }

    bool IsReady() {
        return bReady;
    }
    int GetFrameSize() {
        if (!bReady) {
            return 0;
        }
        return pDec->GetFrameSize();
    }
    int GetWidth() {
        if (!bReady) {
            return 0;
        }
        return pDec->GetWidth();
    }
    int GetHeight() {
        if (!bReady) {
            return 0;
        }
        return pDec->GetHeight();
    }

    bool GetNextFrame(uint8_t *pDst, int nDstPitch = 0, bool bDevice = false) {
        if (!bReady) {
            return false;
        }
        uint8_t *dpFrame = vdpFrame.front();
        vdpFrame.pop_front();
        ck(cuCtxPushCurrent(cuContext));
        CopyImage((CUdeviceptr)dpFrame, pDec->GetWidth() * (pDec->GetBitDepth() == 8 ? 1 : 2), pDec->GetHeight() * 3 / 2, pDst, nDstPitch, bDevice);
        ck(cuCtxPopCurrent(NULL));
        pDec->UnlockFrame(&dpFrame, 1);
        if (vdpFrame.empty()) {
            bReady = DecodeNextFrame();
        }
        return true;
    }
    bool GetNextImageAsBgra(uint8_t *pDst, int nDstPitch = 0, bool bDevice = false, cudaStream_t stream = 0) {
        return GetNextImageAsFormat(pDst, nDstPitch, bDevice, pDec->GetWidth() * 4, pDec->GetHeight(), Nv12ToBgra32, P016ToBgra32, pDec->GetVideoFormatInfo().video_signal_description.matrix_coefficients, stream);
    }
    bool GetNextImageAsBgrPlanar(uint8_t *pDst, int nDstPitch = 0, bool bDevice = false, cudaStream_t stream = 0) {
        return GetNextImageAsFormat(pDst, nDstPitch, bDevice, pDec->GetWidth(), pDec->GetHeight() * 3, Nv12ToBgrPlanar, P016ToBgrPlanar, pDec->GetVideoFormatInfo().video_signal_description.matrix_coefficients, stream);
    }
    bool GetNextImageAsBgra64(uint8_t *pDst, int nDstPitch = 0, bool bDevice = false, cudaStream_t stream = 0) {
        return GetNextImageAsFormat(pDst, nDstPitch, bDevice, pDec->GetWidth() * 8, pDec->GetHeight(), Nv12ToBgra64, P016ToBgra64, pDec->GetVideoFormatInfo().video_signal_description.matrix_coefficients, stream);
    }

private:
    bool DecodeNextFrame() {
        AVPacket *pkt = NULL;
        do {
            pDemuxer->Demux(&pkt);
            uint8_t **ppFrame = NULL;
            NvFrameInfo *pInfo = NULL;
            int nFrameDecoded = pDec->DecodeLockFrame(pkt->data, pkt->size, &ppFrame, &pInfo);
            vdpFrame.insert(vdpFrame.end(), ppFrame, ppFrame + nFrameDecoded);
            vFrameInfo.insert(vFrameInfo.end(), pInfo, pInfo + nFrameDecoded);
        } while (vdpFrame.empty() && pkt->size);
        return !vdpFrame.empty();
    }
    void CopyImage(CUdeviceptr dpSrc, int nWidthInBytes, int nHeight, uint8_t *pDst, int nDstPitch, bool bDevice) {
        CUDA_MEMCPY2D m = { 0 };
        m.WidthInBytes = nWidthInBytes;
        m.Height = nHeight;
        m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        m.srcDevice = dpSrc;
        m.srcPitch = m.WidthInBytes;
        m.dstMemoryType = bDevice ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST;
        m.dstDevice = (CUdeviceptr)(m.dstHost = pDst);
        m.dstPitch = nDstPitch ? nDstPitch : m.WidthInBytes;
        ck(cuMemcpy2D(&m));
    }
    bool GetNextImageAsFormat(uint8_t *pDst, int nDstPitch, bool bDevice, int nWidthInBytes, int nHeight, ConvertFormatFuncType Nv12ToFormat, ConvertFormatFuncType P016ToFormat, int iMatrix, cudaStream_t stream) {
        if (!bReady) {
            return false;
        }
        uint8_t *dpFrame = vdpFrame.front();
        vdpFrame.pop_front();
        uint8_t *dpDst = bDevice ? pDst : (uint8_t *)dpImage;
        int nDeviceDstPitch = (bDevice && nDstPitch) ? nDstPitch : nWidthInBytes;
        ck(cuCtxPushCurrent(cuContext));
        if (pDec->GetBitDepth() == 8) {
            Nv12ToFormat(dpFrame, pDec->GetWidth(), dpDst, nDeviceDstPitch, pDec->GetWidth(), pDec->GetHeight(), iMatrix, stream);
        } else {
            P016ToFormat(dpFrame, pDec->GetWidth() * 2, dpDst, nDeviceDstPitch, pDec->GetWidth(), pDec->GetHeight(), iMatrix, stream);
        }
        if (!bDevice) {
            CopyImage(dpImage, nWidthInBytes, nHeight, pDst, nDstPitch, false);
        }
        ck(cuCtxPopCurrent(NULL));
        pDec->UnlockFrame(&dpFrame, 1);
        if (vdpFrame.empty()) {
            bReady = DecodeNextFrame();
        }
        return true;
    }

    Demuxer *pDemuxer = NULL;
    CUcontext cuContext = NULL;
    NvDecLite *pDec = NULL;
    bool bReady = false;
    CUdeviceptr dpImage = 0;
    std::deque<uint8_t *> vdpFrame;
    std::vector<NvFrameInfo> vFrameInfo;
};
