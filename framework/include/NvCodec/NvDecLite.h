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

#include "nvcuvid.h"
#include <stdint.h>

#define INVALID_TIMESTAMP ((int64_t)0x8000000000000000ll)

typedef struct NvFrameInfo {
    int nWidth, nHeight, nFramePitch, nBitDepth, nFrameSize;
    CUVIDPICPARAMS picParams;
    CUVIDPARSERDISPINFO dispInfo;
} NvFrameInfo;

#include <vector>
#include <map>
#include <mutex>
#include <assert.h>
#include <vector_types.h>
#include <algorithm>

class NvDecLite {
public:
    struct Rect {
        Rect(int l = 0, int t = 0, int r = 0, int b = 0) : l(l), t(t), r(r), b(b) {}
        int l, t, r, b;
    };

    struct Dim {
        Dim(int w = 0, int h = 0) : w(w), h(h) {}
        int w, h;
    };

public:
    NvDecLite(CUcontext cuContext, bool bUseDeviceFrame, cudaVideoCodec eCodec, 
        bool bLowLatency = false, bool bDeviceFramePitched = false, const Rect *pCropRect = NULL, const Dim *pResizeDim = NULL);
    virtual ~NvDecLite();
    CUcontext GetContext() {return cuContext;}
    int GetWidth() {assert(m_nWidth); return m_nWidth;}
    int GetHeight() {assert(m_nHeight); return m_nHeight;}
    int GetFramePitch() {assert(m_nWidth); return m_nDeviceFramePitch ? (int)m_nDeviceFramePitch : m_nWidth * (m_nBitDepthMinus8? 2 : 1);}
    int GetBitDepth() {assert(m_nWidth); return m_nBitDepthMinus8 + 8;}
    int GetFrameSize() {assert(m_nWidth); return m_nWidth * m_nHeight * 3 / (m_nBitDepthMinus8 ? 1 : 2);}
    CUVIDEOFORMAT GetVideoFormatInfo() {assert(m_nWidth); return m_videoFormat;}
    virtual int Decode(const uint8_t *pData, int nSize, uint8_t ***pppFrame, NvFrameInfo **ppFrameInfo, 
        uint32_t flags = 0, int64_t timestamp = 0, CUstream stream = 0);
    int DecodeLockFrame(const uint8_t *pData, int nSize, uint8_t ***pppFrame, NvFrameInfo **ppFrameInfo, 
        uint32_t flags = 0, int64_t timestamp = 0, CUstream stream = 0);
    void UnlockFrame(uint8_t **ppFrame, int nFrame);

private:
    static int CUDAAPI HandleVideoSequenceProc(void *pUserData, CUVIDEOFORMAT *pVideoFormat) {return ((NvDecLite *)pUserData)->HandleVideoSequence(pVideoFormat);}
    static int CUDAAPI HandlePictureDecodeProc(void *pUserData, CUVIDPICPARAMS *pPicParams) {return ((NvDecLite *)pUserData)->HandlePictureDecode(pPicParams);}
    static int CUDAAPI HandlePictureDisplayProc(void *pUserData, CUVIDPARSERDISPINFO *pDispInfo) {return ((NvDecLite *)pUserData)->HandlePictureDisplay(pDispInfo);}
    static int CUDAAPI HandleOperatingPointProc(void *pUserData, CUVIDOPERATINGPOINTINFO *pOPInfo) {return ((NvDecLite *)pUserData)->HandleGetOperatingPoint(pOPInfo); }
    int HandleVideoSequence(CUVIDEOFORMAT *pVideoFormat);
    int HandlePictureDecode(CUVIDPICPARAMS *pPicParams);
    int HandlePictureDisplay(CUVIDPARSERDISPINFO *pDispInfo);
    int HandleGetOperatingPoint(CUVIDOPERATINGPOINTINFO *pOPInfo);
    bool DoDecode(const uint8_t *pData, int nSize, uint32_t flags, int64_t timestamp, CUstream stream);
    bool DoCacheDecode(uint8_t **ppFrame, CUstream stream);
    void ReleaseFrame(uint8_t *pFrame);

protected:
    CUcontext cuContext = NULL;

private:
    CUvideoctxlock m_ctxLock;
    CUvideoparser m_hParser = NULL;
    CUvideodecoder m_hDecoder = NULL;
    bool m_bUseDeviceFrame;
    // dimension of the output
    int m_nWidth = 0, m_nHeight = 0;
    cudaVideoCodec m_eCodec = cudaVideoCodec_NumCodecs;
    cudaVideoChromaFormat m_eChromaFormat = cudaVideoChromaFormat_420;
    int m_nBitDepthMinus8 = 0;
    unsigned int m_nOperatingPoint = 0;
    bool  m_bDispAllLayers = false;
    CUVIDEOFORMAT m_videoFormat = {};
    // stock of frames
    std::vector<uint8_t *> m_vpFrame;
    // decoded frames for return
    std::vector<uint8_t *> m_vpFrameRet;
    // frame info of decoded frames
    std::vector<NvFrameInfo> m_vFrameInfo;
    // round queue of CUVIDPICPARAMS of decoded frames
    std::vector<CUVIDPICPARAMS> m_vPicParam;
    std::map<uint8_t *, int2> m_pFrame2dim;
    bool m_bReleaseOldFrames = false;
    // state of m_vpFrame
    int m_nDecodedFrame = 0, m_nDecodedFrameReturned = 0;
    // frames may be unlocked asynchronously, so should be protected
    std::mutex m_mtx_vpFrame;
    CUstream m_cuvidStream = 0;
    bool m_bDeviceFramePitched = false;
    size_t m_nDeviceFramePitch = 0;
    Rect m_cropRect = {};
    Dim m_resizeDim = {};
};

#if defined AVCODEC_AVCODEC_H
inline cudaVideoCodec FFmpeg2NvCodecId(AVCodecID id) {
    switch (id) {
    case AV_CODEC_ID_MPEG1VIDEO : return cudaVideoCodec_MPEG1;
    case AV_CODEC_ID_MPEG2VIDEO : return cudaVideoCodec_MPEG2;
    case AV_CODEC_ID_MPEG4      : return cudaVideoCodec_MPEG4;
    case AV_CODEC_ID_VC1        : return cudaVideoCodec_VC1;
    case AV_CODEC_ID_H264       : return cudaVideoCodec_H264;
    case AV_CODEC_ID_HEVC       : return cudaVideoCodec_HEVC;
    case AV_CODEC_ID_VP8        : return cudaVideoCodec_VP8;
    case AV_CODEC_ID_VP9        : return cudaVideoCodec_VP9;
    case AV_CODEC_ID_MJPEG      : return cudaVideoCodec_JPEG;
    default                     : return cudaVideoCodec_NumCodecs;
    }
}
#endif