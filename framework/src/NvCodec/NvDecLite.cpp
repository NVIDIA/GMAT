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

#include "NvCodec/NvDecLite.h"
#include "NvCodec/NvCommon.h"

using namespace std;

static const char * GetVideoCodecString(cudaVideoCodec eCodec) {
    static struct {
        cudaVideoCodec eCodec;
        const char *name;
    } aCodecName [] = {
        { cudaVideoCodec_MPEG1,     "MPEG-1"       },
        { cudaVideoCodec_MPEG2,     "MPEG-2"       },
        { cudaVideoCodec_MPEG4,     "MPEG-4 (ASP)" },
        { cudaVideoCodec_VC1,       "VC-1/WMV"     },
        { cudaVideoCodec_H264,      "AVC/H.264"    },
        { cudaVideoCodec_JPEG,      "M-JPEG"       },
        { cudaVideoCodec_H264_SVC,  "H.264/SVC"    },
        { cudaVideoCodec_H264_MVC,  "H.264/MVC"    },
        { cudaVideoCodec_HEVC,      "H.265/HEVC"   },
        { cudaVideoCodec_VP8,       "VP8"          },
        { cudaVideoCodec_VP9,       "VP9"          },
        { cudaVideoCodec_NumCodecs, "Invalid"      },
        { cudaVideoCodec_YUV420,    "YUV  4:2:0"   },
        { cudaVideoCodec_YV12,      "YV12 4:2:0"   },
        { cudaVideoCodec_NV12,      "NV12 4:2:0"   },
        { cudaVideoCodec_YUYV,      "YUYV 4:2:2"   },
        { cudaVideoCodec_UYVY,      "UYVY 4:2:2"   },
    };

    if (eCodec >= 0 && eCodec <= cudaVideoCodec_NumCodecs) {
        return aCodecName[eCodec].name;
    }
    for (int i = cudaVideoCodec_NumCodecs + 1; i < sizeof(aCodecName) / sizeof(aCodecName[0]); i++) {
        if (eCodec == aCodecName[i].eCodec) {
            return aCodecName[eCodec].name;
        }
    }
    return "Unknown";
}

static const char * GetVideoChromaFormatString(cudaVideoChromaFormat eChromaFormat) {
    static struct {
        cudaVideoChromaFormat eChromaFormat;
        const char *name;
    } aChromaFormatName[] = {
        { cudaVideoChromaFormat_Monochrome, "YUV 400 (Monochrome)" },
        { cudaVideoChromaFormat_420,        "YUV 420"              },
        { cudaVideoChromaFormat_422,        "YUV 422"              },
        { cudaVideoChromaFormat_444,        "YUV 444"              },
    };

    if (eChromaFormat >= 0 && eChromaFormat < sizeof(aChromaFormatName) / sizeof(aChromaFormatName[0])) {
        return aChromaFormatName[eChromaFormat].name;
    }
    return "Unknown";
}

int NvDecLite::HandleVideoSequence(CUVIDEOFORMAT *pVideoFormat) {
    LOG(INFO) << "Video Input Information" << endl
        << "\tCodec        : " << GetVideoCodecString(pVideoFormat->codec) << endl
        << "\tFrame rate   : " << pVideoFormat->frame_rate.numerator << "/" << pVideoFormat->frame_rate.denominator 
            << " = " << 1.0 * pVideoFormat->frame_rate.numerator / pVideoFormat->frame_rate.denominator << " fps" << endl
        << "\tSequence     : " << (pVideoFormat->progressive_sequence ? "Progressive" : "Interlaced") << endl
        << "\tCoded size   : [" << pVideoFormat->coded_width << ", " << pVideoFormat->coded_height << "]" << endl
        << "\tDisplay area : [" << pVideoFormat->display_area.left << ", " << pVideoFormat->display_area.top << ", " 
            << pVideoFormat->display_area.right << ", " << pVideoFormat->display_area.bottom << "]" << endl
        << "\tChroma       : " << GetVideoChromaFormatString(pVideoFormat->chroma_format) << endl
        << "\tBit depth    : " << pVideoFormat->bit_depth_luma_minus8 + 8
    ;

    int nDecodeSurface = pVideoFormat->min_num_decode_surfaces;;

    if (m_nWidth && m_nHeight) {
    // cuvidCreateDecoder() has been called before, and now there's possible config change
        if (m_eCodec == cudaVideoCodec_VP9) {
        // For VP9, driver will handle the change
            return nDecodeSurface;
        }
        if (pVideoFormat->coded_width == m_videoFormat.coded_width && pVideoFormat->coded_height == m_videoFormat.coded_height) {
        // No resolution change
            return nDecodeSurface;
        }

        LOG(INFO) << "Dynamic resolution change detected";
        /* Old frames should be released at two timings:
           1) When a new-sized frame is fully decoded (HandlePictureDisplay()), unused old frames must be released (or we'll use wrong-sized frame); 
              old frames still in use haven't be retrieved, so they can't be released yet.
           2) When new data is sent in (DoDecode()), all old frames should be released, since they're guaranteed to have been retrieved*/
        m_bReleaseOldFrames = true;
        if (m_hDecoder) {
            ck(cuvidDestroyDecoder(m_hDecoder));
        }
    }

    // eCodec has been set in the constructor (for parser). Here it's set again for potential correction
    m_eCodec = pVideoFormat->codec;
    m_eChromaFormat = pVideoFormat->chroma_format;
    m_nBitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;
    m_videoFormat = *pVideoFormat;

    CUVIDDECODECREATEINFO videoDecodeCreateInfo = { 0 };
    videoDecodeCreateInfo.CodecType = pVideoFormat->codec;
    videoDecodeCreateInfo.ChromaFormat = pVideoFormat->chroma_format;
    videoDecodeCreateInfo.OutputFormat = pVideoFormat->bit_depth_luma_minus8 ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
    videoDecodeCreateInfo.bitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;
    videoDecodeCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    videoDecodeCreateInfo.ulNumOutputSurfaces = 2;
    // With PreferCUVID, JPEG is still decoded by CUDA while video is decoded by NVDEC hardware
    videoDecodeCreateInfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    videoDecodeCreateInfo.ulNumDecodeSurfaces = nDecodeSurface;
    videoDecodeCreateInfo.vidLock = m_ctxLock;
    videoDecodeCreateInfo.ulWidth = pVideoFormat->coded_width;
    videoDecodeCreateInfo.ulHeight = pVideoFormat->coded_height;

    m_nWidth = pVideoFormat->display_area.right - pVideoFormat->display_area.left;
    m_nHeight = pVideoFormat->display_area.bottom - pVideoFormat->display_area.top;
    videoDecodeCreateInfo.display_area.left = 0;
    videoDecodeCreateInfo.display_area.top = 0;
    videoDecodeCreateInfo.display_area.right = m_nWidth;
    videoDecodeCreateInfo.display_area.bottom = m_nHeight;

    if (m_cropRect.r && m_cropRect.b) {
        m_nWidth = m_cropRect.r - m_cropRect.l;
        m_nHeight = m_cropRect.b - m_cropRect.t;
        videoDecodeCreateInfo.display_area.left = m_cropRect.l;
        videoDecodeCreateInfo.display_area.top = m_cropRect.t;
        videoDecodeCreateInfo.display_area.right = m_cropRect.r;
        videoDecodeCreateInfo.display_area.bottom = m_cropRect.b;
    }
    if (m_resizeDim.w && m_resizeDim.h) {
        m_nWidth = m_resizeDim.w;
        m_nHeight = m_resizeDim.h;
    }

    videoDecodeCreateInfo.ulTargetWidth = m_nWidth;
    videoDecodeCreateInfo.ulTargetHeight = m_nHeight;

    LOG(INFO) << "Video Decoding Params:" << endl
        << "\tNum Surfaces : " << videoDecodeCreateInfo.ulNumDecodeSurfaces << endl
        << "\tCrop         : [" << videoDecodeCreateInfo.display_area.left << ", " << videoDecodeCreateInfo.display_area.top << ", "
        << videoDecodeCreateInfo.display_area.right << ", " << videoDecodeCreateInfo.display_area.bottom << "]" << endl
        << "\tResize       : " << videoDecodeCreateInfo.ulTargetWidth << "x" << videoDecodeCreateInfo.ulTargetHeight << endl
        << "\tDeinterlace  : " << std::vector<const char *>{"Weave", "Bob", "Adaptive"}[videoDecodeCreateInfo.DeinterlaceMode] 
    ;
    ck(cuCtxPushCurrent(cuContext));
    ck(cuvidCreateDecoder(&m_hDecoder, &videoDecodeCreateInfo));
    ck(cuCtxPopCurrent(NULL));

    m_vPicParam.resize(nDecodeSurface);
    return nDecodeSurface;
}

int NvDecLite::HandlePictureDecode(CUVIDPICPARAMS *pPicParams) {
    if (!m_hDecoder) {
        LOG(ERROR) << "Decoder not initialized.";
        return false;
    }

    m_vPicParam[pPicParams->CurrPicIdx] = *pPicParams;
    ck(cuvidDecodePicture(m_hDecoder, pPicParams));
    return 1;
}

int NvDecLite::HandlePictureDisplay(CUVIDPARSERDISPINFO *pDispInfo) {
    CUVIDPROCPARAMS videoProcessingParameters = {};
    videoProcessingParameters.progressive_frame = pDispInfo->progressive_frame;
    videoProcessingParameters.second_field = pDispInfo->repeat_first_field + 1;
    videoProcessingParameters.top_field_first = pDispInfo->top_field_first;
    videoProcessingParameters.unpaired_field = pDispInfo->repeat_first_field < 0;
    videoProcessingParameters.output_stream = m_cuvidStream;

    CUdeviceptr dpSrcFrame = 0;
    unsigned int nSrcPitch = 0;
    ck(cuvidMapVideoFrame(m_hDecoder, pDispInfo->picture_index, &dpSrcFrame,
        &nSrcPitch, &videoProcessingParameters));

    m_mtx_vpFrame.lock();
    if (m_bReleaseOldFrames) {
        // Relase frames not in used
        for (int i = 0; i < (int)m_vpFrame.size() - m_nDecodedFrame; i++) {
            uint8_t *pFrame = m_vpFrame.back();
            m_vpFrame.pop_back();
            ReleaseFrame(pFrame);
            m_pFrame2dim.erase(pFrame);
        }
    }
    if ((unsigned)++m_nDecodedFrame > m_vpFrame.size()) {
    // Not enough frames in stock
        uint8_t *pFrame = NULL;
        if (m_bUseDeviceFrame) {
            ck(cuCtxPushCurrent(cuContext));
            if (m_bDeviceFramePitched) {
                ck(cuMemAllocPitch((CUdeviceptr *)&pFrame, &m_nDeviceFramePitch, m_nWidth * (m_nBitDepthMinus8? 2 : 1), m_nHeight * 3 / 2, 16));
            } else {
                ck(cuMemAlloc((CUdeviceptr *)&pFrame, GetFrameSize()));
            }
            ck(cuCtxPopCurrent(NULL));
        } else {
            pFrame = new uint8_t[GetFrameSize()];
        }
        m_pFrame2dim[pFrame] = int2{m_nWidth, m_nHeight};
        m_vpFrame.push_back(pFrame);
        LOG(TRACE) << "New frame added into stock: m_vpFrame.size()=" << m_vpFrame.size();
    }
    uint8_t *pFrame = m_vpFrame[m_nDecodedFrame - 1];
    m_mtx_vpFrame.unlock();

    ck(cuCtxPushCurrent(cuContext));
    CUDA_MEMCPY2D m = { 0 };
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.srcDevice = dpSrcFrame;
    m.srcPitch = nSrcPitch;
    m.dstMemoryType = m_bUseDeviceFrame ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST;
    m.dstDevice = (CUdeviceptr)(m.dstHost = pFrame);
    m.dstPitch = (m_bUseDeviceFrame && m_nDeviceFramePitch) ? m_nDeviceFramePitch : m_nWidth * (m_nBitDepthMinus8? 2 : 1);
    m.WidthInBytes = m_nWidth * (m_nBitDepthMinus8? 2 : 1);
    m.Height = m_nHeight * 3 / 2;
    ck(cuMemcpy2DAsync(&m, m_cuvidStream));
    ck(cuStreamSynchronize(m_cuvidStream));
    ck(cuCtxPopCurrent(NULL));

    if ((int)m_vFrameInfo.size() < m_nDecodedFrame) {
        m_vFrameInfo.resize(m_vpFrame.size());
    }
    m_vFrameInfo[m_nDecodedFrame - 1] = NvFrameInfo{m_nWidth, m_nHeight, GetFramePitch(), GetBitDepth(), GetFrameSize(), m_vPicParam[pDispInfo->picture_index], *pDispInfo};

    ck(cuvidUnmapVideoFrame(m_hDecoder, dpSrcFrame));
    return 1;
}

int NvDecLite::HandleGetOperatingPoint(CUVIDOPERATINGPOINTINFO *pOPInfo)
{
    if (pOPInfo->codec != cudaVideoCodec_AV1 || pOPInfo->av1.operating_points_cnt <= 1) return -1;
    
    // clip has SVC enabled
    if (m_nOperatingPoint >= pOPInfo->av1.operating_points_cnt)
        m_nOperatingPoint = 0;

    printf("AV1 SVC clip: operating point count %d  ", pOPInfo->av1.operating_points_cnt);
    printf("Selected operating point: %d, IDC 0x%x bOutputAllLayers %d\n", m_nOperatingPoint, pOPInfo->av1.operating_points_idc[m_nOperatingPoint], m_bDispAllLayers);
    return (m_nOperatingPoint | (m_bDispAllLayers << 10));
}

NvDecLite::NvDecLite(CUcontext cuContext, bool bUseDeviceFrame, cudaVideoCodec eCodec, 
    bool bLowLatency, bool bDeviceFramePitched, const Rect *pCropRect, const Dim *pResizeDim) :
    cuContext(cuContext), m_bUseDeviceFrame(bUseDeviceFrame), m_eCodec(eCodec), m_bDeviceFramePitched(bDeviceFramePitched)
{
    if (pCropRect) m_cropRect = *pCropRect;
    if (pResizeDim) m_resizeDim = *pResizeDim;

    ck(cuvidCtxLockCreate(&m_ctxLock, cuContext));

    CUVIDPARSERPARAMS videoParserParameters = {};
    videoParserParameters.CodecType = eCodec;
    videoParserParameters.ulMaxNumDecodeSurfaces = 1;
    videoParserParameters.ulMaxDisplayDelay = bLowLatency ? 0 : 1;
    videoParserParameters.pUserData = this;
    videoParserParameters.pfnSequenceCallback = HandleVideoSequenceProc;
    videoParserParameters.pfnDecodePicture = HandlePictureDecodeProc;
    videoParserParameters.pfnDisplayPicture = HandlePictureDisplayProc;
    videoParserParameters.pfnGetOperatingPoint = HandleOperatingPointProc;
    ck(cuvidCreateVideoParser(&m_hParser, &videoParserParameters));
}

NvDecLite::~NvDecLite() {
    if (m_hParser) {
        ck(cuvidDestroyVideoParser(m_hParser));
    }

    if (m_hDecoder) {
        ck(cuvidDestroyDecoder(m_hDecoder));
    }

    m_mtx_vpFrame.lock();
    for (uint8_t *pFrame : m_vpFrame) {
        ReleaseFrame(pFrame);
        m_pFrame2dim.erase(pFrame);
    }
    if (!m_pFrame2dim.empty()) {
        LOG(WARNING) << "Not all frames in stock are released (#Remained=" << m_pFrame2dim.size() << ")";
    }
    m_mtx_vpFrame.unlock();

    ck(cuvidCtxLockDestroy(m_ctxLock));
}

void NvDecLite::ReleaseFrame(uint8_t *pFrame) {
    if (m_bUseDeviceFrame) {
        ck(cuCtxPushCurrent(cuContext));
        ck(cuMemFree((CUdeviceptr)pFrame));
        ck(cuCtxPopCurrent(NULL));
    } else {
        delete[] pFrame;
    }
}

bool NvDecLite::DoDecode(const uint8_t *pData, int nSize, uint32_t flags, int64_t timestamp, CUstream stream) {
    if (!m_hParser) {
        LOG(ERROR) << "Parser not initialized.";
        return false;
    }

    if (m_bReleaseOldFrames) {
        m_bReleaseOldFrames = false;
        vector<uint8_t *> vpFrameNew;
        m_mtx_vpFrame.lock();
        for (uint8_t *pFrame : m_vpFrame) {
            int2 dim = m_pFrame2dim[pFrame];
            if (dim.x == m_nWidth && dim.y == m_nHeight) {
                vpFrameNew.push_back(pFrame);
                continue;
            }
            ReleaseFrame(pFrame);
            m_pFrame2dim.erase(pFrame);
        }
        m_vpFrame = vpFrameNew;
        m_mtx_vpFrame.unlock();
    }

    m_nDecodedFrame = 0;
    CUVIDSOURCEDATAPACKET packet = {0};
    packet.payload = pData;
    packet.payload_size = nSize;
    if (timestamp != INVALID_TIMESTAMP) {
        packet.flags = flags | CUVID_PKT_TIMESTAMP;
        packet.timestamp = timestamp;
    }
    if (!pData || nSize == 0) {
        packet.flags |= CUVID_PKT_ENDOFSTREAM;
    }
    m_cuvidStream = stream;
    ck(cuvidParseVideoData(m_hParser, &packet));
    m_cuvidStream = 0;
    return true;
}

int NvDecLite::Decode(const uint8_t *pData, int nSize, uint8_t ***pppFrame, NvFrameInfo **ppFrameInfo, uint32_t flags, int64_t timestamp, CUstream stream) {
    if (!DoDecode(pData, nSize, flags, timestamp, stream)) {
        return false;
    }
    if (m_nDecodedFrame > 0) {
        if (pppFrame) {
            m_vpFrameRet.clear();
            m_mtx_vpFrame.lock();
            m_vpFrameRet.insert(m_vpFrameRet.begin(), m_vpFrame.begin(), m_vpFrame.begin() + m_nDecodedFrame);
            m_mtx_vpFrame.unlock();
            *pppFrame = &m_vpFrameRet[0];
        }
        if (ppFrameInfo) {
            *ppFrameInfo = &m_vFrameInfo[0];
        }
    }
    return m_nDecodedFrame;
}

int NvDecLite::DecodeLockFrame(const uint8_t *pData, int nSize, uint8_t ***pppFrame, NvFrameInfo **ppFrameInfo, uint32_t flags, int64_t timestamp, CUstream stream) {
    if (!pppFrame) {
        LOG(ERROR) << "Frame pointers must be returned from " << __FUNCTION__;
        return false;
    }
    int ret = Decode(pData, nSize, pppFrame, ppFrameInfo, flags, timestamp, stream);
    m_mtx_vpFrame.lock();
    m_vpFrame.erase(m_vpFrame.begin(), m_vpFrame.begin() + m_nDecodedFrame);
    m_mtx_vpFrame.unlock();
    return ret;
}

void NvDecLite::UnlockFrame(uint8_t **ppFrame, int nFrame) {
    m_mtx_vpFrame.lock();
    for (int i = 0; i < nFrame; i++) {
        uint8_t *pFrame = ppFrame[i];
        if (m_pFrame2dim.find(pFrame) == m_pFrame2dim.end()) {
            LOG(WARNING) << "Invalid frame pointer: 0x" << pFrame;
            continue;
        }
        int2 dim = m_pFrame2dim[pFrame];
        if (dim.x == m_nWidth && dim.y == m_nHeight) {
            m_vpFrame.push_back(pFrame);
            continue;
        }
        ReleaseFrame(pFrame);
        m_pFrame2dim.erase(pFrame);
    }
    m_mtx_vpFrame.unlock();
}
