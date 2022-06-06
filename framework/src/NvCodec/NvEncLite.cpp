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

#ifndef WIN32
#include <dlfcn.h>
#endif
#include "NvCodec/NvEncLite.h"
#include "NvCodec/NvCommon.h"

NvEncLite::NvEncLite(CUcontext cuContext, int nWidth, int nHeight, NV_ENC_BUFFER_FORMAT eBufferFormat, NvEncoderInitParam *pInitParam, int nExtraOutputDelay) :
    NvEncLiteUnbuffered(NV_ENC_DEVICE_TYPE_CUDA, cuContext, nWidth, nHeight, eBufferFormat, pInitParam, nExtraOutputDelay)
{
    if (!hEncoder) {
        return;
    }
    ck(cuCtxPushCurrent(cuContext));
    for (int i = 0; i < nEncoderBuffer; i++) {
        CUdeviceptr pDeviceFrame;
        ck(cuMemAlloc(&pDeviceFrame, GetDeviceFrameBufferSize()));
        vpDeviceFrame.push_back(pDeviceFrame);

        NV_ENC_REGISTER_RESOURCE registerResource = { NV_ENC_REGISTER_RESOURCE_VER };
        registerResource.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
        registerResource.resourceToRegister = (void *)pDeviceFrame;
        registerResource.width = nWidth;
        registerResource.height = nHeight;
        registerResource.pitch = GetDeviceFrameBufferPitch();
        registerResource.bufferFormat = eBufferFormat;
        ck(nvenc.nvEncRegisterResource(hEncoder, &registerResource));
        vRegisteredResource.push_back(registerResource.registeredResource);

        NV_ENC_MAP_INPUT_RESOURCE mapInputResource = { NV_ENC_MAP_INPUT_RESOURCE_VER };
        mapInputResource.registeredResource = registerResource.registeredResource;
        ck(nvenc.nvEncMapInputResource(hEncoder, &mapInputResource));
        vDeviceInputBuffer.push_back(mapInputResource.mappedResource);
    }
    ck(cuCtxPopCurrent(NULL));
}

NvEncLite::~NvEncLite() {
    if (!hEncoder) {
        return;
    }
    ck(cuCtxPushCurrent(CUcontext(pDevice)));
    for (int i = 0; i < nEncoderBuffer; i++) {
        ck(nvenc.nvEncUnmapInputResource(hEncoder, vDeviceInputBuffer[i]));
        ck(nvenc.nvEncUnregisterResource(hEncoder, vRegisteredResource[i]));
        ck(cuMemFree(vpDeviceFrame[i]));
    }
    ck(cuCtxPopCurrent(NULL));
}

bool NvEncLite::EncodeFrame(uint8_t *pFrame, bool bDeviceFrame, int nFramePitch, std::vector<std::vector<uint8_t>> &vPacket, std::vector<NvPacketInfo> *pvPacketInfo, NV_ENC_PIC_PARAMS *pPicParams) {
    vPacket.clear();
    if (pvPacketInfo) {
        pvPacketInfo->clear();
    }
    if (!ReadyForEncode()) {
        return false;
    }

    if (!pFrame) {
        return EndEncode(vPacket, pvPacketInfo);
    }

    int i = iToSend % nEncoderBuffer;
    // Buffer is umapped for data to be copied in, and remapped afterwards
    ck(nvenc.nvEncUnmapInputResource(hEncoder, vDeviceInputBuffer[i]));
    CopyFrame(pFrame, bDeviceFrame, nFramePitch ? nFramePitch : GetPlaneWidthInBytes(), vpDeviceFrame[i], GetDeviceFrameBufferPitch());
    NV_ENC_MAP_INPUT_RESOURCE mapInputResource = { NV_ENC_MAP_INPUT_RESOURCE_VER };
    mapInputResource.registeredResource = vRegisteredResource[i];
    ck(nvenc.nvEncMapInputResource(hEncoder, &mapInputResource));
    vDeviceInputBuffer[i] = mapInputResource.mappedResource;

    return DoEncode(vDeviceInputBuffer[i], vPacket, pvPacketInfo, pPicParams);
}

bool NvEncLite::EncodeDeviceFrame(uint8_t *pDeviceFrame, int nFramePitch, std::vector<std::vector<uint8_t>> &vPacket, std::vector<NvPacketInfo> *pvPacketInfo, NV_ENC_PIC_PARAMS *pPicParams) {
    return EncodeFrame(pDeviceFrame, true, nFramePitch, vPacket, pvPacketInfo, pPicParams);
}

bool NvEncLite::EncodeHostFrame(uint8_t *pHostFrame, int nFramePitch, std::vector<std::vector<uint8_t>> &vPacket, std::vector<NvPacketInfo> *pvPacketInfo, NV_ENC_PIC_PARAMS *pPicParams) {
    return EncodeFrame(pHostFrame, false, nFramePitch, vPacket, pvPacketInfo, pPicParams);
}

void NvEncLite::CopyFrame(uint8_t *pSrcFrame, bool bDeviceFrame, int nSrcPitch, CUdeviceptr pDstFrame, int nDstPitch) {
    int h = 0, hh = 0;
    GetPlaneHeights(h, hh);
    ck(cuCtxPushCurrent(CUcontext(pDevice)));
    if (nSrcPitch == nDstPitch) {
        if (bDeviceFrame) {
            ck(cuMemcpyDtoD(pDstFrame, (CUdeviceptr)pSrcFrame, nSrcPitch * h + nSrcPitch / 2 * hh));
        } else {
            ck(cuMemcpyHtoD(pDstFrame, pSrcFrame, nSrcPitch * h + nSrcPitch / 2 * hh));
        }
        ck(cuCtxPopCurrent(NULL));
        return;
    }
    int w = GetPlaneWidthInBytes();
    CUDA_MEMCPY2D m = { 0 };
    m.srcMemoryType = bDeviceFrame ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST;
    m.srcDevice = (CUdeviceptr)(m.srcHost = pSrcFrame);
    m.srcPitch = nSrcPitch;
    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    m.dstDevice = pDstFrame;
    m.dstPitch = nDstPitch;
    m.WidthInBytes = w;
    m.Height = h;
    ck(cuMemcpy2D(&m));
    if (hh) {
        m.srcDevice = (CUdeviceptr)(m.srcHost = pSrcFrame + nSrcPitch * h);
        m.srcPitch = nSrcPitch / 2;
        m.dstDevice = (CUdeviceptr)((uint8_t *)pDstFrame + nDstPitch * h);
        m.dstPitch = nDstPitch / 2;
        m.WidthInBytes = w / 2;
        m.Height = hh;
        ck(cuMemcpy2D(&m));
    }
    ck(cuCtxPopCurrent(NULL));
}
