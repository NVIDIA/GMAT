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

#include "NvCodec/NvEncLiteUnbuffered.h"
#include "NvCodec/NvCommon.h"

NV_ENCODE_API_FUNCTION_LIST NvEncLiteUnbuffered::nvenc = NvEncLiteUnbuffered::LoadNvEncApi();

NV_ENCODE_API_FUNCTION_LIST NvEncLiteUnbuffered::LoadNvEncApi() {
    uint32_t version = 0;
    uint32_t currentVersion = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;
    ck(NvEncodeAPIGetMaxSupportedVersion(&version));
    if (currentVersion > version) {
        LOG(FATAL) << "Current Driver Version does not support this NvEncodeAPI version, please upgrade driver";
        return nvenc;
    }

    nvenc = { NV_ENCODE_API_FUNCTION_LIST_VER };
    ck(NvEncodeAPICreateInstance(&nvenc));
    return nvenc;
}

NvEncLiteUnbuffered::NvEncLiteUnbuffered(NV_ENC_DEVICE_TYPE eDeviceType, void *pDevice, int nWidth, int nHeight, NV_ENC_BUFFER_FORMAT eBufferFormat, 
    NvEncoderInitParam *pInitParam, int nExtraOutputDelay, bool stillImage) :
    pDevice(pDevice), eDeviceType(eDeviceType), nWidth(nWidth), nHeight(nHeight), eBufferFormat(eBufferFormat) 
{
    if (!nvenc.nvEncOpenEncodeSession) {
        LOG(FATAL) << "API entries not loaded";
        return;
    }

    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS encodeSessionExParams = { NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER };
    encodeSessionExParams.device = pDevice;
    encodeSessionExParams.deviceType = eDeviceType;
    encodeSessionExParams.apiVersion = NVENCAPI_VERSION;
    
    void *hEncoder = NULL;
    if (!ck(nvenc.nvEncOpenEncodeSessionEx(&encodeSessionExParams, &hEncoder))) {
        return;
    }

    NV_ENC_TUNING_INFO tuningInfo = NV_ENC_TUNING_INFO_LOW_LATENCY;

    encodeConfig.version = NV_ENC_CONFIG_VER;
    initializeParams.version = NV_ENC_INITIALIZE_PARAMS_VER;
    initializeParams.encodeConfig = &encodeConfig;

    initializeParams.encodeGUID = pInitParam ? pInitParam->GetEncodeGUID() : NV_ENC_CODEC_H264_GUID;
    initializeParams.presetGUID = pInitParam ? pInitParam->GetPresetGUID() : NV_ENC_PRESET_P1_GUID;
    initializeParams.encodeWidth = nWidth;
    initializeParams.encodeHeight = nHeight;
    initializeParams.darWidth = nWidth;
    initializeParams.darHeight = nHeight;
    initializeParams.frameRateNum = 25;
    initializeParams.frameRateDen = 1;
    initializeParams.enablePTD = 1;
    initializeParams.reportSliceOffsets = 0;
    initializeParams.enableSubFrameWrite = 0;
    initializeParams.maxEncodeWidth = nWidth;
    initializeParams.maxEncodeHeight = nHeight;
    initializeParams.tuningInfo = tuningInfo;

    NV_ENC_PRESET_CONFIG presetConfig = { NV_ENC_PRESET_CONFIG_VER, { NV_ENC_CONFIG_VER } };
    ck(nvenc.nvEncGetEncodePresetConfigEx(hEncoder, initializeParams.encodeGUID, initializeParams.presetGUID, tuningInfo, &presetConfig));
    encodeConfig = presetConfig.presetCfg;
    encodeConfig.frameIntervalP = stillImage ? 0 : 1;
    encodeConfig.gopLength = stillImage ? 1 : 300;
    encodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
    encodeConfig.rcParams.constQP = {28, 31, 25};

    if (pInitParam && !pInitParam->SetInitParams(&initializeParams)) {
        LOG(ERROR) << "Error setting encoder parameter";
        ck(nvenc.nvEncDestroyEncoder(hEncoder));
        return;
    }
    
    // set other necessary params if not set yet
    if (initializeParams.encodeGUID == NV_ENC_CODEC_H264_GUID) {
        if (eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV420_10BIT || eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT) {
            ck(nvenc.nvEncDestroyEncoder(hEncoder));
            LOG(FATAL) << "10-bit format isn't supported by H264 encoder";
            return;
        }
        if (eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444 || eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT) {
            encodeConfig.encodeCodecConfig.h264Config.chromaFormatIDC = 3;
        }
        encodeConfig.encodeCodecConfig.h264Config.idrPeriod = encodeConfig.gopLength;
        encodeConfig.encodeCodecConfig.h264Config.repeatSPSPPS = 1;
    } else if (initializeParams.encodeGUID == NV_ENC_CODEC_HEVC_GUID) {
        encodeConfig.encodeCodecConfig.hevcConfig.pixelBitDepthMinus8 =
            eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV420_10BIT || eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT ? 2 : 0;
        if (eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444 || eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT) {
            encodeConfig.encodeCodecConfig.hevcConfig.chromaFormatIDC = 3;
        } else {
            encodeConfig.encodeCodecConfig.hevcConfig.chromaFormatIDC = 1;
        }
        encodeConfig.encodeCodecConfig.hevcConfig.idrPeriod = encodeConfig.gopLength;
        encodeConfig.encodeCodecConfig.hevcConfig.repeatSPSPPS = 1;
    }
    if (!encodeConfig.rcParams.enableAQ && !encodeConfig.rcParams.enableTemporalAQ) {
        // QP delta map and AQ (either spatial or temporal) are mutually exclusive
        encodeConfig.rcParams.qpMapMode = NV_ENC_QP_MAP_DELTA;
    }
    if (stillImage) {
        encodeConfig.rcParams.enableLookahead = 0;
    }
    
    LOG(INFO) << NvEncoderInitParam().MainParamToString(&initializeParams);
    LOG(TRACE) << NvEncoderInitParam().FullParamToString(&initializeParams);

    if (!ck(nvenc.nvEncInitializeEncoder(hEncoder, &initializeParams))) {
        ck(nvenc.nvEncDestroyEncoder(hEncoder));
        return;
    }
    this->hEncoder = hEncoder;
    nEncoderBuffer = encodeConfig.frameIntervalP + encodeConfig.rcParams.lookaheadDepth + nExtraOutputDelay;
    // still image has 0 frameIntervalP and lookaheadDepth, alloc one more frame to avoid nEncoderBuffer=0
    nEncoderBuffer += stillImage ? 1 : 0;
    nOutputDelay = nEncoderBuffer - 1;
    for (int i = 0; i < nEncoderBuffer; i++) {
        NV_ENC_CREATE_BITSTREAM_BUFFER createBitstreamBuffer = { NV_ENC_CREATE_BITSTREAM_BUFFER_VER };
        // Sufficiently large output buffer
        createBitstreamBuffer.size = GetFrameSize();
        createBitstreamBuffer.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;
        ck(nvenc.nvEncCreateBitstreamBuffer(hEncoder, &createBitstreamBuffer));
        vBitstreamOutputBuffer.push_back(createBitstreamBuffer.bitstreamBuffer);
    }
    // vpCompletionEvent is list of NULLs until set by subclass
    vpCompletionEvent.resize(nEncoderBuffer);
}

NvEncLiteUnbuffered::~NvEncLiteUnbuffered() {
    if (!hEncoder) {
        return;
    }
    for (int i = 0; i < nEncoderBuffer; i++) {
        ck(nvenc.nvEncDestroyBitstreamBuffer(hEncoder, vBitstreamOutputBuffer[i]));
    }
    ck(nvenc.nvEncDestroyEncoder(hEncoder));
    if (lDts.size()) {
        LOG(WARNING) << "Some frames aren't encoded";
    }
}

bool NvEncLiteUnbuffered::GetSequenceParams(uint8_t **ppSequenceParams, int *pnSize) {
    if (!ReadyForEncode()) {
        return false;
    }
    NV_ENC_SEQUENCE_PARAM_PAYLOAD sequenceParamPayload = {NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER};
    sequenceParamPayload.inBufferSize = sizeof(aSequenceParams);
    sequenceParamPayload.spsppsBuffer = aSequenceParams;
    sequenceParamPayload.outSPSPPSPayloadSize = (uint32_t *)pnSize;
    if (!ck(nvenc.nvEncGetSequenceParams(hEncoder, &sequenceParamPayload))) {
        return false;
    }
    *ppSequenceParams = aSequenceParams;
    return true;
}

bool NvEncLiteUnbuffered::EncodeDeviceFrameUnbuffered(void *pDeviceFrame, int nFramePitch, std::vector<std::vector<uint8_t>> &vPacket, std::vector<NvPacketInfo> *pvPacketInfo, NV_ENC_PIC_PARAMS *pPicParams) {
    vPacket.clear();
    if (!ReadyForEncode()) {
        return false;
    }

    if (nEncoderBuffer != 1) {
        // Unbuffered encoding requires one frame in and one packet out in one call
        LOG(FATAL) << __FUNCTION__ << " requires b-frame number, lookahead depth and nExtraOutputDelay all to be 0.";
        return false;
    }

    if (!pDeviceFrame) {
        return EndEncode(vPacket, pvPacketInfo);
    }

    // Data buffer is mapped and used in place - there's no copy and no internel buffer used
    NV_ENC_REGISTER_RESOURCE registerResource = { NV_ENC_REGISTER_RESOURCE_VER };
    registerResource.resourceType = 
        eDeviceType == NV_ENC_DEVICE_TYPE_DIRECTX ? NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX : 
        (eDeviceType == NV_ENC_DEVICE_TYPE_OPENGL ? NV_ENC_INPUT_RESOURCE_TYPE_OPENGL_TEX : 
        NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR);
    registerResource.resourceToRegister = (void *)pDeviceFrame;
    registerResource.width = nWidth;
    registerResource.height = nHeight;
    registerResource.pitch = eDeviceType == NV_ENC_DEVICE_TYPE_DIRECTX ? 0 : (nFramePitch ? nFramePitch : GetPlaneWidthInBytes());
    registerResource.bufferFormat = eBufferFormat;
    ck(nvenc.nvEncRegisterResource(hEncoder, &registerResource));
    
    NV_ENC_MAP_INPUT_RESOURCE mapInputResource = { NV_ENC_MAP_INPUT_RESOURCE_VER };
    mapInputResource.registeredResource = registerResource.registeredResource;
    ck(nvenc.nvEncMapInputResource(hEncoder, &mapInputResource));

    bool r = DoEncode(mapInputResource.mappedResource, vPacket, pvPacketInfo, pPicParams);

    ck(nvenc.nvEncUnmapInputResource(hEncoder, mapInputResource.mappedResource));
    ck(nvenc.nvEncUnregisterResource(hEncoder, registerResource.registeredResource));

    return r;
}

bool NvEncLiteUnbuffered::DoEncode(NV_ENC_INPUT_PTR inputBuffer, std::vector<std::vector<uint8_t>> &vPacket, std::vector<NvPacketInfo> *pvPacketInfo, NV_ENC_PIC_PARAMS *pPicParams) {
    NV_ENC_PIC_PARAMS picParams = {};
    if (pPicParams) {
        picParams = *pPicParams;
    }
    picParams.version = NV_ENC_PIC_PARAMS_VER;
    picParams.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
    picParams.inputBuffer = inputBuffer;
    picParams.bufferFmt = eBufferFormat;
    picParams.inputWidth = nWidth;
    picParams.inputHeight = nHeight;
    picParams.outputBitstream = vBitstreamOutputBuffer[iToSend % nEncoderBuffer];
    picParams.completionEvent = vpCompletionEvent[iToSend % nEncoderBuffer];
    // Offset DTS to ensure DTS <= PTS
    if (lDts.size() == 1) {
        dtsOffset = (int64_t)pPicParams->inputTimeStamp - lDts.front();
        *lDts.begin() -= dtsOffset;
    }
    lDts.push_back((int64_t)picParams.inputTimeStamp - dtsOffset);
    iToSend++;
    NVENCSTATUS nvStatus = nvenc.nvEncEncodePicture(hEncoder, &picParams);
    if (nvStatus == NV_ENC_SUCCESS || nvStatus == NV_ENC_ERR_NEED_MORE_INPUT) {
    // In the case of NV_ENC_ERR_NEED_MORE_INPUT: though more input needed for this call, packets may be ready for last calls
        GetEncodedPacket(vBitstreamOutputBuffer, vPacket, pvPacketInfo, true);
    } else {
        LOG(FATAL) << "nvEncEncodePicture() error=" << nvStatus << " at line " << __LINE__ << " in file " << __FILE__;
        return false;
    }
    return true;
}

bool NvEncLiteUnbuffered::EndEncode(std::vector<std::vector<uint8_t>> &vPacket, std::vector<NvPacketInfo> *pvPacketInfo) {
    vPacket.clear();
    if (!ReadyForEncode()) {
        return false;
    }

    NV_ENC_PIC_PARAMS picParams = { NV_ENC_PIC_PARAMS_VER };
    picParams.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
    picParams.completionEvent = vpCompletionEvent[iToSend % nEncoderBuffer];
    NVENCSTATUS nvStatus = nvenc.nvEncEncodePicture(hEncoder, &picParams);
    if (ck(nvStatus)) {
        GetEncodedPacket(vBitstreamOutputBuffer, vPacket, pvPacketInfo, false);
    }
    return true;
}

void NvEncLiteUnbuffered::GetEncodedPacket(std::vector<NV_ENC_OUTPUT_PTR> &vOutputBuffer, std::vector<std::vector<uint8_t>> &vPacket, std::vector<NvPacketInfo> *pvPacketInfo, bool bOutputDelay) {
    unsigned i = 0;
    int iEnd = bOutputDelay ? iToSend - nOutputDelay : iToSend;
    for (; iGot < iEnd; iGot++) {
        NV_ENC_LOCK_BITSTREAM lockBitstreamData = { NV_ENC_LOCK_BITSTREAM_VER };
        lockBitstreamData.outputBitstream = vOutputBuffer[iGot % nEncoderBuffer];
        lockBitstreamData.doNotWait = false;
        NVENCSTATUS nvStatus = nvenc.nvEncLockBitstream(hEncoder, &lockBitstreamData);
        if (nvStatus != NV_ENC_SUCCESS) {
            LOG(ERROR) << "nvEncLockBitstream() error=" << nvStatus << " in line " << __LINE__ << " of file " << __FILE__;
        }

        uint8_t *pData = (uint8_t *)lockBitstreamData.bitstreamBufferPtr;
        if (vPacket.size() < i + 1) {
            vPacket.push_back(std::vector<uint8_t>());
        }
        vPacket[i].clear();
        vPacket[i].insert(vPacket[i].end(), &pData[0], &pData[lockBitstreamData.bitstreamSizeInBytes]);
        int64_t dts = 0;
        if (lDts.size()) {
            dts = lDts.front();
            lDts.pop_front();
        }
        if (pvPacketInfo) {
            pvPacketInfo->push_back(NvPacketInfo{lockBitstreamData, dts});
        }
        i++;

        ck(nvenc.nvEncUnlockBitstream(hEncoder, lockBitstreamData.outputBitstream));
    }
}

bool NvEncLiteUnbuffered::Reconfigure(NV_ENC_RECONFIGURE_PARAMS *pReconfigureParams) {
    return ck(nvenc.nvEncReconfigureEncoder(hEncoder, pReconfigureParams));
}
