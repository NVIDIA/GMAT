

#ifndef WIN32
#include <dlfcn.h>
#endif

#include "NvCodec/NvHeifWriter.h"
#include "NvCodec/NvCommon.h"

NvHeifWriter::NvHeifWriter() : NvHeifWriter(NV_ENC_CODEC_HEVC_GUID, true){
}

NvHeifWriter::NvHeifWriter(GUID codec, bool stillImage)
{
    writer = HEIF::Writer::Create();

    if (stillImage) {
        majorBrand = HEIF::FourCC("mif1");
        compatibleBrands = HEIF::Array<HEIF::FourCC>{"mif1", "heic"};
    } else {
        majorBrand = HEIF::FourCC("msf1");
        compatibleBrands = HEIF::Array<HEIF::FourCC>{"msf1", "hevc"};
    }

    if (codec == NV_ENC_CODEC_HEVC_GUID) {
        format = HEIF::MediaFormat::HEVC;
    } else {
        LOG(ERROR) << "Codec not supported";
        return;
    }

    decoderInfo = HEIF::Array<HEIF::DecoderSpecificInfo>(3);
    initParameterSets = true;
}

bool NvHeifWriter::write(std::vector<std::vector<uint8_t>> &nalUnits, char* outFilePath, bool useLastParameterSet)
{
    int offset_vps, offset_sps, offset_pps;
    size_t bytes_vps, bytes_sps, bytes_pps, bytes_slices = 0;
    std::vector<std::vector<int>> offset_idr;
    std::vector<std::vector<int>> startCodePos;
    // find all start code positions
    for (int i = 0; i < nalUnits.size(); i++)
    {
        std::vector<int> offsets;
        for (int offset = 0; offset < nalUnits[i].size(); offset++) 
        {
            uint32_t code = reinterpret_cast<uint32_t*>(nalUnits[i].data() + offset)[0];
            bool isStartCode = (code == 0x1000000);
            if (isStartCode)
            {
                offsets.push_back(offset);
            }
            else if (offset == (nalUnits[i].size() - 1))
            {
                offsets.push_back(offset + 1);
            }
        }
        startCodePos.push_back(offsets);
    }

    // identify nalu types
    // each frame can have multiple slices, but should only have one vps/sps/pps
    // all nal units after idr are treated as slices
    for (int i = 0; i < startCodePos.size(); i++)
    {
        std::vector<int> idrPos;
        // bool finished;
        for (int j =0; j < startCodePos[i].size(); j++)
        {
            int offset = startCodePos[i][j];
            if (offset >= nalUnits[i].size()) break;
            uint8_t nalu_type = nalUnits[i][offset + 4] >> 1 & 0x3f;
            switch (nalu_type)
            {
                case 0x20: // VPS
                    offset_vps = offset;
                    bytes_vps = startCodePos[i][j + 1] - offset;
                    continue;
                case 0x21: // SPS
                    offset_sps = offset;
                    bytes_sps = startCodePos[i][j + 1] - offset;
                    continue;
                case 0x22: // PPS
                    offset_pps = offset;
                    bytes_pps = startCodePos[i][j + 1] - offset;
                    continue;
                case 0x13: // IDR
                    idrPos.push_back(offset);
                    idrPos.push_back(startCodePos[i][j + 1]);
                    bytes_slices += startCodePos[i][j + 1] - offset;
                    // finished = true;
                    continue;
            }
        }
        offset_idr.push_back(idrPos);
        // if (finished) break;
    }

    // heif muxer
    HEIF::OutputConfig writerOutputConf{};
    writerOutputConf.fileName        = outFilePath;
    writerOutputConf.progressiveFile = true;
    writerOutputConf.majorBrand = majorBrand;
    writerOutputConf.compatibleBrands = compatibleBrands;
    // initialize writer now that we have all the needed information from reader
    if (writer->initialize(writerOutputConf) != HEIF::ErrorCode::OK)
    {
        return false;
    }

    // DecoderConfiguration outputdecoderConfig{};
    // outputdecoderConfig.decoderConfigId = 1;
    // Array<DecoderSpecificInfo> decInfo(3);
    uint8_t* sliceData = new uint8_t[bytes_slices];
    uint32_t nalLength = static_cast<uint32_t>(bytes_slices) - 4;
    sliceData[0] = static_cast<uint8_t>(nalLength >> 24 & 0xff);
    sliceData[1] = static_cast<uint8_t>(nalLength >> 16 & 0xff);
    sliceData[2] = static_cast<uint8_t>(nalLength >> 8 & 0xff);
    sliceData[3] = static_cast<uint8_t>(nalLength >> 0 & 0xff);
    // copy slices
    uint8_t* sliceAddr = sliceData + 4;
    for (int i = 0; i < offset_idr.size(); i++) {
        for (int j = 0; j < offset_idr[i].size(); j+=2) {
            size_t copyBytes = offset_idr[i][j + 1] - offset_idr[i][j];
            memcpy(sliceAddr, nalUnits[i].data() + offset_idr[i][j] + 4, copyBytes);
            sliceAddr += copyBytes;
        }
    }
    // copy parameter sets
    if (!useLastParameterSet || initParameterSets) {
        vpsArray = new HEIF::Array<uint8_t>(bytes_vps);
        spsArray = new HEIF::Array<uint8_t>(bytes_sps);
        ppsArray = new HEIF::Array<uint8_t>(bytes_pps);
        memcpy(vpsArray->elements, nalUnits[0].data() + offset_vps, bytes_vps);
        memcpy(spsArray->elements, nalUnits[0].data() + offset_sps, bytes_sps);
        memcpy(ppsArray->elements, nalUnits[0].data() + offset_pps, bytes_pps);
        
        initParameterSets = false;
    }
    // for (int i = 0; i < nalUnits.size(); i++)
    // {
    //     size_t accu = 0;
    //     // uint8_t* sliceStart = sliceData + 4;
    //     if (i == 0)
    //     {
    //         memcpy(sliceData + 4, nalUnits[i].data() + offset_idr + 4, nalUnits[i].size() - offset_idr);
    //         accu += nalUnits[i].size() - offset_idr;
    //     } 
    //     else
    //     {
    //         memcpy(sliceData + 4 + accu, nalUnits[i].data() + 4, nalUnits[i].size());
    //         accu += nalUnits[i].size();
    //     }
    // }

    // VPS
    decoderInfo[0].decSpecInfoType = HEIF::DecoderSpecInfoType::HEVC_VPS;
    decoderInfo[0].decSpecInfoData = *vpsArray;
    // SPS
    decoderInfo[1].decSpecInfoType = HEIF::DecoderSpecInfoType::HEVC_SPS;
    decoderInfo[1].decSpecInfoData = *spsArray;
    // PPS
    decoderInfo[2].decSpecInfoType = HEIF::DecoderSpecInfoType::HEVC_PPS;
    decoderInfo[2].decSpecInfoData = *ppsArray;

    // outputdecoderConfig.decoderSpecificInfo = decInfo;

    HEIF::DecoderConfigId outputDecoderConfigId;
    writer->feedDecoderConfig(decoderInfo, outputDecoderConfigId);

    HEIF::Data imageData{};
    imageData.size = bytes_slices;
    imageData.data = sliceData;
    // feed image data to writer
    imageData.mediaFormat     = format;
    imageData.decoderConfigId = outputDecoderConfigId;
    HEIF::MediaDataId outputMediaId;
    writer->feedMediaData(imageData, outputMediaId);
    delete[] sliceData;

    // create new image based on that data:
    HEIF::ImageId outputImageId;
    writer->addImage(outputMediaId, outputImageId);

    // if this input image was the primary image -> also mark output image as primary image
    writer->setPrimaryItem(outputImageId);

    writer->finalize();

    if (!useLastParameterSet) {
        delete vpsArray;
        delete spsArray;
        delete ppsArray;
    }

    return true;
}