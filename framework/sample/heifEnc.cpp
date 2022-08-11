#include <vector>
#include <thread>

#include "NvCodec/NvCommon.h"
#include "NvCodec/NvEncLite.h"
#include "NvCodec/NvHeifWriter.h"

#include <cuda_runtime.h>
#include <heif/reader/heifreader.h>
#include <heif/writer/heifwriter.h>

using namespace std;
using namespace HEIF;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger(simplelogger::TRACE);

void init_encoder(CUcontext cuContext, int nWidth, int nHeight, const char *szOutFilePath, 
    NvEncoderInitParam &initParam, NvEncLite* &enc, FILE* &fpOut) {
    fpOut = fopen(szOutFilePath, "wb");
    if (fpOut == NULL) {
        cout << "Unable to open file: " << szOutFilePath << endl;
        return;
    }

    std::string init_param_string{"-codec hevc -preset p1 -bitrate 2M"};
    initParam = NvEncoderInitParam(init_param_string.c_str());
    NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_NV12;

    enc = new NvEncLite(cuContext, nWidth, nHeight, eFormat, &initParam);
}

void init_encoder(CUcontext cuContext, int nWidth, int nHeight, 
    NvEncoderInitParam &initParam, NvEncLite* &enc, bool stillImage=false) {
    std::string init_param_string{"-codec hevc -preset p1 -bitrate 2M"};
    initParam = NvEncoderInitParam(init_param_string.c_str());
    NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_NV12;

    enc = new NvEncLite(cuContext, nWidth, nHeight, eFormat, &initParam, 0, stillImage);
}

union FourByte {
    uint32_t d;
    uchar4 v;
    struct {
        uint8_t b1, b2, b3, b4;
    }b;
};

int encodeFromYuv(const char* inPath, const char* outPath, CUcontext current)
{
    
    int frameWidth = 1280, frameHeight = 720;
    BufferedFileReader reader(inPath);
    uint8_t *pNv12Buf, *dpNv12Buf;
    size_t nNv12BufSize;
    reader.GetBuffer(&pNv12Buf, &nNv12BufSize);
    ck(cudaMalloc(&dpNv12Buf, nNv12BufSize));
    ck(cudaMemcpy(dpNv12Buf, pNv12Buf, nNv12BufSize, cudaMemcpyHostToDevice));

    // Encoder setup
    NvEncoderInitParam initParam;
    NvEncLite *enc = nullptr;
    FILE *fpOut = nullptr;
    const char* szOutFilePath = "./bin/sample/heif/bus_720_out.hevc";
    init_encoder(current, frameWidth, frameHeight, szOutFilePath,
    initParam, enc, fpOut);

    vector<vector<uint8_t>> vPacket;
    enc->EncodeDeviceFrame(dpNv12Buf, 0, vPacket);

    int offset_vps, offset_sps, offset_pps;
    size_t bytes_vps, bytes_sps, bytes_pps, bytes_slices = 0;
    vector<vector<int>> offset_idr;
    vector<vector<int>> startCodePos;
    // find all start code positions
    for (int i = 0; i < vPacket.size(); i++)
    {
        vector<int> offsets;
        for (int offset = 0; offset < vPacket[i].size(); offset++) 
        {
            uint32_t code = reinterpret_cast<uint32_t*>(vPacket[i].data() + offset)[0];
            bool isStartCode = (code == 0x1000000);
            if (isStartCode)
            {
                offsets.push_back(offset);
            }
            else if (offset == (vPacket[i].size() - 1))
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
        vector<int> idrPos;
        // bool finished;
        for (int j =0; j < startCodePos[i].size(); j++)
        {
            int offset = startCodePos[i][j];
            if (offset >= vPacket[i].size()) break;
            uint8_t nalu_type = vPacket[i][offset + 4] >> 1 & 0x3f;
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
    auto* writer = Writer::Create();
    OutputConfig writerOutputConf{};
    writerOutputConf.fileName        = outPath;
    writerOutputConf.progressiveFile = true;

    FourCC inputMajorBrand{"mif1"};
    writerOutputConf.majorBrand = inputMajorBrand;
    Array<FourCC> inputCompatibleBrands{"heic", "mif1"};
    writerOutputConf.compatibleBrands = inputCompatibleBrands;
    // initialize writer now that we have all the needed information from reader
    if (writer->initialize(writerOutputConf) != ErrorCode::OK)
    {
        return -1;
    }

    DecoderConfiguration outputdecoderConfig{};
    outputdecoderConfig.decoderConfigId = 1;
    Array<DecoderSpecificInfo> decInfo(3);
    Array<uint8_t> vpsArray(bytes_vps), spsArray(bytes_sps), ppsArray(bytes_pps);
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
            memcpy(sliceAddr, vPacket[i].data() + offset_idr[i][j] + 4, copyBytes);
            sliceAddr += copyBytes;
        }
    }
    // copy parameter sets
    memcpy(vpsArray.elements, vPacket[0].data() + offset_vps, bytes_vps);
    memcpy(spsArray.elements, vPacket[0].data() + offset_sps, bytes_sps);
    memcpy(ppsArray.elements, vPacket[0].data() + offset_pps, bytes_pps);
    // for (int i = 0; i < vPacket.size(); i++)
    // {
    //     size_t accu = 0;
    //     // uint8_t* sliceStart = sliceData + 4;
    //     if (i == 0)
    //     {
    //         memcpy(sliceData + 4, vPacket[i].data() + offset_idr + 4, vPacket[i].size() - offset_idr);
    //         accu += vPacket[i].size() - offset_idr;
    //     } 
    //     else
    //     {
    //         memcpy(sliceData + 4 + accu, vPacket[i].data() + 4, vPacket[i].size());
    //         accu += vPacket[i].size();
    //     }
    // }

    // VPS
    decInfo[0].decSpecInfoType = DecoderSpecInfoType::HEVC_VPS;
    decInfo[0].decSpecInfoData = vpsArray;
    // SPS
    decInfo[1].decSpecInfoType = DecoderSpecInfoType::HEVC_SPS;
    decInfo[1].decSpecInfoData = spsArray;
    // PPS
    decInfo[2].decSpecInfoType = DecoderSpecInfoType::HEVC_PPS;
    decInfo[2].decSpecInfoData = ppsArray;

    outputdecoderConfig.decoderSpecificInfo = decInfo;

    DecoderConfigId outputDecoderConfigId;
    writer->feedDecoderConfig(outputdecoderConfig.decoderSpecificInfo, outputDecoderConfigId);

    Data imageData{};
    imageData.size = bytes_slices;
    imageData.data = sliceData;
    // feed image data to writer
    imageData.mediaFormat     = MediaFormat::HEVC;
    imageData.decoderConfigId = outputDecoderConfigId;
    MediaDataId outputMediaId;
    writer->feedMediaData(imageData, outputMediaId);
    delete[] sliceData;

    // create new image based on that data:
    ImageId outputImageId;
    writer->addImage(outputMediaId, outputImageId);

    // if this input image was the primary image -> also mark output image as primary image
    writer->setPrimaryItem(outputImageId);

    writer->finalize();

    Writer::Destroy(writer);

    return 0;
}

int encodeFromHevc(const char* inPath, const char* outPath, CUcontext current)
{
    
    int frameWidth = 1280, frameHeight = 720;
    BufferedFileReader reader(inPath);
    uint8_t *pHevcBuf;
    size_t nHevcBufSize;
    reader.GetBuffer(&pHevcBuf, &nHevcBufSize);
    // ck(cudaMalloc(&dpNv12Buf, nNv12BufSize));
    // ck(cudaMemcpy(dpNv12Buf, pNv12Buf, nNv12BufSize, cudaMemcpyHostToDevice));

    // Encoder setup
    // NvEncoderInitParam initParam;
    // NvEncLite *enc = nullptr;
    // FILE *fpOut = nullptr;
    // const char* szOutFilePath = outPath;
    // init_encoder(current, frameWidth, frameHeight, szOutFilePath,
    // initParam, enc, fpOut);

    vector<vector<uint8_t>> vPacket;
    // enc->EncodeDeviceFrame(dpNv12Buf, 0, vPacket);

    int offset_vps, offset_sps, offset_pps, offset_idr;
    size_t bytes_vps, bytes_sps, bytes_pps, bytes_slices;
    vector<vector<int>> startCodePos;
    // find all start code positions
    vector<int> offsets;
    for (int i = 0; i < nHevcBufSize; i++)
    {
        uint32_t code = reinterpret_cast<uint32_t*>(pHevcBuf + i)[0];
        bool isStartCode = (code == 0x1000000);
        if (isStartCode || i == (nHevcBufSize - 1))
        {
            offsets.push_back(i);
        }
    }
    // for (int i = 0; i < vPacket.size(); i++)
    // {
    //     vector<int> offsets;
    //     for (int offset = 0; offset < vPacket[i].size(); offset++) 
    //     {
    //         uint32_t code = reinterpret_cast<uint32_t*>(vPacket[i].data() + offset)[0];
    //         bool isStartCode = (code == 0x1000000);
    //         if (isStartCode || offset == (vPacket[i].size() - 1))
    //         {
    //             offsets.push_back(offset);
    //         }
    //     }
    //     startCodePos.push_back(offsets);
    // }

    // identify nalu types
    // each frame can have multiple slices, but should only have one vps/sps/pps
    // all nal units after idr are treated as slices
    for (int i = 0; i < offsets.size(); i++)
    {
        int offset = offsets[i];
        uint8_t nalu_type = pHevcBuf[offset + 4] >> 1 & 0x3f;
        switch (nalu_type)
        {
            case 0x20: // VPS
                offset_vps = offset;
                bytes_vps = offsets[i + 1] - offset;
                continue;
            case 0x21: // SPS
                offset_sps = offset;
                bytes_sps = offsets[i + 1] - offset;
                continue;
            case 0x22: // PPS
                offset_pps = offset;
                bytes_pps = offsets[i + 1] - offset;
                continue;
            case 0x13: // IDR
                offset_idr = offset;
                bytes_slices = nHevcBufSize - offset;
                break;
        }
    }
    // for (int i = 0; i < startCodePos.size(); i++)
    // {
    //     bool finished;
    //     for (int j =0; j < startCodePos[i].size(); j++)
    //     {
    //         int offset = startCodePos[i][j];
    //         uint8_t nalu_type = vPacket[i][offset + 4] >> 1 & 0x3f;
    //         switch (nalu_type)
    //         {
    //             case 0x20: // VPS
    //                 offset_vps = offset;
    //                 bytes_vps = startCodePos[i][j + 1] - offset;
    //                 continue;
    //             case 0x21: // SPS
    //                 offset_sps = offset;
    //                 bytes_sps = startCodePos[i][j + 1] - offset;
    //                 continue;
    //             case 0x22: // PPS
    //                 offset_pps = offset;
    //                 bytes_pps = startCodePos[i][j + 1] - offset;
    //                 continue;
    //             case 0x13: // IDR
    //                 offset_idr = offset;
    //                 bytes_slices = startCodePos.back().back() - offset + 1;
    //                 finished = true;
    //                 break;
    //         }
    //     }
    //     if (finished) break;
    // }

    // heif muxer
    auto* writer = Writer::Create();
    OutputConfig writerOutputConf{};
    writerOutputConf.fileName        = outPath;
    writerOutputConf.progressiveFile = true;

    FourCC inputMajorBrand{"msf1"};
    writerOutputConf.majorBrand = inputMajorBrand;
    Array<FourCC> inputCompatibleBrands{"msf1", "heic", "hevc", "mif1", "iso8"};
    writerOutputConf.compatibleBrands = inputCompatibleBrands;
    // initialize writer now that we have all the needed information from reader
    if (writer->initialize(writerOutputConf) != ErrorCode::OK)
    {
        return -1;
    }

    DecoderConfiguration outputdecoderConfig{};
    outputdecoderConfig.decoderConfigId = 1;
    Array<DecoderSpecificInfo> decInfo(3);
    Array<uint8_t> vpsArray(bytes_vps), spsArray(bytes_sps), ppsArray(bytes_pps);
    uint8_t* sliceData = new uint8_t[bytes_slices];
    // uint8_t vpsData[bytes_vps], spsData[bytes_sps], ppsData[bytes_pps];

    memcpy(vpsArray.elements, pHevcBuf + offset_vps, bytes_vps);
    memcpy(spsArray.elements, pHevcBuf + offset_sps, bytes_sps);
    memcpy(ppsArray.elements, pHevcBuf + offset_pps, bytes_pps);
    memcpy(sliceData + 4, pHevcBuf + offset_idr + 4, bytes_slices);
    uint32_t nalLength = static_cast<uint32_t>(bytes_slices) - 4;
    sliceData[0] = static_cast<uint8_t>(nalLength >> 24 & 0xff);
    sliceData[1] = static_cast<uint8_t>(nalLength >> 16 & 0xff);
    sliceData[2] = static_cast<uint8_t>(nalLength >> 8 & 0xff);
    sliceData[3] = static_cast<uint8_t>(nalLength >> 0 & 0xff);
    // for (int i = 0; i < vPacket.size(); i++)
    // {
    //     size_t accu = 0;
    //     if (i == 0)
    //     {
    //         memcpy(sliceData, vPacket[i].data() + offset_idr, vPacket[i].size() - offset_idr);
    //         accu += vPacket[i].size() - offset_idr;
    //     } 
    //     else
    //     {
    //         memcpy(sliceData + accu, vPacket[i].data(), vPacket[i].size());
    //         accu += vPacket[i].size();
    //     }
    // }
    // VPS
    decInfo[0].decSpecInfoType = DecoderSpecInfoType::HEVC_VPS;
    decInfo[0].decSpecInfoData = vpsArray;
    // SPS
    decInfo[1].decSpecInfoType = DecoderSpecInfoType::HEVC_SPS;
    decInfo[1].decSpecInfoData = spsArray;
    // PPS
    decInfo[2].decSpecInfoType = DecoderSpecInfoType::HEVC_PPS;
    decInfo[2].decSpecInfoData = ppsArray;

    outputdecoderConfig.decoderSpecificInfo = decInfo;

    DecoderConfigId outputDecoderConfigId;
    writer->feedDecoderConfig(outputdecoderConfig.decoderSpecificInfo, outputDecoderConfigId);

    Data imageData{};
    imageData.size = bytes_slices;
    imageData.data = sliceData;
    // feed image data to writer
    imageData.mediaFormat     = MediaFormat::HEVC;
    imageData.decoderConfigId = outputDecoderConfigId;
    MediaDataId outputMediaId;
    writer->feedMediaData(imageData, outputMediaId);
    delete[] sliceData;

    // create new image based on that data:
    ImageId outputImageId;
    writer->addImage(outputMediaId, outputImageId);

    // if this input image was the primary image -> also mark output image as primary image
    writer->setPrimaryItem(outputImageId);

    writer->finalize();

    Writer::Destroy(writer);

    return 0;
}

int encodeImageSequenceFromYuv(const char* inPath, const char* outPath, CUcontext current)
{
    
    int frameWidth = 1280, frameHeight = 720;
    size_t frameSize = frameWidth * frameHeight * 3 / 2;
    BufferedFileReader reader(inPath);
    uint8_t *pNv12Buf, *dpNv12Buf;
    size_t nNv12BufSize;
    reader.GetBuffer(&pNv12Buf, &nNv12BufSize);
    ck(cudaMalloc(&dpNv12Buf, frameSize));
    ck(cudaMemcpy(dpNv12Buf, pNv12Buf, frameSize, cudaMemcpyHostToDevice));

    // Encoder setup
    NvEncoderInitParam initParam;
    NvEncLite *enc = nullptr;
    FILE *fpOut = nullptr;
    const char* szOutFilePath = "./bin/sample/heif/demo_720_out.hevc";
    init_encoder(current, frameWidth, frameHeight, szOutFilePath,
    initParam, enc, fpOut);

    vector<vector<uint8_t>> vPacket;
    enc->EncodeDeviceFrame(dpNv12Buf, 0, vPacket);

    int offset_vps, offset_sps, offset_pps;
    size_t bytes_vps, bytes_sps, bytes_pps, bytes_slices = 0;
    vector<vector<int>> offset_idr;
    vector<vector<int>> startCodePos;
    // find all start code positions
    for (int i = 0; i < vPacket.size(); i++)
    {
        vector<int> offsets;
        for (int offset = 0; offset < vPacket[i].size() - 4; offset++) 
        {
            uint32_t code = reinterpret_cast<uint32_t*>(vPacket[i].data() + offset)[0];
            bool isStartCode = (code == 0x1000000);
            if (isStartCode)
            {
                offsets.push_back(offset);
                if ((vPacket[i][offset + 4] >> 1 & 0x3f) == 0x13) break;
            }
            // else if (offset == (vPacket[i].size() - 1))
            // {
            //     offsets.push_back(offset + 1);
            // }
        }
        offsets.push_back(vPacket[i].size());
        startCodePos.push_back(offsets);
    }

    // identify nalu types
    // each frame can have multiple slices, but should only have one vps/sps/pps
    // all nal units after idr are treated as slices
    for (int i = 0; i < startCodePos.size(); i++)
    {
        vector<int> slicePos;
        // bool finished;
        for (int j =0; j < startCodePos[i].size(); j++)
        {
            int offset = startCodePos[i][j];
            if (offset >= vPacket[i].size()) break;
            uint8_t nalu_type = vPacket[i][offset + 4] >> 1 & 0x3f;
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
                case 0x1:  // P/B frame
                    slicePos.push_back(offset);
                    slicePos.push_back(startCodePos[i][j + 1]);
                    bytes_slices += startCodePos[i][j + 1] - offset;
                    // finished = true;
                    continue;
            }
        }
        offset_idr.push_back(slicePos);
        // if (finished) break;
    }

    // heif muxer
    auto* writer = Writer::Create();
    OutputConfig writerOutputConf{};
    writerOutputConf.fileName        = outPath;
    writerOutputConf.progressiveFile = true;

    FourCC inputMajorBrand{"msf1"};
    writerOutputConf.majorBrand = inputMajorBrand;
    Array<FourCC> inputCompatibleBrands{"hevc", "msf1"};
    writerOutputConf.compatibleBrands = inputCompatibleBrands;
    // initialize writer now that we have all the needed information from reader
    if (writer->initialize(writerOutputConf) != ErrorCode::OK)
    {
        return -1;
    }

    DecoderConfiguration outputdecoderConfig{};
    outputdecoderConfig.decoderConfigId = 1;
    Array<DecoderSpecificInfo> decInfo(3);
    Array<uint8_t> vpsArray(bytes_vps), spsArray(bytes_sps), ppsArray(bytes_pps);
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
            size_t copyBytes = offset_idr[i][j + 1] - offset_idr[i][j] - 4;
            memcpy(sliceAddr, vPacket[i].data() + offset_idr[i][j] + 4, copyBytes);
            sliceAddr += copyBytes;
        }
    }
    // copy parameter sets
    memcpy(vpsArray.elements, vPacket[0].data() + offset_vps, bytes_vps);
    memcpy(spsArray.elements, vPacket[0].data() + offset_sps, bytes_sps);
    memcpy(ppsArray.elements, vPacket[0].data() + offset_pps, bytes_pps);
    // for (int i = 0; i < vPacket.size(); i++)
    // {
    //     size_t accu = 0;
    //     // uint8_t* sliceStart = sliceData + 4;
    //     if (i == 0)
    //     {
    //         memcpy(sliceData + 4, vPacket[i].data() + offset_idr + 4, vPacket[i].size() - offset_idr);
    //         accu += vPacket[i].size() - offset_idr;
    //     } 
    //     else
    //     {
    //         memcpy(sliceData + 4 + accu, vPacket[i].data() + 4, vPacket[i].size());
    //         accu += vPacket[i].size();
    //     }
    // }

    // VPS
    decInfo[0].decSpecInfoType = DecoderSpecInfoType::HEVC_VPS;
    decInfo[0].decSpecInfoData = vpsArray;
    // SPS
    decInfo[1].decSpecInfoType = DecoderSpecInfoType::HEVC_SPS;
    decInfo[1].decSpecInfoData = spsArray;
    // PPS
    decInfo[2].decSpecInfoType = DecoderSpecInfoType::HEVC_PPS;
    decInfo[2].decSpecInfoData = ppsArray;

    outputdecoderConfig.decoderSpecificInfo = decInfo;

    DecoderConfigId outputDecoderConfigId;
    writer->feedDecoderConfig(outputdecoderConfig.decoderSpecificInfo, outputDecoderConfigId);

    // Add video track
    Rational tb{1, 1000};
    SequenceId seqId;
    SequenceId imgSeqId;
    writer->addVideoTrack(tb, seqId);
    CodingConstraints constrain{false, true, 15};
    writer->addImageSequence(tb, constrain, imgSeqId);

    Data imageData{};
    imageData.size = bytes_slices;
    imageData.data = sliceData;
    // feed image data to writer
    imageData.mediaFormat     = MediaFormat::HEVC;
    imageData.decoderConfigId = outputDecoderConfigId;
    MediaDataId outputMediaId;
    writer->feedMediaData(imageData, outputMediaId);
    delete[] sliceData;

    // create new image based on that data:
    ImageId outputImageId;
    // Assuming framerate is 25, duration of one frame is 40
    SampleInfo sampleInfo{40, 0, 1};
    SequenceImageId sampleId, seqImgId;
    writer->addVideo(seqId, outputMediaId, sampleInfo, sampleId);
    writer->addImage(imgSeqId, outputMediaId, sampleInfo, seqImgId);

    // if this input image was the primary image -> also mark output image as primary image
    writer->setPrimaryItem(outputImageId);

    for (int i = 0; i < 25; i ++) {
        ck(cudaMemcpy(dpNv12Buf, pNv12Buf + i * frameSize, frameSize, cudaMemcpyHostToDevice));

        vector<vector<uint8_t>> vPacket1;
        bytes_slices = 0;
        offset_idr.clear();
        startCodePos.clear();
        enc->EncodeDeviceFrame(dpNv12Buf, 0, vPacket1);
        for (int i = 0; i < vPacket1.size(); i++)
        {
            vector<int> offsets;
            for (int offset = 0; offset < vPacket1[i].size() - 4; offset++) 
            {
                uint32_t code = reinterpret_cast<uint32_t*>(vPacket1[i].data() + offset)[0];
                bool isStartCode = (code == 0x1000000);
                if (isStartCode)
                {
                    offsets.push_back(offset);
                    // if ((vPacket1[i][offset + 4] >> 1 & 0x3f) == 0x13) break;
                }
                // else if (offset == (vPacket1[i].size() - 1))
                // {
                //     offsets.push_back(offset + 1);
                // }
            }
            offsets.push_back(vPacket1[i].size());
            startCodePos.push_back(offsets);
        }

        // identify nalu types
        // each frame can have multiple slices, but should only have one vps/sps/pps
        // all nal units after idr are treated as slices
        for (int i = 0; i < startCodePos.size(); i++)
        {
            vector<int> slicePos;
            // bool finished;
            for (int j =0; j < startCodePos[i].size(); j++)
            {
                int offset = startCodePos[i][j];
                if (offset >= vPacket1[i].size()) break;
                uint8_t nalu_type = vPacket1[i][offset + 4] >> 1 & 0x3f;
                switch (nalu_type)
                {
                    // case 0x20: // VPS
                    //     offset_vps = offset;
                    //     bytes_vps = startCodePos[i][j + 1] - offset;
                    //     continue;
                    // case 0x21: // SPS
                    //     offset_sps = offset;
                    //     bytes_sps = startCodePos[i][j + 1] - offset;
                    //     continue;
                    // case 0x22: // PPS
                    //     offset_pps = offset;
                    //     bytes_pps = startCodePos[i][j + 1] - offset;
                    //     continue;
                    case 0x13: // IDR
                    case 0x1:  // P/B frame
                        slicePos.push_back(offset);
                        slicePos.push_back(startCodePos[i][j + 1]);
                        bytes_slices += startCodePos[i][j + 1] - offset;
                        // finished = true;
                        continue;
                }
            }
            offset_idr.push_back(slicePos);
            // if (finished) break;
        }
        sliceData = new uint8_t[bytes_slices];
        nalLength = static_cast<uint32_t>(bytes_slices) - 4;
        sliceData[0] = static_cast<uint8_t>(nalLength >> 24 & 0xff);
        sliceData[1] = static_cast<uint8_t>(nalLength >> 16 & 0xff);
        sliceData[2] = static_cast<uint8_t>(nalLength >> 8 & 0xff);
        sliceData[3] = static_cast<uint8_t>(nalLength >> 0 & 0xff);
        // copy slices
        sliceAddr = sliceData + 4;
        for (int i = 0; i < offset_idr.size(); i++) {
            for (int j = 0; j < offset_idr[i].size(); j+=2) {
                size_t copyBytes = offset_idr[i][j + 1] - offset_idr[i][j] - 4;
                memcpy(sliceAddr, vPacket1[i].data() + offset_idr[i][j] + 4, copyBytes);
                sliceAddr += copyBytes;
            }
        }

        imageData.size = bytes_slices;
        imageData.data = sliceData;
        // feed image data to writer
        imageData.mediaFormat     = MediaFormat::HEVC;
        imageData.decoderConfigId = outputDecoderConfigId;
        writer->feedMediaData(imageData, outputMediaId);
        delete[] sliceData;

        writer->addVideo(seqId, outputMediaId, sampleInfo, sampleId);
        writer->addImage(imgSeqId, outputMediaId, sampleInfo, seqImgId);

    }

    writer->finalize();

    Writer::Destroy(writer);

    return 0;
}
void encodeClass(const char* inPath, char* outPath, CUcontext current)
{
    int frameWidth = 1920, frameHeight = 1080;
    BufferedFileReader reader(inPath);
    uint8_t *pNv12Buf, *dpNv12Buf;
    size_t nNv12BufSize;
    reader.GetBuffer(&pNv12Buf, &nNv12BufSize);
    ck(cudaMalloc(&dpNv12Buf, nNv12BufSize));
    ck(cudaMemcpy(dpNv12Buf, pNv12Buf, nNv12BufSize, cudaMemcpyHostToDevice));

    // Encoder setup
    NvEncoderInitParam initParam;
    NvEncLite *enc = nullptr;
    // FILE *fpOut = nullptr;
    // const char* szOutFilePath = "./bin/sample/heif/bus_720_out.hevc";
    init_encoder(current, frameWidth, frameHeight,
    initParam, enc, true);

    int N_RUNS = 1000;
    std::vector<std::thread> threadVector(N_RUNS);
    StopWatch w, w2;
    double heifTime = 0;
    auto clock_start = chrono::steady_clock::now();
    w.Start();
    NvHeifWriter heifWriter{outPath};
    for (int i = 0; i < N_RUNS; i++) {
        vector<vector<uint8_t>> vPacket;
        enc->EncodeDeviceFrame(dpNv12Buf, 0, vPacket);

        // w2.Start();
        
        // std::thread heifWriterThread(heifWriter.write, vPacket, outPath);
        // threadVector.emplace_back([=](){NvHeifWriter heifWriter{outPath};heifWriter.writeStillImage(vPacket);});
        heifWriter.writeStillImage(vPacket);
        // heifTime += w2.Stop();
    }
    for (auto &thread : threadVector) {
        if (thread.joinable()) thread.join();
    }
    auto clock_stop = chrono::steady_clock::now();
    double t = w.Stop();
    chrono::duration<double> diff_clock = clock_stop - clock_start;
    cout << "Average FPS of " << N_RUNS << " runs: " << N_RUNS / diff_clock.count() << ", average latency:" << diff_clock.count() / N_RUNS << " sec\n";
    // cout << "Average FPS of " << N_RUNS << " runs (heif): "<< heifTime / N_RUNS << "\n";

    delete enc;
    ck(cudaFree(dpNv12Buf));
}
void encodeClassSequence(const char* inPath, char* outPath, CUcontext current)
{
    int frameWidth = 1920, frameHeight = 1080;
    BufferedFileReader reader(inPath);
    uint8_t *pNv12Buf, *dpNv12Buf;
    size_t nNv12BufSize;
    reader.GetBuffer(&pNv12Buf, &nNv12BufSize);
    ck(cudaMalloc(&dpNv12Buf, nNv12BufSize));
    ck(cudaMemcpy(dpNv12Buf, pNv12Buf, nNv12BufSize, cudaMemcpyHostToDevice));

    // Encoder setup
    NvEncoderInitParam initParam;
    NvEncLite *enc = nullptr;
    // FILE *fpOut = nullptr;
    // const char* szOutFilePath = "./bin/sample/heif/bus_720_out.hevc";
    init_encoder(current, frameWidth, frameHeight,
    initParam, enc, true);

    int N_RUNS = 1000;
    std::vector<std::thread> threadVector(N_RUNS);
    StopWatch w, w2;
    double heifTime = 0;
    auto clock_start = chrono::steady_clock::now();
    w.Start();
    NvHeifWriter heifWriter{outPath, false, NV_ENC_CODEC_HEVC_GUID};
    for (int i = 0; i < 100; i++) {
        vector<vector<uint8_t>> vPacket;
        enc->EncodeDeviceFrame(dpNv12Buf, 0, vPacket);

        // w2.Start();
        
        // std::thread heifWriterThread(heifWriter.write, vPacket, outPath);
        heifWriter.addImageToSequence(vPacket, false);
        // heifWriter.write(vPacket, outPath);
        // heifTime += w2.Stop();
    }
    heifWriter.writeSequence();
    auto clock_stop = chrono::steady_clock::now();
    double t = w.Stop();
    chrono::duration<double> diff_clock = clock_stop - clock_start;
    cout << "Average FPS of " << N_RUNS << " runs: " << N_RUNS / diff_clock.count() << ", average latency:" << diff_clock.count() / N_RUNS << " sec\n";
    // cout << "Average FPS of " << N_RUNS << " runs (heif): "<< heifTime / N_RUNS << "\n";

    delete enc;
    ck(cudaFree(dpNv12Buf));
}
int main(){
    cudaSetDevice(0);
    CUcontext current;
    ck(cuDevicePrimaryCtxRetain(&current, 0));
    ck(cuCtxPushCurrent(current));

    // encodeFromYuv("./bin/sample/heif/bus_720.yuv", "bus_720_out.heic", current);
    // encodeFromHevc("/workspace/ffmpeg-gpu-demo/heif/Bins/one_face_out.hevc", "one_face_1440x960.heic", current);
    encodeClass("./bin/sample/heif/bus_1080.yuv", "bus_720_out.heic", current);
    // encodeImageSequenceFromYuv("../ffmpeg-gpu/output/demo_JHH_720.yuv", "demo_JHH_720_video.heif", current);
    // encodeClassSequence("../ffmpeg-gpu/output/demo_JHH_720.yuv", "demo_JHH_720_video.heif", current);
}