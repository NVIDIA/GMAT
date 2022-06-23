#include <vector>

#include "NvCodec/NvCommon.h"
#include "NvCodec/NvEncLite.h"
#include "NvCodec/NvHeifWriter.h"

#include <cuda_runtime.h>
#include <heif/heifreader.h>
#include <heif/heifwriter.h>

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

    std::string init_param_string{"-codec hevc -preset p3 -bitrate 2M"};
    initParam = NvEncoderInitParam(init_param_string.c_str());
    NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_NV12;

    enc = new NvEncLite(cuContext, nWidth, nHeight, eFormat, &initParam);
}

void init_encoder(CUcontext cuContext, int nWidth, int nHeight, 
    NvEncoderInitParam &initParam, NvEncLite* &enc) {
    std::string init_param_string{"-codec hevc -preset p7 -bitrate 2M"};
    initParam = NvEncoderInitParam(init_param_string.c_str());
    NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_NV12;

    enc = new NvEncLite(cuContext, nWidth, nHeight, eFormat, &initParam);
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

void encodeClass(const char* inPath, char* outPath, CUcontext current)
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
    // FILE *fpOut = nullptr;
    // const char* szOutFilePath = "./bin/sample/heif/bus_720_out.hevc";
    init_encoder(current, frameWidth, frameHeight,
    initParam, enc);

    int N_RUNS = 300;
    StopWatch w;
    auto clock_start = chrono::steady_clock::now();
    w.Start();
    for (int i = 0; i < N_RUNS; i++) {
        vector<vector<uint8_t>> vPacket;
        enc->EncodeDeviceFrame(dpNv12Buf, 0, vPacket);

        NvHeifWriter heifWriter{};
        heifWriter.write(vPacket, outPath);
    }
    auto clock_stop = chrono::steady_clock::now();
    double t = w.Stop();
    chrono::duration<double> diff_clock = clock_stop - clock_start;
    cout << "Average latency of " << N_RUNS << " runs: "<< diff_clock.count() / N_RUNS << " s\n";
    cout << "Average latency of " << N_RUNS << " runs (Stopwatch): "<< t / N_RUNS << " s\n";
}

int main(){
    cudaSetDevice(0);
    CUcontext current;
    ck(cuDevicePrimaryCtxRetain(&current, 0));
    ck(cuCtxPushCurrent(current));

    // encodeFromYuv("./bin/sample/heif/bus_720.yuv", "bus_720_out.heic", current);
    // encodeFromHevc("/workspace/ffmpeg-gpu-demo/heif/Bins/one_face_out.hevc", "one_face_1440x960.heic", current);
    encodeClass("./bin/sample/heif/bus_720.yuv", "bus_720_out.heic", current);
}