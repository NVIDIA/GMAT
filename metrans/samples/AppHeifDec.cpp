#include <vector>
#include <thread>

#include "NvCodec/NvCommon.h"
#include "NvCodec/NvDecLite.h"
#include "NvCodec/NvHeifReader.h"
// #include "Utils.h"

#include <cuda_runtime.h>
#include <heif/reader/heifreader.h>
#include <heif/writer/heifwriter.h>

using namespace std;
using namespace HEIF;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger(simplelogger::TRACE);

void demuxDecodeImage(const char* inPath, CUcontext current) {
    int frameWidth = 1920, frameHeight = 1080;
    // BufferedFileReader reader(inPath);
    // uint8_t *pNv12Buf, *dpNv12Buf;
    // size_t nNv12BufSize;
    // reader.GetBuffer(&pNv12Buf, &nNv12BufSize);
    // ck(cudaMalloc(&dpNv12Buf, nNv12BufSize));
    // ck(cudaMemcpy(dpNv12Buf, pNv12Buf, nNv12BufSize, cudaMemcpyHostToDevice));

    NvDecLite dec(current, false, cudaVideoCodec_HEVC, false, false);
    
    auto* reader = Reader::Create();
    if (reader->initialize(inPath) != ErrorCode::OK)
    {
        Reader::Destroy(reader);
        return;
    }

    // get information about all input file content
    FileInformation fileInfo{};
    reader->getFileInformation(fileInfo);

    // Verify that the file has one or several images in the MetaBox
    if (!(fileInfo.features & FileFeatureEnum::HasSingleImage || fileInfo.features & FileFeatureEnum::HasImageCollection))
    {
        cout << "No image found\n";
        return;
    }
    // Array<ImageId> itemIds;
    // reader->getMasterImages(itemIds);
    // getDecoderCodeType(const ImageId& imageId, FourCC& type) const = 0;
    
    // FourCC hvc1{"hvc1"}, hev1{"hev1"};
    for (const auto& image : fileInfo.rootMetaBoxInformation.itemInformations) {
        if (image.features & ItemFeatureEnum::IsMasterImage) {
            if (image.type != FourCC("hvc1") && image.type != FourCC("hev1")) {
                LOG(ERROR) << "Only supports hevc image\n";
                return;
            }

            DecoderConfiguration inputdecoderConfig{};
            reader->getDecoderParameterSets(image.itemId, inputdecoderConfig);

            size_t parameterSetBytes = 0;
            for (int i = 0; i < inputdecoderConfig.decoderSpecificInfo.size; i++) {
                parameterSetBytes += inputdecoderConfig.decoderSpecificInfo[i].decSpecInfoData.size;
            }

            size_t pktBytes = parameterSetBytes + image.size;
            uint8_t *pktData = new uint8_t[pktBytes];

            uint8_t *psData = pktData;
            for (int i = 0; i < inputdecoderConfig.decoderSpecificInfo.size; i++) {
                memcpy(psData, 
                       inputdecoderConfig.decoderSpecificInfo[i].decSpecInfoData.elements, 
                       inputdecoderConfig.decoderSpecificInfo[i].decSpecInfoData.size);
                psData += inputdecoderConfig.decoderSpecificInfo[i].decSpecInfoData.size;
            }

            reader->getItemData(image.itemId, pktData + parameterSetBytes, pktBytes);
            uint8_t **ppFrame = NULL;
            NvFrameInfo *pInfo = NULL;
            int nFrameReturned = dec.Decode(pktData, pktBytes, &ppFrame, &pInfo);
        }
        
    }

}

void demuxDecodeImageSequence(const char* inPath, CUcontext current) {
    int frameWidth = 1920, frameHeight = 1080;
    // BufferedFileReader reader(inPath);
    // uint8_t *pNv12Buf, *dpNv12Buf;
    // size_t nNv12BufSize;
    // reader.GetBuffer(&pNv12Buf, &nNv12BufSize);
    // ck(cudaMalloc(&dpNv12Buf, nNv12BufSize));
    // ck(cudaMemcpy(dpNv12Buf, pNv12Buf, nNv12BufSize, cudaMemcpyHostToDevice));

    NvDecLite dec(current, false, cudaVideoCodec_HEVC, false, false);
    ofstream fOut("out.yuv", ios::out | ios::binary);
    
    auto* reader = Reader::Create();
    if (reader->initialize(inPath) != ErrorCode::OK)
    {
        Reader::Destroy(reader);
        return;
    }

    // get information about all input file content
    FileInformation fileInfo{};
    reader->getFileInformation(fileInfo);

    // Verify that the file has one or several images in the MetaBox
    if (!(fileInfo.features & FileFeatureEnum::HasImageSequence))
    {
        cout << "No image found\n";
        return;
    }
    // Array<ImageId> itemIds;
    // reader->getMasterImages(itemIds);
    // getDecoderCodeType(const ImageId& imageId, FourCC& type) const = 0;
    
    // FourCC hvc1{"hvc1"}, hev1{"hev1"};
    // for (const auto& image : fileInfo.rootMetaBoxInformation.itemInformations) {
    for (const auto& trackProperties : fileInfo.trackInformation) {
        const auto sequenceId = trackProperties.trackId;
        cout << "Track ID " << sequenceId.get() << endl;  // Context ID corresponds to the track ID

        if (trackProperties.features & TrackFeatureEnum::IsMasterImageSequence || trackProperties.features & TrackFeatureEnum::IsVideoTrack)
        {
            cout << "This is a master image sequence / video sequence." << endl;

            for (const auto& sampleProperties : trackProperties.sampleProperties) {
                // A sample might have decoding dependencies. The simplest way to handle this is just to always ask and
                // decode all dependencies.
                Array<SequenceImageId> itemsToDecode;
                reader->getDecodeDependencies(sequenceId, sampleProperties.sampleId, itemsToDecode);
                for (auto dependencyId : itemsToDecode)
                {
                    uint64_t size    = 1024 * 1024;
                    auto* sampleData = new uint8_t[size];
                    reader->getItemDataWithDecoderParameters(sequenceId, dependencyId, sampleData, size);

                    uint8_t **ppFrame = NULL;
                    NvFrameInfo *pInfo = NULL;
                    int nFrameReturned = dec.Decode(sampleData, size, &ppFrame, &pInfo);
                    for (int i = 0; i < nFrameReturned; i++) {
                        fOut.write(reinterpret_cast<char*>(ppFrame[i]), pInfo[i].nFrameSize);

                        // Feed data to decoder...

                        delete[] sampleData;
                    }
                    // Store or show the image...
                }
            }
            break;
        }        
    }
}

void demuxDecodeImageClass(const char* inPath, CUcontext current) {
    NvHeifReader heifReader(inPath);
    NvDecLite dec(current, false, cudaVideoCodec_HEVC, true, false);
    ofstream fOut("image_out.yuv", ios::out | ios::binary);

    uint8_t *pktData = nullptr;
    size_t pktSize = 0;

    StopWatch s;
    size_t N_RUNS = 1000;
    s.Start();
    for (int i = 0; i < N_RUNS; i++){
    heifReader.readImage(pktData, pktSize);
    // cout << "Packet size: " << pktSize << " bytes.\n";
    
    uint8_t **ppFrame = NULL;
    NvFrameInfo *pInfo = NULL;
    int nFrameReturned = dec.Decode(pktData, pktSize, &ppFrame, &pInfo, CUVID_PKT_ENDOFPICTURE);
    if (nFrameReturned == 0) {
        nFrameReturned = dec.Decode(nullptr, 0, &ppFrame, &pInfo);
    }
    for (int i = 0; i < nFrameReturned; i++) {
        fOut.write(reinterpret_cast<char*>(ppFrame[i]), pInfo[i].nFrameSize);
    }
    }
    double t = s.Stop();
    cout << "FPS of " << N_RUNS << " runs: " << N_RUNS / t << endl;
}

void demuxDecodeImageSequenceClass(const char* inPath, CUcontext current) {
    NvHeifReader heifReader(inPath);
    NvDecLite dec(current, false, cudaVideoCodec_HEVC, false, false);
    ofstream fOut("image_sequence_out.yuv", ios::out | ios::binary);

    uint8_t *pktData = nullptr;
    size_t pktSize = 0;
    int nFrame = 0;

    do {
        vector<uint8_t*> v_pktData;
        vector<size_t> v_pktSize;
        heifReader.readVideoFrame(v_pktData, v_pktSize);
        for (int i = 0; i < v_pktData.size(); i++) {
            uint8_t **ppFrame = NULL;
            NvFrameInfo *pInfo = NULL;
            int nFrameReturned = dec.Decode(v_pktData[i], v_pktSize[i], &ppFrame, &pInfo);
            for (int k = 0; k < nFrameReturned; k++) {
                fOut.write(reinterpret_cast<char *>(ppFrame[k]), pInfo[k].nFrameSize);
            }
            nFrame += nFrameReturned;

            delete[] v_pktData[i];
        }

        if (v_pktData.size() == 0) {
            uint8_t **ppFrame = NULL;
            NvFrameInfo *pInfo = NULL;
            int nFrameReturned = dec.Decode(nullptr, 0, &ppFrame, &pInfo);
            for (int k = 0; k < nFrameReturned; k++) {
                fOut.write(reinterpret_cast<char *>(ppFrame[k]), pInfo[k].nFrameSize);
            }
            nFrame += nFrameReturned;

            break;
        }
    } while (true);

    // heifReader.readImage(pktData, pktSize);
    // cout << "Packet size: " << pktSize << " bytes.\n";
    
    // uint8_t **ppFrame = NULL;
    // NvFrameInfo *pInfo = NULL;
    // int nFrameReturned = dec.Decode(pktData, pktSize, &ppFrame, &pInfo);
    // if (nFrameReturned == 0) {
    //     nFrameReturned = dec.Decode(nullptr, 0, &ppFrame, &pInfo);
    // }
    // for (int i = 0; i < nFrameReturned; i++) {
    //     fOut.write(reinterpret_cast<char*>(ppFrame[i]), pInfo[i].nFrameSize);
    // }
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    CUcontext current;
    ck(cuDevicePrimaryCtxRetain(&current, 0));
    ck(cuCtxPushCurrent(current));

    // demuxDecodeImage("../heif/heif_conformance/conformance_files/C002.heic", current);
    // demuxDecodeImageSequence("./demo_JHH_720_video.heif", current);
    demuxDecodeImageClass("../heif/heif_conformance/conformance_files/C002.heic", current);
    // demuxDecodeImageSequenceClass("./starfield_animation.heic", current);
}