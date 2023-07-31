#include <libavutil/log.h>
#include <vector>
#include <thread>

#include "NvCodec/NvCommon.h"
#include "NvCodec/NvEncLite.h"
#include "NvCodec/NvDecLite.h"
#include "NvCodec/NvHeifWriter.h"
#include "NvCodec/NvHeifReader.h"

#include <cuda_runtime.h>
#include <heif/reader/heifreader.h>
#include <heif/writer/heifwriter.h>

using namespace std;
using namespace HEIF;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger(simplelogger::INFO);

extern "C" {

vector<vector<uint8_t>>* Create_PktVector() {
    return new vector<vector<uint8_t>>();
}

void Delete_PktVector(void* vPkt) {
    vector<vector<uint8_t>>* vPacket = static_cast<vector<vector<uint8_t>>*>(vPkt);
    delete vPacket;
}

NvEncLite* NvEncLite_InitStill(int nWidth, int nHeight) {
    CUcontext cuContext = 0;
    ck(cuCtxGetCurrent(&cuContext));
    if (!cuContext) {
        LOG(ERROR) << "No CUDA context in current thread";
        return nullptr;
    }
    // std::string init_param_string{initParamString};
    NvEncoderInitParam initParam {"-codec hevc -preset p1 -bitrate 4M"};
    // NvEncoderInitParam initParam {"-codec hevc -tune lossless"};
    NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_NV12;
    NvEncLite *enc = new NvEncLite(cuContext, nWidth, nHeight, eFormat, &initParam, 0, true);
    return enc;
}
void NvEncLite_Delete(NvEncLite* enc) {
    delete enc;
}

// int NvEncLite_EncodeDeviceFrame(NvEncLite* enc, uint8_t* dpFrame, void* vPkt) {
//     vector<vector<uint8_t>>* vPacket = static_cast<vector<vector<uint8_t>>*>(vPkt);
//     return enc->EncodeDeviceFrame(dpFrame, 0, *vPacket);
// }

int NvEncLite_EncodeDeviceFrame(NvEncLite* enc, uint8_t* dpFrame, void* vPkt) {
    vector<vector<uint8_t>>* vPacket = static_cast<vector<vector<uint8_t>>*>(vPkt);
    vector<vector<uint8_t>> p;
    int nFrameEncoded = 0;
    nFrameEncoded = enc->EncodeDeviceFrame(dpFrame, 0, *vPacket);
    nFrameEncoded += enc->EncodeDeviceFrame(NULL, 0, p);
    vPacket->insert(vPacket->end(), p.begin(), p.end());
    return nFrameEncoded;
}

NvHeifWriter* NvHeifWriter_Init() {
    return new NvHeifWriter();
}

void NvHeifWriter_Delete(NvHeifWriter* writer) {
    // LOG(INFO) << "Delete HEIF writer";
    delete writer;
}

int NvHeifWriter_WriteStillImage(NvHeifWriter* writer, void* vPkt) {
    vector<vector<uint8_t>>* vPacket = static_cast<vector<vector<uint8_t>>*>(vPkt);
    return writer->writeStillImage(*vPacket);
}

uint8_t* NvHeifWriter_GetBufferData(NvHeifWriter* writer) {
    return writer->getBufferData();
}

uint64_t NvHeifWriter_GetBufferSize(NvHeifWriter* writer) {
    return writer->getBufferSize();
}

void NvHeifWriter_WriteToNp(NvHeifWriter* writer, uint8_t* np_data) {
    memcpy(np_data, writer->getBufferData(), writer->getBufferSize());
}

NvHeifReader* NvHeifReader_Init(uint8_t* buffer_ptr, int64_t size) {
    return new NvHeifReader(buffer_ptr, size);
}

size_t NvHeifReader_ReadImage(NvHeifReader* reader, uint8_t** pktData) {
    size_t pktSize;
    reader->readImage(*pktData, pktSize);
    return pktSize;
}

void NvHeifReader_Delete(NvHeifReader* reader) {
    // LOG(INFO) << "Delete HEIF reader";
    delete reader;
}

NvDecLite* NvDecLite_Init() {
    CUcontext cuContext = 0;
    ck(cuCtxGetCurrent(&cuContext));
    if (!cuContext) {
        LOG(ERROR) << "No CUDA context in current thread";
        return nullptr;
    }
    return new NvDecLite(cuContext, true, cudaVideoCodec_HEVC, true, false);
}

void NvDecLite_Delete(NvDecLite* dec) {
    delete dec;
}

int NvDecLite_DecodeStill(NvDecLite* dec, const uint8_t **pktData, int pktSize, uint8_t **pFrame, int* width, int* height, int* linesize) {
    NvFrameInfo *pInfo = NULL;
    uint8_t **ppFrame = NULL;
    int nFrameReturned = dec->Decode(*pktData, pktSize, &ppFrame, &pInfo, CUVID_PKT_ENDOFPICTURE);
    if (nFrameReturned == 0) {
        nFrameReturned = dec->Decode(*pktData, pktSize, &ppFrame, &pInfo);
    }

    *width = pInfo[0].nWidth;
    *height = pInfo[0].nHeight;
    *linesize = pInfo[0].nFramePitch;
    *pFrame = ppFrame[0];

    return nFrameReturned;
}

} // extern "C"