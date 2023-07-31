
#ifndef WIN32
#include <dlfcn.h>
#endif
#include <vector>
#include <cstdlib>
#include <heif/reader/heifreader.h>
#include <heif/writer/heifwriter.h>

#include "NvCodec/NvEncLite.h"
#include "NvCodec/NvCommon.h"

extern simplelogger::Logger *logger;

class StreamBufferOut : public HEIF::OutputStreamInterface
{
public:
    StreamBufferOut(uint8_t* outputBuffer) : outputBuffer(outputBuffer){}
    StreamBufferOut() {
        // streambuf using std::vector
        // bufferVector.resize(2 << 15, 0);
        // outputBuffer = bufferVector.data();

        outputBuffer = static_cast<uint8_t*>(std::malloc(2 << 15));
        if (outputBuffer == nullptr) { throw std::bad_alloc(); }
        bufSize = 2 << 15;
        ownBuffer = true;
    }
    void seekp(std::uint64_t position) {
        this->position = position;
    }
    std::uint64_t tellp() {
        return position;
    }
    // streambuf using std::vector
    // void write(const void* buffer, std::uint64_t size) {
    //     if (ownBuffer && (size + position) >= bufferVector.size()) {
    //         bufferVector.resize(size + position, 0);
    //         outputBuffer = bufferVector.data();
    //     }
    //     memcpy(outputBuffer + position, buffer, size);
    //     position += size;
    //     return;
    // }
    void write(const void* buffer, std::uint64_t size) {
        if (ownBuffer && (size + position) > bufSize) {
            void* newPtr = std::realloc(outputBuffer, size + position);
            if (newPtr == nullptr) { throw std::bad_alloc(); }
            outputBuffer = static_cast<uint8_t*>(newPtr);
            bufSize = size + position;
        }
        memcpy(outputBuffer + position, buffer, size);
        position += size;
        return;
    }
    void remove() {
        position = 0;
        return;
    };
    uint8_t* getOutputBuffer() { return outputBuffer; }
    std::uint64_t getPosition() { return position; }
    ~StreamBufferOut() {
        if (ownBuffer) {
            // delete[] outputBuffer;
            std::free(outputBuffer);
        }
    }

private:
    bool ownBuffer = false;
    // std::vector<uint8_t> bufferVector;
    uint8_t* outputBuffer;
    size_t bufSize;
    std::uint64_t position = 0;
};

class NvHeifWriter {
public:
    NvHeifWriter();
    NvHeifWriter(char* outFilePath);
    NvHeifWriter(char* outFilePath, bool stillImage, GUID codec);
    ~NvHeifWriter();
    bool writeStillImage(std::vector<std::vector<uint8_t>> nalUnits, bool useLastParameterSet = true);
    bool addImageToSequence(std::vector<std::vector<uint8_t>> nalUnits, bool primaryImage, bool useLastParameterSet = true);
    bool writeSequence();
    uint8_t* getBufferData();
    uint64_t getBufferSize();
private:
    HEIF::Writer* writer;
    HEIF::Array<uint8_t> *vpsArray = nullptr, *spsArray = nullptr, *ppsArray = nullptr;
    bool initParameterSets;
    HEIF::Array<HEIF::DecoderSpecificInfo> decoderInfo;
    HEIF::FourCC majorBrand;
    HEIF::Array<HEIF::FourCC> compatibleBrands;
    HEIF::MediaFormat format;
    HEIF::SequenceId vidTrackId, imgSeqId;
    HEIF::DecoderConfigId outputDecoderConfigId;

    StreamBufferOut* buffer;
};