#include <cstdint>
#include <cstring>
#ifndef WIN32
#include <dlfcn.h>
#endif
#include <vector>
#include <heif/reader/heifreader.h>
#include <heif/writer/heifwriter.h>
#include <heif/common/heifstreaminterface.h>

#include "NvCodec/NvEncLite.h"
#include "NvCodec/NvCommon.h"

extern simplelogger::Logger *logger;

class StreamBufferIn : public HEIF::StreamInterface
{
public:
    StreamBufferIn (uint8_t* buffer_ptr, int64_t size) : buffer(buffer_ptr), bufferSize(size) {}
    ~StreamBufferIn() {}

    int64_t read(char* buffer, int64_t size) {
        if (position >= bufferSize) {
            return 0;
        }
        if (size + position > bufferSize) {
            size = bufferSize - position;
        }
        else if (size + position < 0) {
            position = 0;
        }
        memcpy(buffer, this->buffer + position, size);
        position += size;
        return size;
    }

    bool absoluteSeek(offset_t offset) {
        position = offset;
        return true;
    }

    offset_t tell() {
        return position;
    }
    offset_t size() {
        return bufferSize;
    }

private:
    uint8_t* buffer;
    int64_t bufferSize;
    int64_t position = 0;
};

class NvHeifReader {
    public:
    NvHeifReader(uint8_t* buffer_ptr, int64_t size);
    NvHeifReader(const char* outFilePath);
    ~NvHeifReader();

    bool readImage(uint8_t* &pktData, size_t &pktBytes);

    bool readVideoFrame(std::vector<uint8_t*> &v_pktData, std::vector<size_t> &v_pktBytes);

private:
    HEIF::Reader *m_reader;
    HEIF::FileInformation m_fileInfo{};
    HEIF::TrackInformation m_trackInfo;
    uint8_t* m_pktData;
    int m_index;
    StreamBufferIn* m_buffer;
};