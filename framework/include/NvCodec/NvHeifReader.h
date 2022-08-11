#ifndef WIN32
#include <dlfcn.h>
#endif
#include <vector>
#include <heif/reader/heifreader.h>
#include <heif/writer/heifwriter.h>

#include "NvCodec/NvEncLite.h"
#include "NvCodec/NvCommon.h"

extern simplelogger::Logger *logger;

class NvHeifReader {
    public:
    NvHeifReader(char* outFilePath);
    ~NvHeifReader();

    bool readImage(uint8_t* &pktData, size_t &pktBytes);

    bool readVideoFrame();

private:
    HEIF::Reader *m_reader;
    HEIF::FileInformation m_fileInfo{};
    HEIF::TrackInformation m_trackInfo;
    uint8_t* m_pktData;
    int m_index;
};