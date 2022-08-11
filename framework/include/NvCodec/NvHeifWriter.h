
#ifndef WIN32
#include <dlfcn.h>
#endif
#include <vector>
#include <heif/reader/heifreader.h>
#include <heif/writer/heifwriter.h>

#include "NvCodec/NvEncLite.h"
#include "NvCodec/NvCommon.h"

extern simplelogger::Logger *logger;

class NvHeifWriter {
public:
    NvHeifWriter(char* outFilePath);
    NvHeifWriter(char* outFilePath, bool stillImage, GUID codec);
    ~NvHeifWriter();
    bool writeStillImage(std::vector<std::vector<uint8_t>> nalUnits, bool useLastParameterSet = true);
    bool addImageToSequence(std::vector<std::vector<uint8_t>> nalUnits, bool primaryImage, bool useLastParameterSet = true);
    bool writeSequence();

private:
    HEIF::Writer* writer;
    HEIF::Array<uint8_t> *vpsArray, *spsArray, *ppsArray;
    bool initParameterSets;
    HEIF::Array<HEIF::DecoderSpecificInfo> decoderInfo;
    HEIF::FourCC majorBrand;
    HEIF::Array<HEIF::FourCC> compatibleBrands;
    HEIF::MediaFormat format;
    HEIF::SequenceId vidTrackId, imgSeqId;
    HEIF::DecoderConfigId outputDecoderConfigId;
};