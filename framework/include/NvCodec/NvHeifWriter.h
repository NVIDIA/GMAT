
#ifndef WIN32
#include <dlfcn.h>
#endif
#include <vector>
#include <heif/heifreader.h>
#include <heif/heifwriter.h>

#include "NvCodec/NvEncLite.h"
#include "NvCodec/NvCommon.h"

extern simplelogger::Logger *logger;

class NvHeifWriter {
public:
    NvHeifWriter();
    NvHeifWriter(GUID codec, bool stillImage);
    bool write(std::vector<std::vector<uint8_t>> &nalUnits, char* outFilePath, bool useLastParameterSet = true);

private:
    HEIF::Writer* writer;
    HEIF::Array<uint8_t> *vpsArray, *spsArray, *ppsArray;
    bool initParameterSets;
    HEIF::Array<HEIF::DecoderSpecificInfo> decoderInfo;
    HEIF::FourCC majorBrand;
    HEIF::Array<HEIF::FourCC> compatibleBrands;
    HEIF::MediaFormat format;
};