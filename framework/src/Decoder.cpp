#include "Decoder.h"

namespace ffgd{
    FileReader::FileReader(std::string filePath, bool keepAudio) {
        reader = new BufferedFileReader{filePath.c_str()};
        reader.GetBuffer(&pBuf, &nBufSize);
        demuxer = new Demuxer{pBuf, nBufSize, keepAudio};
    }

    FileReader::~FileReader() {
        delete reader;
        delete demuxer;
    }
}