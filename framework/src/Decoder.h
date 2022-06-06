#pragma once

#include <string>
#include <cuda_runtime.h>
#include <torch/torch.h>

// #include "NvCodec/NvDecoderImageProvider.h"
#include "AvToolkit/Demuxer.h"
#include "NvCodec/NvCommon.h"

namespace ffgd {
    enum PixelFormat {
        
    };
    class Image {
        public:
        Image(torch::Tensor ten, );
    }

    class FileReader {
        public:
        FileReader(std::string filePath, bool keepAudio=false);
        ~FileReader();

        private:
        uint8_t *pBuf;
        size_t nBufSize;
        Demuxer *demuxer;
        BufferedFileReader *reader;
    };

    class Decoder {
        public:
        Decoder(FileReader &reader, int batchSize=1);
        ~Decoder();

        uint32_t GetFrameSize();
        uint32_t GetWidth();
        uint32_t GetHeight();
        torch::Tensor GetNextImageAsTensor(torch::Tensor ten, PixelFormat fmt);

    };
}