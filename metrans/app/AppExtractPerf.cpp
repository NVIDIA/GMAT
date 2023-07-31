#include "FrameExtractor.h"
#include "NvCodec/NvCommon.h"
#include "NvCodec/NvDecLite.h"
#include "NvCodec/NvHeifWriter.h"
#include "NvCodec/NvEncLite.h"
#include <iostream>
#include <stdint.h>
#include <experimental/filesystem>
#include <string>

using namespace std;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger(simplelogger::INFO);

void init_encoder(CUcontext cuContext, int nWidth, int nHeight, 
    NvEncoderInitParam &initParam, NvEncLite* &enc, bool stillImage=false) {
    std::string init_param_string{"-codec hevc -preset p1 -bitrate 2M"};
    initParam = NvEncoderInitParam(init_param_string.c_str());
    NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_NV12;

    enc = new NvEncLite(cuContext, nWidth, nHeight, eFormat, &initParam, 0, stillImage);
}

// void encodeHeif(const char* inPath, char* outPath, CUcontext current)
// {
//     int frameWidth = 1920, frameHeight = 1080;
//     BufferedFileReader reader(inPath);
//     uint8_t *pNv12Buf, *dpNv12Buf;
//     size_t nNv12BufSize;
//     reader.GetBuffer(&pNv12Buf, &nNv12BufSize);
//     ck(cudaMalloc(&dpNv12Buf, nNv12BufSize));
//     ck(cudaMemcpy(dpNv12Buf, pNv12Buf, nNv12BufSize, cudaMemcpyHostToDevice));

//     // Encoder setup
//     NvEncoderInitParam initParam;
//     NvEncLite *enc = nullptr;
//     // FILE *fpOut = nullptr;
//     // const char* szOutFilePath = "./bin/sample/heif/bus_720_out.hevc";
//     init_encoder(current, frameWidth, frameHeight,
//     initParam, enc, true);

//     int N_RUNS = 1000;
//     std::vector<std::thread> threadVector(N_RUNS);
//     StopWatch w, w2;
//     double heifTime = 0;
//     auto clock_start = chrono::steady_clock::now();
//     w.Start();
//     NvHeifWriter heifWriter{outPath};
//     for (int i = 0; i < N_RUNS; i++) {
//         vector<vector<uint8_t>> vPacket;
//         enc->EncodeDeviceFrame(dpNv12Buf, 0, vPacket);

//         // w2.Start();
        
//         // std::thread heifWriterThread(heifWriter.write, vPacket, outPath);
//         // threadVector.emplace_back([=](){NvHeifWriter heifWriter{outPath};heifWriter.writeStillImage(vPacket);});
//         heifWriter.writeStillImage(vPacket);
//         // heifTime += w2.Stop();
//     }
//     for (auto &thread : threadVector) {
//         if (thread.joinable()) thread.join();
//     }
//     auto clock_stop = chrono::steady_clock::now();
//     double t = w.Stop();
//     chrono::duration<double> diff_clock = clock_stop - clock_start;
//     cout << "Average FPS of " << N_RUNS << " runs: " << N_RUNS / diff_clock.count() << ", average latency:" << diff_clock.count() / N_RUNS << " sec\n";
//     // cout << "Average FPS of " << N_RUNS << " runs (heif): "<< heifTime / N_RUNS << "\n";

//     delete enc;
//     ck(cudaFree(dpNv12Buf));
// }

template<class T> 
void ExtractSave(CUcontext cuContext, uint8_t * const pMem, size_t nMemSize, T interval) {
    CUstream stm = 0;
    ck(cuStreamCreate(&stm, 0));
    FrameExtractor extractor(pMem, nMemSize, cuContext);
    extractor.SetInterval(interval);
    NvEncoderInitParam initParam;
    NvEncLite *enc = nullptr;
    // FILE *fpOut = nullptr;
    // string outPath {"./extract.heic"};
    char* outPath = "extract.heic";
    init_encoder(cuContext, extractor.GetWidth(), extractor.GetHeight(), initParam, enc, true);
    NvHeifWriter heifWriter{outPath};
    uint8_t *pFrame;
    int i = 0;
    while ((pFrame = extractor.Extract(stm))) {
        i++;
        vector<vector<uint8_t>> vPacket;
        enc->EncodeHostFrame(pFrame, 0, vPacket);
        // outPath = to_string(i) + outPath;
        heifWriter.writeStillImage(vPacket);
    }
    ck(cuStreamDestroy(stm));
}

template<class T> 
void Extract(CUcontext cuContext, uint8_t * const pMem, size_t nMemSize, T interval) {
    CUstream stm = 0;
    ck(cuStreamCreate(&stm, 0));
    FrameExtractor extractor(pMem, nMemSize, cuContext);
    extractor.SetInterval(interval);
    uint8_t *pFrame;
    while (pFrame = extractor.Extract(stm))
        ;
    ck(cuStreamDestroy(stm));
}

void ExtractNormal(CUcontext cuContext, uint8_t * const pMem, size_t nMemSize) {
    VideoDemuxer demuxer(pMem, nMemSize);
    cudaVideoCodec codec = FFmpeg2NvCodecId(demuxer.GetVideoStream()->codecpar->codec_id);
    NvDecLite dec(cuContext, true, codec);
    double target = 0, delta = 1.0 / 3.0;
    int nFrameExtracted = 0;
    while (true) {
        AVPacket *pkt;
        demuxer.Demux(&pkt);

        uint8_t **ppFrame = NULL;
        NvFrameInfo *pInfo = NULL;
        nFrameExtracted += dec.Decode(pkt->data, pkt->size, &ppFrame, &pInfo, CUVID_PKT_ENDOFPICTURE);
        
        if (!pkt->size) break;
    }
    cout << "nFrameExtracted=" << nFrameExtracted << endl;
}

void ShowHelpAndExit(const char *szBadOption = NULL) {
    if (szBadOption) {
        cout << "Error parsing \"" << szBadOption << "\"" << endl;
    }
    cout << "Options:" << endl
        << "-i             Input file path" << endl
        << "-gpu           Ordinal of GPU to use" << endl
	<< "-time         Time Internal" << endl
	<< "-frame        Frame Internal" << endl
        << "-case          0: Fast decoding with seeking and non-ref frame skipping" << endl
        << "               1: Normal decoding" << endl
        ;
    cout << endl;
    exit(1);
}

void ParseCommandLine(int argc, char *argv[], char *szInputDirPath, int &iGpu, double &timeInterval, int &nFrameInterval, int &iCase)
{
    ostringstream oss;
    int i;
    for (i = 1; i < argc; i++) {
        if (!_stricmp(argv[i], "-h")) {
            ShowHelpAndExit();
        }
        if (!_stricmp(argv[i], "-i")) {
            if (++i == argc) {
                ShowHelpAndExit("-i");
            }
            sprintf(szInputDirPath, "%s", argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-gpu")) {
            if (++i == argc) {
                ShowHelpAndExit("-gpu");
            }
            iGpu = atoi(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-time")) {
            if (++i == argc) {
                ShowHelpAndExit("-time");
            }
            timeInterval = atof(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-frame")) {
            if (++i == argc) {
                ShowHelpAndExit("-frame");
            }
            nFrameInterval = atoi(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-case")) {
            if (++i == argc) {
                ShowHelpAndExit("-case");
            }
            iCase = atoi(argv[i]);
            continue;
        }
        ShowHelpAndExit(argv[i]);
    }
}

int main(int argc, char *argv[]) {
    char szInDirPath[256] = "video";
    int iGpu = 0;
    double timeInterval = 0;
    int nFrameInterval = 0;
    int iCase = 0;
    ParseCommandLine(argc, argv, szInDirPath, iGpu, timeInterval, nFrameInterval, iCase);

    av_log_set_level(AV_LOG_WARNING);
    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    if (iGpu < 0 || iGpu >= nGpu) {
        cout << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << endl;
        return 1;
    }
    CUdevice cuDevice = 0;
    ck(cuDeviceGet(&cuDevice, iGpu));
    char szDeviceName[80];
    ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    cout << "GPU in use: " << szDeviceName << endl;
    CUcontext cuContext = NULL;
    ck(cuCtxCreate(&cuContext, 0, cuDevice));

    if (timeInterval > 0) {
        cout << "Extract files from '" << szInDirPath << "', interval = " << timeInterval << " sec" << endl;
    } else {
        cout << "Extract files from '" << szInDirPath << "', interval = " << nFrameInterval << " frames" << endl;
    }

    vector<BufferedFileReader *> vpReader;
    namespace fs = experimental::filesystem;
    for (auto& f: fs::directory_iterator(szInDirPath)) {
        if (f.path().extension() != ".mp4") {
            continue;
        }
        vpReader.push_back(new BufferedFileReader(string(f.path()).c_str()));
    }

    const int nRound = 1;
    StopWatch w;
    w.Start();
    for (int i = 0; i < nRound; i++) {
        for (auto pReader: vpReader) {
            uint8_t *pBuf;
            size_t nSize;
            pReader->GetBuffer(&pBuf, &nSize);
            switch(iCase) {
            default:
            case 0: timeInterval ? Extract(cuContext, pBuf, nSize, timeInterval) : Extract(cuContext, pBuf, nSize, nFrameInterval); break;
            case 1: ExtractNormal(cuContext, pBuf, nSize); break;
            }
        }
    }
    cout << "Decoding time: " << w.Stop() << endl;
    return 0;
}
