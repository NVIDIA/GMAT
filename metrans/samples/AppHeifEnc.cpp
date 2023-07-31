#include <vector>
#include <thread>

#include "NvCodec/NvCommon.h"
#include "NvCodec/NvEncLite.h"
#include "NvCodec/NvHeifWriter.h"

#include <cuda_runtime.h>
#include <heif/reader/heifreader.h>
#include <heif/writer/heifwriter.h>

using namespace std;
using namespace HEIF;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger(simplelogger::WARNING);

void init_encoder(CUcontext cuContext, int nWidth, int nHeight, const char *szOutFilePath, 
    NvEncoderInitParam &initParam, NvEncLite* &enc, FILE* &fpOut) {
    fpOut = fopen(szOutFilePath, "wb");
    if (fpOut == NULL) {
        cout << "Unable to open file: " << szOutFilePath << endl;
        return;
    }

    std::string init_param_string{"-codec hevc -preset p1 -bitrate 2M"};
    initParam = NvEncoderInitParam(init_param_string.c_str());
    NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_NV12;

    enc = new NvEncLite(cuContext, nWidth, nHeight, eFormat, &initParam);
}

void init_encoder(CUcontext cuContext, int nWidth, int nHeight, 
    NvEncoderInitParam &initParam, NvEncLite* &enc, bool stillImage=false) {
    std::string init_param_string{"-codec hevc -tune lossless"};
    initParam = NvEncoderInitParam(init_param_string.c_str());
    NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_NV12;

    enc = new NvEncLite(cuContext, nWidth, nHeight, eFormat, &initParam, 0, stillImage);
}

union FourByte {
    uint32_t d;
    uchar4 v;
    struct {
        uint8_t b1, b2, b3, b4;
    }b;
};

void encodeClass(const char* inPath, char* outPath, CUcontext current)
{
    int frameWidth = 1920, frameHeight = 1080;
    BufferedFileReader reader(inPath);
    uint8_t *pNv12Buf, *dpNv12Buf;
    size_t nNv12BufSize;
    reader.GetBuffer(&pNv12Buf, &nNv12BufSize);
    ck(cudaMalloc(&dpNv12Buf, nNv12BufSize));
    ck(cudaMemcpy(dpNv12Buf, pNv12Buf, nNv12BufSize, cudaMemcpyHostToDevice));

    // Encoder setup
    NvEncoderInitParam initParam;
    NvEncLite *enc = nullptr;
    // FILE *fpOut = nullptr;
    // const char* szOutFilePath = "./bin/sample/heif/bus_720_out.hevc";
    init_encoder(current, frameWidth, frameHeight,
    initParam, enc, true);

    int N_RUNS = 1000;
    // std::vector<std::thread> threadVector(N_RUNS);
    StopWatch w, w2;
    double heifTime = 0;
    auto clock_start = chrono::steady_clock::now();
    w.Start();
    NvHeifWriter heifWriter{outPath};
    for (int i = 0; i < N_RUNS; i++) {
        vector<vector<uint8_t>> vPacket;
        enc->EncodeDeviceFrame(dpNv12Buf, 0, vPacket);

        // w2.Start();
        
        // std::thread heifWriterThread(heifWriter.write, vPacket, outPath);
        // threadVector.emplace_back([=](){NvHeifWriter heifWriter{outPath};heifWriter.writeStillImage(vPacket);});
        heifWriter.writeStillImage(vPacket);
        // heifTime += w2.Stop();
    }
    // for (auto &thread : threadVector) {
    //     if (thread.joinable()) thread.join();
    // }
    auto clock_stop = chrono::steady_clock::now();
    double t = w.Stop();
    chrono::duration<double> diff_clock = clock_stop - clock_start;
    cout << "Average FPS of " << N_RUNS << " runs: " << N_RUNS / diff_clock.count() << ", average latency:" << diff_clock.count() / N_RUNS << " sec\n";
    // cout << "Average FPS of " << N_RUNS << " runs (heif): "<< heifTime / N_RUNS << "\n";

    delete enc;
    ck(cudaFree(dpNv12Buf));
}
void encodeClassSequence(const char* inPath, char* outPath, CUcontext current)
{
    int frameWidth = 1920, frameHeight = 1080;
    BufferedFileReader reader(inPath);
    uint8_t *pNv12Buf, *dpNv12Buf;
    size_t nNv12BufSize;
    reader.GetBuffer(&pNv12Buf, &nNv12BufSize);
    ck(cudaMalloc(&dpNv12Buf, nNv12BufSize));
    ck(cudaMemcpy(dpNv12Buf, pNv12Buf, nNv12BufSize, cudaMemcpyHostToDevice));

    // Encoder setup
    NvEncoderInitParam initParam;
    NvEncLite *enc = nullptr;
    // FILE *fpOut = nullptr;
    // const char* szOutFilePath = "./bin/sample/heif/bus_720_out.hevc";
    init_encoder(current, frameWidth, frameHeight,
    initParam, enc, true);

    int N_RUNS = 1000;
    std::vector<std::thread> threadVector(N_RUNS);
    StopWatch w, w2;
    double heifTime = 0;
    auto clock_start = chrono::steady_clock::now();
    w.Start();
    NvHeifWriter heifWriter{outPath, false, NV_ENC_CODEC_HEVC_GUID};
    for (int i = 0; i < N_RUNS; i++) {
        vector<vector<uint8_t>> vPacket;
        enc->EncodeDeviceFrame(dpNv12Buf, 0, vPacket);

        // w2.Start();
        
        // std::thread heifWriterThread(heifWriter.write, vPacket, outPath);
        heifWriter.addImageToSequence(vPacket, false);
        // heifWriter.write(vPacket, outPath);
        // heifTime += w2.Stop();
    }
    heifWriter.writeSequence();
    auto clock_stop = chrono::steady_clock::now();
    double t = w.Stop();
    chrono::duration<double> diff_clock = clock_stop - clock_start;
    cout << "Average FPS of " << N_RUNS << " runs: " << N_RUNS / diff_clock.count() << ", average latency:" << diff_clock.count() / N_RUNS << " sec\n";
    // cout << "Average FPS of " << N_RUNS << " runs (heif): "<< heifTime / N_RUNS << "\n";

    delete enc;
    ck(cudaFree(dpNv12Buf));
}

void ShowHelpAndExit(const char *szBadOption = NULL) {
    if (szBadOption) {
        cout << "Error parsing \"" << szBadOption << "\"" << endl;
    }
    cout << "Options:" << endl
        << "-i           Input file path" << endl
        << "-s           Input resolution in this form: WxH" << endl
        << "-if          Input format: iyuv nv12 yuv444 p010 yuv444p16 bgra" << endl
        << "-gpu         Ordinal of GPU to use" << endl
        << "-frame       Number of frames to encode per thread (default is 1000)" << endl
        << "-thread      Number of encoding thread (default is 2)" << endl
        << "-single      (No value) Use single context (this may result in suboptimal performance; default is multiple contexts)" << endl
        << "-pause       (No value) Pause before exit" << endl
        ;
    cout << NvEncoderInitParam().GetHelpMessage();
    exit(1);
}

void ParseCommandLine(int argc, char *argv[], char *szInputFileName, char *szOutputFileName, int &nWidth, int &nHeight, 
    NV_ENC_BUFFER_FORMAT &eFormat, int &iGpu, int &nFrame, int &nThread, 
    bool &bSingle, bool &bPause, NvEncoderInitParam &initParam) 
{
    ostringstream oss;
    for (int i = 1; i < argc; i++) {
        if (!_stricmp(argv[i], "-h")) {
            ShowHelpAndExit();
        }
        if (!_stricmp(argv[i], "-i")) {
            if (++i == argc) {
                ShowHelpAndExit("-i");
            }
            sprintf(szInputFileName, "%s", argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-o")) {
            if (++i == argc) {
                ShowHelpAndExit("-o");
            }
            sprintf(szOutputFileName, "%s", argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-s")) {
            if (++i == argc || 2 != sscanf(argv[i], "%dx%d", &nWidth, &nHeight)) {
                ShowHelpAndExit("-s");
            }
            continue;
        }
        vector<string> vszFileFormatName = {
            "iyuv", "nv12", "yv12", "yuv444", "p010", "yuv444p16", "bgra", "argb10", "ayuv", "abgr", "abgr10"
        };
        NV_ENC_BUFFER_FORMAT aFormat[] = {
            NV_ENC_BUFFER_FORMAT_IYUV,
            NV_ENC_BUFFER_FORMAT_NV12,
            NV_ENC_BUFFER_FORMAT_YV12,
            NV_ENC_BUFFER_FORMAT_YUV444,
            NV_ENC_BUFFER_FORMAT_YUV420_10BIT,
            NV_ENC_BUFFER_FORMAT_YUV444_10BIT,
            NV_ENC_BUFFER_FORMAT_ARGB,
            NV_ENC_BUFFER_FORMAT_ARGB10,
            NV_ENC_BUFFER_FORMAT_AYUV,
            NV_ENC_BUFFER_FORMAT_ABGR,
            NV_ENC_BUFFER_FORMAT_ABGR10,
        };
        if (!_stricmp(argv[i], "-if")) {
            if (++i == argc) {
                ShowHelpAndExit("-if");
            }
            auto it = find(vszFileFormatName.begin(), vszFileFormatName.end(), argv[i]);
            if (it == vszFileFormatName.end()) {
                ShowHelpAndExit("-if");
            }
            eFormat = aFormat[it - vszFileFormatName.begin()];
            continue;
        }
        if (!_stricmp(argv[i], "-gpu")) {
            if (++i == argc) {
                ShowHelpAndExit("-gpu");
            }
            iGpu = atoi(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-frame")) {
            if (++i == argc) {
                ShowHelpAndExit("-frame");
            }
            nFrame = atoi(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-thread")) {
            if (++i == argc) {
                ShowHelpAndExit("-thread");
            }
            nThread = atoi(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-single")) {
            bSingle = true;
            continue;
        }
        if (!_stricmp(argv[i], "-pause")) {
            bPause = true;
            continue;
        }
        // Regard as encoder parameter
        if (argv[i][0] != '-') {
            ShowHelpAndExit(argv[i]);
        }
        oss << argv[i] << " ";
        while (i + 1 < argc && argv[i + 1][0] != '-') {
            oss << argv[++i] << " ";
        }
    }
    initParam = NvEncoderInitParam(oss.str().c_str());
}

int main(int argc, char *argv[]){
    vector<thread *> vThread;
    char szInFilePath[256] = "./build/bus_1080.yuv";
    char szOutFilePath[256];
    int nWidth = 1920, nHeight = 1080;
    NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_IYUV;
    int iGpu = 0;
    int nFrame = 5000;
    int nThread = 2;
    bool bSingle = false;
    bool bPause = false;
    NvEncoderInitParam initParam;
    CheckDefaultFileExists(szInFilePath);
    ParseCommandLine(argc, argv, szInFilePath, szOutFilePath, nWidth, nHeight, eFormat, 
        iGpu, nFrame, nThread, bSingle, bPause, initParam);

    cudaSetDevice(0);
    CUcontext current;
    ck(cuDevicePrimaryCtxRetain(&current, 0));
    ck(cuCtxPushCurrent(current));

    for (int i = 0; i < nThread; i++) {
        vThread.push_back(new thread(encodeClass, szInFilePath, nullptr, current));
    }
    for (int i = 0; i < nThread; i++) {
        vThread[i]->join();
    }
}