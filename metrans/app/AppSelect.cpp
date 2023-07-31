#include <iostream>
#include <stdint.h>
#include "FrameSelect.h"
#include "NvCodec/NvCommon.h"
#include "NvCodec/NvDecLite.h"

#include "libavutil/log.h"

using namespace std;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger(simplelogger::INFO);

void ShowHelpAndExit(const char *szBadOption = NULL) {
    if (szBadOption) {
        cout << "Error parsing \"" << szBadOption << "\"" << endl;
    }
    cout << "Options:" << endl
        << "-i             Input file path" << endl
        << "-o             Output file path" << endl
        << "-gpu           Ordinal of GPU to use" << endl
        << "-time          Time interval to extract frames" << endl
        << "-frame         Frame interval to extract frames" << endl
        ;
    cout << endl;
    exit(1);
}

void ParseCommandLine(int argc, char *argv[], char *szInputFileName, char *szOutputFileName, int &iGpu, double &timeInterval, int &nFrameInterval)
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
        ShowHelpAndExit(argv[i]);
    }
}

int main(int argc, char *argv[]) {
    char szInFilePath[256] = "bunny.mp4",
        szOutFilePath[256] = "out.native";
    int iGpu = 0;
    double timeInterval = 0;
    int nFrameInterval = 0;
    ParseCommandLine(argc, argv, szInFilePath, szOutFilePath, iGpu, timeInterval, nFrameInterval);

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
    // ck(cuCtxCreate(&cuContext, 0, cuDevice));
    ck(cudaSetDevice(0));
    ck(cuCtxGetCurrent(&cuContext));

    av_log_set_level(AV_LOG_DEBUG);
    FrameSelect selector(szInFilePath, "gt(scene,0.4)", cuContext);
    
    ofstream fOut(szOutFilePath, ios::out | ios::binary);
    uint8_t *dpFrame;
    auto pFrame = make_unique<uint8_t[]>(selector.GetFrameSize());
    CUstream stm = 0;
    ck(cuStreamCreate(&stm, 0));
    while (dpFrame = selector.Extract(stm)) {
        ck(cuMemcpyDtoHAsync(pFrame.get(), (CUdeviceptr)dpFrame, selector.GetFrameSize(), stm));
        ck(cuStreamSynchronize(stm));
        // fOut.write(reinterpret_cast<char *>(pFrame.get()), selector.GetFrameSize());
    }
    ck(cuStreamDestroy(stm));

    return 0;
}
