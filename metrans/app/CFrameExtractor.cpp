#include "FrameExtractor.h"
#include "NvCodec/NvCommon.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger(simplelogger::INFO);

extern "C" FrameExtractor *FrameExtractor_InitFromFile(const char *szFilePath) {
    CUcontext cuContext = 0;
    ck(cuCtxGetCurrent(&cuContext));
    if (!cuContext) {
        LOG(ERROR) << "No CUDA context in current thread";
        return nullptr;
    }
    CUdevice dev = -1;
    ck(cuCtxGetDevice(&dev));
    FrameExtractor *h = new FrameExtractor(szFilePath, cuContext);
    return h;
}
extern "C" FrameExtractor *FrameExtractor_InitFromBuffer(uint8_t * const pMem, size_t nMemSize) {
    CUcontext cuContext = 0;
    cuCtxGetCurrent(&cuContext);
    if (!cuContext) {
        LOG(ERROR) << "No CUDA context in current thread";
        return nullptr;
    }
    return new FrameExtractor(pMem, nMemSize, cuContext);
}
extern "C" void FrameExtractor_Delete(FrameExtractor *pFrameExtractor) {
    delete pFrameExtractor;
}
extern "C" void FrameExtractor_SetFrameInterval(FrameExtractor *pFrameExtractor, int nFrameInterval) {
    pFrameExtractor->SetInterval(nFrameInterval);
}
extern "C" void FrameExtractor_SetTimeInterval(FrameExtractor *pFrameExtractor, double timeInterval) {
    pFrameExtractor->SetInterval(timeInterval);
}
extern "C" int FrameExtractor_GetWidth(FrameExtractor *pFrameExtractor) {
    return pFrameExtractor->GetWidth();
}
extern "C" int FrameExtractor_GetHeight(FrameExtractor *pFrameExtractor) {
    return pFrameExtractor->GetHeight();
}
extern "C" int FrameExtractor_GetFrameSize(FrameExtractor *pFrameExtractor) {
    return pFrameExtractor->GetFrameSize();
}
extern "C" bool FrameExtractor_ExtractToBuffer(FrameExtractor *pFrameExtractor, uint8_t* pframe, CUstream stream) {
    return pFrameExtractor->ExtractToBuffer(pframe, stream);
}
extern "C" bool FrameExtractor_ExtractToDeviceBuffer(FrameExtractor *pFrameExtractor, float *dpBgrp, CUstream stream) {
    return pFrameExtractor->ExtractToDeviceBuffer(dpBgrp, stream);
}
