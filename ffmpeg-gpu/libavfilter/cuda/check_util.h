#ifndef CUDA_VERSION
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "libavutil/log.h"

static inline int check_cu(CUresult e, void *ctx, int iLine, const char *szFile) {
    if (e != CUDA_SUCCESS) {
        const char* pStr;
        cuGetErrorName(e, &pStr);
        av_log(ctx, AV_LOG_ERROR, "CUDA driver API error: %s, at line %d in file %s\n",
        pStr, iLine, szFile);
        return 0;
    }
    return 1;
}

static inline int check(cudaError_t e, void *ctx, int iLine, const char *szFile) {
    if (e != cudaSuccess) {
        av_log(ctx, AV_LOG_ERROR, "CUDA runtime API error: %s, at line %d in file %s\n",
            cudaGetErrorName(e), iLine, szFile);
        return 0;
    }
    return 1;
}

#define FF_CUDA_CK(call) check(call, s, __LINE__, __FILE__)
#define FF_CU_CK(call) check_cu(call, s, __LINE__, __FILE__)
