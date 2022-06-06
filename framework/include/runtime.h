#pragma once

#include <cuda_runtime.h>
#include "Logger.h"

namespace ffgd{
    class Runtime{
    public:
        Runtime();
        
    private:
        cudaMemPool_t gpuMemPool;
    };
}

