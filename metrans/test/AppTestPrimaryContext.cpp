#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <thread>

using namespace std;

inline bool check(CUresult e, int iLine, const char *szFile) {
    if (e != CUDA_SUCCESS) {
        const char *szErrName = NULL;
        cuGetErrorName(e, &szErrName);
        cerr << "CUDA driver API error " << szErrName << " at line " << iLine << " in file " << szFile;
        return false;
    }
    return true;
}

inline bool check(cudaError_t e, int iLine, const char *szFile) {
    if (e != cudaSuccess) {
        cerr << "CUDA runtime API error " << cudaGetErrorName(e) << " at line " << iLine << " in file " << szFile;
        return false;
    }
    return true;
}

#define ck(call) check(call, __LINE__, __FILE__)

void Proc() {
    ck(cudaFree(0));

    CUcontext context = nullptr;
    ck(cuCtxGetCurrent(&context));
    cout << "context=" << context << endl;
}

int main() {
    ck(cudaFree(0));

    thread th(Proc);
    th.join();

    return 0;
}