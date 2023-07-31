/*
* Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#include <cuda_runtime.h>
#include <stdint.h>

static __global__ void ConvertUInt8ToUInt16Kernel(uint8_t *dpUInt8, uint16_t *dpUInt16, int n) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    while (x < n) {
        *(uchar2 *)&dpUInt16[x] = uchar2{0, dpUInt8[x]};
        x += gridDim.x * blockDim.x;
    }
}

static __global__ void ConvertUInt16ToUInt8Kernel(uint16_t *dpUInt16, uint8_t *dpUInt8, int n) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    while (x < n) {
        dpUInt8[x] = ((uchar2 *)&dpUInt16[x])->y;
        x += gridDim.x * blockDim.x;
    }
}

void ConvertUInt8ToUInt16(uint8_t *dpUInt8, uint16_t *dpUInt16, int n) {
    ConvertUInt8ToUInt16Kernel<<<4096, 256>>>(dpUInt8, dpUInt16, n);
}

void ConvertUInt16ToUInt8(uint16_t *dpUInt16, uint8_t *dpUInt8, int n) {
    ConvertUInt16ToUInt8Kernel<<<4096, 256>>>(dpUInt16, dpUInt8, n);
}
