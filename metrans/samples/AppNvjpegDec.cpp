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

#include <iostream>
#include <vector>
#include <cuda.h>
#include <nvjpeg.h>
#include "NvCodec/NvCommon.h"
#include "Logger.h"

using namespace std;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

class NvjpegDecoder {
public:
    NvjpegDecoder() {

    }
private:
    
};

int main(int argc, char **argv) {
    char const *szJpegPath = "bunny.jpg";
    BufferedFileReader reader(szJpegPath);
    uint8_t *pBuf;
    size_t nSize;
    reader.GetBuffer(&pBuf, &nSize);

    nvjpegHandle_t nvjpeg_handle;
    nvjpegCreate(NVJPEG_BACKEND_DEFAULT, nullptr, &nvjpeg_handle);
    nvjpegJpegState_t jpeg_state;
    nvjpegJpegStateCreate(nvjpeg_handle, &jpeg_state);

    // Retrieve the componenet and size info.
    int nComponent = 0;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    if (NVJPEG_STATUS_SUCCESS != nvjpegGetImageInfo(nvjpeg_handle, pBuf, nSize, &nComponent, &subsampling, widths, heights))
    {
        std::cerr << "Error decoding JPEG header: " << szJpegPath << std::endl;
        return 1;
    }

    // image information
    std::cout << "Image is " << nComponent << " channels." << std::endl;
    for (int i = 0; i < nComponent; i++)
    {
        std::cout << "Channel #" << i << " size: "  << widths[i]  << " x " << heights[i] << std::endl;    
    }

    uint8_t *dpFrame;
    ck(cudaMalloc(&dpFrame, widths[0]*heights[0]*3));

    nvjpegImage_t imgdesc = 
    {
        {
            dpFrame,
            dpFrame + widths[0]*heights[0],
            dpFrame + widths[0]*heights[0]*2,
            nullptr
        },
        {
            (unsigned int)widths[0],
            (unsigned int)widths[0],
            (unsigned int)widths[0],
            0
        }
    };

    nvjpegDecode(nvjpeg_handle, jpeg_state, pBuf, nSize, NVJPEG_OUTPUT_BGR, &imgdesc, NULL);

    vector<uint8_t> pFrame(widths[0]*heights[0]*3);
    ck(cudaMemcpy(pFrame.data(), dpFrame, pFrame.size(), cudaMemcpyDeviceToHost));

    char const *szOutFilePath = "out.bgrp";
    ofstream fOut(szOutFilePath, ios::out | ios::binary);
    fOut.write((char *)pFrame.data(), pFrame.size());

    nvjpegDestroy(nvjpeg_handle);

    return 0;
}
