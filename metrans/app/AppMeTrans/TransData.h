#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include "NvCodec/NvDecLite.h"

using namespace std;

class TransDataConverter;

struct TransData {
    TransData() {}
    TransData(uint8_t *dpVideoFrame, int nPitch, int nWidth, int nHeight, int64_t pts, NvDecLite *pRecycler) 
        : dpVideoFrame(dpVideoFrame), nPitch(nPitch), nWidth(nWidth), nHeight(nHeight), pts(pts), pDec(pRecycler) {}
    TransData(uint8_t *dpVideoFrame, int nPitch, int nWidth, int nHeight, int64_t pts, TransDataConverter *pRecycler) 
        : dpVideoFrame(dpVideoFrame), nPitch(nPitch), nWidth(nWidth), nHeight(nHeight), pts(pts), pConverter(pRecycler) {}
    TransData(uint8_t *pAudioPacket, int nPacketSize, int64_t pts, int64_t dts) : pts(pts), dts(dts) {
        audioPacket.insert(audioPacket.end(), pAudioPacket, pAudioPacket + nPacketSize);
    }
    void Free();

    uint8_t *dpVideoFrame = NULL;
    int nPitch = 0, nWidth = 0, nHeight = 0;
    vector<uint8_t> audioPacket;
    int64_t pts = 0, dts = 0;
    NvDecLite *pDec = NULL;
    TransDataConverter *pConverter = NULL;
};
