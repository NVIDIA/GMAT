#pragma once

#include "TransData.h"

class RoundQueue {
public:
    RoundQueue(int nTransData, int nEnc) : nTransData(nTransData), nEnc(nEnc) {
        aTransData = new TransData[nTransData];
        aiFrameEnc = new int[nEnc];
        memset((void *)aiFrameEnc, 0, nEnc * sizeof(int));
    }
    ~RoundQueue() {
        for (int i = 0; i < nTransData; i++) {
            aTransData[i].Free();
        }
        delete[] aTransData;
        delete[] aiFrameEnc;
    }

    bool Get(int iEnc, TransData *pTransData) {
        if (aiFrameEnc[iEnc] == iFrameDec) {
            return false;
        }
        *pTransData = aTransData[aiFrameEnc[iEnc] % nTransData];
        aiFrameEnc[iEnc]++;
        return true;
    }
    bool Append(const TransData &transData) {
        if (iFrameDec - FindMinEncFrame() == nTransData) {
            return false;
        }
        aTransData[iFrameDec % nTransData].Free();
        aTransData[iFrameDec % nTransData] = transData;
        iFrameDec++;
        return true;
    }
    void SetEof() {
        bEof = true;
    }
    bool IsEof(int iEnc) {
        return bEof && aiFrameEnc[iEnc] == iFrameDec;
    }

private:
    int FindMinEncFrame() {
        int r = INT_MAX;
        for (int i = 0; i < nEnc; i++) {
            if (aiFrameEnc[i] < r) {
                r = aiFrameEnc[i];
            }
        }
        return r;
    }

    TransData *aTransData = NULL;
    int nTransData = 0;
    // next frame to be decoded. aTransData[iFrameDec] is unoccupied when iFrameDec - iFrameEnc < nFrame
    int iFrameDec = 0;
    // iFrameEnc=aiFrameEnc[i]: next frame to be encoded. aTransData[iFrameEnc] is eligible for encoding when iEnc < iDec
    int *aiFrameEnc = NULL;
    int nEnc = 0;
    bool bEof = false;
};
