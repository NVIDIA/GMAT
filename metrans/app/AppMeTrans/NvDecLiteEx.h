#pragma once
#include <queue>
#include "NvCodec/NvDecLite.h"

class NvDecLiteEx : public NvDecLite {
public:
    NvDecLiteEx(CUcontext cuContext, bool bUseDeviceFrame, cudaVideoCodec eCodec, 
            bool bLowLatency = false, bool bDeviceFramePitched = false, const Rect *pCropRect = NULL, const Dim *pResizeDim = NULL, bool bOriginalPts = false) :
        NvDecLite(cuContext, bUseDeviceFrame, eCodec, bLowLatency, bDeviceFramePitched, pCropRect, pResizeDim), bOriginalPts(bOriginalPts){}
    virtual int Decode(const uint8_t *pData, int nSize, uint8_t ***pppFrame, NvFrameInfo **ppFrameInfo, int *pnFrameReturned, 
        uint32_t flags, int64_t timestamp, CUstream stream) 
    {
        int ret = NvDecLite::Decode(pData, nSize, pppFrame, ppFrameInfo, flags, timestamp, stream);
        if (!bOriginalPts && pData && timestamp != INVALID_TIMESTAMP && pnFrameReturned && ppFrameInfo) {
            qPts.push(timestamp);
        }
        if (pnFrameReturned && ppFrameInfo) {
            for (int i = 0; i < *pnFrameReturned && !qPts.empty(); i++) {
                (*ppFrameInfo)[i].dispInfo.timestamp = qPts.top();
                qPts.pop();
            }
        }
        return ret;
    }
private:
    bool bOriginalPts;
    std::priority_queue<int64_t, std::vector<int64_t>, std::greater<int64_t>> qPts;
};
