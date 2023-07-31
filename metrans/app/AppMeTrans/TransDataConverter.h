#pragma once

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
}
#include <cuda.h>
#include "NvCodec/NvCommon.h"
#include "Logger.h"
#include "TransData.h"

class TransDataConverter {
public:
    TransDataConverter(CUcontext cuContext) : cuContext(cuContext) {}
    ~TransDataConverter() {
        if (pFrameDst) delete[] pFrameDst;
        if (vdpFrame.size() != dpFrame2TransData.size()) {
            LOG(WARNING) << "Frames are not fully recycled (" << vdpFrame.size() << "<" << dpFrame2TransData.size() << ")";
        }
        ck(cuCtxPushCurrent(cuContext));
        for (uint8_t *dpFrame : vdpFrame) {
            ck(cuMemFree((CUdeviceptr)dpFrame));
        }
        ck(cuCtxPopCurrent(NULL));
    }
    bool FrmToD(vector<AVFrame *> &vFrm, std::vector<TransData> &vTransData) {
        ck(cuCtxPushCurrent(cuContext));
        for (AVFrame *frm : vFrm) {
            uint8_t *dpFrame = NULL;
            for (int i = vdpFrame.size() - 1; i >= 0; i--) {
                dpFrame = vdpFrame[i];
                vdpFrame.pop_back();
                TransData &data = dpFrame2TransData[dpFrame];
                if (data.nWidth == frm->width && data.nHeight == frm->height) {
                    break;
                } else {
                    ck(cuMemFree((CUdeviceptr)dpFrame));
                    dpFrame2TransData.erase(dpFrame);
                    dpFrame = NULL;
                }
            }
            if (!dpFrame) {
                size_t nPitch = 0;
                ck(cuMemAllocPitch((CUdeviceptr *)&dpFrame, &nPitch, frm->width, frm->height * 3 / 2, 16));
                dpFrame2TransData[dpFrame] = TransData(dpFrame, nPitch, frm->width, frm->height, frm->pts, this);
            }
            
            if (nWidthDst != frm->width || nHeightDst != frm->height) {
                if (pFrameDst) delete[] pFrameDst;
                nWidthDst = frm->width;
                nHeightDst = frm->height;
                pFrameDst = new uint8_t[nWidthDst * nHeightDst * 3 / 2];
            }
            av_image_copy_plane(pFrameDst, nWidthDst, frm->data[0], frm->linesize[0], nWidthDst, nHeightDst);
            av_image_copy_plane(pFrameDst + nWidthDst * nHeightDst, nWidthDst, frm->data[1], frm->linesize[1], nWidthDst, nHeightDst / 2);

            CUDA_MEMCPY2D m = { 0 };
            m.srcMemoryType = CU_MEMORYTYPE_HOST;
            m.srcHost = pFrameDst;
            m.srcPitch = nWidthDst;
            m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            m.dstDevice = (CUdeviceptr)dpFrame;
            m.dstPitch = dpFrame2TransData[dpFrame].nPitch;
            m.WidthInBytes = nWidthDst;
            m.Height = nHeightDst * 3 / 2;
            ck(cuMemcpy2D(&m));

            dpFrame2TransData[dpFrame].pts = frm->pts;
            vTransData.push_back(dpFrame2TransData[dpFrame]);
        }
        ck(cuCtxPopCurrent(NULL));
        return true;
    }
    void Recycle(uint8_t *dpFrame) {
        if (dpFrame2TransData.find(dpFrame) == dpFrame2TransData.end()) {
            LOG(WARNING) << "Device frame 0x" << dpFrame << " can't be recycled (as it's not registered)";
            return;
        }
        vdpFrame.push_back(dpFrame);
    }

private:
    CUcontext cuContext;
    map<uint8_t *, TransData> dpFrame2TransData;
    vector<uint8_t *> vdpFrame;
    uint8_t *pFrameSrc = NULL, *pFrameDst = NULL;
    int nWidthSrc = 0, nWidthDst = 0; 
    int nHeightSrc = 0, nHeightDst = 0;
};
