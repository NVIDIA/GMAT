#pragma once

#include <map>
#include <vector>
#include <cuda.h>
#include "NvCodec/NvCommon.h"
#include "AvToolkit/VidFilt.h"
#include "Logger.h"
#include "TransData.h"
#include "TransDataConverter.h"

using namespace std;

class VidFiltEx : public VidFilt {
public:
    VidFiltEx(CUcontext context, AVPixelFormat eInputFormat, int nWidth, int nHeight, AVRational timebase, AVRational sar, const char *szFilterDesc) : 
        VidFilt(eInputFormat, nWidth, nHeight, timebase, sar, szFilterDesc, true), 
        m_converter(m_cuContext), m_cuContext(context), m_nWidth(nWidth), m_nHeight(nHeight) {
    }
    ~VidFiltEx() {
        for (AVFrame *frm : m_vFrm) {
            delete frm->data[0];
            av_frame_free(&frm);
        }
    }
    bool FilterFrmToD(vector<AVFrame *> vFrmSrc, std::vector<TransData> &vTransData) {
        vector<AVFrame *> vFrm;
        if (!Filter(vFrmSrc, vFrm)) {
            return false;
        }
        return m_converter.FrmToD(vFrm, vTransData);
    }
    bool FilterDtoD(uint8_t *dpFrame, int nPitch, int64_t pts, std::vector<TransData> &vTransData) {
        return FilterDtoD(std::vector<TransData>{TransData(dpFrame, nPitch, m_nWidth, m_nHeight, pts, (TransDataConverter *)NULL)}, vTransData);
    }
    bool FilterDtoD(const std::vector<TransData> &vTransDataSrc, std::vector<TransData> &vTransDataDst) {
        vector<AVFrame *> vFrm;
        int iFrom = (int)m_vFrm.size() - 1;
        for (const TransData &src : vTransDataSrc) {
            AVFrame *frm = NULL;
            if (!AllocateAndFill(src, &frm, iFrom)) {
                return false;
            }
            vFrm.push_back(frm);
        }
        return FilterFrmToD(vFrm, vTransDataDst);
    }

private:
    bool AllocateAndFill(const TransData &src, AVFrame **pFrm, int &iFrom) {
        *pFrm = NULL;
        while (iFrom >= 0) {
            AVFrame *frm = m_vFrm[iFrom--];
            if (frm->width == src.nWidth && frm->height == src.nHeight) {
                *pFrm = frm;
                break;
            }
            delete frm->data[0];
            av_frame_free(&frm);
            m_vFrm.pop_back();
        }
        if (!*pFrm) {
            AVFrame *frm = av_frame_alloc();
            frm->width = src.nWidth;
            frm->height = src.nHeight;
            frm->format = AV_PIX_FMT_NV12;
            uint8_t *pFrame = new uint8_t[src.nWidth * src.nHeight * 3 / 2];
            av_image_fill_arrays(frm->data, frm->linesize, pFrame, AV_PIX_FMT_NV12, src.nWidth, src.nHeight, 1);
            *pFrm = frm;
            m_vFrm.push_back(frm);
        }
        CUDA_MEMCPY2D m = { 0 };
        m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        m.srcDevice = (CUdeviceptr)src.dpVideoFrame;
        m.srcPitch = src.nPitch;
        m.dstMemoryType = CU_MEMORYTYPE_HOST;
        m.dstHost = (*pFrm)->data[0];
        m.dstPitch = src.nWidth;
        m.WidthInBytes = src.nWidth;
        m.Height = src.nHeight * 3 / 2;
        ck(cuCtxPushCurrent(m_cuContext));
        ck(cuMemcpy2D(&m));
        ck(cuCtxPopCurrent(NULL));
        (*pFrm)->pts = src.pts;
        return true;
    }

private:
    CUcontext m_cuContext;
    int m_nWidth = 0, m_nHeight = 0;
    std::vector<AVFrame *> m_vFrm;
    TransDataConverter m_converter;
};
