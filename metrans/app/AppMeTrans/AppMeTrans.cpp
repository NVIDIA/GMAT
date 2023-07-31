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

#include <stdio.h>
#include <time.h>
#include <iostream>
#include <thread>
#include <memory>
#include <cuda_runtime.h>
#include <boost/algorithm/string/replace.hpp>
#include "AvToolkit/Demuxer.h"
#include "AvToolkit/Muxer.h"
#include "AvToolkit/AudDec.h"
#include "AvToolkit/AudEnc.h"
#include "AvToolkit/AudFilt.h"
#include "NvDecLiteEx.h"
#include "NvCodec/NvEncLite.h"
#include "NvCodec/NvCommon.h"
#include "Options.h"
#include "RoundQueue.h"
#include "TransData.h"
#include "TransDataConverter.h"
#include "VidFiltEx.h"
#include "VidDecEx.h"
#include "FpsLimiter.h"

using namespace std;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

void EncodeAudioFrame(AudEnc *&pAudEnc, vector<LazyMuxer *> vpMuxer, AVRational timebase, 
    AVFrame * frm, RoundQueue &queue, const Options &options)
{
    if (!pAudEnc) {
        AVCodecParameters *par = avcodec_parameters_alloc();
        par->bit_rate = options.nAudioBitRate;
        par->sample_rate = options.nAudioSampleRate;
        AVCodec const *codec = NULL;
        if (options.strAudioCodec.size()) {
            codec = avcodec_find_encoder_by_name(options.strAudioCodec.c_str());
            if (!codec) {
                cout << "Audio codec " << options.strAudioCodec << " not found" << endl;
                return;
            }
        }
        pAudEnc = new AudEnc((AVSampleFormat)frm->format, frm->sample_rate, frm->channel_layout, codec ? codec->id : AV_CODEC_ID_AAC, NULL, timebase, par);
        avcodec_parameters_free(&par);
        for (LazyMuxer *pMuxer : vpMuxer) {
            pMuxer->SetAudioStream(pAudEnc->GetCodecParameters(), pAudEnc->GetCodecContext()->time_base);
        }
    }

    vector<AVPacket *> vPkt;
    pAudEnc->Encode(frm, vPkt);
    for (AVPacket *pkt : vPkt) {
        TransData data(pkt->data, pkt->size, pkt->pts, pkt->dts);
        while (!queue.Append(data)) {
            this_thread::sleep_for(chrono::milliseconds(1));
        }        
    }
}

void EncodeVideoProc(RoundQueue *pQueue, int iEnc, VidFiltEx *pVidFilt, NvEncLite *pEnc, LazyMuxer *pMuxer, int nFpsLimit) {
    ck(cuCtxSetCurrent((CUcontext)pEnc->GetDevice()));
    uint8_t *dpFrameResized;
    ck(cuMemAlloc((CUdeviceptr *)&dpFrameResized, pEnc->GetFrameSize()));
    
    vector<vector<uint8_t>> vPacket;
    FpsLimiter fpsLimiter(nFpsLimit);
    while (!pQueue->IsEof(iEnc)) {
        TransData data;
        if (!pQueue->Get(iEnc, &data)) {
            this_thread::sleep_for(chrono::milliseconds(1));
            continue;
        }
        if (!data.dpVideoFrame) {
            pMuxer->MuxAudio(data.audioPacket.data(), data.audioPacket.size(), data.pts, data.dts);
            continue;
        }

        ScaleNv12(data.dpVideoFrame, data.nPitch, data.nWidth, data.nHeight,
            dpFrameResized, pEnc->GetWidth(), pEnc->GetWidth(), pEnc->GetHeight());
        
        vector<TransData> vTransData;
        if (pVidFilt) {
            pVidFilt->FilterDtoD(dpFrameResized, pEnc->GetWidth(), data.pts, vTransData);
        } else {
            vTransData.push_back(TransData(dpFrameResized, pEnc->GetWidth(), pEnc->GetWidth(), pEnc->GetHeight(), data.pts, (NvDecLite *)NULL));
        }
        
        for (TransData &data : vTransData) {
            if (data.nWidth != pEnc->GetWidth() || data.nHeight != pEnc->GetHeight()) {
                LOG(ERROR) << "SW filter changed resolution. To exit.";
                exit(1);
            }

            NV_ENC_PIC_PARAMS param = {};
            param.inputTimeStamp = data.pts;
            vector<NvPacketInfo> vPacketInfo;
            pEnc->EncodeDeviceFrame(data.dpVideoFrame, data.nPitch, vPacket, &vPacketInfo, &param);
            for (unsigned i = 0; i < vPacket.size(); i++) {
                pMuxer->MuxVideo(vPacket[i].data(), vPacket[i].size(), vPacketInfo[i].info.outputTimeStamp, vPacketInfo[i].dts);
                fpsLimiter.CheckAndSleep();
            }
            data.Free();
        }
    }

    vector<NvPacketInfo> vPacketInfo;
    pEnc->EndEncode(vPacket, &vPacketInfo);
    for (unsigned i = 0; i < vPacket.size(); i++) {
        pMuxer->MuxVideo(vPacket[i].data(), vPacket[i].size(), vPacketInfo[i].info.outputTimeStamp, vPacketInfo[i].dts);
    }

    ck(cuMemFree((CUdeviceptr)dpFrameResized));
}

void DecodeAndFilter(AVPacket * pkt, AVRational timebaseVideoStream, NvDecLite * pDec, VidFiltEx * pVidFiltOne, vector<TransData> &vTransData)
{
    bool bConvertNvPts = pkt->size && pkt->pts == INVALID_TIMESTAMP;
    uint8_t **ppFrameReturned = NULL;
    NvFrameInfo *pNvFrameInfo = NULL;
    int nFrameReturned = pDec->DecodeLockFrame(pkt->data, pkt->size, &ppFrameReturned, &pNvFrameInfo, CUVID_PKT_ENDOFPICTURE, pkt->pts);
    if (!nFrameReturned) {
        return;
    }

    for (int k = 0; k < nFrameReturned; k++) {
        int64_t pts = pNvFrameInfo[k].dispInfo.timestamp;
        if (bConvertNvPts) pts = av_rescale_q_rnd(pts, AVRational{ 1, 10000000 }, timebaseVideoStream, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
        vTransData.push_back(TransData(ppFrameReturned[k], pNvFrameInfo[k].nFramePitch, pNvFrameInfo[k].nWidth, pNvFrameInfo[k].nHeight, pts, pDec));
    }

    if (pVidFiltOne) {
        vector<TransData> vTransDataFiltered;
        pVidFiltOne->FilterDtoD(vTransData, vTransDataFiltered);
        for (TransData &data : vTransData) {
            data.Free();
        }
        vTransData = vTransDataFiltered;
    }
}

void DecodeAndFilter(AVPacket * pkt, VidDecEx *pDec, VidFiltEx * pVidFiltOne, vector<TransData> &vTransData)
{
    vTransData.clear();
    if (!pVidFiltOne) {
        pDec->DecodeFrmToD(pkt, vTransData);
        return;
    }

    vector<AVFrame *> vFrm;
    pDec->Decode(pkt, vFrm);
    pVidFiltOne->FilterFrmToD(vFrm, vTransData);
}

void DecodeVideoAndTransAudio(RoundQueue *pQueue, Demuxer *pDemuxer, NvDecLite *pNvDec, VidDecEx *pVidDec, 
    VidFiltEx *pVidFiltOne, vector<LazyMuxer *> &vpMuxer, const Options &options, const int iThread, volatile int *pnFps) 
{
    AudDec *pAudDec = NULL;
    AudEnc *pAudEnc = NULL;
    AudFilt *pAudFilt = NULL;

    AVPacket *pkt;
    bool bAudio;
    int nFps = 0;
    time_t t = time(NULL);
    do {
        pDemuxer->Demux(&pkt, &bAudio);
        if (bAudio) {
            if (!pAudDec) pAudDec = new AudDec(pDemuxer->GetAudioStream()->codecpar);
            vector<AVFrame *> vFrm;
            pAudDec->Decode(pkt, vFrm);
            for (AVFrame *frm : vFrm) {
                if (!options.strAudioFilterDesc.size()) {
                    EncodeAudioFrame(pAudEnc, vpMuxer, pDemuxer->GetAudioStream()->time_base, frm, *pQueue, options);
                    continue;
                }
                if (!pAudFilt) {
                    pAudFilt = new AudFilt((AVSampleFormat)frm->format, frm->sample_rate, frm->channel_layout, pDemuxer->GetAudioStream()->time_base, options.strAudioFilterDesc.c_str());
                }
                vector<AVFrame *> vFrmFilted;
                pAudFilt->Filter(frm, vFrmFilted);
                for (AVFrame *frmFiltered : vFrmFilted) {
                    EncodeAudioFrame(pAudEnc, vpMuxer, pAudFilt->GetOutputTimebase(), frmFiltered, *pQueue, options);
                }
            }
            continue;
        }
        if (!pkt->size && pAudEnc) {
            EncodeAudioFrame(pAudEnc, vpMuxer, AVRational{}, NULL, *pQueue, options);
        }

        vector<TransData> vTransData;
        if (pNvDec) {
            DecodeAndFilter(pkt, pDemuxer->GetVideoStream()->time_base, pNvDec, pVidFiltOne, vTransData);
        } else {
            DecodeAndFilter(pkt, pVidDec, pVidFiltOne, vTransData);
        }

        for (TransData &data : vTransData) {
            while (!pQueue->Append(data)) {
                this_thread::sleep_for(chrono::milliseconds(1));
            }

            nFps++;
            if (t != time(NULL)) {
                pnFps[iThread] = nFps;
                t = time(NULL);
                nFps = 0;
            }
        }
    } while (pkt->size);

    if (pAudDec) delete pAudDec;
    if (pAudEnc) delete pAudEnc;
    if (pAudFilt) delete pAudFilt;
}

void TransProc(CUcontext cuContext, const Options &options, int iSession, volatile int *pnFps, volatile int *pbEnd) {
    Demuxer demuxer(options.strInputFile.c_str(), false, true);
    if (demuxer.GetVideoStream() == NULL) {
        cout << "No video stream in " << options.strInputFile.c_str() << endl;
        return;
    }

    const int nTransData = 8;
    int nEnc = (int)options.vRes.size();
    RoundQueue *pQueue = new RoundQueue(nTransData, nEnc);
    VidFiltEx *pVidFiltOne = NULL;
    if (options.strVideoFilterDesc.size()) {
        pVidFiltOne = new VidFiltEx(cuContext, AV_PIX_FMT_NV12, 
            demuxer.GetVideoStream()->codecpar->width, demuxer.GetVideoStream()->codecpar->height, 
            demuxer.GetVideoStream()->time_base, AVRational{1,1}, options.strVideoFilterDesc.c_str());
    }

    vector<VidFiltEx *> vpVidFilt;
    vector<NvEncLite *> vpEnc;
    vector<LazyMuxer *> vpMuxer;
    for (int i = 0; i < nEnc; i++) {
        const Options::Resolution &res = options.vRes[i];
        VidFiltEx *pVidFilt = NULL;
        if (res.strVideoFilterDesc.size()) {
            pVidFilt = new VidFiltEx(cuContext, AV_PIX_FMT_NV12, res.nWidth, res.nHeight, 
                pVidFiltOne ? pVidFiltOne->GetOutputTimebase() : demuxer.GetVideoStream()->time_base, 
                AVRational{1,1}, res.strVideoFilterDesc.c_str());
        }
        vpVidFilt.push_back(pVidFilt);

        string strParam = options.strVideoEncParam + (res.strVideoEncParamSuffix.size() ? (string(":") + res.strVideoEncParamSuffix) : "");
        vpEnc.push_back(new NvEncLite(cuContext, res.nWidth, res.nHeight, NV_ENC_BUFFER_FORMAT_NV12, std::shared_ptr<NvEncoderInitParam>(new NvEncoderInitParam(strParam.c_str())).get()));
        
        string strName = boost::replace_all_copy(res.strOutputFile, "#", to_string(iSession));
        vpMuxer.push_back(new LazyMuxer(strName.c_str(), res.strOutputFormat.size() ? res.strOutputFormat.c_str() : NULL));
        AVCodecParameters *vpar = ExtractAVCodecParameters(vpEnc[i]);
        vpMuxer[i]->SetVideoStream(vpar, pVidFilt ? pVidFilt->GetOutputTimebase() : 
            (pVidFiltOne ? pVidFiltOne->GetOutputTimebase() : demuxer.GetVideoStream()->time_base));
        avcodec_parameters_free(&vpar);
        if (demuxer.GetAudioStream() == NULL) vpMuxer[i]->SetAudioStream(NULL, AVRational{0, 1});
    }
    
    NvDecLite *pNvDec = NULL;
    VidDecEx *pVidDec = NULL;
    if (options.bUseSwVideoDecoder) {
        pVidDec = new VidDecEx(cuContext, demuxer.GetVideoStream()->avg_frame_rate, demuxer.GetVideoStream()->time_base, demuxer.GetVideoStream()->codecpar);
    } else {
        pNvDec = new NvDecLiteEx(cuContext, true, FFmpeg2NvCodecId(demuxer.GetVideoStream()->codecpar->codec_id), false, true);
    }

    vector<thread *> vpth;
    for (int i = 0; i < nEnc; i++) {
        vpth.push_back(new thread(EncodeVideoProc, pQueue, i, vpVidFilt[i], vpEnc[i], vpMuxer[i], i == 0 ? options.nFpsLimit : 0));
    }
    DecodeVideoAndTransAudio(pQueue, &demuxer, pNvDec, pVidDec, pVidFiltOne, vpMuxer, options, iSession, pnFps);
    pQueue->SetEof();
    for (auto pth : vpth) {
        pth->join();
        delete pth;
    }

    for (int i = 0; i < nEnc; i++) {
        if (vpVidFilt[i]) delete vpVidFilt[i];
        delete vpEnc[i];
        delete vpMuxer[i];
    }
    // pNvDec/pVidDec and pVidFiltOne must be alive when deleting pQueue
    delete pQueue;
    if (pNvDec) delete pNvDec;
    if (pVidDec) delete pVidDec;
    if (pVidFiltOne) delete pVidFiltOne;

    pbEnd[iSession] = 1;
}

int main(int argc, char *argv[]) {
    if (argc == 1) {
        cout << "Usage: " << argv[0] << " <options_xml> <#session>(optional)" << endl;
        cout << NvEncoderInitParam().GetHelpMessage(false, false, true);
        return 1;
    }

    Options options;
    try {
        options.Load(argv[1]);
    } catch (const std::exception& e) {
        cout << e.what() << endl;
        return 1;
    }
    if (argc >= 3) {
        int nSession = atoi(argv[2]);
        if (nSession > 0) {
            options.nSession = nSession;
        }
    }

    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    if (options.iGpu < 0 || options.iGpu >= nGpu) {
        cout << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << endl;
        return 1;
    }
    CUdevice cuDevice = 0;
    ck(cuDeviceGet(&cuDevice, options.iGpu));
    char szDeviceName[80];
    ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    cout << "GPU: " << szDeviceName << endl;
    CUcontext cuContext = NULL;
    ck(cuCtxCreate(&cuContext, 0, cuDevice));

    vector<int> vnFps(options.nSession);
    vector<int> vbEnd(options.nSession);
    vector<thread *> vpth;
    for (int i = 0; i < options.nSession; i++) {
        vpth.push_back(new thread(TransProc, cuContext, options, i, vnFps.data(), vbEnd.data()));
    }

    bool bAllEnd;
    do {
        cout << "FPS(" << options.nSession << ")";
        double sum = 0;
        for (int nFps : vnFps) {
            sum += nFps;
            cout << " " << nFps;
        }
        cout << " AVG=" << sum / options.nSession << "\t\t\r";
        cout.flush();
        bAllEnd = true;
        for (int bEnd : vbEnd) {
            if (!bEnd) {
                this_thread::sleep_for(chrono::seconds(1));
                bAllEnd = false;
                break;
            }
        }
    } while (!bAllEnd);
    cout << endl;

    for (thread *th : vpth) {
        th->join();
        delete th;
    }

    return 0;
}
