#include <iostream>
#include <fstream>
#include <set>
#include <memory>
#include "AvToolkit/Demuxer.h"
#include "NvCodec/NvDecLite.h"
#include "NvCodec/NvCommon.h"
#include "HevcParser/HevcParser.h"
#include "Logger.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

using namespace std;

namespace HEVC {

void PrintRefPicSet(ShortTermRefPicSet &set, int poc) {
    cout << "REF: #" << poc << " negative=" << set.num_negative_pics << " (";
    for (unsigned u : set.delta_poc_s0_minus1) {
        cout << u << ",";
    }
    cout << "), positive=" << set.num_positive_pics << " (";
    for (unsigned u : set.delta_poc_s1_minus1) {
        cout << u << ",";
    }
    cout << ")";

    if (!poc) {
        cout << endl;
        return;
    }
    cout << ": ";
    for (int u : set.delta_poc_s0_minus1) {
        cout << poc - (u + 1) << ",";
    }
    for (int u : set.delta_poc_s1_minus1) {
        cout << poc + (u + 1) << ",";
    }
    cout << endl;
}

class SimpleConsumer: public Parser::Consumer {
public:
    virtual ~SimpleConsumer() {}
    virtual void onNALUnit(std::shared_ptr<NALUnit> pNALUnit, const Parser::Info *pInfo) override {
        static std::shared_ptr<HEVC::SPS> sps;
        switch (pNALUnit->m_nalHeader.type) {
        case NAL_SPS: {
            sps = std::dynamic_pointer_cast<HEVC::SPS>(pNALUnit);
            for (unsigned stRpsIdx = 0; stRpsIdx < sps->short_term_ref_pic_set.size(); stRpsIdx++) {
                ShortTermRefPicSet &set = sps->short_term_ref_pic_set[stRpsIdx];
                if (!set.inter_ref_pic_set_prediction_flag) {
                    continue;
                }

                ShortTermRefPicSet &srcSet = sps->short_term_ref_pic_set[stRpsIdx - (set.delta_idx_minus1 + 1)];
                std::size_t srcNumDeltaPocs = 0;
                if (srcSet.inter_ref_pic_set_prediction_flag) {
                    for (std::size_t i = 0; i < srcSet.used_by_curr_pic_flag.size(); i++) {
                        if (srcSet.used_by_curr_pic_flag[i] || srcSet.use_delta_flag[i]) {
                            srcNumDeltaPocs++;
                        }
                    }
                } else {
                    srcNumDeltaPocs = srcSet.num_negative_pics + srcSet.num_positive_pics;
                }

                int deltaRps = (1 - 2 * set.delta_rps_sign) * (set.abs_delta_rps_minus1 + 1);
                for (int j = srcSet.num_positive_pics - 1; j >= 0; j--) {
                    unsigned dPoc = srcSet.delta_poc_s1_minus1[j] + 1 + deltaRps;
                    if (dPoc < 0 && set.use_delta_flag[srcSet.num_negative_pics + j]) {
                        set.delta_poc_s0_minus1.push_back(-dPoc - 1);
                        set.used_by_curr_pic_s0_flag.push_back(set.used_by_curr_pic_flag[srcSet.num_negative_pics + j]);
                    }
                }
                if (deltaRps < 0 && set.use_delta_flag[srcNumDeltaPocs]) {
                    set.delta_poc_s0_minus1.push_back(-deltaRps - 1);
                    set.used_by_curr_pic_s0_flag.push_back(set.used_by_curr_pic_flag[srcNumDeltaPocs]);
                }
                for (int j = 0; j < srcSet.num_negative_pics; j++) {
                    int dPoc = -(srcSet.delta_poc_s0_minus1[j] + 1) + deltaRps;
                    if (dPoc < 0 && set.use_delta_flag[j]) {
                        set.delta_poc_s0_minus1.push_back(-dPoc - 1);
                        set.used_by_curr_pic_s0_flag.push_back(set.used_by_curr_pic_flag[j]);
                    }
                }
                set.num_negative_pics = set.delta_poc_s0_minus1.size();

                for (int j = srcSet.num_negative_pics - 1; j >= 0; j--) {
                    int dPoc = -(srcSet.delta_poc_s0_minus1[j] + 1) + deltaRps;
                    if (dPoc > 0 && set.use_delta_flag[j]) {
                        set.delta_poc_s1_minus1.push_back(dPoc - 1);
                        set.used_by_curr_pic_s1_flag.push_back(set.used_by_curr_pic_flag[j]);
                    }
                }
                if (deltaRps > 0 && set.use_delta_flag[srcNumDeltaPocs]) {
                    set.delta_poc_s1_minus1.push_back(deltaRps - 1);
                    set.used_by_curr_pic_s1_flag.push_back(set.used_by_curr_pic_flag[srcNumDeltaPocs]);
                }
                for (int j = 0; j < srcSet.num_positive_pics; j++) {
                    int dPoc = srcSet.delta_poc_s1_minus1[j] + 1 + deltaRps;
                    if (dPoc > 0 && set.use_delta_flag[srcSet.num_negative_pics + j]) {
                        set.delta_poc_s1_minus1.push_back(dPoc - 1);
                        set.used_by_curr_pic_s1_flag.push_back(set.used_by_curr_pic_flag[srcSet.num_negative_pics + j]);
                    }
                }
                set.num_positive_pics = set.delta_poc_s1_minus1.size();
            }
            break;
        }
        case NAL_TRAIL_N:
        case NAL_TRAIL_R:
        case NAL_TSA_N:
        case NAL_TSA_R:
        case NAL_STSA_N:
        case NAL_STSA_R:
        case NAL_RADL_N:
        case NAL_RADL_R:
        case NAL_RASL_N:
        case NAL_RASL_R:
        case NAL_BLA_W_LP:
        case NAL_BLA_W_RADL:
        case NAL_BLA_N_LP:
        case NAL_IDR_W_RADL:
        case NAL_IDR_N_LP:
        case NAL_CRA_NUT: {
            std::shared_ptr<HEVC::Slice> slice = std::dynamic_pointer_cast<HEVC::Slice>(pNALUnit);
            if (slice->short_term_ref_pic_set_sps_flag) {
                slice->short_term_ref_pic_set = sps->short_term_ref_pic_set[slice->short_term_ref_pic_set_idx];
            }
            break;
        }
        }
    }
    virtual void onWarning(const std::string &warning, const Parser::Info *pInfo, Parser::WarningType type) override {
        LOG(WARNING) << "Parser warning: " << warning;
    }
};
}

std::set<uint32_t> GetRefPicSet(const HEVC::Slice &slice) {
    std::set<uint32_t> ret;
    for (int u : slice.short_term_ref_pic_set.delta_poc_s0_minus1) {
        ret.insert(slice.slice_pic_order_cnt_lsb - (u + 1));
    }
    for (int u : slice.short_term_ref_pic_set.delta_poc_s1_minus1) {
        ret.insert(slice.slice_pic_order_cnt_lsb + (u + 1));
    }
    return ret;
}

int DecodeFile(CUcontext cuContext, const char *szInFilePath, const char *szOutFilePath, bool bSlow)
{
    Demuxer demuxer(szInFilePath);
    cudaVideoCodec eCodec = FFmpeg2NvCodecId(demuxer.GetVideoStream()->codecpar->codec_id);
    NvDecLite dec(cuContext, false, eCodec);

    auto pParser = unique_ptr<HEVC::Parser>(HEVC::Parser::create());
    HEVC::SimpleConsumer consumer;
    pParser->addConsumer(&consumer);

    shared_ptr<HEVC::Slice> pLastSlice;
    vector<uint8_t> vLastPacket;
    int nTotal = 0, nSkip = 0;

    AVPacket *pkt;
    int nFrame = 0;
    ofstream fOut(szOutFilePath, ios::out | ios::binary);
    do {
        demuxer.Demux(&pkt);
        uint8_t **ppFrame = NULL;
        NvFrameInfo *pInfo = NULL;
        int nFrameReturned = 0;
        if (!pkt->data) {
            nFrameReturned = dec.Decode(NULL, 0, &ppFrame, &pInfo);
        } else {
            nTotal++;
            vector<HEVC::Data_NALUnit> vData_NALUnit = pParser->parse(pkt->data, pkt->size);
            shared_ptr<HEVC::Slice> pSlice;
            for (HEVC::Data_NALUnit dn : vData_NALUnit) {
                if (dn.pNALUnit->m_nalHeader.type <= HEVC::NAL_CRA_NUT) {
                    pSlice = std::dynamic_pointer_cast<HEVC::Slice>(dn.pNALUnit);
                    break;
                }
            }
            if (pLastSlice) {
                auto s = GetRefPicSet(*pSlice);
                if (s.find(pLastSlice->slice_pic_order_cnt_lsb) != s.end()) {
                    cout << "Decode " << pLastSlice->slice_pic_order_cnt_lsb << endl;
                    nFrameReturned = dec.Decode(vLastPacket.data(), vLastPacket.size(), &ppFrame, &pInfo);
                } else {
                    cout << "Skip " << pLastSlice->slice_pic_order_cnt_lsb << endl;
                    nSkip++;
                }
            }
            pLastSlice = pSlice;
            vLastPacket.clear();
            vLastPacket.insert(vLastPacket.begin(), pkt->data, pkt->data + pkt->size);
        }
        for (int i = 0; i < nFrameReturned; i++) {
            fOut.write(reinterpret_cast<char *>(ppFrame[i]), pInfo[i].nFrameSize);
        }
        nFrame += nFrameReturned;
    } while (pkt->size);

    cout << "Skipped: " << nSkip << "/" << nTotal << " = " << 1.0 * nSkip / nTotal << endl;
    return nFrame;
}

int run_test() {
    const char *szFilePath = "ref_in_slice.265";
    ifstream file(szFilePath, ios::in | ios::binary);
    if (!file.is_open()) {
        cout << "File not opened: " << szFilePath << endl;
        return 1;
    }
    streampos beg, end;
    beg = file.tellg();
    file.seekg(0, ios::end);
    end = file.tellg();
    size_t nByte = end - beg;
    shared_ptr<uint8_t[]> buf(new uint8_t[nByte]);
    file.seekg(0, ios::beg);
    file.read((char *)buf.get(), nByte);
    if (file.gcount() != nByte) {
        cout << "IO error: " << szFilePath << ", file.gcount()=" << file.gcount() << ", nByte=" << nByte << endl;
        cerr << "Error: " << strerror(errno) << endl;
        return 1;
    }
    file.close();

    HEVC::Parser *pParser = HEVC::Parser::create();
    HEVC::SimpleConsumer consumer;
    pParser->addConsumer(&consumer);
    vector<HEVC::Data_NALUnit> vData_NALUnit = pParser->parse(buf.get(), file.gcount());
    HEVC::Parser::release(pParser);

    shared_ptr<HEVC::Slice> pLastSlice;
    for (HEVC::Data_NALUnit dn : vData_NALUnit) {
        if (dn.pNALUnit->m_nalHeader.type > HEVC::NAL_CRA_NUT) {
            continue;
        }
        shared_ptr<HEVC::Slice> pSlice = std::dynamic_pointer_cast<HEVC::Slice>(dn.pNALUnit);
        PrintRefPicSet(pSlice->short_term_ref_pic_set, pSlice->slice_pic_order_cnt_lsb);
        std::set<uint32_t> s = GetRefPicSet(*pSlice);
        pLastSlice = pSlice;
    }

    return 0;
}

int main(int argc, char *argv[]) {
//    run_test();
    const char *szFilePath = "ref_in_sps.265";
    if (argc > 1) {
        szFilePath = argv[1];
    }

    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    int iGpu = 0;
    if (iGpu < 0 || iGpu >= nGpu) {
        cout << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << endl;
        return 1;
    }
    CUdevice cuDevice = 0;
    ck(cuDeviceGet(&cuDevice, iGpu));
    char szDeviceName[80];
    ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    cout << "GPU in use: " << szDeviceName << endl;
    CUcontext cuContext = NULL;
    ck(cuCtxCreate(&cuContext, 0, cuDevice));

    DecodeFile(cuContext, szFilePath, "out.yuv", false);
}
