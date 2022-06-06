#pragma once
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <iterator>
#include <cstring>
#include <cctype>
#include <functional>
#include "Logger.h"
#include "NvCodec/nvEncodeAPI.h"

extern simplelogger::Logger *logger;

#ifndef _WIN32
#include <cstring>
inline bool operator==(const GUID &guid1, const GUID &guid2) {
    return !memcmp(&guid1, &guid2, sizeof(GUID));
}

inline bool operator!=(const GUID &guid1, const GUID &guid2) {
    return !(guid1 == guid2);
}
#endif

class NvEncoderInitParam {
public:
    NvEncoderInitParam(const char *szParam = "", 
        std::function<void(NV_ENC_INITIALIZE_PARAMS *pParams)> *pfuncInit = NULL) 
        : strParam(szParam)
    {
        if (pfuncInit) {
            funcInit = *pfuncInit;
        }

        strParam = LeftTrimString(strParam);
        if (strParam.length() && strParam[0] != '-') {
            strParam = std::string("-") + ReplaceString(ReplaceString(strParam, ":", " -"), "=", " ");
        }

        std::transform(strParam.begin(), strParam.end(), strParam.begin(), [](unsigned char c){ return std::tolower(c);});
        std::istringstream ss(strParam);
        tokens = std::vector<std::string> {
            std::istream_iterator<std::string>(ss),
            std::istream_iterator<std::string>() 
        };

        for (unsigned i = 0; i < tokens.size(); i++) {
            if (tokens[i] == "-codec" && ++i != tokens.size()) {
                ParseString("-codec", tokens[i], vCodec, szCodecNames, &guidCodec);
                continue;
            }
            if (tokens[i] == "-preset" && ++i != tokens.size()) {
                ParseString("-preset", tokens[i], vPreset, szPresetNames, &guidPreset);
                continue;
            }
        }
    }
    virtual ~NvEncoderInitParam() {}
    virtual bool IsCodecH264() {
        return GetEncodeGUID() == NV_ENC_CODEC_H264_GUID;
    }
    std::string GetHelpMessage(bool bMeOnly = false, bool bUnbuffered = false, bool bHide444 = false) {
        std::ostringstream oss;
            oss << "-codec       Codec: " << szCodecNames << std::endl
                << "-preset      Preset: " << szPresetNames << std::endl
                << "-profile     H264: " << szH264ProfileNames << "; HEVC: " << szHevcProfileNames << std::endl;
        if (!bHide444) {
            oss << "-444         (Only for RGB input) YUV444 encode" << std::endl;
        }
        if (bMeOnly) return oss.str();
            oss << "-rc          Rate control mode: " << szRcModeNames << std::endl
                << "-fps         Frame rate" << std::endl
                << "-gop         Length of GOP (Group of Pictures)" << std::endl;
        if (!bUnbuffered) {
            oss << "-bf          Number of consecutive B-frames" << std::endl;
        }
            oss << "-bitrate     Average bit rate, can be in unit of 1, K, M" << std::endl
                << "-maxbitrate  Max bit rate, can be in unit of 1, K, M" << std::endl
                << "-vbvbufsize  VBV buffer size in bits, can be in unit of 1, K, M" << std::endl
                << "-vbvinit     VBV initial delay in bits, can be in unit of 1, K, M" << std::endl
                << "-aq          Enable spatial AQ and set its stength (range 1-15, 0-auto)" << std::endl
                << "-temporalaq  (No value) Enable temporal AQ" << std::endl;
        if (!bUnbuffered) {
            oss << "-lookahead   Maximum depth of lookahead (range 0-32)" << std::endl;
        }
            oss << "-cq          Target constant quality level for VBR mode (range 1-51, 0-auto)" << std::endl
                << "-qmin        Min QP value" << std::endl
                << "-qmax        Max QP value" << std::endl
                << "-initqp      Initial QP value" << std::endl
                << "-constqp     QP value for constqp rate control mode" << std::endl
                << "Note: QP value can be in the form of qp_of_P_B_I or qp_P,qp_B,qp_I (no space)" << std::endl;
        if (bUnbuffered) {
            oss << "Note: Options -bf and -lookahead are unavailable for this app" << std::endl;
        }
        return oss.str();
    }
    std::string MainParamToString(const NV_ENC_INITIALIZE_PARAMS *pParams) {
        std::ostringstream os;
        os 
            << "Encoding Parameters:" 
            << std::endl << "\tcodec        : " << ConvertValueToString(vCodec, szCodecNames, pParams->encodeGUID)
            << std::endl << "\tpreset       : " << ConvertValueToString(vPreset, szPresetNames, pParams->presetGUID)
            << std::endl << "\tprofile      : " << ConvertValueToString(vProfile, szProfileNames, pParams->encodeConfig->profileGUID)
            << std::endl << "\tchroma       : " << ConvertValueToString(vChroma, szChromaNames, (pParams->encodeGUID == NV_ENC_CODEC_H264_GUID) ? pParams->encodeConfig->encodeCodecConfig.h264Config.chromaFormatIDC : pParams->encodeConfig->encodeCodecConfig.hevcConfig.chromaFormatIDC)
            << std::endl << "\tbitdepth     : " << ((pParams->encodeGUID == NV_ENC_CODEC_H264_GUID) ? 0 : pParams->encodeConfig->encodeCodecConfig.hevcConfig.pixelBitDepthMinus8) + 8
            << std::endl << "\trc           : " << ConvertValueToString(vRcMode, szRcModeNames, pParams->encodeConfig->rcParams.rateControlMode)
            ;
            if (pParams->encodeConfig->rcParams.rateControlMode == NV_ENC_PARAMS_RC_CONSTQP) {
                os << " (P,B,I=" << pParams->encodeConfig->rcParams.constQP.qpInterP << "," << pParams->encodeConfig->rcParams.constQP.qpInterB << "," << pParams->encodeConfig->rcParams.constQP.qpIntra << ")";
            }
        os
            << std::endl << "\tfps          : " << pParams->frameRateNum << "/" << pParams->frameRateDen
            << std::endl << "\tgop          : " << (pParams->encodeConfig->gopLength == NVENC_INFINITE_GOPLENGTH ? "INF" : std::to_string(pParams->encodeConfig->gopLength))
            << std::endl << "\tbf           : " << pParams->encodeConfig->frameIntervalP - 1
            << std::endl << "\tsize         : " << pParams->encodeWidth << "x" << pParams->encodeHeight
            << std::endl << "\tbitrate      : " << pParams->encodeConfig->rcParams.averageBitRate
            << std::endl << "\tmaxbitrate   : " << pParams->encodeConfig->rcParams.maxBitRate
            << std::endl << "\tvbvbufsize   : " << pParams->encodeConfig->rcParams.vbvBufferSize
            << std::endl << "\tvbvinit      : " << pParams->encodeConfig->rcParams.vbvInitialDelay
            << std::endl << "\taq           : " << (pParams->encodeConfig->rcParams.enableAQ ? (pParams->encodeConfig->rcParams.aqStrength ? std::to_string(pParams->encodeConfig->rcParams.aqStrength) : "auto") : "disabled")
            << std::endl << "\ttemporalaq   : " << (pParams->encodeConfig->rcParams.enableTemporalAQ ? "enabled" : "disabled")
            << std::endl << "\tlookahead    : " << (pParams->encodeConfig->rcParams.enableLookahead ? std::to_string(pParams->encodeConfig->rcParams.lookaheadDepth) : "disabled")
            << std::endl << "\tcq           : " << (int)pParams->encodeConfig->rcParams.targetQuality
            << std::endl << "\tqmin         : P,B,I=" << pParams->encodeConfig->rcParams.minQP.qpInterP << "," << pParams->encodeConfig->rcParams.minQP.qpInterB << "," << pParams->encodeConfig->rcParams.minQP.qpIntra
            << std::endl << "\tqmax         : P,B,I=" << pParams->encodeConfig->rcParams.maxQP.qpInterP << "," << pParams->encodeConfig->rcParams.maxQP.qpInterB << "," << pParams->encodeConfig->rcParams.maxQP.qpIntra
            << std::endl << "\tinitqp       : P,B,I=" << pParams->encodeConfig->rcParams.initialRCQP.qpInterP << "," << pParams->encodeConfig->rcParams.initialRCQP.qpInterB << "," << pParams->encodeConfig->rcParams.initialRCQP.qpIntra
            ;
        return os.str();
    }

protected:
    virtual GUID GetEncodeGUID() { return guidCodec; }
    virtual GUID GetPresetGUID() { return guidPreset; }
    virtual bool SetInitParams(NV_ENC_INITIALIZE_PARAMS *pParams) {
        NV_ENC_CONFIG &config = *pParams->encodeConfig;
        for (unsigned i = 0; i < tokens.size(); i++) {
            if (
                i < tokens.size() && tokens[i] == "-codec"      && ++i ||
                i < tokens.size() && tokens[i] == "-preset"     && ++i ||
                i < tokens.size() && tokens[i] == "-profile"    && ++i != tokens.size() && (IsCodecH264() ?  ParseString("-profile", tokens[i], vH264Profile, szH264ProfileNames, &config.profileGUID) :  ParseString("-profile", tokens[i], vHevcProfile, szHevcProfileNames, &config.profileGUID)) ||
                i < tokens.size() && tokens[i] == "-rc"         && ++i != tokens.size() && ParseString("-rc",          tokens[i], vRcMode, szRcModeNames, &config.rcParams.rateControlMode)                    ||
                i < tokens.size() && tokens[i] == "-fps"        && ++i != tokens.size() && ParseFps("-fps",            tokens[i], &pParams->frameRateNum, &pParams->frameRateDen)                              ||
                i < tokens.size() && tokens[i] == "-gop"        && ++i != tokens.size() && ParseInt("-gop",            tokens[i], &config.gopLength)                                                           ||
                i < tokens.size() && tokens[i] == "-bf"         && ++i != tokens.size() && ParseInt("-bf",             tokens[i], &config.frameIntervalP) && ++config.frameIntervalP                           ||
                i < tokens.size() && tokens[i] == "-bitrate"    && ++i != tokens.size() && ParseBitRate("-bitrate",    tokens[i], &config.rcParams.averageBitRate)                                             ||
                i < tokens.size() && tokens[i] == "-maxbitrate" && ++i != tokens.size() && ParseBitRate("-maxbitrate", tokens[i], &config.rcParams.maxBitRate)                                                 ||
                i < tokens.size() && tokens[i] == "-vbvbufsize" && ++i != tokens.size() && ParseBitRate("-vbvbufsize", tokens[i], &config.rcParams.vbvBufferSize)                                              ||
                i < tokens.size() && tokens[i] == "-vbvinit"    && ++i != tokens.size() && ParseBitRate("-vbvinit",    tokens[i], &config.rcParams.vbvInitialDelay)                                            ||
                i < tokens.size() && tokens[i] == "-lookahead"  && ++i != tokens.size() && ParseInt("-lookahead",      tokens[i], &config.rcParams.lookaheadDepth) && (config.rcParams.enableLookahead = true) ||
                i < tokens.size() && tokens[i] == "-cq"         && ++i != tokens.size() && ParseInt("-cq",             tokens[i], &config.rcParams.targetQuality)                                              ||
                i < tokens.size() && tokens[i] == "-initqp"     && ++i != tokens.size() && ParseQp("-initqp",          tokens[i], &config.rcParams.initialRCQP) && (config.rcParams.enableInitialRCQP = true)  ||
                i < tokens.size() && tokens[i] == "-qmin"       && ++i != tokens.size() && ParseQp("-qmin",            tokens[i], &config.rcParams.minQP) && (config.rcParams.enableMinQP = true)              ||
                i < tokens.size() && tokens[i] == "-qmax"       && ++i != tokens.size() && ParseQp("-qmax",            tokens[i], &config.rcParams.maxQP) && (config.rcParams.enableMaxQP = true)              ||
                i < tokens.size() && tokens[i] == "-constqp"    && ++i != tokens.size() && ParseQp("-constqp",         tokens[i], &config.rcParams.constQP)                                                    ||
                i < tokens.size() && tokens[i] == "-temporalaq" && (config.rcParams.enableTemporalAQ = true)
            ) {
                continue;
            }
            int aqStrength;
            if (i < tokens.size() && tokens[i] == "-aq" && ++i != tokens.size() && ParseInt("-aq", tokens[i], &aqStrength)) {
                config.rcParams.enableAQ = true;
                config.rcParams.aqStrength = aqStrength;
                continue;
            }
            if (i < tokens.size() && tokens[i] == "-444") {
                if (IsCodecH264()) {
                    config.encodeCodecConfig.h264Config.chromaFormatIDC = 3;
                } else {
                    config.encodeCodecConfig.hevcConfig.chromaFormatIDC = 3;
                }
                continue;
            }
            LOG(ERROR) << "Wrong parameter: " << (i < tokens.size() ? tokens[i] : tokens[tokens.size() - 1]) << std::endl << GetHelpMessage();
            return false;
        }
        funcInit(pParams);
        return true;
    }

private:
    template<typename T>
    bool ParseString(const std::string &strName, const std::string &strValue, const std::vector<T> &vValue, const std::string &strValueNames, T *pValue) {
        std::vector<std::string> vstrValueName = Split(strValueNames, ' ');
        auto it = std::find(vstrValueName.begin(), vstrValueName.end(), strValue);
        if (it == vstrValueName.end()) {
            LOG(ERROR) << strName << " options: " << strValueNames;
            return false;
        }
        *pValue = vValue[it - vstrValueName.begin()];
        return true;
    }
    template<typename T>
    std::string ConvertValueToString(const std::vector<T> &vValue, const std::string &strValueNames, T value) {
        auto it = std::find(vValue.begin(), vValue.end(), value);
        if (it == vValue.end()) {
            LOG(ERROR) << "Invalid value. Allowed values: " << strValueNames;
            return std::string();
        }
        return Split(strValueNames, ' ')[it - vValue.begin()];
    }
    bool ParseBitRate(const std::string &strName, const std::string &strValue, unsigned *pBitRate) {
        try {
            size_t l;
            double r = std::stod(strValue, &l);
            char c = strValue[l];
            if (c != 0 && c != 'k' && c != 'm') {
                LOG(ERROR) << strName << " units: 1, K, M (lower case also allowed)";
            }
            *pBitRate = (unsigned)((c == 'm' ? 1000000 : (c == 'k' ? 1000 : 1)) * r);
        } catch (std::invalid_argument) {
            return false;
        }
        return true;
    }
    bool ParseFps(const std::string &strName, const std::string &strValue, uint32_t *pFpsNum, uint32_t *pFpsDen) {
        size_t m = strValue.find('/');
        if (m == std::string::npos) {
            if (!ParseInt(strName, strValue, pFpsNum)) {
                return false;
            }
            *pFpsDen = 1;
            return true;
        }
        return ParseInt(strName, strValue.substr(0, m), pFpsNum) && ParseInt(strName, strValue.substr(m + 1), pFpsDen);
    }
    template<typename T>
    bool ParseInt(const std::string &strName, const std::string &strValue, T *pInt) {
        try {
            int v = std::stoi(strValue);
            if (v < 0) {
                throw std::invalid_argument("");
            }
            *pInt = (T)v;
        } catch (std::invalid_argument) {
            LOG(ERROR) << strName << " need a value of positive number";
            return false;
        }
        return true;
    }
    bool ParseQp(const std::string &strName, const std::string &strValue, NV_ENC_QP *pQp) {
        std::vector<std::string> vQp = Split(strValue, ',');
        try {
            if (vQp.size() == 1) {
                unsigned qp = (unsigned)std::stoi(vQp[0]);
                *pQp = {qp, qp, qp};
            } else if (vQp.size() == 3) {
                *pQp = {(unsigned)std::stoi(vQp[0]), (unsigned)std::stoi(vQp[1]), (unsigned)std::stoi(vQp[2])};
            } else {
                LOG(ERROR) << strName << " qp_for_P_B_I or qp_P,qp_B,qp_I (no space is allowed)";
                return false;
            }
        } catch (std::invalid_argument) {
            return false;
        }
        return true;
    }
    static std::vector<std::string> Split(const std::string &s, char delim) {
        std::stringstream ss(s);
        std::string token;
        std::vector<std::string> tokens;
        while (getline(ss, token, delim)) {
            tokens.push_back(token);
        }
        return tokens;
    }
    static std::string LeftTrimString(std::string s) {
        while (s.length()) {
            if (!isspace(s[0])) {
                break;
            }
            s.replace(0, 1, "");
        }
        return s;
    }
    static std::string ReplaceString(std::string subject, const std::string& search, const std::string& replace) {
        size_t pos = 0;
        while ((pos = subject.find(search, pos)) != std::string::npos) {
             subject.replace(pos, search.length(), replace);
             pos += replace.length();
        }
        return subject;
    }

private:
    std::string strParam;
    std::function<void(NV_ENC_INITIALIZE_PARAMS *pParams)> funcInit = [](NV_ENC_INITIALIZE_PARAMS *pParams){};
    std::vector<std::string> tokens;
    GUID guidCodec = NV_ENC_CODEC_H264_GUID;
    GUID guidPreset = NV_ENC_PRESET_P1_GUID;
    
    const char *szCodecNames = "h264 hevc";
    std::vector<GUID> vCodec = std::vector<GUID> {
        NV_ENC_CODEC_H264_GUID,
        NV_ENC_CODEC_HEVC_GUID
    };
    
    const char *szChromaNames = "yuv420 yuv444";
    std::vector<uint32_t> vChroma = std::vector<uint32_t> {
        1, 3
    };
    
    const char *szPresetNames = "p1 p2 p3 p4 p5 p6 p7";
    std::vector<GUID> vPreset = std::vector<GUID> {
        NV_ENC_PRESET_P1_GUID,
        NV_ENC_PRESET_P2_GUID,
        NV_ENC_PRESET_P3_GUID,
        NV_ENC_PRESET_P4_GUID,
        NV_ENC_PRESET_P5_GUID,
        NV_ENC_PRESET_P6_GUID,
        NV_ENC_PRESET_P7_GUID,
    };

    const char *szH264ProfileNames = "baseline main high high444";
    std::vector<GUID> vH264Profile = std::vector<GUID> {
        NV_ENC_H264_PROFILE_BASELINE_GUID,
        NV_ENC_H264_PROFILE_MAIN_GUID,
        NV_ENC_H264_PROFILE_HIGH_GUID,
        NV_ENC_H264_PROFILE_HIGH_444_GUID,
    };
    const char *szHevcProfileNames = "main main10 frext";
    std::vector<GUID> vHevcProfile = std::vector<GUID> {
        NV_ENC_HEVC_PROFILE_MAIN_GUID,
        NV_ENC_HEVC_PROFILE_MAIN10_GUID,
        NV_ENC_HEVC_PROFILE_FREXT_GUID,
    };
    const char *szProfileNames = "(default) auto baseline(h264) main(h264) high(h264) high444(h264)"
        " stereo(h264) progressiv_high(h264) constrained_high(h264)"
        " main(hevc) main10(hevc) frext(hevc)";
    std::vector<GUID> vProfile = std::vector<GUID> {
        GUID{},
        NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID,
        NV_ENC_H264_PROFILE_BASELINE_GUID,
        NV_ENC_H264_PROFILE_MAIN_GUID,
        NV_ENC_H264_PROFILE_HIGH_GUID,
        NV_ENC_H264_PROFILE_HIGH_444_GUID,
        NV_ENC_H264_PROFILE_STEREO_GUID,
        NV_ENC_H264_PROFILE_PROGRESSIVE_HIGH_GUID,
        NV_ENC_H264_PROFILE_CONSTRAINED_HIGH_GUID,
        NV_ENC_HEVC_PROFILE_MAIN_GUID,
        NV_ENC_HEVC_PROFILE_MAIN10_GUID,
        NV_ENC_HEVC_PROFILE_FREXT_GUID,
    };

    const char *szRcModeNames = "constqp vbr cbr cbr_ll_hq cbr_hq vbr_hq";
    std::vector<NV_ENC_PARAMS_RC_MODE> vRcMode = std::vector<NV_ENC_PARAMS_RC_MODE> {
        NV_ENC_PARAMS_RC_CONSTQP,
        NV_ENC_PARAMS_RC_VBR,
        NV_ENC_PARAMS_RC_CBR,
        NV_ENC_PARAMS_RC_CBR_LOWDELAY_HQ,
        NV_ENC_PARAMS_RC_CBR_HQ,
        NV_ENC_PARAMS_RC_VBR_HQ,
    };

    const char *szTuningInfo = "undefined high_quality low_latency ultra_low_latency lossless";
    std::vector<NV_ENC_TUNING_INFO> vTuningInfo = std::vector<NV_ENC_TUNING_INFO> {
        NV_ENC_TUNING_INFO_UNDEFINED,
        NV_ENC_TUNING_INFO_HIGH_QUALITY,
        NV_ENC_TUNING_INFO_LOW_LATENCY,
        NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY,
        NV_ENC_TUNING_INFO_LOSSLESS,
    };

    const char *szQpMapMode = "disabled emphasis delta map";
    std::vector<NV_ENC_QP_MAP_MODE> vQpMapMode = std::vector<NV_ENC_QP_MAP_MODE> {
        NV_ENC_QP_MAP_DISABLED,
        NV_ENC_QP_MAP_EMPHASIS,
        NV_ENC_QP_MAP_DELTA,
        NV_ENC_QP_MAP,
    };

    const char *szMultiPass = "disabled qres fullres";
    std::vector<NV_ENC_MULTI_PASS> vMultiPass = std::vector<NV_ENC_MULTI_PASS> {
        NV_ENC_MULTI_PASS_DISABLED,
        NV_ENC_TWO_PASS_QUARTER_RESOLUTION,
        NV_ENC_TWO_PASS_FULL_RESOLUTION,
    };

    const char *szCuSize = "auto 8x8 16x16 32x32 64x64";
    std::vector<NV_ENC_HEVC_CUSIZE> vCuSize = std::vector<NV_ENC_HEVC_CUSIZE> {
        NV_ENC_HEVC_CUSIZE_AUTOSELECT,
        NV_ENC_HEVC_CUSIZE_8x8,
        NV_ENC_HEVC_CUSIZE_16x16,
        NV_ENC_HEVC_CUSIZE_32x32,
        NV_ENC_HEVC_CUSIZE_64x64,
    };

    friend class NvEncLiteUnbuffered;

public:
    std::string FullParamToString(const NV_ENC_INITIALIZE_PARAMS *pInitializeParams) {
        std::ostringstream os;
        os << "NV_ENC_INITIALIZE_PARAMS:" << std::endl
            << "encodeGUID: " << ConvertValueToString(vCodec, szCodecNames, pInitializeParams->encodeGUID) << std::endl
            << "presetGUID: " << ConvertValueToString(vPreset, szPresetNames, pInitializeParams->presetGUID) << std::endl
            << "encodeWidth: " << pInitializeParams->encodeWidth << std::endl
            << "encodeHeight: " << pInitializeParams->encodeHeight << std::endl
            << "darWidth: " << pInitializeParams->darWidth << std::endl
            << "darHeight: " << pInitializeParams->darHeight << std::endl
            << "frameRateNum: " << pInitializeParams->frameRateNum << std::endl
            << "frameRateDen: " << pInitializeParams->frameRateDen << std::endl
            << "enableEncodeAsync: " << pInitializeParams->enableEncodeAsync << std::endl
            << "enablePTD" << pInitializeParams->enablePTD << std::endl
            << "reportSliceOffsets: " << pInitializeParams->reportSliceOffsets << std::endl
            << "enableSubFrameWrite: " << pInitializeParams->enableSubFrameWrite << std::endl
            << "enableExternalMEHints: " << pInitializeParams->enableExternalMEHints << std::endl
            << "enableMEOnlyMode: " << pInitializeParams->enableMEOnlyMode << std::endl
            << "enableWeightedPrediction: " << pInitializeParams->enableWeightedPrediction << std::endl
            << "enableOutputInVidmem" << pInitializeParams->enableOutputInVidmem << std::endl
            << "maxEncodeWidth: " << pInitializeParams->maxEncodeWidth << std::endl
            << "maxEncodeHeight: " << pInitializeParams->maxEncodeHeight << std::endl
            << "maxMEHintCountsPerBlock: " << pInitializeParams->maxMEHintCountsPerBlock << std::endl
            << "tuningInfo: " << ConvertValueToString(vTuningInfo, szTuningInfo, pInitializeParams->tuningInfo) << std::endl
        ;
        NV_ENC_CONFIG *pConfig = pInitializeParams->encodeConfig;
        os << "NV_ENC_CONFIG:" << std::endl
            << "profile: " << ConvertValueToString(vProfile, szProfileNames, pConfig->profileGUID) << std::endl
            << "gopLength: " << pConfig->gopLength << std::endl
            << "frameIntervalP: " << pConfig->frameIntervalP << std::endl
            << "monoChromeEncoding: " << pConfig->monoChromeEncoding << std::endl
            << "frameFieldMode: " << pConfig->frameFieldMode << std::endl
            << "mvPrecision: " << pConfig->mvPrecision << std::endl
            << "NV_ENC_RC_PARAMS:" << std::endl
            << "    rateControlMode: 0x" << std::hex << pConfig->rcParams.rateControlMode << std::dec << std::endl
            << "    constQP: " << pConfig->rcParams.constQP.qpInterP << ", " << pConfig->rcParams.constQP.qpInterB << ", " << pConfig->rcParams.constQP.qpIntra << std::endl
            << "    averageBitRate:  " << pConfig->rcParams.averageBitRate << std::endl
            << "    maxBitRate:      " << pConfig->rcParams.maxBitRate << std::endl
            << "    vbvBufferSize:   " << pConfig->rcParams.vbvBufferSize << std::endl
            << "    vbvInitialDelay: " << pConfig->rcParams.vbvInitialDelay << std::endl
            << "    enableMinQP: " << pConfig->rcParams.enableMinQP << std::endl
            << "    enableMaxQP: " << pConfig->rcParams.enableMaxQP << std::endl
            << "    enableInitialRCQP: " << pConfig->rcParams.enableInitialRCQP << std::endl
            << "    enableAQ: " << pConfig->rcParams.enableAQ << std::endl
            << "    enableLookahead: " << pConfig->rcParams.enableLookahead << std::endl
            << "    disableIadapt: " << pConfig->rcParams.disableIadapt << std::endl
            << "    disableBadapt: " << pConfig->rcParams.disableBadapt << std::endl
            << "    enableTemporalAQ: " << pConfig->rcParams.enableTemporalAQ << std::endl
            << "    zeroReorderDelay: " << pConfig->rcParams.zeroReorderDelay << std::endl
            << "    enableNonRefP: " << pConfig->rcParams.enableNonRefP << std::endl
            << "    strictGOPTarget: " << pConfig->rcParams.strictGOPTarget << std::endl
            << "    aqStrength: " << pConfig->rcParams.aqStrength << std::endl
            << "    minQP: " << pConfig->rcParams.minQP.qpInterP << ", " << pConfig->rcParams.minQP.qpInterB << ", " << pConfig->rcParams.minQP.qpIntra << std::endl
            << "    maxQP: " << pConfig->rcParams.maxQP.qpInterP << ", " << pConfig->rcParams.maxQP.qpInterB << ", " << pConfig->rcParams.maxQP.qpIntra << std::endl
            << "    initialRCQP: " << pConfig->rcParams.initialRCQP.qpInterP << ", " << pConfig->rcParams.initialRCQP.qpInterB << ", " << pConfig->rcParams.initialRCQP.qpIntra << std::endl
            << "    temporallayerIdxMask: " << pConfig->rcParams.temporallayerIdxMask << std::endl
            << "    temporalLayerQP: " << (int)pConfig->rcParams.temporalLayerQP[0] << ", " << (int)pConfig->rcParams.temporalLayerQP[1] << ", " << (int)pConfig->rcParams.temporalLayerQP[2] << ", " << (int)pConfig->rcParams.temporalLayerQP[3] << ", " << (int)pConfig->rcParams.temporalLayerQP[4] << ", " << (int)pConfig->rcParams.temporalLayerQP[5] << ", " << (int)pConfig->rcParams.temporalLayerQP[6] << ", " << (int)pConfig->rcParams.temporalLayerQP[7] << std::endl
            << "    targetQuality: " << (int)pConfig->rcParams.targetQuality << std::endl
            << "    targetQualityLSB: " << (int)pConfig->rcParams.targetQualityLSB << std::endl
            << "    lookaheadDepth: " << pConfig->rcParams.lookaheadDepth << std::endl
            << "    lowDelayKeyFrameScale: " << (int)pConfig->rcParams.lowDelayKeyFrameScale << std::endl
            << "    qpMapMode: " << ConvertValueToString(vQpMapMode, szQpMapMode, pConfig->rcParams.qpMapMode) << std::endl
            << "    multiPass: " << ConvertValueToString(vMultiPass, szMultiPass, pConfig->rcParams.multiPass) << std::endl
            ;
        if (pInitializeParams->encodeGUID == NV_ENC_CODEC_H264_GUID) {
            os  
            << "NV_ENC_CODEC_CONFIG (H264):" << std::endl
            << "    enableTemporalSVC: " << pConfig->encodeCodecConfig.h264Config.enableTemporalSVC << std::endl
            << "    enableStereoMVC: " << pConfig->encodeCodecConfig.h264Config.enableStereoMVC << std::endl
            << "    hierarchicalPFrames: " << pConfig->encodeCodecConfig.h264Config.hierarchicalPFrames << std::endl
            << "    hierarchicalBFrames: " << pConfig->encodeCodecConfig.h264Config.hierarchicalBFrames << std::endl
            << "    outputBufferingPeriodSEI: " << pConfig->encodeCodecConfig.h264Config.outputBufferingPeriodSEI << std::endl
            << "    outputPictureTimingSEI: " << pConfig->encodeCodecConfig.h264Config.outputPictureTimingSEI << std::endl
            << "    outputAUD: " << pConfig->encodeCodecConfig.h264Config.outputAUD << std::endl
            << "    disableSPSPPS: " << pConfig->encodeCodecConfig.h264Config.disableSPSPPS << std::endl
            << "    outputFramePackingSEI: " << pConfig->encodeCodecConfig.h264Config.outputFramePackingSEI << std::endl
            << "    outputRecoveryPointSEI: " << pConfig->encodeCodecConfig.h264Config.outputRecoveryPointSEI << std::endl
            << "    enableIntraRefresh: " << pConfig->encodeCodecConfig.h264Config.enableIntraRefresh << std::endl
            << "    enableConstrainedEncoding: " << pConfig->encodeCodecConfig.h264Config.enableConstrainedEncoding << std::endl
            << "    repeatSPSPPS: " << pConfig->encodeCodecConfig.h264Config.repeatSPSPPS << std::endl
            << "    enableVFR: " << pConfig->encodeCodecConfig.h264Config.enableVFR << std::endl
            << "    enableLTR: " << pConfig->encodeCodecConfig.h264Config.enableLTR << std::endl
            << "    qpPrimeYZeroTransformBypassFlag: " << pConfig->encodeCodecConfig.h264Config.qpPrimeYZeroTransformBypassFlag << std::endl
            << "    useConstrainedIntraPred: " << pConfig->encodeCodecConfig.h264Config.useConstrainedIntraPred << std::endl
            << "    enableFillerDataInsertion: " << pConfig->encodeCodecConfig.h264Config.enableFillerDataInsertion << std::endl
            << "    disableSVCPrefixNalu: " << pConfig->encodeCodecConfig.h264Config.disableSVCPrefixNalu << std::endl
            << "    enableScalabilityInfoSEI: " << pConfig->encodeCodecConfig.h264Config.enableScalabilityInfoSEI << std::endl
            << "    level: " << pConfig->encodeCodecConfig.h264Config.level << std::endl
            << "    idrPeriod: " << pConfig->encodeCodecConfig.h264Config.idrPeriod << std::endl
            << "    separateColourPlaneFlag: " << pConfig->encodeCodecConfig.h264Config.separateColourPlaneFlag << std::endl
            << "    disableDeblockingFilterIDC: " << pConfig->encodeCodecConfig.h264Config.disableDeblockingFilterIDC << std::endl
            << "    numTemporalLayers: " << pConfig->encodeCodecConfig.h264Config.numTemporalLayers << std::endl
            << "    spsId: " << pConfig->encodeCodecConfig.h264Config.spsId << std::endl
            << "    ppsId: " << pConfig->encodeCodecConfig.h264Config.ppsId << std::endl
            << "    adaptiveTransformMode: " << pConfig->encodeCodecConfig.h264Config.adaptiveTransformMode << std::endl
            << "    fmoMode: " << pConfig->encodeCodecConfig.h264Config.fmoMode << std::endl
            << "    bdirectMode: " << pConfig->encodeCodecConfig.h264Config.bdirectMode << std::endl
            << "    entropyCodingMode: " << pConfig->encodeCodecConfig.h264Config.entropyCodingMode << std::endl
            << "    stereoMode: " << pConfig->encodeCodecConfig.h264Config.stereoMode << std::endl
            << "    intraRefreshPeriod: " << pConfig->encodeCodecConfig.h264Config.intraRefreshPeriod << std::endl
            << "    intraRefreshCnt: " << pConfig->encodeCodecConfig.h264Config.intraRefreshCnt << std::endl
            << "    maxNumRefFrames: " << pConfig->encodeCodecConfig.h264Config.maxNumRefFrames << std::endl
            << "    sliceMode: " << pConfig->encodeCodecConfig.h264Config.sliceMode << std::endl
            << "    sliceModeData: " << pConfig->encodeCodecConfig.h264Config.sliceModeData << std::endl
            << "    NV_ENC_CONFIG_H264_VUI_PARAMETERS:" << std::endl
            << "        overscanInfoPresentFlag: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.overscanInfoPresentFlag << std::endl
            << "        overscanInfo: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.overscanInfo << std::endl
            << "        videoSignalTypePresentFlag: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.videoSignalTypePresentFlag << std::endl
            << "        videoFormat: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.videoFormat << std::endl
            << "        videoFullRangeFlag: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.videoFullRangeFlag << std::endl
            << "        colourDescriptionPresentFlag: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.colourDescriptionPresentFlag << std::endl
            << "        colourPrimaries: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.colourPrimaries << std::endl
            << "        transferCharacteristics: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.transferCharacteristics << std::endl
            << "        colourMatrix: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.colourMatrix << std::endl
            << "        chromaSampleLocationFlag: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.chromaSampleLocationFlag << std::endl
            << "        chromaSampleLocationTop: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.chromaSampleLocationTop << std::endl
            << "        chromaSampleLocationBot: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.chromaSampleLocationBot << std::endl
            << "        bitstreamRestrictionFlag: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.bitstreamRestrictionFlag << std::endl
            << "    ltrNumFrames: " << pConfig->encodeCodecConfig.h264Config.ltrNumFrames << std::endl
            << "    ltrTrustMode: " << pConfig->encodeCodecConfig.h264Config.ltrTrustMode << std::endl
            << "    chromaFormatIDC: " << pConfig->encodeCodecConfig.h264Config.chromaFormatIDC << std::endl
            << "    maxTemporalLayers: " << pConfig->encodeCodecConfig.h264Config.maxTemporalLayers << std::endl
            << "    useBFramesAsRef: " << pConfig->encodeCodecConfig.h264Config.useBFramesAsRef << std::endl
            << "    numRefL0: " << pConfig->encodeCodecConfig.h264Config.numRefL0 << std::endl
            << "    numRefL1: " << pConfig->encodeCodecConfig.h264Config.numRefL1 << std::endl
            ;
        } else if (pInitializeParams->encodeGUID == NV_ENC_CODEC_HEVC_GUID) {
            os  
            << "NV_ENC_CODEC_CONFIG (HEVC):" << std::endl
            << "    level: " << pConfig->encodeCodecConfig.hevcConfig.level << std::endl
            << "    tier: " << pConfig->encodeCodecConfig.hevcConfig.tier << std::endl
            << "    minCUSize: " << ConvertValueToString(vCuSize, szCuSize, pConfig->encodeCodecConfig.hevcConfig.minCUSize) << std::endl
            << "    maxCUSize: " << ConvertValueToString(vCuSize, szCuSize, pConfig->encodeCodecConfig.hevcConfig.maxCUSize) << std::endl
            << "    useConstrainedIntraPred: " << pConfig->encodeCodecConfig.hevcConfig.useConstrainedIntraPred << std::endl
            << "    disableDeblockAcrossSliceBoundary: " << pConfig->encodeCodecConfig.hevcConfig.disableDeblockAcrossSliceBoundary << std::endl
            << "    outputBufferingPeriodSEI: " << pConfig->encodeCodecConfig.hevcConfig.outputBufferingPeriodSEI << std::endl
            << "    outputPictureTimingSEI: " << pConfig->encodeCodecConfig.hevcConfig.outputPictureTimingSEI << std::endl
            << "    outputAUD: " << pConfig->encodeCodecConfig.hevcConfig.outputAUD << std::endl
            << "    enableLTR: " << pConfig->encodeCodecConfig.hevcConfig.enableLTR << std::endl
            << "    disableSPSPPS: " << pConfig->encodeCodecConfig.hevcConfig.disableSPSPPS << std::endl
            << "    repeatSPSPPS: " << pConfig->encodeCodecConfig.hevcConfig.repeatSPSPPS << std::endl
            << "    enableIntraRefresh: " << pConfig->encodeCodecConfig.hevcConfig.enableIntraRefresh << std::endl
            << "    chromaFormatIDC: " << pConfig->encodeCodecConfig.hevcConfig.chromaFormatIDC << std::endl
            << "    pixelBitDepthMinus8: " << pConfig->encodeCodecConfig.hevcConfig.pixelBitDepthMinus8 << std::endl
            << "    enableFillerDataInsertion: " << pConfig->encodeCodecConfig.hevcConfig.enableFillerDataInsertion << std::endl
            << "    enableConstrainedEncoding: " << pConfig->encodeCodecConfig.hevcConfig.enableConstrainedEncoding << std::endl
            << "    enableAlphaLayerEncoding: " << pConfig->encodeCodecConfig.hevcConfig.enableAlphaLayerEncoding << std::endl
            << "    idrPeriod: " << pConfig->encodeCodecConfig.hevcConfig.idrPeriod << std::endl
            << "    intraRefreshPeriod: " << pConfig->encodeCodecConfig.hevcConfig.intraRefreshPeriod << std::endl
            << "    intraRefreshCnt: " << pConfig->encodeCodecConfig.hevcConfig.intraRefreshCnt << std::endl
            << "    maxNumRefFramesInDPB: " << pConfig->encodeCodecConfig.hevcConfig.maxNumRefFramesInDPB << std::endl
            << "    ltrNumFrames: " << pConfig->encodeCodecConfig.hevcConfig.ltrNumFrames << std::endl
            << "    vpsId: " << pConfig->encodeCodecConfig.hevcConfig.vpsId << std::endl
            << "    spsId: " << pConfig->encodeCodecConfig.hevcConfig.spsId << std::endl
            << "    ppsId: " << pConfig->encodeCodecConfig.hevcConfig.ppsId << std::endl
            << "    sliceMode: " << pConfig->encodeCodecConfig.hevcConfig.sliceMode << std::endl
            << "    sliceModeData: " << pConfig->encodeCodecConfig.hevcConfig.sliceModeData << std::endl
            << "    maxTemporalLayersMinus1: " << pConfig->encodeCodecConfig.hevcConfig.maxTemporalLayersMinus1 << std::endl
            << "    NV_ENC_CONFIG_HEVC_VUI_PARAMETERS:" << std::endl
            << "        overscanInfoPresentFlag: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.overscanInfoPresentFlag << std::endl
            << "        overscanInfo: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.overscanInfo << std::endl
            << "        videoSignalTypePresentFlag: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.videoSignalTypePresentFlag << std::endl
            << "        videoFormat: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.videoFormat << std::endl
            << "        videoFullRangeFlag: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.videoFullRangeFlag << std::endl
            << "        colourDescriptionPresentFlag: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.colourDescriptionPresentFlag << std::endl
            << "        colourPrimaries: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.colourPrimaries << std::endl
            << "        transferCharacteristics: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.transferCharacteristics << std::endl
            << "        colourMatrix: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.colourMatrix << std::endl
            << "        chromaSampleLocationFlag: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.chromaSampleLocationFlag << std::endl
            << "        chromaSampleLocationTop: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.chromaSampleLocationTop << std::endl
            << "        chromaSampleLocationBot: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.chromaSampleLocationBot << std::endl
            << "        bitstreamRestrictionFlag: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.bitstreamRestrictionFlag << std::endl
            << "    ltrTrustMode: " << pConfig->encodeCodecConfig.hevcConfig.ltrTrustMode << std::endl
            << "    useBFramesAsRef: " << pConfig->encodeCodecConfig.hevcConfig.useBFramesAsRef << std::endl
            << "    numRefL0: " << pConfig->encodeCodecConfig.hevcConfig.numRefL0 << std::endl
            << "    numRefL1: " << pConfig->encodeCodecConfig.hevcConfig.numRefL1 << std::endl
            ;
        }
        return os.str();
    }
};
