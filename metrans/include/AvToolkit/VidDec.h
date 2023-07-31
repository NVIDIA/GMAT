#pragma once

#include "AvDec.h"

class VidDec : public AvDec {
public:
    VidDec(AVCodecParameters *par, const char *szCodecName = NULL) : AvDec(par, szCodecName) {}
};
