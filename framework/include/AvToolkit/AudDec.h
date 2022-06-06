#pragma once

#include "AvDec.h"

class AudDec : public AvDec {
public:
    AudDec(AVCodecParameters *par, const char *szCodecName = NULL) : AvDec(par, szCodecName) {}
};
