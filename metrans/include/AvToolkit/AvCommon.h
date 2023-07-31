#pragma once
extern "C" {
#include <libavutil/avutil.h>
}
#include <map>

extern simplelogger::Logger *logger;

inline bool CheckAvError(int e, int iLine, const char *szFile) {
    char buf[1024];
    av_strerror(e, buf, sizeof(buf));
    if (e < 0) {
        LOG(ERROR) << "FFmpeg error " << e << " (" << buf << ")" << " at line " << iLine << " in file " << szFile;
        return false;
    }
    return true;
}
#define ckav(call) CheckAvError(call, __LINE__, __FILE__)

template <typename T>
inline T *CheckNotNull(T *p, const char *szCall, int iLine, const char *szFile) {
    if (p == NULL) {
        LOG(ERROR) << szCall << " returned NULL at line " << iLine << " in file " << szFile;
    }
    return p;
}
template <typename T>
inline T const *CheckNotNull(T const *p, const char *szCall, int iLine, const char *szFile) {
    if (p == NULL) {
        LOG(ERROR) << szCall << " returned NULL at line " << iLine << " in file " << szFile;
    }
    return p;
}
#define cknn(call) CheckNotNull(call, #call, __LINE__, __FILE__)

inline std::ostream & operator << (std::ostream &o, AVRational r) {
    return o << "{" << r.num << "," << r.den << "}";
}

static std::map<int, int> pitch2alignment;
static int PitchToAlignment(int nPitch, int nWidth) {
    auto it = pitch2alignment.find(nPitch);
    if (it != pitch2alignment.end()) {
        return it->second;
    }
    for (int a = nPitch - nWidth + 1; a <= nPitch; a++) {
        if (nPitch % a == 0 && (nWidth + a - 1) / a == nPitch / a) {
            pitch2alignment.insert(std::make_pair(nPitch, a));
            return a;
        }
    }
    return 0;
}
