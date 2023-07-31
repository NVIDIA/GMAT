#include <iostream>
#include <stdint.h>
#include "AvToolkit/VidDec.h"
#include "AvToolkit/Demuxer.h"

using namespace std;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

void SaveFrame(AVFrame *frm, FILE *fpOut) {
    for (int i = 0; i < frm->height; i++) {
        fwrite(frm->data[0] + i * frm->linesize[0], 1, frm->width, fpOut);
    }
    for (int i = 0; i < frm->height / 2; i++) {
        fwrite(frm->data[1] + i * frm->linesize[1], 1, frm->width / 2, fpOut);
    }
    for (int i = 0; i < frm->height / 2; i++) {
        fwrite(frm->data[2] + i * frm->linesize[2], 1, frm->width / 2, fpOut);
    }
}

void DecodeVideo(const char *szInFilePath, const char *szOutFilePath) {
    Demuxer demuxer(szInFilePath);
    if (!demuxer.GetVideoStream()) {
        cout << "No video stream in file " << szInFilePath << endl;
        return;
    }

    VidDec dec(demuxer.GetVideoStream()->codecpar);
    int n = 0;
    AVPacket *pkt = NULL;
    vector<AVFrame *> vFrm;
    FILE *fpOut = cknn(fopen(szOutFilePath, "wb"));    
    do {
        demuxer.Demux(&pkt);
        dec.Decode(pkt, vFrm);
        for (AVFrame *frm : vFrm) {
            SaveFrame(frm, fpOut);
            n++;
        }
    } while (pkt->size);
    fclose(fpOut);
    cout << "Decoded samples: " << n << endl;
}

int main(int argc, char *argv[]) {
    DecodeVideo("bunny.mp4", "bunny.iyuv");
    return 0;
}
