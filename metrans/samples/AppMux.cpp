#include <iostream>
#include <stdint.h>
#include "AvToolkit/Demuxer.h"
#include "AvToolkit/Muxer.h"

using namespace std;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

void DemuxV() {
    Demuxer demuxer("bunny.mp4");
    AVPacket *pkt = NULL;
    FILE *fpOut = fopen("out.h264", "wb");
    while (demuxer.Demux(&pkt)) {
        fwrite(pkt->data, 1, pkt->size, fpOut);
    }
    fclose(fpOut);
}

void DemuxAV() {
    // It's ok to keep AVCC format here, and Muxer will convert it to Annex-B
    Demuxer demuxer("bunny.mp4", true, true);
    AVPacket *pkt = NULL;
    bool bAudio = false;
    Muxer vmuxer(demuxer.GetVideoStream()->codecpar, demuxer.GetVideoStream()->time_base, NULL, AVRational{}, "bunny.h264");
    Muxer amuxer(NULL, AVRational{}, demuxer.GetAudioStream()->codecpar, demuxer.GetAudioStream()->time_base, "bunny.aac");
    while (demuxer.Demux(&pkt, &bAudio)) {
        if (bAudio) {
            amuxer.MuxAudio(pkt);
        } else {
            vmuxer.MuxVideo(pkt);
        }
    }
}

void Remux() {
    // We have to keep AVCC format here, otherwise Muxer doesn't have enough info to fully initialize
    Demuxer demuxer("bunny.mp4", true, true);
    AVPacket *pkt = NULL;
    bool bAudio = false;
    Muxer muxer(demuxer.GetVideoStream()->codecpar, demuxer.GetVideoStream()->time_base, 
        demuxer.GetAudioStream()->codecpar, demuxer.GetAudioStream()->time_base, "out.flv");
    while (demuxer.Demux(&pkt, &bAudio)) {
        if (bAudio) {
            muxer.MuxAudio(pkt);
        } else {
            muxer.MuxVideo(pkt);
        }
    }
}

void Mux() {
    // TODO: For ac3, codec frame size returned by demuxer is 0. It's better to be fixed.
    Demuxer vdemuxer("bunny.h264"), ademuxer("bunny.aac", false, true);
    AVPacket *vpkt = NULL, *apkt = NULL;
    bool voef = false, aeof = false;
    int vi = 0, ai = 0;
    Muxer muxer(vdemuxer.GetVideoStream()->codecpar, vdemuxer.GetVideoStream()->time_base,
    		ademuxer.GetAudioStream()->codecpar, ademuxer.GetAudioStream()->time_base, "out.mp4");
    while (true) {
        if (vpkt == NULL && !voef) {
            voef = !vdemuxer.Demux(&vpkt, NULL);
        }
        if (apkt == NULL && !aeof) {
            aeof = !ademuxer.Demux(&apkt, NULL);
        }
        if (!vpkt && !apkt && voef && aeof) {
            break;
        }
        if (!vpkt) {
            muxer.MuxAudio(apkt);
            apkt = NULL;
            continue;
        }
        if (vpkt->dts == AV_NOPTS_VALUE) {
            vpkt->pts = vpkt->dts = av_rescale_q_rnd(vi++, av_inv_q(vdemuxer.GetVideoStream()->avg_frame_rate),
                    vdemuxer.GetVideoStream()->time_base, (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
        }
        if (!apkt) {
            muxer.MuxVideo(vpkt);
            vpkt = NULL;
            continue;
        }
        int64_t dts4v = av_rescale_q_rnd(vpkt->dts, vdemuxer.GetVideoStream()->time_base,
                ademuxer.GetAudioStream()->time_base, (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
        if (dts4v <= apkt->dts) {
            muxer.MuxVideo(vpkt);
            vpkt = NULL;
        } else {
            muxer.MuxAudio(apkt);
            apkt = NULL;
        }
    }
}

int main(int argc, char *argv[]) {
    DemuxV();
    DemuxAV();
    Remux();
    Mux();
    return 0;
}
