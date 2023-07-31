#include <iostream>
#include <stdint.h>
#include "AvToolkit/DemuxerEx.h"
#include "NvCodec/NvCommon.h"

AVPixelFormat get_format(struct AVCodecContext *s, const enum AVPixelFormat * fmt) {
    return AV_PIX_FMT_CUDA;
}

void RunDecoder() {
    Demuxer demuxer("/home/gji/video/liwei/1020.h264", true);
    // Demuxer demuxer("/home/gji/git/metrans/build/bunny.h264", true);

    auto *Codec = avcodec_find_decoder(demuxer.GetVideoStream()->codecpar->codec_id);
    if (Codec == nullptr) {
        cout << "Video codec not found" << endl;
        return;
    }

    AVCodecContext *CodecContext = avcodec_alloc_context3(Codec);
    if (CodecContext == nullptr) {
        cout << "Could not allocate video codec context." << endl;
        return;
    }
    if (avcodec_parameters_to_context(CodecContext, demuxer.GetVideoStream()->codecpar) < 0) {
        cout << "Could not copy video decoder parameters." << endl;
        return;
    }
    CodecContext->thread_count = 1;
    // CodecContext->delay = 0;
    CodecContext->flags |= AV_CODEC_FLAG_LOW_DELAY;
    // CodecContext->skip_frame = AVDISCARD_NONREF;
    // CodecContext->has_b_frames = 15;

    for (int i = 0;; i++) {
        const AVCodecHWConfig *config = avcodec_get_hw_config(Codec, i);
        if (!config) {
            break;
        }
        if (config->device_type == AV_HWDEVICE_TYPE_CUDA) {
            CodecContext->get_format = get_format;
            if (av_hwdevice_ctx_create(&CodecContext->hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0) {
                cout << "Failed to create specified HW device." << endl;
                return;
            }
        }
    }

    if (avcodec_open2(CodecContext, Codec, nullptr) < 0) {
        cout << "Could not open video codec" << endl;
        return;
    }

    AVPacket *pkt = NULL;
    AVFrame *tmpFrame = nullptr, *DecodeFrame = nullptr;
    DecodeFrame = av_frame_alloc();
    tmpFrame = av_frame_alloc();
    //std::ofstream f("frame.gray", std::ios::out | std::ios::binary);
    int ret0, ret1;
    StopWatch w;
    w.Start();
    int nDecoded = 0;
    while (demuxer.Demux(&pkt)) {
        ret0 = avcodec_send_packet(CodecContext, pkt);
        ret1 = avcodec_receive_frame(CodecContext, tmpFrame);
        //cout << pkt->size << ", ret0=" << ret0 << ", ret1=" << ret1 << endl;
        if (ret1 == 0) {
            nDecoded++;
            // if (tmpFrame->format == AV_PIX_FMT_CUDA) {
            //     if (av_hwframe_transfer_data(DecodeFrame, tmpFrame, 0) < 0) {
            //         cout << "av_hwframe_transfer_data() failed" << endl;
            //         return;
            //     }
            //     // cout << "DecodeFrame->format=" << DecodeFrame->format << endl;
            // } else {
            //     // cout << "std::swap(DecodeFrame, tmpFrame)" << endl;
            //     std::swap(DecodeFrame, tmpFrame);
            // }

            //f.write((char *)DecodeFrame->data[0], DecodeFrame->width * DecodeFrame->height);
        }
    }
    cout << "nDecoded=" << nDecoded << ", Decode time: " << w.Stop() * 1000 << " ms" << endl;
}
