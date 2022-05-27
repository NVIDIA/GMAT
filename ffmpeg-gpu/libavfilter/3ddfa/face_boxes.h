/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#pragma once

#include <vector>
#include <utility>

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torchvision/vision.h>
#include <torchvision/ops/nms.h>

#include "../trt_lite/trt_lite_utils.h"
#include "../trt_lite/trt_lite.h"

#include "3ddfa_kernels.h"

using namespace torch::indexing;
using namespace std;

struct config{
    string name="FaceBoxes";
    vector<vector<uint32_t>> min_sizes{{32, 64, 128},{256}, {512}};
    vector<uint32_t> steps{32, 64, 128};
    pair<float, float> variance{0.1, 0.2};
    bool clip=false;
    bool scale_flag=false;
}cfg;

void prior_box(unique_ptr<float[]>& anchors, size_t& anchor_num, float img_h, float img_w) {
    uint32_t anchor_idx = 0;
    anchor_num = 0;
    vector<pair<uint32_t, uint32_t>> feature_maps(cfg.steps.size());
    for (int i = 0; i < cfg.steps.size(); i++) {
        feature_maps[i].first = ceil(img_h / cfg.steps[i]);
        feature_maps[i].second = ceil(img_w / cfg.steps[i]);
    }

    for (int i = 0; i < cfg.min_sizes.size(); i++) {
        for (int j = 0; j < cfg.min_sizes[i].size(); j++) {
            switch (cfg.min_sizes[i][j]) {
                case 32:
                anchor_num += feature_maps[i].first * feature_maps[i].second * 16;
                break;

                case 64:
                anchor_num += feature_maps[i].first * feature_maps[i].second * 4;
                break;

                default:
                anchor_num += feature_maps[i].first * feature_maps[i].second;
            }
        }
    }

    anchors = unique_ptr<float[]>(new float[anchor_num * 4]);
    // float* anchors_float = reinterpret_cast<float*>(*anchors.get());
    for (int k = 0; k < feature_maps.size(); k++) {
        for (int i = 0; i < feature_maps[k].first; i++) {
            for (int j = 0; j < feature_maps[k].second; j++) {
                for (auto min_size : cfg.min_sizes[k]) {
                    float s_kx = min_size / img_w;
                    float s_ky = min_size / img_h;

                    switch (min_size) {
                        case 32:
                        #pragma unroll
                        for (int m = 0; m < 4; m++) {
                            #pragma unroll
                            for (int n = 0; n < 4; n++) {
                                anchors[4 * anchor_idx] = (j + n * 0.25f) * cfg.steps[k] / img_w;
                                anchors[4 * anchor_idx + 1] = (i + m * 0.25f) * cfg.steps[k] / img_h;
                                anchors[4 * anchor_idx + 2] = s_kx;
                                anchors[4 * anchor_idx + 3] = s_ky;

                                anchor_idx += 1;
                            }
                        }
                        break;

                        case 64:
                        #pragma unroll
                        for (int m = 0; m < 2; m++) {
                            #pragma unroll
                            for (int n = 0; n < 2; n++) {
                                anchors[4 * anchor_idx] = (j + n * 0.25f) * cfg.steps[k] / img_w;
                                anchors[4 * anchor_idx + 1] = (i + m * 0.25f) * cfg.steps[k] / img_h;
                                anchors[4 * anchor_idx + 2] = s_kx;
                                anchors[4 * anchor_idx + 3] = s_ky;

                                anchor_idx += 1;
                            }
                        }
                        break;

                        default:
                        anchors[4 * anchor_idx] = (j + 0.5f) * cfg.steps[k] / img_w;
                        anchors[4 * anchor_idx + 1] = (i + 0.5f) * cfg.steps[k] / img_h;
                        anchors[4 * anchor_idx + 2] = s_kx;
                        anchors[4 * anchor_idx + 3] = s_ky;
                        anchor_idx += 1;
                    }
                }
            }
        }
    }
}

class FaceBoxes_ONNX {
public:
    FaceBoxes_ONNX(const char* trt_path, int height=720, int width=1280) :
    h(height), w(width){
        trt = unique_ptr<TrtLite>(TrtLiteCreator::Create(trt_path));
        trt->PrintInfo();

        int nBatch = n, nChannel = c;
        int numBoxes = 0;
        i2shape.insert(make_pair(0, Dims{4, {nBatch, nChannel, height, width}}));
        if (trt->GetEngine()->hasImplicitBatchDimension()) {
            vInfo = trt->ConfigIO(nBatch);
        } else {
            vInfo = trt->ConfigIO(i2shape);
        }

        for (auto info : vInfo) {
            cout << info.to_string() << endl;

            void *dpBuf = nullptr;
            ck(cudaMalloc(&dpBuf, info.GetNumBytes()));
            vdpBuf.push_back(dpBuf);

            if (info.name.compare("output") == 0)
            {
                ck(cudaMalloc(&dp_anchors, info.GetNumBytes()));
                ck(cudaMalloc(&dp_boxes, info.GetNumBytes()));
                numBoxes = info.GetNumBytes() / sizeof(float) / 4;
            }
        }
    // CUcontext dummy;
    // (cuCtxGetCurrent(&dummy));
    // cudaSetDevice(0);
    // std::cout << "Current context: " << dummy << std::endl;
        auto options =
            torch::TensorOptions()
                .dtype(torch::kFloat32)
                .layout(torch::kStrided)
                .device(torch::kCUDA)
                .requires_grad(false);
    // (cuCtxGetCurrent(&dummy));
    // std::cout << "Current context: " << dummy << std::endl;
        // boxes_tensor = torch.cuda.FloatTensor(numBoxes, 4);
        // conf_tensor = torch.cuda.FloatTensor(numBoxes, 2);
        boxes_tensor = torch::zeros({numBoxes, 4}, options);
        conf_tensor = torch::zeros({numBoxes, 2}, options);
    // (cuCtxGetCurrent(&dummy));
    // std::cout << "Current context: " << dummy << std::endl;

        // unique_ptr<float[]> anchors;
        // size_t anchor_num = 0;
        prior_box(anchors, anchor_num, height, width);
        // for (int i = 0; i < 100; i++) cout << anchors[i] << "  ";
        // cout << "Anchor numbers: " << anchor_num << endl;
    }

    ~FaceBoxes_ONNX(){
        for (auto dpBuf : vdpBuf) {
            ck(cudaFree(dpBuf));
        }
        ck(cudaFree(dp_anchors));
        ck(cudaFree(dp_boxes));
    }

    template<typename T>
    torch::Tensor forward(T *d_input_img, int img_h, int img_w, cudaStream_t stream=0){
        float scale = 1.0f;
        int h_s = img_h;
        int w_s = img_w;
        if (scale_flag) {
            if (img_h > h)
                scale = h / img_h;
            if (img_w * scale > w)
                scale *= w / (img_w * scale);
            h_s = static_cast<int>(scale * img_h);
            w_s = static_cast<int>(scale * img_w);
        }

        for (int i = 0; i < vInfo.size(); i++) {
            auto &info = vInfo[i];
            if (info.bInput) {
                // ck(cudaMemcpyAsync(vdpBuf[i], d_input_img, info.GetNumBytes(), cudaMemcpyHostToDevice, stream));
                vdpBuf[i] = d_input_img;
            }
        }
        ck(cudaGetLastError());
        if (trt->GetEngine()->hasImplicitBatchDimension())
            trt->Execute(n, vdpBuf, stream);
        else
            trt->Execute(i2shape, vdpBuf, stream);
        ck(cudaGetLastError());

        // DEBUG
        #ifdef DEBUG
        vector<void*> out(2);
        for (int i = 0; i < vInfo.size(); i++) {
            auto &info = vInfo[i];
            if (!info.bInput) {
                out[i % 2] = new uint8_t[info.GetNumBytes()];
                ck(cudaMemcpyAsync(out[i % 2], vdpBuf[i], info.GetNumBytes(), cudaMemcpyDeviceToHost, stream));
            }
        }
        #endif

        ck(cudaMemcpyAsync(dp_anchors, anchors.get(), anchor_num * 4 * sizeof(float), cudaMemcpyHostToDevice, stream));
        decode_locations(reinterpret_cast<float*>(vdpBuf[1]), reinterpret_cast<float*>(dp_anchors), anchor_num,
            cfg.variance, reinterpret_cast<float*>(dp_boxes));
        ck(cudaGetLastError());

        #ifdef DEBUG
        // ck(cudaDeviceSynchronize());
        uint8_t* boxes_host = new uint8_t[anchor_num * 4 * sizeof(float)];
        ck(cudaMemcpy(boxes_host, dp_boxes, anchor_num * 4 * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < 100; i++) {
            cout << reinterpret_cast<float*>(boxes_host)[i] << "  ";
        }
        #endif
        torch::Tensor scale_bbox = torch::ones({4});
        scale_bbox[0] = w_s;
        scale_bbox[1] = h_s;
        scale_bbox[2] = w_s;
        scale_bbox[3] = h_s;

        scale_bbox = scale_bbox.cuda();
        float* boxes_ptr = boxes_tensor.data_ptr<float>();
        float* conf_ptr = conf_tensor.data_ptr<float>();
        ck(cudaMemcpyAsync(boxes_ptr, dp_boxes, anchor_num * 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        ck(cudaMemcpyAsync(conf_ptr, vdpBuf[2], anchor_num * 2 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        ck(cudaStreamSynchronize(stream));

        boxes_tensor = boxes_tensor * scale_bbox / scale;
        torch::Tensor scores_tensor = conf_tensor.index({"...", 1}).squeeze(-1);

        auto indices = torch::where(scores_tensor > 0.05);
        // auto indices = (scores_tensor > 0.05);
        // cout << indices[0] << endl;
        auto boxes = boxes_tensor.index({indices[0], "..."});
        auto scores = scores_tensor.index({indices[0], "..."});

        // cout << "Scores: " << scores_tensor << endl;
        // cout << "boxes_tensor: \n";
        // for (int i = 0; i < 10; i++){
        //     cout << boxes_tensor[i] << endl;
        // }
        // cout << "scores_tensor: \n";
        // for (int i = 0; i < 10; i++){
        //     cout << scores_tensor[i] << endl;
        // }
        // auto keep = nms_kernel_gpu(boxes, scores, 0.3);
        // cout << keep << endl;
        auto keep_tv = vision::ops::nms(boxes, scores, 0.3);
        // cout << keep_tv << endl;
        auto boxes_keep = boxes.index({keep_tv, "..."});
        auto scores_keep = scores.index({keep_tv});

        auto thres_mask = (scores_keep > 0.5);
        auto det_boxes = boxes_keep.index({thres_mask});
        auto det_scores = scores_keep.index({thres_mask});

        auto dets = torch::cat({det_boxes, det_scores.unsqueeze(-1)}, 1);

        return dets;

        // torch::Tensor det_boxes = torch::zeros({5});
        // for (int i = 0; i < scores_keep.sizes()[0]; i++){
        //     float score = scores_keep[i].item<float>();
        //     if (score > 0.5){
        //         boxes_keep.resize_({5});
        //         boxes_keep[4] = scores_keep[i];
        //         // torch::stack({det_boxes, boxes_keep});
        //         // torch::stack({det_boxes, torch::concat({boxes_keep[i], scores_keep[i]}, 1)}, 0);
        //     }
        // }

        // cout << dets << endl;
        // return det_boxes;

        // cout << "NMS results: " << boxes_keep << endl;
    }
private:
    unique_ptr<TrtLite> trt;
    vector<IOInfo> vInfo;
    vector<void *> vdpBuf;
    map<int, Dims> i2shape;
    unique_ptr<float[]> anchors;
    size_t anchor_num = 0;
    void* dp_anchors;
    void* dp_boxes;

    torch::Tensor boxes_tensor, conf_tensor;

    bool scale_flag = false;
    int n = 1, c = 3;
    float h, w;
};
