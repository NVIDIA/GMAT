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

#include "../trt_lite/trt_lite_utils.h"
#include "../trt_lite/trt_lite.h"

#include "3ddfa_kernels.h"

static float param_mean_array[62] = {
    3.4926363e-04,  2.5279013e-07, -6.8751979e-07,  6.0167957e+01,
    -6.2955132e-07,  5.7572004e-04, -5.0853912e-05,  7.4278198e+01,
    5.4009172e-07,  6.5741384e-05,  3.4420125e-04, -6.6671577e+01,
    -3.4660369e+05, -6.7468234e+04,  4.6822266e+04, -1.5262047e+04,
    4.3505889e+03, -5.4261453e+04, -1.8328033e+04, -1.5843289e+03,
    -8.4566344e+04,  3.8359607e+03, -2.0811361e+04,  3.8094930e+04,
    -1.9967855e+04, -9.2413701e+03, -1.9600715e+04,  1.3168090e+04,
    -5.2591440e+03,  1.8486478e+03, -1.3030662e+04, -2.4355562e+03,
    -2.2542065e+03, -1.4396562e+04, -6.1763291e+03, -2.5621920e+04,
    2.2639447e+02, -6.3261235e+03, -1.0867251e+04,  8.6846509e+02,
    -5.8311479e+03,  2.7051238e+03, -3.6294177e+03,  2.0439901e+03,
    -2.4466162e+03,  3.6586970e+03, -7.6459897e+03, -6.6744526e+03,
    1.1638839e+02,  7.1855972e+03, -1.4294868e+03,  2.6173665e+03,
    -1.2070955e+00,  6.6907924e-01, -1.7760828e-01,  5.6725528e-02,
    3.9678156e-02, -1.3586316e-01, -9.2239931e-02, -1.7260718e-01,
    -1.5804484e-02, -1.4168486e-01};

static float param_std_array[62] = {
    1.76321526e-04, 6.73794348e-05, 4.47084894e-04, 2.65502319e+01,
    1.23137695e-04, 4.49302170e-05, 7.92367064e-05, 6.98256302e+00,
    4.35044407e-04, 1.23148900e-04, 1.74000015e-04, 2.08030396e+01,
    5.75421125e+05, 2.77649062e+05, 2.58336844e+05, 2.55163125e+05,
    1.50994375e+05, 1.60086109e+05, 1.11277305e+05, 9.73117812e+04,
    1.17198453e+05, 8.93173672e+04, 8.84935547e+04, 7.22299297e+04,
    7.10802109e+04, 5.00139531e+04, 5.59685820e+04, 4.75255039e+04,
    4.95150664e+04, 3.81614805e+04, 4.48720586e+04, 4.62732383e+04,
    3.81167695e+04, 2.81911621e+04, 3.21914375e+04, 3.60061719e+04,
    3.25598926e+04, 2.55511172e+04, 2.42675098e+04, 2.75213984e+04,
    2.31665312e+04, 2.11015762e+04, 1.94123242e+04, 1.94522031e+04,
    1.74549844e+04, 2.25376230e+04, 1.61742812e+04, 1.46716406e+04,
    1.51156885e+04, 1.38700732e+04, 1.37463125e+04, 1.26631338e+04,
    1.58708346e+00, 1.50770092e+00, 5.88135779e-01, 5.88974476e-01,
    2.13278517e-01, 2.63020128e-01, 2.79642940e-01, 3.80302161e-01,
    1.61628410e-01, 2.55969286e-01
};

void parse_roi_box_from_bbox(torch::Tensor bbox, torch::Tensor& roi_box){
    assert(bbox.numel() == 5);

    roi_box = torch::zeros({4});
    float left, top, right, bottom;
    left = bbox[0].item<float>();
    top = bbox[1].item<float>();
    right = bbox[2].item<float>();
    bottom = bbox[3].item<float>();

    float old_size = (right - left + bottom - top) / 2;
    float center_x = right - (right - left) / 2.0;
    float center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14;
    int size = static_cast<int>(old_size * 1.58);

    roi_box[0] = center_x - size / 2;
    roi_box[1] = center_y - size / 2;
    roi_box[2] = roi_box[0] + size;
    roi_box[3] = roi_box[1] + size;
}

template<typename T>
void reorder_to_chw(cv::Mat const &mat, std::vector<float> &out, T mean={0,0,0}, float normalize=1.0f) {
    assert(mat.channels() == 3);
    assert(out.size() == mat.channels()* mat.rows * mat.cols);

    // T mean = {104, 117, 123};
    for (int y = 0; y < mat.rows; ++y) {
        for (int x = 0; x < mat.cols; ++x) {
            T rgb = static_cast<T>(mat.at<cv::Vec3b>(y, x)) - mean;
            for (int c = 0; c < mat.channels(); ++c) {
                out[c * (mat.rows * mat.cols) + y * mat.cols + x] =
                    rgb[c];
                if (abs(normalize - 1.0f) > 1e-5)
                    out[c * (mat.rows * mat.cols) + y * mat.cols + x] /= normalize;
            }
        }
    }
}

template void reorder_to_chw<cv::Vec3i>(cv::Mat const &mat, std::vector<float> &out, cv::Vec3i mean={0,0,0}, float normalize=1.0f);
template void reorder_to_chw<cv::Vec3f>(cv::Mat const &mat, std::vector<float> &out, cv::Vec3f mean={0,0,0}, float normalize=1.0f);

class TDDFA_ONNX {
    public:
    TDDFA_ONNX(const char* backbone_path, const char* bfm_path, int batch = 1) : n(batch){
        // Load backbone model
        backbone_trt = unique_ptr<TrtLite>(TrtLiteCreator::Create(backbone_path));
        backbone_trt->PrintInfo();

        int nBatch = n;
        // int numBoxes = 0;
        backbone_i2shape.insert(make_pair(0, Dims{4, {nBatch, 3, 120, 120}}));
        backbone_vInfo = backbone_trt->ConfigIO(backbone_i2shape);

        for (auto info : backbone_vInfo) {
            cout << info.to_string() << endl;

            void *dpBuf = nullptr;
            ck(cudaMalloc(&dpBuf, info.GetNumBytes()));
            backbone_vdpBuf.push_back(dpBuf);

            // if (info.name.compare("output") == 0)
            // {
            //     ck(cudaMalloc(&dp_anchors, info.GetNumBytes()));
            //     ck(cudaMalloc(&dp_boxes, info.GetNumBytes()));
            //     numBoxes = info.GetNumBytes() / sizeof(float) / 4;
            // }
        }

        // Load bfm model
        bfm_trt = unique_ptr<TrtLite>(TrtLiteCreator::Create(bfm_path));
        bfm_trt->PrintInfo();

        // int numBoxes = 0;
        bfm_i2shape.insert(make_pair(0, Dims{2, {3, 3}})); // R
        bfm_i2shape.insert(make_pair(1, Dims{2, {3, 1}})); // offset
        bfm_i2shape.insert(make_pair(2, Dims{2, {40, 1}})); // alpha_shp
        bfm_i2shape.insert(make_pair(3, Dims{2, {10, 1}})); //alpha_exp

        bfm_vInfo = bfm_trt->ConfigIO(bfm_i2shape);

        // size_t num_vertices = 0;
        for (auto info : bfm_vInfo) {
            cout << info.to_string() << endl;

            void *dpBuf = nullptr;
            ck(cudaMalloc(&dpBuf, info.GetNumBytes()));
            bfm_vdpBuf.push_back(dpBuf);

            if (!info.bInput)
            {
                // ck(cudaMalloc(&dp_anchors, info.GetNumBytes()));
                // ck(cudaMalloc(&dp_boxes, info.GetNumBytes()));
                // numBoxes = info.GetNumBytes() / sizeof(float) / 4;
                num_vertices = info.GetNumBytes() / sizeof(float);
            }
        }
        cub_dry_run(cub_d_temp_storage, cub_temp_storage_bytes, num_vertices);
        // cub::DeviceReduce::Min(cub_d_temp_storage, cub_temp_storage_bytes,
        //     reinterpret_cast<float*>(bfm_vdpBuf[4]), cub_out, num_vertices);
        // ck(cudaMalloc(&cub_d_temp_storage, cub_temp_storage_bytes));
        ck(cudaMalloc(&cub_out, sizeof(float)));
    }

    ~TDDFA_ONNX(){
        for (auto dpBuf : backbone_vdpBuf) {
            ck(cudaFree(dpBuf));
        }
        for (auto dpBuf : bfm_vdpBuf) {
            ck(cudaFree(dpBuf));
        }
        ck(cudaFree(cub_out));
        ck(cudaFree(cub_d_temp_storage));
    }

    void forward(cv::Mat image_ori, torch::Tensor face_dets, vector<torch::Tensor>& param_vec,
        vector<torch::Tensor>& roi_box_vec, cudaStream_t stream=0){
        auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32);
        auto tensor_options_cuda = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        for (int i = 0; i < face_dets.size(0); i++){

            auto backbone_pre_start = chrono::steady_clock::now();

            torch::Tensor roi_box;
            torch::Tensor det_cpu = face_dets[i].to(torch::kCPU);
            parse_roi_box_from_bbox(det_cpu, roi_box);
            // cout << roi_box << endl;

            cv::Mat image_120;
            cv::Mat cropped(roi_box[3].item<int>() - roi_box[1].item<int>(),
                roi_box[2].item<int>() - roi_box[0].item<int>(),
                CV_8UC3);
            cropped.setTo(0);

            int sx, sy, dw, dh;
            sx = roi_box[0].item<int>() < 0 ? 0 : roi_box[0].item<int>();
            sy = roi_box[1].item<int>() < 0 ? 0 : roi_box[1].item<int>();
            dw = roi_box[2].item<int>() > image_ori.cols ? (image_ori.cols - sx) : (roi_box[2].item<int>() - sx);
            dh = roi_box[3].item<int>() > image_ori.rows ? (image_ori.rows - sy) : (roi_box[3].item<int>() - sy);

            int dsx, dsy;
            dsx = roi_box[0].item<int>() < 0 ? -roi_box[0].item<int>() : 0;
            dsy = roi_box[1].item<int>() < 0 ? -roi_box[1].item<int>() : 0;

            int roi_h = roi_box[3].item<int>() - roi_box[1].item<int>();
            int roi_w = roi_box[2].item<int>() - roi_box[0].item<int>();
            // void *d_image_120, *d_image_crop, *d_cropped;
            // cudaMallocAsync(&d_cropped, roi_h * roi_w * 3, stream);

            // cv::Mat image_crop = image(cv::Rect{
            //     roi_box[0].item<int>(),
            //     roi_box[1].item<int>(),
            //     roi_box[2].item<int>() - roi_box[0].item<int>(),
            //     roi_box[3].item<int>() - roi_box[1].item<int>()});
            cv::Mat image_crop = image_ori(cv::Rect{sx, sy, dw, dh});
            image_crop.copyTo(cropped(cv::Rect{
                dsx,
                dsy,
                dw,
                dh
                }));

            cv::resize(cropped, image_120, cv::Size{120, 120});
            vector<float> vImg_120_chw(image_120.channels() * image_120.cols * image_120.rows);
            reorder_to_chw<cv::Vec3f>(image_120, vImg_120_chw, cv::Vec3f{127.5, 127.5, 127.5}, 128.0f);

            for (int i = 0; i < backbone_vInfo.size(); i++) {
                if (backbone_vInfo[i].name.compare("input") == 0){
                    ck(cudaMemcpyAsync(backbone_vdpBuf[i], vImg_120_chw.data(), backbone_vInfo[i].GetNumBytes(),
                        cudaMemcpyHostToDevice, stream));
                }
            }

            auto backbone_pre_stop = chrono::steady_clock::now();
            chrono::duration<double> backbone_pre_duration = backbone_pre_stop - backbone_pre_start;

            backbone_trt->Execute(backbone_i2shape, backbone_vdpBuf, stream);

            torch::Tensor param_mean_tensor = torch::from_blob(param_mean_array, {62}, tensor_options).to(torch::kCUDA);
            torch::Tensor param_std_tensor = torch::from_blob(param_std_array, {62}, tensor_options).to(torch::kCUDA);
            torch::Tensor param_tensor = torch::zeros({62}).to(torch::kCUDA);

            float* param_ptr = param_tensor.data_ptr<float>();
            ck(cudaMemcpyAsync(param_ptr, backbone_vdpBuf[1], backbone_vInfo[1].GetNumBytes(),
                cudaMemcpyDeviceToDevice, stream));
            // cout << param_tensor << endl;

            param_tensor = param_tensor * param_std_tensor + param_mean_tensor;
            // cout << param_tensor << endl;

            float* d_bfm_out_ptr;
            auto bfm_pre_start = chrono::steady_clock::now();

            for (int i = 0; i < bfm_vInfo.size(); i++){
                string binding_name = bfm_vInfo[i].name;
                torch::Tensor param_slice;
                if (binding_name.compare("R") == 0){
                    param_slice = param_tensor.index({Slice(0, 12)}).reshape({3, -1});
                    param_slice = param_slice.index({"...", Slice(0, 3)}).clone();
                    // cout << "R: " << param_slice << endl;
                }
                else if (binding_name.compare("offset") == 0){
                    param_slice = param_tensor.index({Slice(0, 12)}).reshape({3, -1});
                    param_slice = param_slice.index({"...", -1}).reshape({3, 1}).clone();
                }
                else if (binding_name.compare("alpha_shp") == 0){
                    param_slice = param_tensor.index({Slice(12, 52)}).clone();
                }
                else if (binding_name.compare("alpha_exp") == 0){
                    param_slice = param_tensor.index({Slice(52, 62)}).clone();
                }
                else{
                    d_bfm_out_ptr = static_cast<float*>(bfm_vdpBuf[i]);
                    continue;
                }
                // auto param_slice = param_tensor.index({"...", Slice(0, offset)});
                ck(cudaMemcpyAsync(bfm_vdpBuf[i], param_slice.data_ptr<float>(), bfm_vInfo[i].GetNumBytes(), cudaMemcpyDeviceToDevice, stream));
            }

            auto bfm_pre_stop = chrono::steady_clock::now();
            chrono::duration<double> bfm_pre_duration = bfm_pre_stop - bfm_pre_start;

            // cudaStreamSynchronize(stream);
            // #define DEBUG
            // #ifdef DEBUG
            // torch::Tensor R_tensor = torch::from_blob(bfm_vdpBuf[0], {3, 3}, tensor_options_cuda);
            // cout << R_tensor << endl;
            // #endif
            // #undef DEBUG

            bfm_trt->Execute(bfm_i2shape, bfm_vdpBuf, stream);
            // cudaStreamSynchronize(stream);

            torch::Tensor pts3d_tensor = torch::from_blob(bfm_vdpBuf[bfm_vdpBuf.size() - 1], {3, 38365}, tensor_options_cuda);
            // cout << pts3d_tensor.index({0, Slice(0, 10)}) << endl;
            param_vec.push_back(param_tensor);

            similar_transform_transpose(d_bfm_out_ptr, &d_vertices_out, num_vertices, roi_box, cub_d_temp_storage, cub_temp_storage_bytes, cub_out, stream);
            // cout << "Time of backbone preprocess: " << backbone_pre_duration.count() << "s\n";
            // cout << "Time of bfm preprocess: " << bfm_pre_duration.count() << "s\n";
        }
    }

    float* vertex_data(){
        // return reinterpret_cast<float*>(bfm_vdpBuf[4]);
        return d_vertices_out;
    }

    size_t vertex_num(){
        return num_vertices;
    }

    private:
    unique_ptr<TrtLite> backbone_trt;
    unique_ptr<TrtLite> bfm_trt;

    vector<IOInfo> backbone_vInfo;
    vector<IOInfo> bfm_vInfo;
    vector<void *> backbone_vdpBuf;
    vector<void *> bfm_vdpBuf;
    map<int, Dims> backbone_i2shape;
    map<int, Dims> bfm_i2shape;

    // cub temp storage
    void     *cub_d_temp_storage = NULL;
    size_t   cub_temp_storage_bytes = 0;
    float*   cub_out;
    float*   d_vertices_out;

    int n;
    int size=120;
    size_t num_vertices = 0;
};
