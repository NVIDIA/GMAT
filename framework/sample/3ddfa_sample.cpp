#define STB_IMAGE_IMPLEMENTATION
#include <vector>
#include <map>
#include <utility>
#include <string>
#include <iostream>
#include <cmath>
#include <chrono>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "torch/torch.h"
#include <torchvision/vision.h>
#include <torchvision/ops/nms.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cv_cuda.h>

#include "common_utils.h"
#include "tddfa_kernels.h"
#include "TrtLite/TrtLiteUtils.h"
#include "TrtLite/TrtLite.h"
#include "NvCodec/NvDecoderImageProvider.h"
#include "NvCodec/NvEncLite.h"
#include "NvCodec/NvCommon.h"
#include "format_cuda.h"

// #include "param_mean_std.h"
#include "cnpy.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "GLUtils.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>
// #include "app/cnpy.h"
// #include <GLFW/glfw3.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

#include <cuda_gl_interop.h>
#include <nvToolsExt.h> 
#include <cuda_profiler_api.h>

// #define DEBUG
#define BATCH_SIZE  1

using namespace cv;
using namespace std;
using namespace torch::indexing;
simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger(simplelogger::TRACE);

enum OutputFormat {
    native = 0, bgrp, bgra, bgra64
};
vector<string> vstrOutputFormatName = {
    "native", "bgrp", "bgra", "bgra64"
};
static unsigned int indices_pic[] = {
    0, 1, 2,
    2, 3, 0
};

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

const char *srcVertexShader_face = R"###(
#version 330 core
layout (location = 0) in vec3 aPos;
out vec2 texPos;

uniform float W, H;
uniform mat4 projection;

void main() {
    // vec4 p = projection * vec4(aPos, 1.0f);
    // gl_Position = vec4(p.x, -p.y, p.z, 1.0f);
    gl_Position = vec4(aPos.x / W * 2 - 1.0f, aPos.y / H * 2 - 1.0f, aPos.z / 10000.0f, 1.0f);
    texPos = vec2((gl_Position.x + 1.0f) / 2, (gl_Position.y + 1.0f) / 2);
}
)###";

const char *srcFragmentShader_face = R"###(
#version 330 core
in vec2 texPos;
out vec4 FragColor;

uniform sampler2D tex0;

void main() {
    // FragColor = texture(tex0, texPos);
    FragColor = vec4(1.0f, 1.0f, 1.0f, 0.3f);
}
)###";

const char *srcVertexShader_pic = R"###(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexPos;
out vec2 texPos;

void main() {
    gl_Position = vec4(aPos.x, aPos.y, 0.99f, 1.0f);
    texPos = aTexPos;
}
)###";

const char *srcFragmentShader_pic = R"###(
#version 330 core
in vec2 texPos;
out vec4 FragColor;

uniform sampler2D tex0;

void main() {
    FragColor = texture(tex0, texPos);
}
)###";

void ck_egl(EGLBoolean ret) {
    if (ret) return;
    cout << "EGL error" << endl;
	EGLint error = eglGetError();
	if (error != EGL_SUCCESS) {
		stringstream s;
		s << "EGL error 0x" << std::hex << error;
		throw runtime_error(s.str());
	}
}

struct config{
    ~config(){
        delete shader_face;
        delete shader_pic;
    }

    string name="FaceBoxes";
    vector<vector<uint32_t>> min_sizes{{32, 64, 128},{256}, {512}};
    vector<uint32_t> steps{32, 64, 128};
    pair<float, float> variance{0.1, 0.2};
    bool clip=false;
    bool scale_flag=false;

    unsigned int tex_image, tex_face, VAO_face, VAO_pic, PBO_fb, VBO_face;
    size_t index_num;
    Shader *shader_face;
    Shader *shader_pic;
    cudaGraphicsResource *cuda_mesh_resource;
    cudaGraphicsResource *cuda_texture_resource;
    cudaGraphicsResource *cuda_fb_resource;
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

class FaceBoxes_ONNX {
public:
    FaceBoxes_ONNX(const char* trt_path){
        trt = unique_ptr<TrtLite>(TrtLiteCreator::Create(trt_path));
        trt->PrintInfo();

        int nBatch = n, nChannel = c, nHeight = h, nWidth = w;
        int numBoxes = 0;
        i2shape.insert(make_pair(0, Dims{4, {nBatch, nChannel, nHeight, nWidth}}));
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
        auto options =
            torch::TensorOptions()
                .dtype(torch::kFloat32)
                .layout(torch::kStrided)
                .device(torch::kCUDA)
                .requires_grad(false);
        // boxes_tensor = torch.cuda.FloatTensor(numBoxes, 4);
        // conf_tensor = torch.cuda.FloatTensor(numBoxes, 2);
        boxes_tensor = torch::zeros({numBoxes, 4}, options);
        conf_tensor = torch::zeros({numBoxes, 2}, options);
    
        // prepare anchors
        unique_ptr<float[]> anchors;
        prior_box(anchors, anchor_num, h, w);
        // for (int i = 0; i < 100; i++) cout << anchors[i] << "  ";
        // cout << "Anchor numbers: " << anchor_num << endl;
        ck(cudaMemcpy(dp_anchors, anchors.get(), anchor_num * 4 * sizeof(float), cudaMemcpyHostToDevice));
        
        scale = 1.0f;
        int h_s = h;
        int w_s = w;
        if (scale_flag) {
            if (h > h)
                scale = h / h;
            if (w * scale > w)
                scale *= w / (w * scale);
            h_s = static_cast<int>(scale * h);
            w_s = static_cast<int>(scale * w);
        }
        scale_bbox = torch::ones({4});
        scale_bbox[0] = w_s;
        scale_bbox[1] = h_s;
        scale_bbox[2] = w_s;
        scale_bbox[3] = h_s;
        scale_bbox = scale_bbox.to(torch::kCUDA);
    }

    ~FaceBoxes_ONNX(){
        for (auto dpBuf : vdpBuf) {
            ck(cudaFree(dpBuf));
        }
    }

    template<typename T>
    torch::Tensor forward(T *input_img, int h, int img_w, bool deviceInput=false, cudaStream_t stream=0){
        
        for (int i = 0; i < vInfo.size(); i++) {
            auto &info = vInfo[i];
            if (info.bInput) {
                if (deviceInput)
                    vdpBuf[i] = input_img;
                else
                    ck(cudaMemcpyAsync(vdpBuf[i], input_img, info.GetNumBytes(), cudaMemcpyHostToDevice, stream));
            }
        }
        if (trt->GetEngine()->hasImplicitBatchDimension())
            trt->Execute(n, vdpBuf, stream);
        else
            trt->Execute(i2shape, vdpBuf, stream);

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

        decode_locations(reinterpret_cast<float*>(vdpBuf[1]), reinterpret_cast<float*>(dp_anchors), anchor_num,
            cfg.variance, reinterpret_cast<float*>(dp_boxes), stream);
        ck(cudaGetLastError());

        #ifdef DEBUG
        // ck(cudaDeviceSynchronize());
        uint8_t* boxes_host = new uint8_t[anchor_num * 4 * sizeof(float)];
        ck(cudaMemcpy(boxes_host, dp_boxes, anchor_num * 4 * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < 100; i++) {
            cout << reinterpret_cast<float*>(boxes_host)[i] << "  ";
        }
        #endif

        float* boxes_ptr = boxes_tensor.data_ptr<float>();
        float* conf_ptr = conf_tensor.data_ptr<float>();
        ck(cudaMemcpyAsync(boxes_ptr, dp_boxes, anchor_num * 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        ck(cudaMemcpyAsync(conf_ptr, vdpBuf[2], anchor_num * 2 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        // ck(cudaStreamSynchronize(stream));

        boxes_tensor = boxes_tensor * scale_bbox / scale;
        torch::Tensor scores_tensor = conf_tensor.index({"...", 1}).squeeze(-1);

        auto indices = torch::where(scores_tensor > 0.05);
        // auto indices = (scores_tensor > 0.05);
        // cout << indices[0] << endl;
        ck(cudaDeviceSynchronize());
        nvtxRangePush(__FUNCTION__);
        nvtxMark("Torch indexing");
        auto boxes = boxes_tensor.index({indices[0], "..."});
        auto scores = scores_tensor.index({indices[0], "..."});
        nvtxRangePop();

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
    void* dp_anchors;
    void* dp_boxes;
    size_t anchor_num;

    torch::Tensor boxes_tensor, conf_tensor;

    bool scale_flag = false;
    int n = 1, c = 3;
    float h = 720, w = 1280;
    int h_s, w_s;
    float scale;
    torch::Tensor scale_bbox;
};

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

        // Allocate device memory for cub
        cub_dry_run(cub_d_temp_storage, cub_temp_storage_bytes, num_vertices);
        // cub::DeviceReduce::Min(cub_d_temp_storage, cub_temp_storage_bytes,
        //     reinterpret_cast<float*>(bfm_vdpBuf[4]), cub_out, num_vertices);
        // ck(cudaMalloc(&cub_d_temp_storage, cub_temp_storage_bytes));
        ck(cudaMalloc(&cub_out, sizeof(float)));

        // Init cv-cuda ops
        cuda_op::DataShape img_shape{BATCH_SIZE, 3, 720, 1280};
        cuda_op::DataShape resize_shape{BATCH_SIZE, 3, 120, 120};
        crop_op = new cuda_op::CustomCrop(img_shape, img_shape);
        resize_op = new cuda_op::Resize(resize_shape, resize_shape);
        ck(cudaMalloc(&crop_workspace, crop_op->calBufferSize(img_shape, img_shape, cuda_op::kCV_8U)));
        ck(cudaMalloc(&resize_workspace, resize_op->calBufferSize(resize_shape, resize_shape, cuda_op::kCV_8U)));
        ck(cudaMalloc(&crop_output, 3 * 720 * 1280));

        // Allocate space for vertex output
        ck(cudaMalloc(&d_vertices_out, num_vertices * sizeof(float)));
    }

    ~TDDFA_ONNX(){
        delete crop_op;
        cudaFree(cub_out);
        cudaFree(crop_workspace);
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

            // cv::Mat image_crop = image(cv::Rect{
            //     roi_box[0].item<int>(),
            //     roi_box[1].item<int>(),
            //     roi_box[2].item<int>() - roi_box[0].item<int>(),
            //     roi_box[3].item<int>() - roi_box[1].item<int>()});
            cv::Mat image_crop = image_ori(cv::Rect{sx, sy, dw, dh});
            image_crop.copyTo(cropped(Rect{
                dsx,
                dsy,
                dw,
                dh
                }));

            cv::resize(cropped, image_120, Size{120, 120});
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

            similar_transform_transpose(d_bfm_out_ptr, d_vertices_out, num_vertices, roi_box, cub_d_temp_storage, cub_temp_storage_bytes, cub_out, stream);
            // cout << "Time of backbone preprocess: " << backbone_pre_duration.count() << "s\n";
            // cout << "Time of bfm preprocess: " << bfm_pre_duration.count() << "s\n";
        }
    }

    void forward(torch::Tensor image_ori, torch::Tensor face_dets, vector<torch::Tensor>& param_vec,
        vector<torch::Tensor>& roi_box_vec, cudaStream_t stream=0){
        auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32);
        auto tensor_options_cuda = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
        for (int i = 0; i < face_dets.size(0); i++){

            auto backbone_pre_start = chrono::steady_clock::now();

            torch::Tensor roi_box;
            torch::Tensor det_cpu = face_dets[i].to(torch::kCPU);
            parse_roi_box_from_bbox(det_cpu, roi_box);
            // cout << roi_box << endl;
            size_t image_cols = image_ori.size(2);
            size_t image_rows = image_ori.size(1);

            cv::Mat image_120;
            cv::Mat cropped(roi_box[3].item<int>() - roi_box[1].item<int>(),
                roi_box[2].item<int>() - roi_box[0].item<int>(),
                CV_8UC3);
            cropped.setTo(0);

            void *d_cropped;
            size_t cropped_size = (roi_box[3].item<int>() - roi_box[1].item<int>()) * 
                (roi_box[2].item<int>() - roi_box[0].item<int>()) * 4;
            torch::Tensor image_ten_120 = torch::empty({120, 120, 4}, tensor_options_cuda);
            torch::Tensor d_cropped_ten = torch::empty({roi_box[3].item<int>() - roi_box[1].item<int>(), roi_box[2].item<int>() - roi_box[0].item<int>(), 4}, tensor_options_cuda);
            // ck(cudaMalloc(&d_cropped, cropped_size));

            int sx, sy, dw, dh;
            sx = roi_box[0].item<int>() < 0 ? 0 : roi_box[0].item<int>();
            sy = roi_box[1].item<int>() < 0 ? 0 : roi_box[1].item<int>();
            dw = roi_box[2].item<int>() > image_cols ? (image_cols - sx) : (roi_box[2].item<int>() - sx);
            dh = roi_box[3].item<int>() > image_rows ? (image_rows - sy) : (roi_box[3].item<int>() - sy);

            int dsx, dsy;
            dsx = roi_box[0].item<int>() < 0 ? -roi_box[0].item<int>() : 0;
            dsy = roi_box[1].item<int>() < 0 ? -roi_box[1].item<int>() : 0;


            // cv::Mat image_crop = image(cv::Rect{
            //     roi_box[0].item<int>(),
            //     roi_box[1].item<int>(),
            //     roi_box[2].item<int>() - roi_box[0].item<int>(),
            //     roi_box[3].item<int>() - roi_box[1].item<int>()});
            // cuda_op::CopyTo copy_to{};
            // cv::Rect roi{sx, sy, dw, dh};
            // cuda_op::DataShape img_shape{, 3, , 3, 720, 1280};
            // crop_op->infer(&image_ori.data_ptr<uint8_t>(), &crop_output, crop_workspace, roi, img_shape, 
            //     cuda_op::kHWC, cuda_op::kCV_8U);

            // crop_op->infer(&crop_output, &d_cropped, crop_workspace, cv::Rect{sx, sy, dw, dh}, {1, 3, dw, dh}, 
            //     cuda_op::kHWC, cuda_op::kCV_8U);

            int4 srcRoi{sx, sy, dw, dh};
            int4 dstRoi{dsx, dsy, dw, dh};
            cv::Size dsize{120, 120};
            torch::Tensor mean = torch::empty({1}).toType(torch::kFloat32);
            mean[0] = 127.5;
            mean = mean.to(torch::kCUDA);

            // DEBUG
            // auto image_ori_rgb = image_ori.to(torch::kCPU).index({Slice(None), Slice(None), Slice(None), Slice(None, 3)});
            // auto image_ori_rgb_gpu = image_ori.index({Slice(None), Slice(None), Slice(None), Slice(None, 3)});
            // image_ori.index({Slice(None), Slice(None), Slice(3)}) = 128;
            // // cout << "image_ori: \n" << image_ori_rgb.index({Slice(None), Slice(None), Slice(None, 10), Slice(None)}) << endl;
            // uint8_t h_rgb[1280*720*4];
            // ck(cudaMemcpy(h_rgb, image_ori_rgb_gpu.data_ptr(), 1280*720*3, cudaMemcpyDeviceToHost));
            // cout << "cropped image: \n";
            // for (int i = 0; i < 100; i++) {
            //     // cout << image_ori_rgb.data_ptr<uint8_t>()[i] << ", ";
            //     printf("%d / %d, ", image_ori_rgb.data_ptr<uint8_t>()[i], h_rgb[i]);
            // }
            // cout << endl;
            // stbi_write_bmp("image120_debug_out.bmp", image_cols, image_rows, 3, image_ori_rgb_gpu.to(torch::kCPU).data_ptr<uint8_t>());

            uint8_t *image_ten_120_ptr = image_ten_120.data_ptr<uint8_t>();
            uint8_t *cropped_ten_ptr = d_cropped_ten.data_ptr<uint8_t>();
            
            cuda_op::DataShape input_shape{BATCH_SIZE, 4, dh, dw};
            cropWithRoi<ffgd::BGRA32>(image_ori.data_ptr<uint8_t>(), cropped_ten_ptr, srcRoi, dstRoi, image_ori.size(2) * 4, dw * 4, 
                image_ori.size(1), dh, stream);

            // DEBUG
            // uint8_t h_cropped[cropped_size];
            // ck(cudaMemcpy(h_cropped, d_cropped, cropped_size, cudaMemcpyDeviceToHost));
            // cout << "cropped image: \n";
            // for (int i = 0; i < 100; i++) {
            //     // cout << h_cropped[i] << ", ";
            //     printf("%d, ", h_cropped[i]);
            // }
            // cout << endl;

            resize_op->infer((void**)&cropped_ten_ptr, (void**)&image_ten_120_ptr, resize_workspace, dsize, .0f, .0f,
                cv::INTER_LINEAR, input_shape, cuda_op::kHWC, cuda_op::kCV_8U, stream);

            // bgra to bgr using torch slicing
            auto bgr_120 = image_ten_120.index({Slice(), Slice(), Slice(None, 3)});
            bgr_120 = bgr_120.permute({2, 0, 1}).toType(torch::kFloat);
            // std::cout << bgr_120.index({0, 0, Slice(None, 100)}) << std::endl;
            bgr_120 = bgr_120.sub_(127.5f).div_(128.0f);
            
            // cv::Mat image_crop = image_ori(cv::Rect{sx, sy, dw, dh});
            // image_crop.copyTo(cropped(Rect{
            //     dsx,
            //     dsy,
            //     dw,
            //     dh
            //     }));

            // cv::resize(cropped, image_120, Size{120, 120});
            // vector<float> vImg_120_chw(image_120.channels() * image_120.cols * image_120.rows);
            // reorder_to_chw<cv::Vec3f>(image_120, vImg_120_chw, cv::Vec3f{127.5, 127.5, 127.5}, 128.0f);

            for (int i = 0; i < backbone_vInfo.size(); i++) {
                if (backbone_vInfo[i].name.compare("input") == 0){
                    // ck(cudaMemcpyAsync(backbone_vdpBuf[i], vImg_120_chw.data(), backbone_vInfo[i].GetNumBytes(),
                    //     cudaMemcpyHostToDevice, stream));
                    backbone_vdpBuf[i] = bgr_120.data_ptr();

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

            similar_transform_transpose(d_bfm_out_ptr, d_vertices_out, num_vertices, roi_box, cub_d_temp_storage, cub_temp_storage_bytes, cub_out, stream);
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

    void *crop_workspace, *crop_output, *resize_workspace;
    cuda_op::CustomCrop *crop_op = nullptr;
    cuda_op::Resize *resize_op = nullptr;
};

void init_egl(int device_index=0) {
    int const MAX_DEVICES = 32;
    EGLDeviceEXT eglDevs[MAX_DEVICES];
	EGLint numDevices;

    PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT =(PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT = (PFNEGLGETPLATFORMDISPLAYEXTPROC) eglGetProcAddress("eglGetPlatformDisplayEXT");

    eglQueryDevicesEXT(MAX_DEVICES, eglDevs, &numDevices);
    cout << "numDevices=" << numDevices << endl;
    EGLDisplay display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, eglDevs[device_index], nullptr);;
    ck_egl(eglInitialize(display, NULL, NULL));
    ck_egl(eglBindAPI(EGL_OPENGL_API));

    EGLint attributes[] = {
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_NONE
    };
    EGLConfig config = {};
    EGLint num_config = 0;
    ck_egl(eglChooseConfig(display, attributes, &config, 1, &num_config));

    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, NULL);
    ck_egl(eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context));
    ck_egl(eglWaitClient());
}

// // process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// void processInput(GLFWwindow *window) {
//     if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
//         glfwSetWindowShouldClose(window, true);
//     }
// }
// // glfw: whenever the window size changed (by OS or user resize) this callback function executes
// void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
//     // make sure the viewport matches the new window dimensions; note that width and
//     // height will be significantly larger than specified on retina displays.
//     glViewport(0, 0, width, height);
// }

// void init_glfw(GLFWwindow** window) {
//     const int W = 1280;
//     const int H = 720;

//     // glfw: initialize and configure
//     glfwInit();
//     glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//     glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
//     glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

//     // glfw window creation
//     *window = glfwCreateWindow(W, H, "LearnOpenGL", NULL, NULL);
//     if (window == NULL) {
//         std::cout << "Failed to create GLFW window" << std::endl;
//         glfwTerminate();
//         return;
//     }
//     glfwMakeContextCurrent(*window);
//     glfwSetFramebufferSizeCallback(*window, framebuffer_size_callback);

// }

void config_gl(const char* index_path, size_t vertices_num, const int W=1280, const int H=720) {
    glViewport(0, 0, W, H);
    glCheckError();

    GLuint fbo, rbo;
    glGenFramebuffers(1, &fbo);
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB, W, H);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo);

    GLuint rbo_depth;
    glGenRenderbuffers(1, &rbo_depth);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo_depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, W, H);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo_depth);

    // Shader shader_pic(srcVertexShader_pic, srcFragmentShader_pic);
    cfg.shader_pic = new Shader(srcVertexShader_pic, srcFragmentShader_pic);
    glCheckError();

    unsigned int tex_image = -1;
    glGenTextures(1, &tex_image);
    glBindTexture(GL_TEXTURE_2D, tex_image);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W, H, 0,
        GL_BGRA, GL_UNSIGNED_BYTE, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);
    glCheckError();

    // unsigned int PBO_cuda_unpack;
    // glGenBuffers(1, &PBO_cuda_unpack);
    // glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO_cuda_unpack);
    // glBufferData(GL_PIXEL_UNPACK_BUFFER, vertices_num * sizeof(float),
    //             NULL, GL_DYNAMIC_COPY);

    // unsigned tex_pic = LoadTexture("./one_face_1280.jpg", true);
    unsigned int VBO_pic, VAO_pic;
    glGenVertexArrays(1, &VAO_pic);
    glCheckError();
    glGenBuffers(1, &VBO_pic);
    glCheckError();

    // 0. bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO_pic);
    glCheckError();
    glBindBuffer(GL_ARRAY_BUFFER, VBO_pic);
    glCheckError();

    float vertices_pic[] = {
        -1.0, -1.0, 0.0, 0.0,
        -1.0,  1.0, 0.0, 1.0,
         1.0,  1.0, 1.0, 1.0,
         1.0, -1.0, 1.0, 0.0,
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices_pic), vertices_pic, GL_STATIC_DRAW);

    // 1. then set the vertex attributes pointers
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(0 * sizeof(float)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glCheckError();

    unsigned int EBO_pic;
    glGenBuffers(1, &EBO_pic);
    unsigned int indices_pic[] = {
        0, 1, 2,
        2, 3, 0
    };
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_pic);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices_pic), indices_pic, GL_STATIC_DRAW);

    unsigned tex_face = LoadTexture("./bin/sample/3ddfa/color.1024x1024.jpg", false);
    // Shader shader_face(srcVertexShader_face, srcFragmentShader_face);
    cfg.shader_face = new Shader(srcVertexShader_face, srcFragmentShader_face);
    glCheckError();

    unsigned int VAO_face;
    unsigned int VBO_face;
    glGenBuffers(1, &VBO_face);
    glGenVertexArrays(1, &VAO_face);
    glBindVertexArray(VAO_face);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_face);
    // cnpy::NpyArray face = cnpy::npy_load("vert_0.npy");
    glBufferData(GL_ARRAY_BUFFER, vertices_num * sizeof(float),
        NULL, GL_DYNAMIC_DRAW);
    glCheckError();
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(0 * sizeof(float)));
    glEnableVertexAttribArray(0);

    // glBindBuffer(GL_ARRAY_BUFFER, 0);
    glCheckError();

    unsigned int EBO_face;
    glGenBuffers(1, &EBO_face);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_face);
    cnpy::NpyArray index = cnpy::npy_load(index_path);
    cfg.index_num = index.num_vals;
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, index.num_bytes(), index.data<int>(), GL_STATIC_DRAW);
    // cout << "number of vertices: " << index.num_vals << endl;

    unsigned int PBO_fb;
    glGenBuffers(1, &PBO_fb);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, PBO_fb);
    glBufferData(GL_PIXEL_PACK_BUFFER, W * H * 4, NULL, GL_DYNAMIC_READ);

    // unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    cfg.tex_image = tex_image;
    cfg.tex_face = tex_face;
    cfg.VAO_face = VAO_face;
    cfg.VAO_pic = VAO_pic;
    cfg.PBO_fb = PBO_fb;
    cfg.VBO_face = VBO_face;
    ck(cudaGraphicsGLRegisterBuffer(&cfg.cuda_mesh_resource, VBO_face, cudaGraphicsRegisterFlagsNone));
    ck(cudaGraphicsGLRegisterBuffer(&cfg.cuda_fb_resource, PBO_fb, cudaGraphicsRegisterFlagsNone));
    ck(cudaGraphicsGLRegisterImage(&cfg.cuda_texture_resource, tex_image, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
}

void draw_gl(torch::Tensor& image, float* d_vertices, torch::Tensor& out, cudaStream_t stream=0) {
    
    size_t W = image.size(2);
    size_t H = image.size(1);

    void *cuda_mapped_pointer, *pack_mapped_pointer;
    size_t cuda_mapped_size = 0;
    cudaArray_t texture_array;
    // float* h_vertices = new float[vertices_num];
    // cudaMemcpy(h_vertices, d_vertices, vertices_num * sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "Vertices: \n";
    // for (int i = 0; i < 100; i++){
    //     std::cout << h_vertices[38365 + i] << "  ";
    // }
    // std::cout << std::endl;
    CUcontext current;
    ck(cuCtxGetCurrent(&current));
    // std::cout << current << std::endl;
    ck(cudaGraphicsMapResources(1, &cfg.cuda_texture_resource, stream));
    ck(cudaGraphicsSubResourceGetMappedArray(&texture_array, cfg.cuda_texture_resource, 0, 0));
    ck(cudaMemcpy2DToArrayAsync(texture_array, 0, 0, image.data_ptr(), W * 4,
        W * 4, H, cudaMemcpyDeviceToDevice, stream));
    ck(cudaGraphicsUnmapResources(1, &cfg.cuda_texture_resource, stream));

    ck(cudaGraphicsMapResources(1, &cfg.cuda_mesh_resource, stream));
    ck(cudaGraphicsResourceGetMappedPointer(&cuda_mapped_pointer, &cuda_mapped_size, cfg.cuda_mesh_resource));

    ck(cudaMemcpyAsync(cuda_mapped_pointer, d_vertices, cuda_mapped_size, cudaMemcpyDeviceToDevice, stream));

    ck(cudaGraphicsUnmapResources(1, &cfg.cuda_mesh_resource, stream));

    glm::mat4 projection = glm::ortho(0.0f, (float)W, 0.0f, (float)H, -10000.0f, 10000.0f);
    // print(projection, "projection");

    glCheckError();
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // glFinish();
    glCheckError();
    // render loop
    // while (!glfwWindowShouldClose(window)) {
        // input
        // processInput(window);
        // GLenum check_return = glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER);
        // cout << glGetString(check_return) << endl;
        // render
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        // glCheckError();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glCheckError();

        cfg.shader_pic->Use();
        // glCheckError();
        glActiveTexture(GL_TEXTURE0);
        glCheckError();
        glBindTexture(GL_TEXTURE_2D, cfg.tex_image);
        glCheckError();
        cfg.shader_pic->SetUniform("tex0", 0);
        glBindVertexArray(cfg.VAO_pic);
        glCheckError();
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glDrawElements(GL_TRIANGLES, sizeof(indices_pic) / sizeof(indices_pic[0]), GL_UNSIGNED_INT, 0);
        glCheckError();

        cfg.shader_face->Use();
        cfg.shader_face->SetUniform("W", (float)W);
        cfg.shader_face->SetUniform("H", (float)H);
        cfg.shader_face->SetUniform("projection", projection);
        glCheckError();

        cfg.shader_face->Use();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, cfg.tex_face);
        cfg.shader_face->SetUniform("tex0", 0);
        glBindVertexArray(cfg.VAO_face);
        glCheckError();
        
        // DEBUG
        // glBindBuffer(GL_ARRAY_BUFFER, cfg.VBO_face);
        // uint8_t h_VBO_face[cuda_mapped_size];
        // glGetBufferSubData(GL_ARRAY_BUFFER, 0, cuda_mapped_size, h_VBO_face);
        // cout << "VBO_face: \n";
        // for (int i = 0; i < 100; i++) {
        //     cout << ((float*)h_VBO_face)[i] << ", ";
        // }
        // cout << endl;
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        // glLineWidth(0.1);
        glDrawElements(GL_TRIANGLES, cfg.index_num, GL_UNSIGNED_INT, 0);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // glfwSwapBuffers(window);
        // glfwPollEvents();
    // }
    glCheckError();
    glBindBuffer(GL_PIXEL_PACK_BUFFER, cfg.PBO_fb);
    glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    // DEBUG
    // glFinish();
    // std::vector<std::uint8_t> buf(W * H * 3);
    // uint8_t h_rgba_out[W * H * 3];
    // glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, buf.data());
    // glCheckError();
    // stbi_write_png("rgb_debug_out.png", W, H, 3, buf.data(), W * 3);


    ck(cudaGraphicsMapResources(1, &cfg.cuda_fb_resource, stream));
    ck(cudaGraphicsResourceGetMappedPointer(&pack_mapped_pointer, &cuda_mapped_size, cfg.cuda_fb_resource));

    // ck(cudaMemcpyAsync(out.data_ptr(), cuda_mapped_pointer, cuda_mapped_size, cudaMemcpyDeviceToDevice, stream));

    // ck(cudaMemcpyAsync(cuda_mapped_pointer, d_vertices, cuda_mapped_size, cudaMemcpyDeviceToDevice, stream));
    ffgd::Color32ToNv12<ffgd::RGBA32>(static_cast<uint8_t*>(pack_mapped_pointer), W * 4, out.data_ptr<uint8_t>(), W, W, H, stream);

    ck(cudaGraphicsUnmapResources(1, &cfg.cuda_fb_resource, stream));
}

void draw_on_image(cv::Mat image, const char* index_path, float* d_vertices, size_t vertices_num,
    const int W=1280, const int H=720, cudaStream_t stream=0) {
    glViewport(0, 0, W, H);
    glCheckError();

    GLuint fbo, rbo;
    glGenFramebuffers(1, &fbo);
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB, W, H);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo);

    GLuint rbo_depth;
    glGenRenderbuffers(1, &rbo_depth);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo_depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, W, H);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo_depth);

    Shader shader_pic(srcVertexShader_pic, srcFragmentShader_pic);
    glCheckError();

    unsigned int tex_image = -1;
    glGenTextures(1, &tex_image);
    glBindTexture(GL_TEXTURE_2D, tex_image);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows, 0,
        image.channels() == 4 ? GL_BGRA : GL_BGR, GL_UNSIGNED_BYTE, image.data);
    glGenerateMipmap(GL_TEXTURE_2D);
    glCheckError();

    // unsigned int PBO_cuda_unpack;
    // glGenBuffers(1, &PBO_cuda_unpack);
    // glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO_cuda_unpack);
    // glBufferData(GL_PIXEL_UNPACK_BUFFER, vertices_num * sizeof(float),
    //             NULL, GL_DYNAMIC_COPY);

    // unsigned tex_pic = LoadTexture("./one_face_1280.jpg", true);
    unsigned int VBO_pic, VAO_pic;
    glGenVertexArrays(1, &VAO_pic);
    glCheckError();
    glGenBuffers(1, &VBO_pic);
    glCheckError();

    // 0. bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO_pic);
    glCheckError();
    glBindBuffer(GL_ARRAY_BUFFER, VBO_pic);
    glCheckError();

    float vertices_pic[] = {
        -1.0, -1.0, 0.0, 0.0,
        -1.0,  1.0, 0.0, 1.0,
         1.0,  1.0, 1.0, 1.0,
         1.0, -1.0, 1.0, 0.0,
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices_pic), vertices_pic, GL_STATIC_DRAW);

    // 1. then set the vertex attributes pointers
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(0 * sizeof(float)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glCheckError();

    unsigned int EBO_pic;
    glGenBuffers(1, &EBO_pic);
    unsigned int indices_pic[] = {
        0, 1, 2,
        2, 3, 0
    };
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_pic);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices_pic), indices_pic, GL_STATIC_DRAW);

    unsigned tex_face = LoadTexture("color.1024x1024.jpg", false);
    Shader shader_face(srcVertexShader_face, srcFragmentShader_face);
    glCheckError();

    unsigned int VBO_face;
    glGenBuffers(1, &VBO_face);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_face);
    // cnpy::NpyArray face = cnpy::npy_load("vert_0.npy");
    glBufferData(GL_ARRAY_BUFFER, vertices_num * sizeof(float),
        NULL, GL_DYNAMIC_DRAW);
    glCheckError();

    unsigned int VAO_face;
    glGenVertexArrays(1, &VAO_face);
    glBindVertexArray(VAO_face);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(0 * sizeof(float)));
    glEnableVertexAttribArray(0);

    // glBindBuffer(GL_ARRAY_BUFFER, 0);
    glCheckError();

    unsigned int EBO_face;
    glGenBuffers(1, &EBO_face);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_face);
    cnpy::NpyArray index = cnpy::npy_load(index_path);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, index.num_bytes(), index.data<int>(), GL_STATIC_DRAW);
    cout << "number of vertices: " << index.num_vals << endl;

    // unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    cudaGraphicsResource *cuda_mesh_resource;
    ck(cudaGraphicsGLRegisterBuffer(&cuda_mesh_resource, VBO_face, cudaGraphicsRegisterFlagsNone));

    void *cuda_mapped_pointer;
    size_t cuda_mapped_size = 0;
    float* h_vertices = new float[vertices_num];
    cudaMemcpy(h_vertices, d_vertices, vertices_num * sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "Vertices: \n";
    // for (int i = 0; i < 100; i++){
    //     std::cout << h_vertices[38365 + i] << "  ";
    // }
    // std::cout << std::endl;
    ck(cudaGraphicsMapResources(1, &cuda_mesh_resource, stream));
    ck(cudaGraphicsResourceGetMappedPointer(&cuda_mapped_pointer, &cuda_mapped_size, cuda_mesh_resource));

    ck(cudaMemcpyAsync(cuda_mapped_pointer, d_vertices, cuda_mapped_size, cudaMemcpyDeviceToDevice, stream));

    ck(cudaGraphicsUnmapResources(1, &cuda_mesh_resource, stream));

    glm::mat4 projection = glm::ortho(0.0f, (float)W, 0.0f, (float)H, -10000.0f, 10000.0f);
    // print(projection, "projection");

    glCheckError();
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glFinish();
    glCheckError();
    // render loop
    // while (!glfwWindowShouldClose(window)) {
        // input
        // processInput(window);
        // GLenum check_return = glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER);
        // cout << glGetString(check_return) << endl;
        // render
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glCheckError();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glCheckError();

        shader_pic.Use();
        glCheckError();
        glActiveTexture(GL_TEXTURE0);
        glCheckError();
        glBindTexture(GL_TEXTURE_2D, tex_image);
        glCheckError();
        shader_pic.SetUniform("tex0", 0);
        glBindVertexArray(VAO_pic);
        glCheckError();
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glDrawElements(GL_TRIANGLES, sizeof(indices_pic) / sizeof(indices_pic[0]), GL_UNSIGNED_INT, 0);
        glCheckError();

        shader_face.Use();
        shader_face.SetUniform("W", (float)W);
        shader_face.SetUniform("H", (float)H);
        shader_face.SetUniform("projection", projection);
        glCheckError();

        shader_face.Use();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex_face);
        shader_face.SetUniform("tex0", 0);
        glBindVertexArray(VAO_face);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        // glLineWidth(0.1);
        glDrawElements(GL_TRIANGLES, index.num_vals, GL_UNSIGNED_INT, 0);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // glfwSwapBuffers(window);
        // glfwPollEvents();
    // }
    glCheckError();
    std::vector<std::uint8_t> buf(W * H * 3);
    glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, buf.data());
    stbi_write_png("rio_out.png", W, H, 3, buf.data(), W * 3);
    // glfw: terminate, clearing all previously allocated GLFW resources.
    // glfwTerminate();
}

void init_encoder(CUcontext cuContext, int nWidth, int nHeight, char *szOutFilePath, 
    NvEncoderInitParam &initParam, NvEncLite* &enc, FILE* &fpOut) {
    fpOut = fopen(szOutFilePath, "wb");
    if (fpOut == NULL) {
        cout << "Unable to open file: " << szOutFilePath << endl;
        return;
    }

    std::string init_param_string{"-codec hevc -preset p7 -bitrate 2M"};
    initParam = NvEncoderInitParam(init_param_string.c_str());
    NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_NV12;

    enc = new NvEncLite(cuContext, nWidth, nHeight, eFormat, &initParam);
}

void encode_device_frame(NvEncLite* enc, uint8_t *device_frame, FILE *fpOut) {
    vector<vector<uint8_t>> vPacket;
    enc->EncodeDeviceFrame(device_frame, 0, vPacket);

    for (vector<uint8_t> &packet : vPacket) {
        fwrite(packet.data(), 1, packet.size(), fpOut);
    }
}

#define STREAM_NUM 3

int main(int argc, char **argv){
    int iDevice = 0;
    if (argc >= 2) iDevice = atoi(argv[1]);
    ck(cudaSetDevice(iDevice));
    CUcontext current;
    (cuCtxGetCurrent(&current));
    std::cout <<"Current cuda context: " << current << std::endl;
    ck(cuDevicePrimaryCtxRetain(&current, iDevice));
    ck(cuCtxPushCurrent(current));
    
    cudaDeviceProp prop = {};
    ck(cudaGetDeviceProperties(&prop, iDevice));
    cout << "Using " << prop.name << endl;

    // cudaStream_t stream = 0;
    // ck(cudaStreamCreate(&stream));

    cudaStream_t stream_array[STREAM_NUM];
    for (int i = 0; i < STREAM_NUM; i++) {
        ck(cudaStreamCreate(&stream_array[i]));
    }

    init_egl();

    // c10::Device torch_cuda_dev(at::kCUDA, iDevice);
    // c10::Stream torch_cuda_stream(torch_cuda_dev, );

    auto tensor_options_cuda_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto tensor_options_cuda_byte = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);

    BuildEngineParam param = {1, 3};
    // int nBatch = 1, nChannel = 3, nHeight = 720, nWidth = 1280;
    // int nFloat = nBatch * param.nChannel * nHeight * nWidth,
    //     nByte = nFloat * sizeof(float);

    // GPU pipeline
    //---------------------------------------------------------------
    FaceBoxes_ONNX face_boxes_gpu{"./bin/sample/3ddfa/FaceBoxesProd_rtx8000_fp16.trt8203"};
    TDDFA_ONNX tddfa_gpu{"./bin/sample/3ddfa/mb1_120x120_rtx8000_fp16_bs1.trt8203",
        "./bin/sample/3ddfa/bfm_noneck_v3_rtx8000_fp16_static.trt8203"};
    vector<torch::Tensor> param_vec_gpu, roi_box_vec_gpu;

    OutputFormat eOutputFormat = native;
    BufferedFileReader reader("./bin/sample/3ddfa/demo_JHH.mp4");
    uint8_t *pBuf;
    size_t nBufSize;
    reader.GetBuffer(&pBuf, &nBufSize);
    Demuxer demuxer(pBuf, nBufSize, false);
    NvDecLiteImageProvider dec(current, &demuxer);
    int frameWidth = dec.GetWidth(), frameHeight = dec.GetHeight();

    // cv::Mat cv_image = imread("./bin/sample/3ddfa/one_face_1280.jpg");
    // cv::cvtColor(cv_image, cv_image, cv::COLOR_BGR2YUV_YV12);

    // Encoder setup
    NvEncoderInitParam initParam;
    NvEncLite *enc = nullptr;
    FILE *fpOut = nullptr;
    init_encoder(current, frameWidth, frameHeight, "./bin/sample/3ddfa/demo_JHH_out.hevc", 
    initParam, enc, fpOut);
    // std::string szOutFilePath{"./bin/sample/3ddfa/demo_JHH_out.hevc"};
    // FILE *fpOut = fopen(szOutFilePath.c_str(), "wb");
    // if (!fpOut) {
    //     cout << "Unable to open output file: " << szOutFilePath << endl;
    //     return;
    // }
    // NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_NV12;
    // NvEncoderInitParam initParam;

    // NvEncLite enc(cuContext, frameWidth, frameHeight, eFormat, pInitParam);
    // if (!enc.ReadyForEncode()) {
    //     cout << "NvEncLite fails to initialize." << endl;
    //     return;
    // }

    int anSize[] = {0, 3, 4, 8};
    int nFrameSize = eOutputFormat == native ? dec.GetFrameSize() : frameWidth * frameHeight * anSize[eOutputFormat];
    int nFrame = 0;
    uint8_t *dpImageRgbp, *dpImageNv12;
    ck(cudaMalloc(&dpImageNv12, nFrameSize));
    ck(cudaMalloc(&dpImageRgbp, frameWidth * frameHeight * 3 * sizeof(float)));
    float shift[3] = {123, 117, 104};
    torch::Tensor image_bgra = torch::zeros({BATCH_SIZE, frameHeight, frameWidth, 4}, tensor_options_cuda_byte);
    torch::Tensor out_nv12 = torch::empty({BATCH_SIZE, frameHeight * 3 / 2, frameWidth}, tensor_options_cuda_byte);
    torch::Tensor image_rgba = torch::empty({BATCH_SIZE, frameHeight, frameWidth, 4}, tensor_options_cuda_byte);
    // torch::Tensor rgba_permute = torch::empty({3}, torch::dtype(torch::kLong));
    // rgba_permute[0] = 2;
    // rgba_permute[1] = 1;
    // rgba_permute[2] = 0;

    cuda_op::DataShape input_shape(BATCH_SIZE, 4, frameHeight, frameWidth);
    cuda_op::CvtColor cvt_color(input_shape, input_shape);
    void *cvt_color_workspace;
    ck(cudaMalloc(&cvt_color_workspace, cvt_color.calBufferSize(input_shape, input_shape, cuda_op::kCV_8U)));

    uint8_t *dpRgbpf[3];
    uint8_t *dpNv12[2];
    dpRgbpf[0] = dpImageRgbp;
    dpRgbpf[1] = dpImageRgbp + frameWidth * frameHeight * sizeof(float);
    dpRgbpf[2] = dpImageRgbp + frameWidth * frameHeight * sizeof(float) * 2;

    dpNv12[0] = dpImageNv12;
    dpNv12[1] = dpImageNv12 + frameWidth * frameHeight;
    config_gl("./bin/sample/3ddfa/bfm_tri.npy", tddfa_gpu.vertex_num(), frameWidth, frameHeight);
    
    ck(cudaProfilerStart());
    auto clock_start = chrono::steady_clock::now();

#define PERF
// #define NVTX_RANGE
#ifdef PERF
    while(true) 
#endif
    {
        bool finish = false;
        for (int stream_count = 0; stream_count < STREAM_NUM; stream_count++)
        {
        cudaStream_t stream = stream_array[stream_count];
        at::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(stream, iDevice));
        if (!(dec.GetNextFrame(dpImageNv12, frameWidth, true, stream)) || nFrame == 9) {
#ifdef PERF
            encode_device_frame(enc, NULL, fpOut);
            fclose(fpOut);
            finish = true;
            break;
#endif
        }
        // ck(cudaMemcpy(dpImageNv12, cv_image.data, frameWidth * frameHeight * 3 / 2, cudaMemcpyHostToDevice));
        nFrame++;
        // if (nFrame == 100) ck(cudaProfilerStart());

#ifdef NVTX_RANGE
        nvtxRangeId_t r1 = nvtxRangeStartA("Colorspcace conversion range");
#endif
        ffgd::Nv12ToBgrpf32Shift(dpImageNv12, frameWidth, dpImageRgbp, frameWidth * sizeof(float),
            frameWidth, frameHeight, 1.0f, shift, stream);
        ffgd::Nv12ToColor32<ffgd::BGRA32>(dpImageNv12, frameWidth, image_bgra.data_ptr<uint8_t>(), frameWidth * 4,
            frameWidth, frameHeight, stream);
        ck(cudaGetLastError());
#ifdef NVTX_RANGE
        nvtxRangeEnd(r1);
#endif

        // DEBUG
        // cudaDeviceSynchronize();
        // uint8_t image_nv12[720*1280*3/2];
        // auto image_cpu = image_bgra.to(torch::kCPU);
        // cudaDeviceSynchronize();
        // ck(cudaMemcpy(image_nv12, dpImageNv12, 720*1280*3/2, cudaMemcpyDeviceToHost));
        // stbi_write_png("bgra_debug_out.png", frameWidth, frameHeight, 4, image_cpu.data_ptr<uint8_t>(), frameWidth);

#ifdef NVTX_RANGE
        nvtxRangeId_t r2 = nvtxRangeStartA("FaceBoxes range");
#endif
        torch::Tensor face_dets_gpu = face_boxes_gpu.forward(reinterpret_cast<float*>(dpImageRgbp), frameHeight, frameWidth, true, stream);
        // std::cout << face_dets_gpu << std::endl;
        if (face_dets_gpu.size(0) == 0){
            encode_device_frame(enc, dpImageNv12, fpOut);
#ifdef PERF
            continue;
#endif
        }
#ifdef NVTX_RANGE
        nvtxRangeEnd(r2);
#endif

#ifdef NVTX_RANGE
        nvtxRangeId_t r3 = nvtxRangeStartA("Tddfa range");
#endif
        tddfa_gpu.forward(image_bgra, face_dets_gpu, param_vec_gpu, roi_box_vec_gpu, stream);
        // std::cout << face_dets_gpu << std::endl;
#ifdef NVTX_RANGE
        nvtxRangeEnd(r3);
#endif

#ifdef NVTX_RANGE
        nvtxRangeId_t r4 = nvtxRangeStartA("GL draw range");
#endif
        image_rgba = image_bgra.index({Slice(None), Slice(None), Slice(None), torch::tensor({2, 1, 0, 3})});
        }
        if (finish) break;

        for (int stream_count = 0; stream_count < STREAM_NUM; stream_count++)
        {
            cudaStream_t stream = stream_array[stream_count];
            draw_gl(image_rgba, tddfa_gpu.vertex_data(), out_nv12, stream);
            
    #ifdef NVTX_RANGE
            nvtxRangeId_t r5 = nvtxRangeStartA("Encoding range");
    #endif
            encode_device_frame(enc, (uint8_t*)out_nv12.data_ptr(), fpOut);
    #ifdef NVTX_RANGE
            nvtxRangeEnd(r5);
    #endif
        }
#ifdef NVTX_RANGE
        nvtxRangeEnd(r4);
#endif

        // DEBUG
        // stbi_write_png("nv12_debug_out.png", frameWidth, frameHeight, 2, out_nv12.to(torch::kCPU).data_ptr<uint8_t>(), frameWidth);

        // uint8_t h_nv12_out[frameWidth * frameHeight * 3 / 2];
        // ck(cudaMemcpy(h_nv12_out, out_nv12.data_ptr<uint8_t>(), frameWidth * frameHeight * 3 / 2, cudaMemcpyDeviceToHost));
        // stbi_write_png("nv12_debug_out.png", frameWidth, frameHeight, 2, h_nv12_out, frameWidth);

        // if (nFrame == 100) ck(cudaProfilerStop());
    }
    // fclose(fpOut);
    
    ck(cudaDeviceSynchronize());
    auto clock_stop = chrono::steady_clock::now();
    chrono::duration<double> diff_clock = clock_stop - clock_start;
    cout << "Average latency of " << nFrame << " runs: "<< diff_clock.count() / nFrame << " s\n";
    ck(cudaProfilerStop());

    // ---------------------------------------------------------------

#ifdef OLD
    // First pipeline, CPU + GPU
    // ---------------------------------------------------------------
    cv::Mat image = imread("./bin/sample/3ddfa/one_face_1280.jpg");
    // cvtColor(image, image, COLOR_BGR2RGB);
    vector<float> img_chw(image.channels() * image.cols * image.rows);
    cv::Vec3i mean = {104, 117, 123};
    reorder_to_chw(image, img_chw, mean, false);

    FaceBoxes_ONNX face_boxes{"./bin/sample/3ddfa/FaceBoxesProd_rtx8000_fp16.trt8203"};
    TDDFA_ONNX tddfa{"./bin/sample/3ddfa/mb1_120x120_rtx8000_fp16_bs1.trt8203",
        "./bin/sample/3ddfa/bfm_noneck_v3_rtx8000_fp16_static.trt8203"};
    vector<torch::Tensor> param_vec, roi_box_vec;

    cudaDeviceSynchronize();
    auto inference_start = chrono::steady_clock::now();

    int N_RUNS = 1;
    for (int i = 0; i < N_RUNS; i++)
    {
        auto face_dets = face_boxes.forward<float>(img_chw.data(), image.rows, image.cols);
        tddfa.forward(image, face_dets, param_vec, roi_box_vec);
    }
    cudaStreamSynchronize(0);
    auto inference_stop = chrono::steady_clock::now();
    chrono::duration<double> diff_inference = inference_stop - inference_start;

    cout << "Time of inference: "<< diff_inference.count() / N_RUNS << " s\n";


    // init_egl();
    // GLFWwindow* window;
    // init_glfw(&window);

    draw_on_image(image, "./bin/sample/3ddfa/bfm_tri.npy", tddfa.vertex_data(), tddfa.vertex_num());
#endif
}