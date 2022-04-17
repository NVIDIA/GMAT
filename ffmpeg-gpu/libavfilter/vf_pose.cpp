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

/**
 * @file
 * implementing a filter to generate & render faces.
 */

// C++ section
#include <bits/stdint-uintn.h>
#include <climits>
#include <cstddef>
#include <libavutil/macros.h>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <onnxruntime/onnxruntime_c_api.h>
#include <onnxruntime/onnxruntime_session_options_config_keys.h>
#include <onnxruntime/onnxruntime_run_options_config_keys.h>
#include <stdint.h>
#include <torch/torch.h>
#include <torchvision/vision.h>
#include <torchvision/ops/nms.h>
#include <Eigen/Dense>
#include "pose/cnpy.h"
#include "pose_kernel.h"
#include "pose_proc.h"
#include "pose_shaders.h"

#include <string>
#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <unistd.h>
/*
 * EGL headers.
 */
#include <EGL/egl.h>

 /*
  * OpenGL headers.
  */
#include <EGL/eglext.h>
#include "pose/GLUtils.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

#include <cuda_gl_interop.h>

using namespace std;
using namespace Ort;
using namespace Eigen;

// extern at::Tensor nms_kernel_gpu(
//     const at::Tensor& dets,
//     const at::Tensor& scores,
//     double iou_threshold);

struct OrtContext{
    OrtContext(){
        pose_mean = VectorXf{{-0.0238f,  0.0275f, -0.0144f,  0.0664f,  0.2380f, 3.4813f}};
        pose_stddev = VectorXf{{0.2353f, 0.5395f, 0.1767f, 0.1320f, 0.1358f, 0.3663f}};

        input_names = {"images"};
        output_names = {"boxes", "labels", "scores", "dofs"};
    }
    ~OrtContext(){
        if (mem_info != nullptr)
            delete mem_info;
        if (ort_session != nullptr)
            delete ort_session;
        if (ort_in_dev)
            cudaFree(ort_in_dev);
    }

    Session *ort_session;
    MemoryInfo *mem_info;
    VectorXf pose_mean;
    VectorXf pose_stddev;
    vector<const char*> input_names;
    vector<const char*> output_names;
    vector<Ort::Value> ort_inputs;
    array<int64_t, 3> image_shape;

    void *ort_in_dev;
};

struct GLContext{
    ~GLContext(){
        if (cuda_image_resource)
            cudaGraphicsUnregisterResource(cuda_image_resource);
        if (cuda_out_resource)
            cudaGraphicsUnregisterResource(cuda_out_resource);

        if (shader_face) delete shader_face;
        if (shader_pic) delete shader_pic;
    }
    size_t num_triangles;
    GLuint VAO_pic, VBO_pic, VAO_face, VBO_face;
    GLuint gl_image, gl_tex, gl_tex_face, gl_out;
    cudaGraphicsResource *cuda_image_resource, *cuda_out_resource;
    Shader *shader_pic, *shader_face;

    Model model;
};

static OrtContext *ort_ctx;
static GLContext gl_ctx;

extern "C"{
    #include "libavutil/frame.h"
    #include "libavutil/log.h"
    #include "libavutil/buffer.h"
    #include "avfilter.h"
    #include "filters.h"
    #include "libavutil/hwcontext.h"
    #include "libavutil/hwcontext_cuda_internal.h"
    typedef struct PoseContext{
        const AVClass *av_class;
        char* model_path;
        char* mask_path;
        size_t ort_gpu_mem_limit;
        size_t ort_arena_strategy;
        AVBufferRef *hw_frames_ctx;
        void *ort_in, *ort_out;

        int gpu;

        EGLDisplay display;
        EGLConfig config;
        EGLContext context;
        EGLSurface surface;
        EGLint num_config;

    }PoseContext;
}

// Self-defined CUDA check functions as cuda_check.h is not available for cpp due to void* function pointers
static inline bool check_cu(CUresult e, void *ctx, CudaFunctions* cu, int iLine, const char *szFile) {
    if (e != CUDA_SUCCESS) {
        const char* pStr;
        cu->cuGetErrorName(e, &pStr);
        av_log(ctx, AV_LOG_ERROR, "CUDA driver API error: %s, at line %d in file %s\n",
        pStr, iLine, szFile);
        return false;
    }
    return true;
}

static inline bool check(cudaError_t e, void *ctx, int iLine, const char *szFile) {
    if (e != cudaSuccess) {
        av_log(ctx, AV_LOG_ERROR, "CUDA runtime API error: %s, at line %d in file %s\n",
            cudaGetErrorName(e), iLine, szFile);
        return false;
    }
    return true;
}

#define ck(call) check(call, s, __LINE__, __FILE__)
#define ck_cu(call) check_cu(call, s, cu, __LINE__, __FILE__)

static unsigned int indices_pic[] = {
    0, 1, 2,
    2, 3, 0
};

static void load_ort(OrtContext *s, char *model_path, array<int64_t, 3> image_shape={3, 720, 1280}){
    string instanceName{"img2pose"};
    const auto& api = Ort::GetApi();

    SessionOptions sessionOptions;
    const char* keys[] = {"max_mem", "arena_extend_strategy", "initial_chunk_size_bytes", "max_dead_bytes_per_chunk", "initial_growth_chunk_size_bytes"};
    const size_t values[] = {0 /*let ort pick default max memory*/, 0, 1024, 0, 256};

    OrtArenaCfg* arena_cfg = nullptr;
    assert(api.CreateArenaCfgV2(keys, values, 5, &arena_cfg) == nullptr);
    MemoryInfo *info_cuda = new MemoryInfo("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
    s->mem_info = info_cuda;

    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                 instanceName.c_str());
    OrtEnv* env_ptr = (OrtEnv*)(env);
    OrtCUDAProviderOptions cuda_provider_options{};
    cuda_provider_options.cudnn_conv_algo_search = (enum OrtCudnnConvAlgoSearch)0;
    cuda_provider_options.gpu_mem_limit = 16 * 1024 * 1024 * 1024ul; // 16 GB
    // ffmpeg only uses default stream
    cuda_provider_options.do_copy_in_default_stream = 1;

    cuda_provider_options.default_memory_arena_cfg = arena_cfg;
    sessionOptions.AppendExecutionProvider_CUDA(cuda_provider_options);
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    assert(api.CreateAndRegisterAllocator(env_ptr, *info_cuda, arena_cfg)==nullptr);

    s->ort_session = new Session(env, model_path, sessionOptions);
    Ort::Allocator cuda_allocator(*s->ort_session, *info_cuda);
    auto allocator_info = cuda_allocator.GetInfo();
    assert(*info_cuda == allocator_info);

    int ort_in_size = image_shape[0] * image_shape[1] * image_shape[2];
    Ort::Value dryrun_tensor = Ort::Value::CreateTensor(*s->mem_info, static_cast<float*>(s->ort_in_dev),
                                                            ort_in_size, image_shape.data(), image_shape.size());

    s->ort_inputs.push_back(move(dryrun_tensor));
    // Dry run
    // cout << "Start session run\n";
    vector<Ort::Value> ort_outputs = s->ort_session->Run(Ort::RunOptions{nullptr}, s->input_names.data(),
                                                s->ort_inputs.data(), s->input_names.size(),
                                                s->output_names.data(), s->output_names.size());
}

// C section, included in extern "C"
extern "C"{
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "avfilter.h"
#include "filters.h"
#include "formats.h"
#include "internal.h"
#include "libavutil/frame.h"
#include "libavutil/pixfmt.h"
#include "libavutil/cuda_check.h"
#include "libavformat/avio.h"
#include "libavutil/opt.h"
#include "pose/stb_image_write.h"

#include <dlfcn.h>

#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, cu, x)

#define OFFSET(x) offsetof(PoseContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM

static const AVOption pose_options[] = {
    {"model", "path to the ONNX modeld file", OFFSET(model_path), AV_OPT_TYPE_STRING, {.str = NULL}, 0,  0,       FLAGS},
    {"gpu_mem", "GPU memory limit for onnxruntime (GB)", OFFSET(ort_gpu_mem_limit), AV_OPT_TYPE_UINT64, {.i64 = 4}, 1,  UINT_MAX, FLAGS},
    {"mask", "path to the digital assets", OFFSET(mask_path), AV_OPT_TYPE_STRING, {.str = "./"}, 0, 0, FLAGS},
    {"arena", "Onnxruntime arena extend strategy", OFFSET(ort_arena_strategy), AV_OPT_TYPE_UINT64, {.i64 = 0}, 0,  1, FLAGS},
    {"gpu", "GPU index", OFFSET(gpu), AV_OPT_TYPE_UINT64, {.i64 = 0}, 0,  32, FLAGS},
    {NULL}
};

AVFILTER_DEFINE_CLASS(pose);

static void print(const glm::mat4 &m, std::string name) {
    std::cout << name << "=np.asarray([";
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << m[j][i] << ",";
        }
        std::cout << std::endl;
    }
    std::cout << "]).reshape(4, 4)" << std::endl;
}

// static void ck_egl(EGLBoolean ret) {
//     if (ret) return;
//     cout << "EGL error" << endl;
// 	EGLint error = eglGetError();
// 	if (error != EGL_SUCCESS) {
// 		stringstream s;
// 		s << "EGL error 0x" << std::hex << error;
// 		throw runtime_error(s.str());
// 	}
// }

static void assertEGLError(const std::string& msg) {
	EGLint error = eglGetError();

	if (error != EGL_SUCCESS) {
		stringstream s;
		s << "EGL error 0x" << std::hex << error << " at " << msg;
		throw runtime_error(s.str());
	}
}

static int init_egl(PoseContext *s){
    static const EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_BLUE_SIZE,8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_DEPTH_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
    };

    constexpr int RENDER_USE_GPU = 0;
    constexpr int MAX_DEVICES = 32;
	EGLDeviceEXT eglDevs[MAX_DEVICES];
	EGLint numDevices;
	PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT =
	(PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");

	eglQueryDevicesEXT(MAX_DEVICES, eglDevs, &numDevices);

	PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
		(PFNEGLGETPLATFORMDISPLAYEXTPROC) eglGetProcAddress("eglGetPlatformDisplayEXT");

    EGLint i = 0;
    for (; i < numDevices; ++i) {
        EGLDisplay disp = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT,
                                                        eglDevs[i], nullptr);
        if (eglGetError() == EGL_SUCCESS && disp != EGL_NO_DISPLAY && RENDER_USE_GPU==i ) {
            s->display = disp;
            cout << "current display is " << i<<endl;
            break;
        }
    }
    assert( RENDER_USE_GPU<numDevices );

    assertEGLError("eglGetDisplay");

    int major=0, minor=-2;
    eglInitialize(s->display, &major, &minor);

	assertEGLError("eglInitialize");

	eglChooseConfig(s->display, configAttribs, &s->config, 1, &s->num_config);
	assertEGLError("eglChooseConfig");

	eglBindAPI(EGL_OPENGL_API);
	assertEGLError("eglBindAPI");

	s->context = eglCreateContext(s->display, s->config, EGL_NO_CONTEXT, NULL);
	assertEGLError("eglCreateContext");

	eglMakeCurrent(s->display, EGL_NO_SURFACE, EGL_NO_SURFACE, s->context);
	assertEGLError("eglMakeCurrent");

    printf("%s\n", glGetString(GL_VENDOR));
    printf("%s\n", glGetString(GL_RENDERER));
    printf("%s\n", glGetString(GL_VERSION));

    return 0;
}

static int init(AVFilterContext *ctx){
    PoseContext *s = (PoseContext*)ctx->priv;
    if (!s->model_path) {
        av_log(s, AV_LOG_ERROR, "No model file provided.\n");
        return AVERROR(EINVAL);
    }

    ort_ctx = new OrtContext();

    if (init_egl(s) != 0){
        av_log(s, AV_LOG_ERROR, "Failed to load OpenGL with EGL\n");
        return AVERROR(EINVAL);
    }

    return 0;
}

static int config_opengl_mask(int width, int height, PoseContext *s, CudaFunctions *cu, CUcontext cuctx, cudaStream_t stream=0){
    glViewport(0, 0, width, height);

    GLuint fbo, rbo;
    glGenFramebuffers(1, &fbo);
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB, width, height);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo);

    GLuint rbo_depth;
    glGenRenderbuffers(1, &rbo_depth);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo_depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo_depth);

    unsigned int cuda_buffer;
    glGenBuffers(1, &cuda_buffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, cuda_buffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4,
                NULL, GL_DYNAMIC_COPY);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    GLenum gl_error = glGetError();
    if (gl_error != GL_NO_ERROR){
        printf("GL error\n");
    }

    unsigned int texture0;
    glGenTextures(1, &texture0);
    glBindTexture(GL_TEXTURE_2D, texture0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGetError();
    glGenerateMipmap(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, 0);

    // use &gl_ctx will cause segfault directly
    unsigned int VBO_pic, VAO_pic;
    glGenVertexArrays(1, &VAO_pic);
    glGenBuffers(1, &VBO_pic);
    gl_ctx.VBO_pic = VBO_pic;
    gl_ctx.VAO_pic = VAO_pic;

    // 0. bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO_pic);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_pic);

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

    unsigned int EBO_pic;
    glGenBuffers(1, &EBO_pic);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_pic);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices_pic), indices_pic, GL_STATIC_DRAW);

    string mask_path{s->mask_path};
    if (mask_path.back() != '/') mask_path += '/';
    mask_path += "medmask.obj";

    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(mask_path, aiProcess_Triangulate | aiProcess_FlipUVs);
    if (!scene) {
        av_log(s, AV_LOG_ERROR, "%s\n", importer.GetErrorString());
        return -1;
    }
    gl_ctx.model = Model::Parse(scene);

    GLuint pack_buffer;
    glGenBuffers(1, &pack_buffer);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pack_buffer);
    glBufferData(GL_PIXEL_PACK_BUFFER, width * height * 4,
        NULL, GL_DYNAMIC_READ);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    // Build shaders
    gl_ctx.shader_face = new Shader(srcVertexShader_face_mask, srcFragmentShader_face_mask);
    gl_ctx.shader_pic = new Shader(srcVertexShader_pic_mask, srcFragmentShader_pic_mask);
    glGetError();

    // unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glGetError();

    CUcontext dummy;
    ck_cu(cu->cuCtxPushCurrent(cuctx));


    // Register texture in CUDA
    ck(cudaGraphicsGLRegisterBuffer(&gl_ctx.cuda_image_resource, cuda_buffer, cudaGraphicsRegisterFlagsNone));
    ck(cudaGraphicsGLRegisterBuffer(&gl_ctx.cuda_out_resource, pack_buffer, cudaGraphicsRegisterFlagsReadOnly));
    gl_ctx.gl_image = cuda_buffer;
    gl_ctx.gl_tex = texture0;
    gl_ctx.gl_out = pack_buffer;
    ck_cu(cu->cuCtxPopCurrent(&dummy));

    return 0;
}

static int query_formats(AVFilterContext *ctx){
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE
    };
    AVFilterFormats *fmts_list = ff_make_format_list((const int*)pix_fmts);
    return ff_set_common_formats(ctx, fmts_list);
}

static int config_props(AVFilterLink *outlink){
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = ctx->inputs[0];
    PoseContext *s = (PoseContext*)ctx->priv;

    AVHWFramesContext *in_frame_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVBufferRef *hw_device_ref = in_frame_ctx->device_ref;
    AVCUDADeviceContext *hw_ctx = (AVCUDADeviceContext*)in_frame_ctx->device_ctx->hwctx;
    CudaFunctions *cu = hw_ctx->internal->cuda_dl;

    AVBufferRef *out_ref = NULL;
    AVHWFramesContext *out_ctx;
    CUcontext dummy;

    int ret;

    int ort_in_size = 3 * inlink->h * inlink->w;
    enum AVPixelFormat fmt = in_frame_ctx->sw_format;

    switch (fmt) {
    case AV_PIX_FMT_NV12:
        break;
    default:
        av_log(s, AV_LOG_ERROR, "Pixel format not supported\n");
        return AVERROR(EINVAL);
    }
    config_opengl_mask(inlink->w, inlink->h, s, cu, hw_ctx->cuda_ctx);

    ck_cu(cu->cuCtxPushCurrent(hw_ctx->cuda_ctx));

    ck_cu(cu->cuMemAlloc((CUdeviceptr*)&ort_ctx->ort_in_dev, ort_in_size * sizeof(float)));

    ort_ctx->image_shape = {3, inlink->h, inlink->w};
    load_ort(ort_ctx, s->model_path, ort_ctx->image_shape);

    outlink->w = inlink->w;
    outlink->h = inlink->h;

    out_ref = av_hwframe_ctx_alloc(hw_device_ref);
    if (!out_ref)
        return AVERROR(ENOMEM);
    out_ctx = (AVHWFramesContext*)out_ref->data;

    out_ctx->format = AV_PIX_FMT_CUDA;
    out_ctx->sw_format = in_frame_ctx->sw_format;
    out_ctx->width = inlink->w;
    out_ctx->height = inlink->h;

    ret = av_hwframe_ctx_init(out_ref);
    if (ret < 0)
    {
        av_buffer_unref(&out_ref);
        return ret;
    }

    ck_cu(cu->cuCtxPopCurrent(&dummy));

    av_buffer_unref(&s->hw_frames_ctx);

    s->hw_frames_ctx = inlink->hw_frames_ctx;
    outlink->hw_frames_ctx = av_buffer_ref(out_ref);
    if (!outlink->hw_frames_ctx)
        return AVERROR(ENOMEM);
    return 0;
}

static int transform_and_draw_mask(PoseContext *s, AVFrame *in, AVFrame *out, float *poses, int nPose, float *scores,
    CudaFunctions *cu, CUcontext cuctx, int W, int H, float scores_thresh=0.9f){
    void *dev_ptr;
    size_t tex_size;
    cudaArray *tex_array;
    CUcontext cur;
    ck_cu(cu->cuCtxPushCurrent(cuctx));

    ck(cudaGraphicsMapResources(1, &gl_ctx.cuda_image_resource, 0));
    ck(cudaGraphicsResourceGetMappedPointer(&dev_ptr, &tex_size, gl_ctx.cuda_image_resource));

    Nv12ToColor32<RGBA32>(in->data, in->linesize[0], static_cast<uint8_t*>(dev_ptr), W * 4, W, H, in->colorspace);
    ck(cudaGetLastError());

    ck(cudaGraphicsUnmapResources(1, &gl_ctx.cuda_image_resource, 0));
    ck_cu(cu->cuCtxPopCurrent(&cur));

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_ctx.gl_image);
    glBindTexture(GL_TEXTURE_2D, gl_ctx.gl_tex);
    glCheckError();
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glCheckError();
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glCheckError();

    glCheckError();

    gl_ctx.shader_pic->Use();
    glCheckError();
    gl_ctx.shader_pic->SetUniform("tex0", 0);
    glCheckError();

    gl_ctx.shader_face->Use();


    gl_ctx.shader_face->SetUniform("W", (float)W);
    gl_ctx.shader_face->SetUniform("H", (float)H);

    glm::mat4 projection = glm::mat4(
        W + H, 0.0, 0.0, 0.0,
        0.0, W + H, 0.0, 0.0,
        W / 2.0f, H / 2.0f, 1.0, 0.0,
        0.0,0.0,0.0,1.0
    );
    gl_ctx.shader_face->SetUniform("projection", projection);
    glCheckError();

    glEnable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glCheckError();

    gl_ctx.shader_pic->Use();
    glBindVertexArray(gl_ctx.VAO_pic);
    glBindBuffer(GL_ARRAY_BUFFER, gl_ctx.VBO_pic);

    int iTexPic = 0;
    glActiveTexture(GL_TEXTURE0 + iTexPic);
    glBindTexture(GL_TEXTURE_2D, gl_ctx.gl_tex);
    gl_ctx.shader_pic->SetUniform("tex0", iTexPic);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDrawElements(GL_TRIANGLES, sizeof(indices_pic) / sizeof(indices_pic[0]), GL_UNSIGNED_INT, 0);
    glCheckError();

    for (int i = 0; i < nPose; i++) {
        if (scores[i] < scores_thresh) continue;
        float *pose = poses + 6 * i;
        glm::vec3 rotvec(pose[0], pose[1], pose[2]);
        float l = glm::length(rotvec);
        glm::mat4 m = glm::toMat4(glm::angleAxis(l, rotvec/l));
        m[3][0] = pose[3];
        m[3][1] = pose[4];
        m[3][2] = pose[5];
        m = glm::scale(m, glm::vec3(10.0f, 10.0f, 10.0f));
        gl_ctx.shader_face->Use();
        gl_ctx.shader_face->SetUniform("model", m);
        glCheckError();
        gl_ctx.model.Draw(*gl_ctx.shader_face);
        glCheckError();
    }
    glCheckError();

    // Read framebuffer into packed buffer
    glBindBuffer(GL_PIXEL_PACK_BUFFER, gl_ctx.gl_out);
    glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glCheckError();
    glFinish();

    ck_cu(cu->cuCtxPushCurrent(cuctx));

    void *out_ptr;
    size_t out_size;
    ck(cudaGraphicsMapResources(1, &gl_ctx.cuda_out_resource, 0));
    ck(cudaGraphicsResourceGetMappedPointer(&out_ptr, &out_size, gl_ctx.cuda_out_resource));

    Color32ToNv12<RGBA32>(static_cast<uint8_t*>(out_ptr), W * 4, out->data[0], out->linesize[0], W, H, in->colorspace);

    ck(cudaGraphicsUnmapResources(1, &gl_ctx.cuda_out_resource));
    ck_cu(cu->cuCtxPopCurrent(&cur));


    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    PoseContext *s = (PoseContext*)ctx->priv;

    AVHWFramesContext *in_frame_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVCUDADeviceContext *hw_ctx = (AVCUDADeviceContext*)in_frame_ctx->device_ctx->hwctx;
    CudaFunctions *cu = hw_ctx->internal->cuda_dl;

    CUcontext dummy;

    vector<Ort::Value> ort_outputs;

    int ret, linesize;
    uint8_t* dp_rgbpf32[3];
    AVFrame *out = av_frame_alloc();
    if (!out)
    {
        ret = AVERROR(ENOMEM);
        av_frame_free(&in);
        av_frame_free(&out);
        return ret;
    }
    ck_cu(cu->cuCtxPushCurrent(hw_ctx->cuda_ctx));


    ret = av_hwframe_get_buffer(outlink->hw_frames_ctx, out, 0);
    if (ret < 0){
        av_frame_free(&in);
        av_frame_free(&out);
        return ret;
    }

    linesize = inlink->w * sizeof(float);
    dp_rgbpf32[0] = static_cast<uint8_t*>(ort_ctx->ort_in_dev);
    dp_rgbpf32[1] = static_cast<uint8_t*>(ort_ctx->ort_in_dev) + linesize * inlink->h;
    dp_rgbpf32[2] = static_cast<uint8_t*>(ort_ctx->ort_in_dev) + linesize * inlink->h * 2;
    nv12_to_rgbpf32(hw_ctx->stream, in->data, in->linesize, dp_rgbpf32,
                    linesize, in->width, in->height, in->colorspace);
    // Sync before launch ort inference, as ort session runs on a different stream
    ck(cudaStreamSynchronize(hw_ctx->stream));

    ort_outputs = ort_ctx->ort_session->Run(Ort::RunOptions{nullptr}, ort_ctx->input_names.data(),
                                                ort_ctx->ort_inputs.data(), ort_ctx->input_names.size(),
                                                ort_ctx->output_names.data(), ort_ctx->output_names.size());

    Ort::Value& dof_value = ort_outputs[3];
    Ort::TensorTypeAndShapeInfo dof_type_info = dof_value.GetTensorTypeAndShapeInfo();
    float* output_dofs = dof_value.GetTensorMutableData<float>();
    Ort::Value& score_value = ort_outputs[2];
    Ort::Value& box_value = ort_outputs[0];
    Ort::TensorTypeAndShapeInfo score_type_info = score_value.GetTensorTypeAndShapeInfo();
    Ort::TensorTypeAndShapeInfo box_type_info = box_value.GetTensorTypeAndShapeInfo();
    vector<int64_t> score_shape = score_type_info.GetShape();
    vector<int64_t> box_shape = box_type_info.GetShape();
    vector<int64_t> dof_shape = dof_type_info.GetShape();

    float* output_scores = score_value.GetTensorMutableData<float>();
    float* output_boxes = box_value.GetTensorMutableData<float>();
    Map<MatrixXfRowMajor> dofs{static_cast<float*>(output_dofs), dof_shape[0], dof_shape[1]};
    Map<MatrixXfRowMajor> scores{static_cast<float*>(output_scores), score_shape[0], score_shape[1]};
    Map<MatrixXfRowMajor> boxes{static_cast<float*>(output_boxes), box_shape[0], box_shape[1]};

    MatrixXfRowMajor global_dofs;
    MatrixXfRowMajor projected_boxes;
    transform_pose_global_project_bbox(boxes, dofs, ort_ctx->pose_mean, ort_ctx->pose_stddev, ort_ctx->image_shape, threed_68_points,
        global_dofs, projected_boxes);
    torch::Tensor boxes_tensor = torch::from_blob(projected_boxes.data(), {projected_boxes.rows(), projected_boxes.cols()});
    torch::Tensor scores_tensor = torch::from_blob(static_cast<float*>(output_scores), {score_shape[0]});
    torch::Tensor dofs_tensor = torch::from_blob(global_dofs.data(), {global_dofs.rows(), global_dofs.cols()});

    if (boxes_tensor.sizes()[0] == 0){
        // No detections, passthrough
        return ff_filter_frame(outlink, in);
    }

    torch::Tensor boxes_gpu = boxes_tensor.to(torch::kCUDA);
    torch::Tensor scores_gpu = scores_tensor.to(torch::kCUDA);
    torch::Tensor dofs_gpu = dofs_tensor.to(torch::kCUDA);
    // auto keep_gpu = nms_kernel_gpu(boxes_gpu, scores_gpu, 0.5);
    auto keep_gpu = vision::ops::nms(boxes_gpu, scores_gpu, 0.5);
    auto scores_out = scores_gpu.index({keep_gpu}).to(torch::kCPU);
    auto dofs_out = dofs_gpu.index({keep_gpu}).to(torch::kCPU);

    torch::cuda::synchronize();
    ck_cu(cu->cuCtxPopCurrent(&dummy));

    transform_and_draw_mask(s, in, out, dofs_out.data<float>(), dofs_out.sizes()[0],
        scores_out.data<float>(), cu, hw_ctx->cuda_ctx, inlink->w, inlink->h);
    glFinish();
    glCheckError();

    ret = av_frame_copy_props(out, in);
    if (ret < 0)
        goto fail;
    av_frame_free(&in);
    ret = ff_filter_frame(outlink, out);

    return ret;

fail:
    av_frame_free(&in);
    av_frame_free(&out);
    return ret;
}

static void uninit(AVFilterContext *ctx){
    PoseContext *s = (PoseContext*)ctx->priv;

    if (ort_ctx) delete ort_ctx;
    if (s->ort_in) cudaFree(s->ort_in);
}

static const AVFilterPad pose_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
    },
    { NULL }
};

static const AVFilterPad pose_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_props,
    },
    { NULL }
};

AVFilter ff_vf_pose = {
    .name          = "pose",
    .description   = NULL_IF_CONFIG_SMALL("Generate and render poses"),
    .inputs        = pose_inputs,
    .outputs       = pose_outputs,
    .priv_class    = &pose_class,
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .priv_size     = sizeof(PoseContext),
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};

} // extern "C"
