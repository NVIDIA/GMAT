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

// #include <GL/gl.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
// #include <cv_cuda.h>
#include "torch/torch.h"
#include <torchvision/vision.h>
#include "trt_lite/trt_lite.h"
#include "trt_lite/trt_lite_utils.h"

#include "3ddfa/face_boxes.h"
#include "3ddfa/3ddfa_shaders.h"
#include "3ddfa/3ddfa_kernels.h"
#include "3ddfa/3ddfa.h"
#include "pose/GLUtils.h"
#include "cnpy/cnpy.h"
#include "format_cuda.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "pose/stb_image_write.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger(simplelogger::TRACE);

static cnpy::NpyArray npy_load(const char* fname) {
    std::string fname_string{fname};
    return cnpy::npy_load(fname_string);
}

extern "C"
{
#include <dirent.h>

#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include "libavutil/cuda_check.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/opt.h"
#include "libavutil/log.h"
#include "libavutil/error.h"
#include "libavutil/pixfmt.h"
#include "libavutil/pixdesc.h"

#include "cuda/check_util.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <cuda_gl_interop.h>

typedef struct TddfaContext{
    const AVClass *av_class;

    int model_h, model_w;
    AVBufferRef *hw_frames_ctx;

    void* d_rgbpf32_img, *d_bgra_img;
    char* path_opt;
    std::string facebox_path;
    std::string mb120_path;
    std::string bfm_path;
    FaceBoxes_ONNX *face_box;
    TDDFA_ONNX *tddfa;

    unsigned int tex_image, tex_face, VAO_face, VAO_pic, PBO_fb;
    size_t index_num;
    Shader *shader_face;
    Shader *shader_pic;
    cudaGraphicsResource *cuda_mesh_resource;
    cudaGraphicsResource *cuda_texture_resource;
    cudaGraphicsResource *cuda_fb_resource;
};

#define OFFSET(x) offsetof(TddfaContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM

static const AVOption tddfa_options[] = {
    // { "pix_fmt", "OpenCV color conversion code", OFFSET(outfmt_opt), AV_OPT_TYPE_STRING, .flags = AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM },
    { "path", "Path containing models and vertices", OFFSET(path_opt), AV_OPT_TYPE_STRING, {.str = "./"}, .flags = AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM},
    // { "cvt_code", "OpenCV color conversion code", OFFSET(cvt_code_opt), AV_OPT_TYPE_STRING, .flags = AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM },
    // { "dtype", "OpenCV data type", OFFSET(dtype_opt), AV_OPT_TYPE_STRING, {.str = "8u"}, .flags = AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM},
    { NULL }
};

AVFILTER_DEFINE_CLASS(tddfa);
static unsigned int indices_pic[] = {
        0, 1, 2,
        2, 3, 0
    };
static void init_egl(int device_index=0) {
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

static void config_opengl(TddfaContext *s, size_t vertex_num, const char* index_path, int W=1280, int H=720, cudaStream_t stream=0){
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

    // Shader shader_pic(tddfaVertexShader_pic, tddfaFragmentShader_pic);
    s->shader_pic = new Shader(tddfaVertexShader_pic, tddfaFragmentShader_pic);
    glCheckError();

    unsigned int tex_image = -1;
    glGenTextures(1, &tex_image);
    glBindTexture(GL_TEXTURE_2D, tex_image);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W, H, 0,
        GL_RGBA, GL_UNSIGNED_BYTE, NULL);
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
    // unsigned int indices_pic[] = {
    //     0, 1, 2,
    //     2, 3, 0
    // };
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_pic);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices_pic), indices_pic, GL_STATIC_DRAW);

    unsigned tex_face = LoadTexture("color.1024x1024.jpg", false);
    // Shader shader_face(tddfaVertexShader_face, tddfaFragmentShader_face);
    s->shader_face = new Shader(tddfaVertexShader_face, tddfaFragmentShader_face);
    glCheckError();

    unsigned int VBO_face;
    glGenBuffers(1, &VBO_face);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_face);
    // cnpy::NpyArray face = cnpy::npy_load("vert_0.npy");
    glBufferData(GL_ARRAY_BUFFER, vertex_num * sizeof(float),
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
    cnpy::NpyArray index = npy_load(index_path);
    s->index_num = index.num_vals;
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
    s->tex_image = tex_image;
    s->tex_face = tex_face;
    s->VAO_face = VAO_face;
    s->VAO_pic = VAO_pic;
    s->PBO_fb = PBO_fb;
    CUcontext current;
    FF_CU_CK(cuCtxGetCurrent(&current));
    FF_CUDA_CK(cudaGraphicsGLRegisterBuffer(&s->cuda_mesh_resource, VBO_face, cudaGraphicsRegisterFlagsNone));
    FF_CUDA_CK(cudaGraphicsGLRegisterBuffer(&s->cuda_fb_resource, PBO_fb, cudaGraphicsRegisterFlagsNone));
    FF_CUDA_CK(cudaGraphicsGLRegisterImage(&s->cuda_texture_resource, tex_image, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
    FF_CUDA_CK(cudaGetLastError());
}

static int query_formats(AVFilterContext *ctx) {
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_CUDA,
        AV_PIX_FMT_NONE
    };
    AVFilterFormats *fmt_list = ff_make_format_list((const int*)pix_fmts);
    if (!fmt_list)
        return AVERROR(ENOMEM);
    return ff_set_common_formats(ctx, fmt_list);
}

static av_cold int init(AVFilterContext *ctx) {
    TddfaContext *s = (TddfaContext*)ctx->priv;
    if (!s->path_opt) {
        av_log(s, AV_LOG_INFO, "No model folder provided, using current path as default.\n");
    }

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (s->path_opt)) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir(dir)) != NULL) {
            printf ("%s\n", ent->d_name);
            std::string entry_string{ent->d_name};
            if (entry_string.find("FaceBoxesProd") != std::string::npos) {
                s->facebox_path = s->path_opt + entry_string;
                continue;
            }
            if (entry_string.find("mb1_120x120") != std::string::npos) {
                s->mb120_path = s->path_opt + entry_string;
                continue;
            }
            if (entry_string.find("bfm_noneck") != std::string::npos) {
                s->bfm_path = s->path_opt + entry_string;
                continue;
            }
        }
        closedir (dir);
    } else {
        /* could not open directory */
        av_log(s, AV_LOG_INFO, "Cannot open model folder.\n");
        return AVERROR(EINVAL);
    }

    init_egl();

    return 0;
}

static int config_props(AVFilterLink *outlink) {
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = ctx->inputs[0];
    TddfaContext *s = (TddfaContext*)ctx->priv;

    AVHWFramesContext *in_frame_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVBufferRef *hw_device_ref = in_frame_ctx->device_ref;
    AVCUDADeviceContext *hw_ctx = (AVCUDADeviceContext*)in_frame_ctx->device_ctx->hwctx;

    AVBufferRef *out_ref = NULL;
    AVHWFramesContext *out_ctx;
    CUcontext dummy;
    int ret;

    int w = inlink->w;
    int h = inlink->h;
    s->model_w = w;
    s->model_h = h;

    if (in_frame_ctx->sw_format != AV_PIX_FMT_NV12) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported input format: %s\n", 
                av_get_pix_fmt_name(in_frame_ctx->sw_format));
        return AVERROR(ENOSYS);
    }

    FF_CU_CK(cuCtxPushCurrent(hw_ctx->cuda_ctx));

    out_ref = av_hwframe_ctx_alloc(hw_device_ref);
    if (!out_ref)
        return AVERROR(ENOMEM);
    out_ctx = (AVHWFramesContext*)out_ref->data;

    out_ctx->format = AV_PIX_FMT_CUDA;
    out_ctx->sw_format = in_frame_ctx->sw_format;
    out_ctx->width = in_frame_ctx->width;
    out_ctx->height = in_frame_ctx->height;
    FF_CUDA_CK(cudaGetLastError());
    ret = av_hwframe_ctx_init(out_ref);
    if (ret < 0)
    {
        av_buffer_unref(&out_ref);
        return ret;
    }
    s->hw_frames_ctx = out_ref;
    outlink->hw_frames_ctx = av_buffer_ref(s->hw_frames_ctx);
    if (!outlink->hw_frames_ctx)
        return AVERROR(ENOMEM);
    FF_CU_CK(cuCtxGetCurrent(&dummy));
    std::cout << "Current context: " << dummy << ", ffmpeg context: " << hw_ctx->cuda_ctx << std::endl;

    s->face_box = new FaceBoxes_ONNX{s->facebox_path.c_str(), h, w};
    FF_CU_CK(cuCtxGetCurrent(&dummy));
    s->tddfa = new TDDFA_ONNX{s->mb120_path.c_str(), s->bfm_path.c_str()};
    FF_CU_CK(cuCtxGetCurrent(&dummy));

    cudaMalloc(&s->d_rgbpf32_img, w * h * 3 * sizeof(float));
    cudaMalloc(&s->d_bgra_img, w * h * 4);
    config_opengl(s, s->tddfa->vertex_num(), "rio_tri.npy", w, h, hw_ctx->stream);

    FF_CU_CK(cuCtxPopCurrent(&dummy));

    return 0;
}

void draw_on_image(TddfaContext *s, void* d_rgba_image, float* d_vertices, size_t vertices_num, AVFrame *out,
    const int W=1280, const int H=720, cudaStream_t stream=0){

    void *cuda_mapped_pointer;
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
    FF_CU_CK(cuCtxGetCurrent(&current));
    // std::cout << current << std::endl;
    FF_CUDA_CK(cudaGraphicsMapResources(1, &s->cuda_texture_resource, stream));
    FF_CUDA_CK(cudaGraphicsSubResourceGetMappedArray(&texture_array, s->cuda_texture_resource, 0, 0));
    FF_CUDA_CK(cudaMemcpy2DToArrayAsync(texture_array, 0, 0, d_rgba_image, s->model_w * 4,
        s->model_w * 4, s->model_h, cudaMemcpyDeviceToDevice, stream));
    FF_CUDA_CK(cudaGraphicsUnmapResources(1, &s->cuda_texture_resource, stream));

    FF_CUDA_CK(cudaGraphicsMapResources(1, &s->cuda_mesh_resource, stream));
    FF_CUDA_CK(cudaGraphicsResourceGetMappedPointer(&cuda_mapped_pointer, &cuda_mapped_size, s->cuda_mesh_resource));

    FF_CUDA_CK(cudaMemcpyAsync(cuda_mapped_pointer, d_vertices, cuda_mapped_size, cudaMemcpyDeviceToDevice, stream));

    FF_CUDA_CK(cudaGraphicsUnmapResources(1, &s->cuda_mesh_resource, stream));

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

        s->shader_pic->Use();
        glCheckError();
        glActiveTexture(GL_TEXTURE0);
        glCheckError();
        glBindTexture(GL_TEXTURE_2D, s->tex_image);
        glCheckError();
        s->shader_pic->SetUniform("tex0", 0);
        glBindVertexArray(s->VAO_pic);
        glCheckError();
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glDrawElements(GL_TRIANGLES, sizeof(indices_pic) / sizeof(indices_pic[0]), GL_UNSIGNED_INT, 0);
        glCheckError();

        s->shader_face->Use();
        s->shader_face->SetUniform("W", (float)W);
        s->shader_face->SetUniform("H", (float)H);
        s->shader_face->SetUniform("projection", projection);
        glCheckError();

        s->shader_face->Use();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, s->tex_face);
        s->shader_face->SetUniform("tex0", 0);
        glBindVertexArray(s->VAO_face);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        // glLineWidth(0.1);
        glDrawElements(GL_TRIANGLES, s->index_num, GL_UNSIGNED_INT, 0);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // glfwSwapBuffers(window);
        // glfwPollEvents();
    // }
    glCheckError();
    glBindBuffer(GL_PIXEL_PACK_BUFFER, s->PBO_fb);
    glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glFinish();

    FF_CUDA_CK(cudaGraphicsMapResources(1, &s->cuda_fb_resource, stream));
    FF_CUDA_CK(cudaGraphicsResourceGetMappedPointer(&cuda_mapped_pointer, &cuda_mapped_size, s->cuda_fb_resource));

    // FF_CUDA_CK(cudaMemcpyAsync(cuda_mapped_pointer, d_vertices, cuda_mapped_size, cudaMemcpyDeviceToDevice, stream));
    Color32ToNv12<RGBA32>(static_cast<uint8_t*>(cuda_mapped_pointer), W * 4, out->data[0], out->linesize[0], W, H, out->colorspace);

    FF_CUDA_CK(cudaGraphicsUnmapResources(1, &s->cuda_fb_resource, stream));
    // std::vector<std::uint8_t> buf(W * H * 3);
    // glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, buf.data());

    // DEBUG
    // #define STB_IMAGE_WRITE_IMPLEMENTATION
    // #define STB_IMAGE_STATIC
    // #include "pose/stb_image_write.h"
    // stbi_write_png("one_face_out.png", W, H, 3, buf.data(), W * 3);
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in) {
    AVFilterContext *ctx = inlink->dst;
    TddfaContext *s = (TddfaContext*)ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    AVHWFramesContext     *frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVCUDADeviceContext *hw_ctx = (AVCUDADeviceContext*)frames_ctx->device_ctx->hwctx;
    // CudaFunctions *cu = hw_ctx->internal->cuda_dl;
    CUstream stream = hw_ctx->stream;
    CUcontext dummy;
    int rgbp_pitch = s->model_w * sizeof(float);
    torch::Tensor face_dets;
    cudaArray_t image_array;
    cv::Mat bgr_image(s->model_h, s->model_w, CV_8UC4);
    vector<torch::Tensor> param_vec, roi_box_vec;
    uint8_t* dp_rgbpf32_data[3];
    // float shift[3] = {104, 117, 123};
    float shift[3] = {123, 117, 104};

    // float* hp_rgbpf32_data = new float[s->model_w * s->model_h * 3];
    // const AVPixFmtDescriptor *in_desc;
    // const AVPixFmtDescriptor *out_desc;
    // CvtColor* cvt_color_class = s->cvt_color_class;
    // volatile AVFrame* in_frame = in;
    // int in_frame_copy_w = in_frame->width * 3;

    int ret;
    AVFrame *out = av_frame_alloc();
    if (!out)
    {
        ret = AVERROR(ENOMEM);
        goto fail;
        return ret;
    }

    FF_CU_CK(cuCtxPushCurrent(hw_ctx->cuda_ctx));
    ret = av_hwframe_get_buffer(s->hw_frames_ctx, out, 0);
    if (ret < 0)
        goto fail;
    FF_CUDA_CK(cudaGetLastError());
    dp_rgbpf32_data[0] = static_cast<uint8_t*>(s->d_rgbpf32_img);
    dp_rgbpf32_data[1] = static_cast<uint8_t*>(s->d_rgbpf32_img) + rgbp_pitch * s->model_h;
    dp_rgbpf32_data[2] = static_cast<uint8_t*>(s->d_rgbpf32_img) + rgbp_pitch * s->model_h * 2;
    nv12_to_bgrpf32_shift(stream, in->data, in->linesize, dp_rgbpf32_data, &rgbp_pitch,
                    s->model_w, s->model_h, 1.0f, shift, in->colorspace);
    FF_CUDA_CK(cudaGetLastError());

    // DEBUG
    // ck(cudaMemcpy(hp_rgbpf32_data, dp_rgbpf32_data[0], rgbp_pitch * s->model_h * 3, cudaMemcpyDeviceToHost));
    // std::cout << "rgbpf32: \n";
    // for (int i = 0; i < 100; i++){
    //     std::cout << hp_rgbpf32_data[i] << "    ";
    // }
    // std::cout << std::endl;

    face_dets = s->face_box->forward<float>((float*)s->d_rgbpf32_img, s->model_h, s->model_w);
    CUcontext current;
    FF_CU_CK(cuCtxGetCurrent(&current));

    Nv12ToColor32<RGBA32>(in->data[0], in->linesize[0], (uint8_t*)s->d_bgra_img, s->model_w * 4, s->model_w, s->model_h, in->colorspace);
    FF_CUDA_CK(cudaMemcpyAsync(bgr_image.data, s->d_bgra_img, s->model_h * s->model_w * 4, cudaMemcpyDeviceToHost, stream));
    cv::cvtColor(bgr_image, bgr_image, cv::COLOR_RGBA2BGR);
    // cv::imwrite("bgr_from_nv12.jpg", bgr_image);
    s->tddfa->forward(bgr_image, face_dets, param_vec, roi_box_vec, stream);
    FF_CU_CK(cuCtxGetCurrent(&current));

    FF_CUDA_CK(cudaGetLastError());

    draw_on_image(s, s->d_bgra_img, s->tddfa->vertex_data(), s->tddfa->vertex_num(), out,
        s->model_w, s->model_h, stream);
    FF_CU_CK(cuCtxPopCurrent(&dummy));

    ret = av_frame_copy_props(out, in);
    if (ret < 0)
        goto fail;

    av_frame_free(&in);
    return ff_filter_frame(outlink, out);
fail:
    av_frame_free(&in);
    av_frame_free(&out);
    return ret;
}

static av_cold void uninit(AVFilterContext *ctx) {
    TddfaContext *s = (TddfaContext*)ctx->priv;

    delete s->shader_pic;
    delete s->shader_face;
    delete s->face_box;
    delete s->tddfa;
}

static const AVFilterPad tddfa_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
    },
    { NULL }
};

static const AVFilterPad tddfa_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_props,
    },
    { NULL }
};

AVFilter ff_vf_tddfa = {
    .name          = "3ddfa",
    .description   = NULL_IF_CONFIG_SMALL("Generate and render poses using the 3DDFA v2 model."),
    .inputs        = tddfa_inputs,
    .outputs       = tddfa_outputs,
    .priv_class    = &tddfa_class,
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .priv_size     = sizeof(TddfaContext),
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};

} // extern "C"