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

#include <map>
#include <stdint.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <chrono>
#include <sstream>


#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C"
{
#endif
    #include "tensorrt.h"

    #include "libavformat/avio.h"
    #include "avfilter.h"
    #include "libavutil/error.h"
    #include "libavutil/macros.h"
    #include "libavutil/hwcontext.h"
    #include "libavfilter/internal.h"
    #include "libavutil/frame.h"
    #include "libavutil/mem.h"
    #include "libavutil/log.h"

    #include <npp.h>
#ifdef __cplusplus
}
#endif

using namespace nvinfer1;
using namespace std;

// ===========================TensorRT section==============================
// Self-defined CUDA check functions as cuda_check.h is not available for cpp due to void* function pointers
inline bool check(CUresult e, void *ctx, CudaFunctions* cu, int iLine, const char *szFile) {
    if (e != CUDA_SUCCESS) {
        const char* pStr;
        cu->cuGetErrorName(e, &pStr);
        av_log(ctx, AV_LOG_ERROR, "CUDA driver API error: %s, at line %d in file %s\n",
        pStr, iLine, szFile);
        return false;
    }
    return true;
}

inline bool check(cudaError_t e, void *ctx, int iLine, const char *szFile) {
    if (e != cudaSuccess) {
        av_log(ctx, AV_LOG_ERROR, "CUDA runtime API error: %s, at line %d in file %s\n",
            cudaGetErrorName(e), iLine, szFile);
        return false;
    }
    return true;
}

inline bool check(bool bSuccess, void *ctx, int iLine, const char *szFile) {
    if (!bSuccess) {
        av_log(ctx, AV_LOG_ERROR, "Error at line %d in file %s\n", iLine, szFile);
        return false;
    }
    return true;
}

#define ck(call, s) check(call, s, __LINE__, __FILE__)
#define ck_cu(call) check(call, s, cu, __LINE__, __FILE__)

inline std::string to_string(nvinfer1::Dims const &dim) {
    std::ostringstream oss;
    oss << "(";
    for (int i = 0; i < dim.nbDims; i++) {
        oss << dim.d[i] << ", ";
    }
    oss << ")";
    return oss.str();
}

struct IOInfo {
    string name;
    bool bInput;
    nvinfer1::Dims dim;
    nvinfer1::DataType dataType;

    string GetDimString() {
        return ::to_string(dim);
    }
    string GetDataTypeString() {
        static string aTypeName[] = {"float", "half", "int8", "int32", "bool"};
        return aTypeName[(int)dataType];
    }
    size_t GetNumBytes() {
        static int aSize[] = {4, 2, 1, 4, 1};
        size_t nSize = aSize[(int)dataType];
        for (int i = 0; i < dim.nbDims; i++) {
            nSize *= dim.d[i];
        }
        return nSize;
    }
    string to_string() {
        ostringstream oss;
        oss << setw(6) << (bInput ? "input" : "output") 
            << " | " << setw(5) << GetDataTypeString() 
            << " | " << GetDimString() 
            << " | " << "size=" << GetNumBytes()
            << " | " << name;
        return oss.str();
    }
};

struct BuildEngineParam {
    int nMaxBatchSize;
    int nChannel, nHeight, nWidth;
    std::size_t nMaxWorkspaceSize;
    bool bFp16, bInt8, bRefit;
};

class TrtLogger : public nvinfer1::ILogger {
public:
    TrtLogger(TensorrtContext *ctx) : ctx(ctx) {}
    void log(Severity severity, const char* msg) noexcept override {
        int log_level = AV_LOG_INFO;
        switch (severity){
            case nvinfer1::ILogger::Severity::kERROR:
            log_level = AV_LOG_ERROR;
            break;
            case nvinfer1::ILogger::Severity::kWARNING:
            log_level = AV_LOG_WARNING;
            break;
            case nvinfer1::ILogger::Severity::kINFO:
            log_level = AV_LOG_INFO;
            break;
            case nvinfer1::ILogger::Severity::kVERBOSE:
            log_level = AV_LOG_DEBUG;
            break;
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            log_level = AV_LOG_FATAL;
            break;
        }
        av_log(ctx, log_level, "%s\n", msg);
    }
private:
    TensorrtContext *ctx = nullptr;
};
    
class TrtLite {
public:
    TrtLite(const char *szEnginePath, TensorrtContext *ctx): ctx(ctx) {
        uint8_t *pBuf = nullptr;
        size_t nSize = 0;

        trt_logger = new TrtLogger(ctx);
        
        read_file_cpp(&pBuf, &nSize, szEnginePath);
        IRuntime *runtime = createInferRuntime(*trt_logger);
        engine = runtime->deserializeCudaEngine(pBuf, nSize);
        runtime->destroy();
        if (!engine) {
            av_log(ctx, AV_LOG_ERROR, "No engine generated\n");
            return;
        }
        delete[] pBuf;
    }
    TrtLite(const char *szOnnxPath, TensorrtContext *ctx, int nProfile, void *pData, bool serialize=true): ctx(ctx) {
        uint8_t *pBuf = nullptr;
        size_t nSize = 0;
        BuildEngineParam *pParam = reinterpret_cast<BuildEngineParam*>(pData);
        trt_logger = new TrtLogger(ctx);

        IBuilder *builder = createInferBuilder(*trt_logger);
        INetworkDefinition *network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, *trt_logger);
        read_file(&pBuf, &nSize, szOnnxPath);
        if (!parser->parse(pBuf, nSize)) {
            return;
        }

        for (int i = 0; i < network->getNbInputs(); i++) {
            ITensor *tensor = network->getInput(i);
            cout << "#" << i << ": " << IOInfo{string(tensor->getName()), true,
                tensor->getDimensions(), tensor->getType()}.to_string() << endl;
        }

        IBuilderConfig *config = builder->createBuilderConfig();
        config->setMaxWorkspaceSize(pParam->nMaxWorkspaceSize);
        if (pParam->bFp16) {
            config->setFlag(BuilderFlag::kFP16);
        }
        engine = builder->buildEngineWithConfig(*network, *config);
        if (serialize)
        {
            string engineCacheName = szOnnxPath;
            engineCacheName += ".trtcache";
            IHostMemory *serializedModel = engine->serialize();
            av_log(ctx, AV_LOG_INFO, "Write engine cache for ONNX model\n");
            write_file(serializedModel->data(), serializedModel->size(), engineCacheName.c_str());
        }
        config->destroy();
        network->destroy();
        builder->destroy();

        if (!engine) {
            av_log(ctx, AV_LOG_ERROR, "No engine created\n");
            return;
        }
        delete[] pBuf;
    }
    virtual ~TrtLite() {
        if (context) {
            context->destroy();
        }
        if (engine) {
            engine->destroy();
        }
        // if (ctx) {
        //     delete ctx;
        // }
    }
    ICudaEngine *GetEngine() {
        return engine;
    }
    void Execute(int nBatch, void* data[], cudaStream_t stm = 0, cudaEvent_t* evtInputConsumed = nullptr) {
        if (!engine) {
            av_log(ctx, AV_LOG_ERROR, "No engine\n");
            return;
        }
        if (!engine->hasImplicitBatchDimension() && nBatch > 1) {
            av_log(ctx, AV_LOG_ERROR, 
                "Engine was built with explicit batch but is executed with batch size != 1. Results may be incorrect.\n");
            return;
        }
        if (engine->getNbBindings() != NUM_TRT_IO) {
            av_log(ctx, AV_LOG_ERROR, "Number of bindings conflicts with input and output\n");
            return;
        }
        if (!context) {
            context = engine->createExecutionContext();
            if (!context) {
                av_log(ctx, AV_LOG_ERROR, "createExecutionContext() failed\n");
                return;
            }
        }
        ck(context->enqueue(nBatch, data, stm, evtInputConsumed), ctx);
    }
    void Execute(map<int, Dims> i2shape, void* data[], cudaStream_t stm = 0, cudaEvent_t* evtInputConsumed = nullptr) {
        if (!engine) {
            av_log(ctx, AV_LOG_ERROR, "No engine\n");
            return;
        }
        if (engine->hasImplicitBatchDimension()) {
            av_log(ctx, AV_LOG_ERROR, "Engine was built with static-shaped input\n");
            return;
        }
        if (engine->getNbBindings() != NUM_TRT_IO) {
            av_log(ctx, AV_LOG_ERROR, "Number of bindings conflicts with input and output\n");
            return;
        }
        if (!context) {
            context = engine->createExecutionContext();
            if (!context) {
                av_log(ctx, AV_LOG_ERROR, "createExecutionContext() failed\n");
                return;
            }
        }
        for (auto &it : i2shape) {
            context->setBindingDimensions(it.first, it.second);
        }
        ck(context->enqueueV2(data, stm, evtInputConsumed), ctx);
    }

    vector<IOInfo> ConfigIO(int nBatchSize) {
        vector<IOInfo> vInfo;
        if (!engine) {
            av_log(ctx, AV_LOG_ERROR, "No engine\n");
            return vInfo;
        }
        if (!engine->hasImplicitBatchDimension()) {
            av_log(ctx, AV_LOG_ERROR, "Engine must be built with implicit batch size (and static shape)\n");
            return vInfo;
        }
        for (int i = 0; i < engine->getNbBindings(); i++) {
            vInfo.push_back({string(engine->getBindingName(i)), engine->bindingIsInput(i), 
                MakeDim(nBatchSize, engine->getBindingDimensions(i)), engine->getBindingDataType(i)});
        }
        return vInfo;
    }
    vector<IOInfo> ConfigIO(map<int, Dims> i2shape) {
        vector<IOInfo> vInfo;
        if (!engine) {
            av_log(ctx, AV_LOG_ERROR, "No engine\n");
            return vInfo;
        }
        if (engine->hasImplicitBatchDimension()) {
            av_log(ctx, AV_LOG_ERROR, "Engine must be built with explicit batch size (to enable dynamic shape)\n");
            return vInfo;
        }
        if (!context) {
            context = engine->createExecutionContext();
            if (!context) {
                av_log(ctx, AV_LOG_ERROR, "createExecutionContext() failed\n");
                return vInfo;
            }
        }
        for (auto &it : i2shape) {
            context->setBindingDimensions(it.first, it.second);
        }
        if (!context->allInputDimensionsSpecified()) {
            av_log(ctx, AV_LOG_ERROR, "Not all binding shape are specified\n");
            return vInfo;
        }
        for (int i = 0; i < engine->getNbBindings(); i++) {
            vInfo.push_back({string(engine->getBindingName(i)), engine->bindingIsInput(i), 
                context->getBindingDimensions(i), engine->getBindingDataType(i)});
        }
        return vInfo;
    }

    void PrintInfo() {
        if (!engine) {
            av_log(ctx, AV_LOG_ERROR, "No engine\n");
            return;
        }
        av_log(ctx, AV_LOG_INFO, "nbBindings: %d\n", engine->getNbBindings());
        // Only contains engine-level IO information: if dynamic shape is used,
        // dimension -1 will be printed
        for (int i = 0; i < engine->getNbBindings(); i++) {
            av_log(ctx, AV_LOG_INFO, "#%d: %s\n", i, IOInfo{string(engine->getBindingName(i)), engine->bindingIsInput(i),
                engine->getBindingDimensions(i), engine->getBindingDataType(i)}.to_string().c_str());
        }
    }
    
private:
    int read_file_cpp(uint8_t **file_buf, size_t *file_size, const char *file_filename) {
        streampos size;
        *file_buf = nullptr;

        ifstream engine_file(file_filename, ios::in|ios::binary|ios::ate);
        if (engine_file.is_open()) {
            size = engine_file.tellg();
            char* buffer = new char[size];
            if (!buffer){
                av_log(ctx, AV_LOG_ERROR, "Error allocating memory for TRT engine.\n");
                return -2;
            }
            engine_file.seekg(0, ios::beg);
            engine_file.read(buffer, size);
            engine_file.close();

            *file_buf = reinterpret_cast<uint8_t*>(buffer);
            *file_size = static_cast<size_t>(size);

            return 0;
        }

        av_log(ctx, AV_LOG_ERROR, "Error reading engine file from disk!\n");
        return -2;
    }
    int read_file(uint8_t **file_buf, size_t *file_size, const char *file_filename) {
        AVIOContext *engine_file_ctx;
        *file_buf = nullptr;

        if (avio_open(&engine_file_ctx, file_filename, AVIO_FLAG_READ) < 0){
            av_log(ctx, AV_LOG_ERROR, "Error reading engine file from disk!\n");
            return -1;
        }

        size_t size = avio_size(engine_file_ctx);
        uint8_t *buffer = (uint8_t*)av_malloc(size);
        if (!buffer){
            avio_closep(&engine_file_ctx);
            av_log(ctx, AV_LOG_ERROR, "Error allocating memory for TRT engine.\n");
            return -2;
        }
        size_t bytes_read = avio_read(engine_file_ctx, buffer, size);
        avio_closep(&engine_file_ctx);
        if (bytes_read != size){
            av_freep(&buffer);
            av_log(ctx, AV_LOG_ERROR, "File size (%lu) does not equal to read size (%lu)\n", size, bytes_read);
            return -3;
        }

        *file_buf = buffer;
        *file_size = size;

        return 0;
    }
    int write_file(void *file_buf, size_t file_size, const char *file_name) {
        AVIOContext *engine_file_ctx;

        if (avio_open(&engine_file_ctx, file_name, AVIO_FLAG_WRITE) < 0){
            av_log(ctx, AV_LOG_ERROR, "Error reading engine file from disk!\n");
            return -1;
        }

        avio_write(engine_file_ctx, reinterpret_cast<uint8_t*>(file_buf), file_size);
        avio_flush(engine_file_ctx);
        avio_closep(&engine_file_ctx);

        return 0;
    }
    static size_t GetBytesOfBinding(int iBinding, ICudaEngine *engine, IExecutionContext *context = nullptr) {
        size_t aValueSize[] = {4, 2, 1, 4, 1};
        size_t nSize = aValueSize[(int)engine->getBindingDataType(iBinding)];
        const Dims &dims = context ? context->getBindingDimensions(iBinding) : engine->getBindingDimensions(iBinding);
        for (int i = 0; i < dims.nbDims; i++) {
            nSize *= dims.d[i];
        }
        return nSize;
    }
    static nvinfer1::Dims MakeDim(int nBatchSize, nvinfer1::Dims dim) {
        nvinfer1::Dims ret(dim);
        for (int i = ret.nbDims; i > 0; i--) {
            ret.d[i] = ret.d[i - 1];
        }
        ret.d[0] = nBatchSize;
        ret.nbDims++;
        return ret;
    }

    ICudaEngine *engine = nullptr;
    IExecutionContext *context = nullptr;
    TrtLogger *trt_logger = nullptr;
    TensorrtContext *ctx = nullptr;
    // vector<void*> device_buffer;
};
// ==========================End of TensorRT section===========================


#ifdef __cplusplus
extern "C"
{
#endif

void init_trt(TensorrtContext *s)
{
    TrtLite *trt_model = nullptr;
    string filename = s->engine_filename;
    string cache_name = filename + ".trtcache";
    s->cached = -1;
    // Only batch_size=1 is supported at this moment
    s->batch_size = 1;

    if (filename.find(".onnx") == filename.size() - 5 || filename.find(".ONNX") == filename.size() - 5)
    {
        av_log(s, AV_LOG_INFO, "Input is ONNX model, looking for engine cache\n");
        s->is_onnx = 1;
        AVIOContext *engine_file_ctx;
        s->cached = avio_open(&engine_file_ctx, cache_name.c_str(), AVIO_FLAG_READ);
        // cout << "Cached: " << s->cached << endl;
        avio_closep(&engine_file_ctx);
        if (s->cached == AVERROR(ENOENT))
        {
            av_log(s, AV_LOG_INFO, "No cached engine found\n");
            return;
        }
        else if (s->cached >= 0)
        {
            av_log(s, AV_LOG_INFO, "Engine cache found\n");
            s->engine_filename = (char*)av_malloc(cache_name.size() + 1);
            snprintf(s->engine_filename, cache_name.size() + 1, "%s", cache_name.c_str());
            printf("Engine filename: %s\n", s->engine_filename);
        }
        else
        {
            av_log(s, AV_LOG_ERROR, "Cannot open engine/model file\n");
            return;
        }
    }
}

int config_props_trt(TensorrtContext *s, AVFilterLink *inlink, CudaFunctions *cu)
{
    vector<IOInfo> io_infos;
    map<int, Dims> i2shape;

    BuildEngineParam param = {1, s->channels, inlink->h, inlink->w};

    TrtLite *trt_model = reinterpret_cast<TrtLite*>(s->trt_model);
    if (s->is_onnx == 1 && s->cached == AVERROR(ENOENT))
    {
        av_log(s, AV_LOG_INFO, "Generate trt engine from the ONNX model\n");
        trt_model = new TrtLite(s->engine_filename, s, 1, &param);
        s->trt_model = trt_model;
        s->dynamic_shape = trt_model->GetEngine()->hasImplicitBatchDimension() ? 0 : 1;
    }
    else
    {
        av_log(s, AV_LOG_INFO, "Load trt engine\n");
        trt_model = new TrtLite(s->engine_filename, s);
        s->trt_model = trt_model;
        s->dynamic_shape = trt_model->GetEngine()->hasImplicitBatchDimension() ? 0 : 1;
    }
    
    if (s->dynamic_shape)
    {
        i2shape.insert(make_pair(0, Dims{4, {s->batch_size, s->channels, inlink->h, inlink->w}}));
        io_infos = trt_model->ConfigIO(i2shape);
    }
    else
        io_infos = trt_model->ConfigIO(s->batch_size);

    if (io_infos.size() != 2)
    {
        av_log(s, AV_LOG_ERROR, "TRT model must have only one input and one output\n");
        return AVERROR(EIO);
    }
    for (int i = 0; i < io_infos.size(); i++)
    {
        if (io_infos[i].bInput)
        {
            s->in_h = io_infos[i].dim.d[2];
            s->in_w = io_infos[i].dim.d[3];
            s->trt_in_index = i;
        }
        else
        {
            s->out_h = io_infos[i].dim.d[2];
            s->out_w = io_infos[i].dim.d[3];
            s->trt_out_index = i;
        }
    }

    if (s->in_h != inlink->h || s->in_w != inlink->w)
    {
        av_log(s, AV_LOG_ERROR, 
            "Frame resolution (%dx%d) does not match model input size(%dx%d).\n", 
            inlink->w, inlink->h, s->in_w, s->in_h);
        
        return AVERROR(EINVAL);
    }

    return 0;
}

void copy_UV_plane(AVFrame *in, AVFrame *out, AVFilterContext *ctx)
{
    TensorrtContext *s = (TensorrtContext*)ctx->priv;
    AVHWDeviceContext *hw_device = (AVHWDeviceContext*)ctx->hw_device_ctx->data;
    AVCUDADeviceContext *hw_ctx = (AVCUDADeviceContext*)hw_device->hwctx;
    CudaFunctions *cu = hw_ctx->internal->cuda_dl;
    AVHWFramesContext *frame_ctx = (AVHWFramesContext*)in->hw_frames_ctx->data;
    int format = frame_ctx->sw_format;

    CUDA_MEMCPY2D copy_param;
    memset(&copy_param, 0, sizeof(copy_param));
    copy_param.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy_param.dstDevice = *(CUdeviceptr*)&out->data[1];
    copy_param.dstPitch = out->linesize[1];
    copy_param.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy_param.srcDevice = *(CUdeviceptr*)&in->data[1];
    copy_param.srcPitch = in->linesize[1];
    copy_param.Height = in->height;

    if (format == AV_PIX_FMT_NV12)
        ck_cu(cu->cuMemcpy2DAsync(&copy_param, hw_ctx->stream));
}

int filter_frame_trt(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    TensorrtContext *s = reinterpret_cast<TensorrtContext*>(ctx->priv);
    AVHWDeviceContext *hw_device = (AVHWDeviceContext*)ctx->hw_device_ctx->data;
    AVCUDADeviceContext *hw_ctx = (AVCUDADeviceContext*)hw_device->hwctx;
    CudaFunctions *cu = hw_ctx->internal->cuda_dl;

    AVFrame *out = NULL;
    TrtLite *trt_model = reinterpret_cast<TrtLite*>(s->trt_model);

    int ret;
    out = av_frame_alloc();
    if (!out)
    {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    ret = av_hwframe_get_buffer(outlink->hw_frames_ctx, out, 0);
    if (ret < 0)
        goto fail;

    s->trt_io[s->trt_in_index] = in->data[0];
    s->trt_io[s->trt_out_index] = out->data[0];

    if (s->dynamic_shape)
    {
        map<int, Dims> i2shape;
        i2shape.insert(make_pair(0, Dims{4, {s->batch_size, s->channels, s->in_h, s->in_w}}));
        trt_model->Execute(i2shape, s->trt_io, hw_ctx->stream);
    }
    else
        trt_model->Execute(s->batch_size, s->trt_io, hw_ctx->stream);

    if (s->channels == 1)
        copy_UV_plane(in, out, ctx);

    ret = av_frame_copy_props(out, in);
    if (ret < 0)
        return ret;
    av_frame_free(&in);
    return ff_filter_frame(outlink, out);

fail:
    av_frame_free(&in);
    av_frame_free(&out);
    return ret;
}

void free_trt(TensorrtContext *s)
{
    if (s->cached >= 0) av_free(s->engine_filename);
    if (s->trt_model)
    {
        delete static_cast<TrtLite*>(s->trt_model);
        s->trt_model = NULL;
    }
}

#ifdef __cplusplus
}
#endif