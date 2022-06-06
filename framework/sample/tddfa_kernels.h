#pragma once

#include <cuda_runtime.h>
#include <utility>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
// #include <torch/torch.h>

// #include <cuda.h>

// using namespace std;

void decode_locations(float* loc, float* priors, int num_prior, std::pair<float, float>variance, float* out_boxes,
    cudaStream_t stream=0);

// at::Tensor nms_kernel_gpu(const at::Tensor& dets, const at::Tensor& scores, double iou_threshold);

// void cub_dry_run(void* d_temp_storage, size_t temp_storage_bytes, size_t num_vertices);
void cub_dry_run(void* &cub_d_temp_storage, size_t &cub_temp_storage_bytes, size_t num_vertices);

void similar_transform(float* vertices, uint32_t vertices_num, at::Tensor& roi_box,
    void* cub_temp_storage, size_t cub_temp_storage_bytes, float* cub_out,
    cudaStream_t stream=0, uint32_t size=120);

void similar_transform_transpose(float* vertices, float* d_vertices_out, uint32_t vertices_num, at::Tensor& roi_box,
    void* cub_temp_storage, size_t cub_temp_storage_bytes, float* cub_out,
    cudaStream_t stream=0, uint32_t size=120);

template<class Rgb>
void cropWithRoi(uint8_t* d_src, uint8_t* d_dst, int4 srcRoi, int4 dstRoi, size_t srcPitch, size_t dstPitch, 
    size_t srcHeight, size_t dstHeight, cudaStream_t stream);