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

#include "3ddfa_kernels.h"
// #include "common_utils.h"

#include <cub/cub.cuh>
#include <iostream>
#include <torch/torch.h>

// using namespace std;

// #define DEBUG

void cub_dry_run(void* &cub_d_temp_storage, size_t &cub_temp_storage_bytes, size_t num_vertices){
    float* dummy_ptr = nullptr;
    cub::DeviceReduce::Min(cub_d_temp_storage, cub_temp_storage_bytes, dummy_ptr, dummy_ptr, num_vertices, 0);
    cudaMalloc(&cub_d_temp_storage, cub_temp_storage_bytes);
}

__global__ void decode_locations_kernel(float* loc, float* priors, int num_prior,
                                        std::pair<float, float>variance, float* out_boxes){
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= num_prior)
        return;

    float4 box4{};
    float4 loc4 = reinterpret_cast<float4*>(loc)[idx];
    float4 prior4 = reinterpret_cast<float4*>(priors)[idx];

    #ifdef DEBUG
    if (idx < 10)
        printf("loc4 %d: %f  %f  %f  %f\n", idx, loc4.x, loc4.y, loc4.z, loc4.w);
    if (idx < 10)
        printf("prior4 %d: %f  %f  %f  %f\n", idx, prior4.x, prior4.y, prior4.z, prior4.w);
    #endif

    box4.x = prior4.x + loc4.x * variance.first * prior4.z;
    box4.y = prior4.y + loc4.y * variance.first * prior4.w;
    box4.z = prior4.z * exp(loc4.z * variance.second);
    box4.w = prior4.w * exp(loc4.w * variance.second);

    box4.x -= box4.z / 2;
    box4.y -= box4.w / 2;
    box4.z += box4.x;
    box4.w += box4.y;

    #ifdef DEBUG
    if (idx < 10)
        printf("out_boxes %d: %f  %f  %f  %f\n", idx, box4.x, box4.y, box4.z, box4.w);
    #endif
    reinterpret_cast<float4*>(out_boxes)[idx] = box4;
}

void decode_locations(float* loc, float* priors, int num_prior, std::pair<float, float>variance, float* out_boxes,
    cudaStream_t stream){

    dim3 block_dim(256);
    dim3 grid_dim((num_prior + 255) / 256);

    decode_locations_kernel<<<grid_dim, block_dim, 0, stream>>>(loc, priors, num_prior, variance, out_boxes);
}

__global__ void similar_transform_kernel(float* vertices, uint32_t vertices_num_channel,
    float scale_x, float scale_y, float sx, float sy, float* min_z, uint32_t size=120){
    int vertex_index = blockIdx.x * blockDim.x + threadIdx.x;
    int xyz_index = blockIdx.y;

    if (vertex_index >= vertices_num_channel) return;

    float vertex = vertices[vertex_index + xyz_index * vertices_num_channel];

    switch (xyz_index){
        case 0:
        vertex -= 1;
        vertex = vertex * scale_x + sx;
        // DEBUG
        // if (vertex_index == 0) printf("Vertex[0].x = %f\n", vertex);
        break;

        case 1:
        vertex = size - vertex;
        vertex = vertex * scale_y + sy;
        // if (vertex_index == 0) printf("Vertex[0].y = %f\n", vertex);
        break;

        case 2:
        vertex -= 1;
        float s = (scale_x + scale_y) / 2;
        vertex *= s;
        vertex -= min_z[0];
    }

    vertices[vertex_index + xyz_index * vertices_num_channel] = vertex;
}

__global__ void similar_transform_kernel_transpose(float* vertices, float* vertices_out, uint32_t vertices_num_channel,
    float scale_x, float scale_y, float sx, float sy, float* min_z, uint32_t size=120){
    int vertex_index = blockIdx.x * blockDim.x + threadIdx.x;
    // int xyz_index = blockIdx.y;

    if (vertex_index >= vertices_num_channel) return;

    float3 xyz;
    float vertex;

    xyz.x = vertices[vertex_index + 0 * vertices_num_channel];
    xyz.x -= 1;
    xyz.x = xyz.x * scale_x + sx;

    xyz.y = vertices[vertex_index + 1 * vertices_num_channel];
    xyz.y = size - xyz.y;
    xyz.y = xyz.y * scale_y + sy;

    xyz.z = vertices[vertex_index + 2 * vertices_num_channel];
    xyz.z -= 1;
    float s = (scale_x + scale_y) / 2;
    xyz.z *= s;
    xyz.z -= min_z[0];

    reinterpret_cast<float3*>(vertices_out)[vertex_index] = xyz;

    // float vertex = vertices[vertex_index + xyz_index * vertices_num_channel];

    // switch (xyz_index){
    //     case 0:
    //     vertex -= 1;
    //     vertex = vertex * scale_x + sx;
    //     // DEBUG
    //     // if (vertex_index == 0) printf("Vertex[0].x = %f\n", vertex);
    //     break;

    //     case 1:
    //     vertex = size - vertex;
    //     vertex = vertex * scale_y + sy;
    //     // if (vertex_index == 0) printf("Vertex[0].y = %f\n", vertex);
    //     break;

    //     case 2:
    //     vertex -= 1;
    //     float s = (scale_x + scale_y) / 2;
    //     vertex *= s;
    //     vertex -= min_z[0];
    // }

    // vertices[vertex_index + xyz_index * vertices_num_channel] = vertex;
}

void similar_transform(float* d_vertices, uint32_t vertices_num, at::Tensor& roi_box,
    void* cub_temp_storage, size_t cub_temp_storage_bytes, float* cub_out,
    cudaStream_t stream, uint32_t size)
{
    dim3 grid_dim((vertices_num + 255) / 256, 3);
    dim3 block_dim(256);

    float scale_x = (roi_box[2].item<float>() - roi_box[0].item<float>()) / size;
    float scale_y = (roi_box[3].item<float>() - roi_box[1].item<float>()) / size;

    // cub::DeviceReduce::Min(cub_temp_storage, cub_temp_storage_bytes, d_vertices, cub_out, vertices_num, stream);
    // (cudaMalloc(&cub_temp_storage, cub_temp_storage_bytes));

    cub::DeviceReduce::Min(cub_temp_storage, cub_temp_storage_bytes, d_vertices, cub_out, vertices_num, stream);

    float* h_vertices = new float[vertices_num];
    // cudaMemcpy(h_vertices, d_vertices, vertices_num * sizeof(float), cudaMemcpyDeviceToHost);
    // using namespace torch::indexing;
    // auto tensor_options_cuda = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    // torch::Tensor pts3d_tensor = torch::from_blob(d_vertices, {3, 38365}, tensor_options_cuda);
    // std::cout << pts3d_tensor.index({0, Slice(0, 10)}) << std::endl;
    // for (int i = 0; i < 100; i++){
    //     std::cout << h_vertices[i] << "  ";
    // }
    // std::cout << roi_box << std::endl;
    // std::cout << "scale_x, scale_y: " << scale_x << ", " << scale_y << std::endl;
    similar_transform_kernel<<<grid_dim, block_dim, 0, stream>>>(d_vertices, vertices_num/3,
        scale_x, scale_y, roi_box[0].item<float>(), roi_box[1].item<float>(), cub_out);
    cudaGetLastError();
    // DEBUG
    // cudaDeviceSynchronize();
    // float* h_vertices = new float[vertices_num];
    // cudaMalloc(&h_vertices, vertices_num * sizeof(float));
    // cudaMemcpy(h_vertices, d_vertices, vertices_num * sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "Vertices: \n";
    // for (int i = 0; i < 100; i++){
    //     std::cout << h_vertices[i] << "  ";
    // }
    // std::cout << std::endl;
}

void similar_transform_transpose(float* d_vertices, float** d_vertices_out, uint32_t vertices_num, at::Tensor& roi_box,
    void* cub_temp_storage, size_t cub_temp_storage_bytes, float* cub_out,
    cudaStream_t stream, uint32_t size)
{
    dim3 grid_dim((vertices_num / 3 + 255) / 256);
    dim3 block_dim(256);

    float scale_x = (roi_box[2].item<float>() - roi_box[0].item<float>()) / size;
    float scale_y = (roi_box[3].item<float>() - roi_box[1].item<float>()) / size;

    float* d_out;
    cudaMalloc(&d_out, vertices_num * sizeof(float));
    *d_vertices_out = d_out;

    // cub::DeviceReduce::Min(cub_temp_storage, cub_temp_storage_bytes, d_vertices, cub_out, vertices_num, stream);
    // cudaMalloc(&cub_temp_storage, cub_temp_storage_bytes);

    cub::DeviceReduce::Min(cub_temp_storage, cub_temp_storage_bytes, d_vertices, cub_out, vertices_num, stream);

    // float* h_vertices = new float[vertices_num];
    // cudaMemcpy(h_vertices, d_vertices, vertices_num * sizeof(float), cudaMemcpyDeviceToHost);
    // using namespace torch::indexing;
    // auto tensor_options_cuda = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    // torch::Tensor pts3d_tensor = torch::from_blob(d_vertices, {3, 38365}, tensor_options_cuda);
    // std::cout << pts3d_tensor.index({0, Slice(0, 10)}) << std::endl;
    // for (int i = 0; i < 100; i++){
    //     std::cout << h_vertices[i] << "  ";
    // }
    // std::cout << roi_box << std::endl;
    // std::cout << "scale_x, scale_y: " << scale_x << ", " << scale_y << std::endl;
    similar_transform_kernel_transpose<<<grid_dim, block_dim, 0, stream>>>(d_vertices, d_out, vertices_num/3,
        scale_x, scale_y, roi_box[0].item<float>(), roi_box[1].item<float>(), cub_out);
    cudaGetLastError();

}

template<typename T>
__global__ void crop_and_copy_kernel(T *d_in, T *d_out, int in_linesize, int out_linesize,
    int sx, int sy, int dw, int dh, int dsx, int dsy) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dw || y >= dh) return;

    d_out[out_linesize * (y + dsy) + x + dsx] = d_in[in_linesize * (y + sy) + x + sx];
}

template<typename T>
void crop_and_copy(T *d_in, T *d_out, int in_width, int out_width,
    int sx, int sy, int dw, int dh, int dsx, int dsy, cudaStream_t stream) {
    dim3 grid_dim((dw * 3 - 1) / 16 + 1, (dh - 1) / 16 + 1);
    dim3 block_dim(16, 16);

    crop_and_copy_kernel<<<grid_dim, block_dim, 0, stream>>>(d_in, d_out, in_width * 3, out_width * 3,
                                                            sx, sy, dw, dh, dsx, dsy);
}