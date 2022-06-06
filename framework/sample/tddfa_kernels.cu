#include "tddfa_kernels.h"
// #include "common_utils.h"

#include <cub/cub.cuh>
#include <iostream>
#include <torch/torch.h>
#include <format_cuda.h>

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


// #define CUDA_1D_KERNEL_LOOP(i, n)                                \
// for (int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < (n); \
//         i += (blockDim.x * gridDim.x))

// template <typename integer>
// constexpr __host__ __device__ inline integer ceil_div(integer n, integer m) {
// return (n + m - 1) / m;
// }
// int const threadsPerBlock = sizeof(unsigned long long) * 8;

// template <typename T>
// __device__ inline bool devIoU(
//     T const* const a,
//     T const* const b,
//     const float threshold) {
//   T left = max(a[0], b[0]), right = min(a[2], b[2]);
//   T top = max(a[1], b[1]), bottom = min(a[3], b[3]);
//   T width = max(right - left, (T)0), height = max(bottom - top, (T)0);
//   using acc_T = at::acc_type<T, /*is_cuda=*/true>;
//   acc_T interS = (acc_T)width * height;
//   acc_T Sa = ((acc_T)a[2] - a[0]) * (a[3] - a[1]);
//   acc_T Sb = ((acc_T)b[2] - b[0]) * (b[3] - b[1]);
//   return (interS / (Sa + Sb - interS)) > threshold;
// }

// template <typename T>
// __global__ void nms_kernel_impl(
//     int n_boxes,
//     double iou_threshold,
//     const T* dev_boxes,
//     unsigned long long* dev_mask) {
//   const int row_start = blockIdx.y;
//   const int col_start = blockIdx.x;

//   if (row_start > col_start)
//     return;

//   const int row_size =
//       min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
//   const int col_size =
//       min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

//   __shared__ T block_boxes[threadsPerBlock * 4];
//   if (threadIdx.x < col_size) {
//     block_boxes[threadIdx.x * 4 + 0] =
//         dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 0];
//     block_boxes[threadIdx.x * 4 + 1] =
//         dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 1];
//     block_boxes[threadIdx.x * 4 + 2] =
//         dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 2];
//     block_boxes[threadIdx.x * 4 + 3] =
//         dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 3];
//   }
//   __syncthreads();

//   if (threadIdx.x < row_size) {
//     const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
//     const T* cur_box = dev_boxes + cur_box_idx * 4;
//     int i = 0;
//     unsigned long long t = 0;
//     int start = 0;
//     if (row_start == col_start) {
//       start = threadIdx.x + 1;
//     }
//     for (i = start; i < col_size; i++) {
//       if (devIoU<T>(cur_box, block_boxes + i * 4, iou_threshold)) {
//         t |= 1ULL << i;
//       }
//     }
//     const int col_blocks = ceil_div(n_boxes, threadsPerBlock);
//     dev_mask[cur_box_idx * col_blocks + col_start] = t;
//   }
// }

// at::Tensor nms_kernel_gpu(
//     const at::Tensor& dets,
//     const at::Tensor& scores,
//     double iou_threshold) {
//   TORCH_CHECK(dets.is_cuda(), "dets must be a CUDA tensor");
//   TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");

//   TORCH_CHECK(
//       dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
//   TORCH_CHECK(
//       dets.size(1) == 4,
//       "boxes should have 4 elements in dimension 1, got ",
//       dets.size(1));
//   TORCH_CHECK(
//       scores.dim() == 1,
//       "scores should be a 1d tensor, got ",
//       scores.dim(),
//       "D");
//   TORCH_CHECK(
//       dets.size(0) == scores.size(0),
//       "boxes and scores should have same number of elements in ",
//       "dimension 0, got ",
//       dets.size(0),
//       " and ",
//       scores.size(0))

//   at::cuda::CUDAGuard device_guard(dets.device());

//   if (dets.numel() == 0) {
//     return at::empty({0}, dets.options().dtype(at::kLong));
//   }

//   auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
//   auto dets_sorted = dets.index_select(0, order_t).contiguous();

//   int dets_num = dets.size(0);

//   const int col_blocks = ceil_div(dets_num, threadsPerBlock);

//   at::Tensor mask =
//       at::empty({dets_num * col_blocks}, dets.options().dtype(at::kLong));

//   dim3 blocks(col_blocks, col_blocks);
//   dim3 threads(threadsPerBlock);
//   cudaStream_t stream = at::cuda::getCurrentCUDAStream();

//   AT_DISPATCH_FLOATING_TYPES_AND_HALF(
//       dets_sorted.scalar_type(), "nms_kernel_gpu", [&] {
//         nms_kernel_impl<scalar_t><<<blocks, threads, 0, stream>>>(
//             dets_num,
//             iou_threshold,
//             dets_sorted.data_ptr<scalar_t>(),
//             (unsigned long long*)mask.data_ptr<int64_t>());
//       });

//   at::Tensor mask_cpu = mask.to(at::kCPU);
//   unsigned long long* mask_host =
//       (unsigned long long*)mask_cpu.data_ptr<int64_t>();

//   std::vector<unsigned long long> remv(col_blocks);
//   memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

//   at::Tensor keep =
//       at::empty({dets_num}, dets.options().dtype(at::kLong).device(at::kCPU));
//   int64_t* keep_out = keep.data_ptr<int64_t>();

//   int num_to_keep = 0;
//   for (int i = 0; i < dets_num; i++) {
//     int nblock = i / threadsPerBlock;
//     int inblock = i % threadsPerBlock;

//     if (!(remv[nblock] & (1ULL << inblock))) {
//       keep_out[num_to_keep++] = i;
//       unsigned long long* p = mask_host + i * col_blocks;
//       for (int j = nblock; j < col_blocks; j++) {
//         remv[j] |= p[j];
//       }
//     }
//   }

//   AT_CUDA_CHECK(cudaGetLastError());
//   return order_t.index(
//       {keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep)
//            .to(order_t.device(), keep.scalar_type())});
// }

// TORCH_LIBRARY_IMPL(torchvision, CUDA, m) {
//   m.impl(TORCH_SELECTIVE_NAME("torchvision::nms"), TORCH_FN(nms_kernel_gpu));
// }

// void cub_dry_run(void* d_temp_storage, size_t& temp_storage_bytes, size_t num_vertices){
//     float *in, *out;
//     // Allocate temp storage bytes for cub
//     cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes,
//         in, out, num_vertices);
// }

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

void similar_transform_transpose(float* d_vertices, float* d_vertices_out, uint32_t vertices_num, at::Tensor& roi_box,
    void* cub_temp_storage, size_t cub_temp_storage_bytes, float* cub_out,
    cudaStream_t stream, uint32_t size)
{
    dim3 grid_dim((vertices_num / 3 + 255) / 256);
    dim3 block_dim(256);

    float scale_x = (roi_box[2].item<float>() - roi_box[0].item<float>()) / size;
    float scale_y = (roi_box[3].item<float>() - roi_box[1].item<float>()) / size;

    // float* d_out;
    // cudaMalloc(&d_out, vertices_num * sizeof(float));
    // *d_vertices_out = d_out;

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
    similar_transform_kernel_transpose<<<grid_dim, block_dim, 0, stream>>>(d_vertices, d_vertices_out, vertices_num/3,
        scale_x, scale_y, roi_box[0].item<float>(), roi_box[1].item<float>(), cub_out);
    cudaGetLastError();
    // DEBUG
    // cudaDeviceSynchronize();
    // float* h_vertices = new float[vertices_num];
    // cudaMemcpy(h_vertices, d_out, vertices_num * sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "Vertices: \n";
    // for (int i = 0; i < 100; i++){
    //     std::cout << h_vertices[i] << "  ";
    // }
    // std::cout << std::endl;
}

template<class Rgb>
__global__ static void cropWithRoiKernel(uint8_t* src, uint8_t* dst, int4 srcRoi, int4 dstRoi, size_t srcPitch, size_t dstPitch) {
    int tidX = threadIdx.x + blockDim.x * blockIdx.x;
    int tidY = threadIdx.y + blockDim.y * blockIdx.y;

    if (tidX >= srcRoi.z || tidY >= srcRoi.w)
        return;

    Rgb *pSrc = reinterpret_cast<Rgb*>(src + srcPitch * (tidY + srcRoi.y));
    Rgb *pDst = reinterpret_cast<Rgb*>(dst + dstPitch * (tidY + dstRoi.y));
    pDst[dstRoi.x + tidX] = pSrc[srcRoi.x + tidX];
}

template<class Rgb>
void cropWithRoi(uint8_t* d_src, uint8_t* d_dst, int4 srcRoi, int4 dstRoi, size_t srcPitch, size_t dstPitch, 
    size_t srcHeight, size_t dstHeight, cudaStream_t stream=0)
{
    if ((srcRoi.x + srcRoi.z) * sizeof(Rgb) > srcPitch
        || (srcRoi.y + srcRoi.w) > srcHeight)
    {
        std::cout << "ROI out of image boundary.\n";
        return;
    }

    if (srcRoi.z != dstRoi.z || srcRoi.w != dstRoi.w)
    {
        std::cout << "ROI must have same width and height.\n";
        return;
    }

    dim3 block_dim(32, 16);
    dim3 grid_dim((srcRoi.z - 1) / 32 + 1, (srcRoi.w - 1) / 16 + 1);

    cropWithRoiKernel<Rgb><<<grid_dim, block_dim, 0, stream>>>(d_src, d_dst, srcRoi, dstRoi, srcPitch, dstPitch);
    cudaGetLastError();
}

template
void cropWithRoi<ffgd::BGRA32>(uint8_t* d_src, uint8_t* d_dst, int4 srcRoi, int4 dstRoi, size_t srcPitch, size_t dstPitch, 
    size_t srcHeight, size_t dstHeight, cudaStream_t stream);