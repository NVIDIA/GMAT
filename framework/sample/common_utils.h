#ifndef UTILS_H_
#define UTILS_H_

#include <vector>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <iostream>
#include <cuda_runtime.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// #ifndef checkCudaErrors
// #define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
// #endif

// // These are the inline versions for all of the SDK helper functions
// inline void __checkCudaErrors(cudaError_t err, const char *file, const int line)
// {
//     if(cudaSuccess != err)
//     {
//         const char *errorStr = NULL;
//         errorStr = cudaGetErrorString(err);
//         fprintf(stderr,
//                 "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
//                 "line %i.\n",
//                 err, errorStr, file, line);
//         exit(1);
//     }
// }

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

template<typename T>
void reorder_to_chw(cv::Mat const &mat, std::vector<float> &out, T mean={0,0,0}, bool normalize=true) {
    assert(mat.channels() == 3);
    assert(out.size() == mat.channels()* mat.rows * mat.cols);

    // T mean = {104, 117, 123};
    for (int y = 0; y < mat.rows; ++y) {
        for (int x = 0; x < mat.cols; ++x) {
            T rgb = static_cast<T>(mat.at<cv::Vec3b>(y, x)) - mean;
            for (int c = 0; c < mat.channels(); ++c) {
                out[c * (mat.rows * mat.cols) + y * mat.cols + x] =
                    rgb[c];
                if (normalize)
                    out[c * (mat.rows * mat.cols) + y * mat.cols + x] /= 255.0f;
            }
        }
    }
}

template void reorder_to_chw<cv::Vec3i>(cv::Mat const &mat, std::vector<float> &out, cv::Vec3i mean={0,0,0}, bool normalize=true);
template void reorder_to_chw<cv::Vec3f>(cv::Mat const &mat, std::vector<float> &out, cv::Vec3f mean={0,0,0}, bool normalize=true);
template void reorder_to_chw<cv::Vec3i>(cv::Mat const &mat, std::vector<float> &out, cv::Vec3i mean={0,0,0}, float normalize=1.0f);
template void reorder_to_chw<cv::Vec3f>(cv::Mat const &mat, std::vector<float> &out, cv::Vec3f mean={0,0,0}, float normalize=1.0f);

auto reorder_to_chw(cv::Mat const &mat) {
    assert(mat.channels() == 3);
    std::vector<float> data(mat.cols * mat.rows * mat.channels());
    for (int y = 0; y < mat.rows; ++y) {
        for (int x = 0; x < mat.cols; ++x) {
            for (int c = 0; c < mat.channels(); ++c) {
                data[c * (mat.rows * mat.cols) + y * mat.cols + x] =
                    mat.at<cv::Vec3b>(y, x)[c];
            }
        }
    }
    return data;
}

#endif
