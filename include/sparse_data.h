#pragma once

#include <cuda_runtime_api.h>
#include <vector>
#include "utils.cuh"
struct d_csr
{
    /*device data */
    int *d_csr_rowPtr = nullptr;
    int *d_csr_colIdx = nullptr;
    float *d_csr_val = nullptr;
    d_csr(std::vector<int> rowPtr, std::vector<int> colIdx, std::vector<float> val)
    {
        gpuErrchk(cudaMalloc((void **)&d_csr_rowPtr, rowPtr.size() * sizeof(int)));
        gpuErrchk(cudaMalloc((void **)&d_csr_colIdx, colIdx.size() * sizeof(int)));
        gpuErrchk(cudaMalloc((void **)&d_csr_val, val.size() * sizeof(float)));
        gpuErrchk(cudaMemcpy(d_csr_rowPtr, rowPtr.data(), rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_csr_colIdx, colIdx.data(), colIdx.size() * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_csr_val, val.data(), val.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    ~d_csr()
    {
        gpuErrchk(cudaFree(d_csr_rowPtr));
        gpuErrchk(cudaFree(d_csr_colIdx));
        gpuErrchk(cudaFree(d_csr_val));
    }
};
struct d_csc_zip
{
    int *d_csc_colPtr = nullptr;
    int *d_csc_rowIdx = nullptr;
    int *d_csc_colIdx_zip = nullptr;
    float *d_csc_val = nullptr;
    d_csc_zip(std::vector<int> colPtr, std::vector<int> rowIdx, std::vector<int> colIdx_zip, std::vector<float> val)
    {
        gpuErrchk(cudaMalloc((void **)&d_csc_colPtr, colPtr.size() * sizeof(int)));
        gpuErrchk(cudaMalloc((void **)&d_csc_rowIdx, rowIdx.size() * sizeof(int)));
        gpuErrchk(cudaMalloc((void **)&d_csc_colIdx_zip, colIdx_zip.size() * sizeof(int)));
        gpuErrchk(cudaMalloc((void **)&d_csc_val, val.size() * sizeof(float)));
        gpuErrchk(cudaMemcpy(d_csc_colPtr, colPtr.data(), colPtr.size() * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_csc_rowIdx, rowIdx.data(), rowIdx.size() * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_csc_colIdx_zip, colIdx_zip.data(), colIdx_zip.size() * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_csc_val, val.data(), val.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    ~d_csc_zip()
    {
        gpuErrchk(cudaFree(d_csc_colPtr));
        gpuErrchk(cudaFree(d_csc_rowIdx));
        gpuErrchk(cudaFree(d_csc_colIdx_zip));
        gpuErrchk(cudaFree(d_csc_val));
    }
};

struct d_csr_zip
{
    int *d_csr_rowPtr = nullptr;
    int *d_csr_colIdx = nullptr;
    int *d_csr_rowIdx_zip = nullptr;
    float *d_csr_val = nullptr;
    d_csr_zip(std::vector<int> rowPtr, std::vector<int> colIdx, std::vector<int> rowIdx_zip, std::vector<float> val)
    {
        gpuErrchk(cudaMalloc((void **)&d_csr_rowPtr, rowPtr.size() * sizeof(int)));
        gpuErrchk(cudaMalloc((void **)&d_csr_colIdx, colIdx.size() * sizeof(int)));
        gpuErrchk(cudaMalloc((void **)&d_csr_rowIdx_zip, rowIdx_zip.size() * sizeof(int)));
        gpuErrchk(cudaMalloc((void **)&d_csr_val, val.size() * sizeof(float)));
        gpuErrchk(cudaMemcpy(d_csr_rowPtr, rowPtr.data(), rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_csr_colIdx, colIdx.data(), colIdx.size() * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_csr_rowIdx_zip, rowIdx_zip.data(), rowIdx_zip.size() * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_csr_val, val.data(), val.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    ~d_csr_zip()
    {
        gpuErrchk(cudaFree(d_csr_rowPtr));
        gpuErrchk(cudaFree(d_csr_colIdx));
        gpuErrchk(cudaFree(d_csr_rowIdx_zip));
        gpuErrchk(cudaFree(d_csr_val));
    }
};