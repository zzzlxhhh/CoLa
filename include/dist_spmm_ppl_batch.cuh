#pragma once
#include <nvshmem.h>
#include <nvshmemx.h>
#include "sparse_data.h"


template <typename T>
std::vector<T> bin2vec(const std::string filename)
{
    // printf("filename: %s\n", filename.c_str());
    std::ifstream file(filename.c_str(), std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error opening file!" << filename << std::endl;
        exit(1);
    }

    file.seekg(0, std::ios::end);
    int fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<T> vec(fileSize / sizeof(T));
    file.read(reinterpret_cast<char *>(vec.data()), fileSize);
    file.close();

    return vec;
}

void nnz_split(std::vector<int> &rowIdx, std::vector<int> &rowPtr, std::vector<int> &colIdx,
               std::vector<int> &target, std::vector<int> &ptr, const int peid, bool local)
{
    if (rowIdx.size() == 0)
        return;
    int avg = colIdx.size() / rowIdx.size();
    int gran = (avg + 32 - 1) / 32 * 32;
    ptr.push_back(0);
    int lb, hb;
    for (int i = 0; i < rowIdx.size(); i++)
    {
        lb = rowPtr[i];
        hb = rowPtr[i + 1];
        for (; lb < hb; lb += gran)
        {
            target.push_back(rowIdx[i]);
            if (lb + gran > hb)
                ptr.push_back(hb);
            else
                ptr.push_back(lb + gran);
        }
    }
    if (local)
        printf("PE-%d, l_split_ratio:%.3f \n", peid, (float)target.size() / rowIdx.size());
    else
        printf("PE-%d, r_split_ratio:%.3f \n", peid, (float)target.size() / rowIdx.size());
    assert(target.size() == ptr.size() - 1);
}

// kernel for request coalescing, coalescing the embedding for later communication
__global__ void batching_data_ppl(
    const float *input_nvsh,
    float *ppl_send_nvsh,
    int *put_colIdx,
    const int dim,
    const int nodesPerPE,
    const int group_size)
{
    int blockId = blockIdx.x;
    int laneId = threadIdx.x;
    int col = put_colIdx[blockId] % nodesPerPE;
    int stages = (dim - laneId + 32 - 1) / 32;
    for (int i = 0; i < stages; i++)
        ppl_send_nvsh[blockId * dim + laneId + i * 32] = input_nvsh[col * dim + laneId + i * 32];
}


template <int CF, int blkWarpNum, bool is_residue>
__global__ void ppl0_batch_SpMM_kernel(
    float *output,
    const float *input, // NVSHMEM
    const int *csr_rowPtr_l,
    const int *csr_colIdx_l,
    const int *csr_rowIdx_zip_l,
    const float *csr_val_l,
    const int *csc_colPtr_r,
    const int *csc_rowIdx_r,
    const int *csc_colIdx_zip_r,
    const float *csc_val_r,
    const int local_row_lb,
    const int remote_col_lb,
    const int local_row_num,
    const int remote_col_num,
    const int dim,
    const int nodesPerPE,
    const int peid,
    const int T)
{
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    int g_warpId = blockIdx.x * blkWarpNum + warpId;
    int offset = blockIdx.y * CF * 32;
    extern __shared__ float tmp[];
    int CrowIdx, r_GPUid, lb, hb, cur_col;
    float res[CF] = {0.0};
    float tmpA = 0.0;
    int CFnum;
    if (g_warpId >= T)
        return;

    if (is_residue)
        CFnum = CEIL_DIV(dim - laneId - offset, 32);

    if (local_row_num <= g_warpId && g_warpId < T)
    {
        int idx;
        if ((peid & 2) == 0)
            idx = g_warpId - local_row_num;
        else
            idx = T - g_warpId - 1;
        idx += remote_col_lb;
        cur_col = csc_colIdx_zip_r[idx];
        r_GPUid = cur_col / nodesPerPE;
        cur_col = cur_col % nodesPerPE;

        nvshmemx_float_get_warp((float *)&tmp[warpId * dim],
                                &input[cur_col * dim], dim, r_GPUid);
        lb = csc_colPtr_r[idx];
        hb = csc_colPtr_r[idx + 1];
        // cache the embedding in registers
        if (is_residue)
        {
#pragma unroll
            for (int j = 0; j < CFnum; j++)
                res[j] = tmp[warpId * dim + j * 32 + laneId];

            for (; lb < hb; lb++)
            {
                CrowIdx = csc_rowIdx_r[lb];
                tmpA = csc_val_r[lb];
#pragma unroll
                for (int j = 0; j < CFnum; j++)
                    atomicAdd(&output[CrowIdx * dim + offset + j * 32 + laneId], res[j] * tmpA);
            }
        }
        else
        {
#pragma unroll
            for (int j = 0; j < CF; j++)
                // res[j] = comm_buffer_nv[idx * dim + j * 32 + laneId];
                res[j] = tmp[warpId * dim + j * 32 + laneId];

            for (; lb < hb; lb++)
            {
                CrowIdx = csc_rowIdx_r[lb];
                tmpA = csc_val_r[lb];
#pragma unroll
                for (int j = 0; j < CF; j++)
                    atomicAdd(&output[CrowIdx * dim + offset + j * 32 + laneId], res[j] * tmpA);
            }
        }
        return;
    }

    // ### local computation ###
    if (g_warpId < local_row_num)
    {
        lb = csr_rowPtr_l[g_warpId];
        hb = csr_rowPtr_l[g_warpId + 1];
        if (is_residue)
        {
            for (; lb < hb; lb++)
            {
                cur_col = csr_colIdx_l[lb];
                tmpA = csr_val_l[lb];
#pragma unroll
                for (int i = 0; i < CFnum; i++)
                    res[i] += input[cur_col * dim + offset + i * 32 + laneId] * tmpA;
            }
            CrowIdx = csr_rowIdx_zip_l[g_warpId];
#pragma unroll
            for (int i = 0; i < CFnum; i++)
                atomicAdd(&output[CrowIdx * dim + offset + i * 32 + laneId], res[i]);
        }
        else
        {
            CrowIdx = csr_rowIdx_zip_l[g_warpId];
            for (; lb < hb; lb++)
            {
                cur_col = csr_colIdx_l[lb];
                tmpA = csr_val_l[lb];
#pragma unroll
                for (int i = 0; i < CF; i++)
                    res[i] += input[cur_col * dim + offset + i * 32 + laneId] * tmpA;
            }
#pragma unroll
            for (int i = 0; i < CF; i++)
                atomicAdd(&output[CrowIdx * dim + offset + i * 32 + laneId], res[i]);
        }
        return;
    }
}

template <int CF, bool is_residue>
void ppl0_batch_SpMM(
    float *output,
    const float *input,      // NVSHMEM
    const int *csr_rowPtr_l, // local CSR
    const int *csr_colIdx_l,
    const int *csr_rowIdx_zip_l,
    const float *csr_val_l,
    const int *csc_colPtr_r, // remote CSC
    const int *csc_rowIdx_r,
    const int *csc_colIdx_zip_r,
    const float *csc_val_r,
    const int local_row_lb,
    const int remote_col_lb,
    const int local_row_num,
    const int remote_col_num,
    const int dim,
    const int nodesPerPE,
    const int peid,
    cudaStream_t stream)
{
    constexpr int blkWarpNum = 8;
    int T = local_row_num + remote_col_num;
    int dim_x, dim_y;
    const int shared_mem = blkWarpNum * 32 * CF * sizeof(float);
    dim_y = CEIL_DIV(dim, CF * 32);
    dim_x = CEIL_DIV(T, blkWarpNum);
    dim3 blockDim(blkWarpNum * 32, 1, 1);
    dim3 gridDim(dim_x, dim_y, 1);

    if (T != 0)
        ppl0_batch_SpMM_kernel<CF, blkWarpNum, is_residue><<<gridDim, blockDim, shared_mem, stream>>>(
            output, input,
            csr_rowPtr_l, csr_colIdx_l, csr_rowIdx_zip_l, csr_val_l,
            csc_colPtr_r, csc_rowIdx_r, csc_colIdx_zip_r, csc_val_r,
            local_row_lb, remote_col_lb, local_row_num, remote_col_num,
            dim, nodesPerPE, peid, T);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error @ppl0_batch_SpMM: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

template <int CF, int blkWarpNum, bool is_residue>
__global__ void ppl0_batch_buffer_SpMM_kernel(
    float *output,
    const float *input, // NVSHMEM
    const float *comm_buffer,
    const int *csr_rowPtr_l,
    const int *csr_colIdx_l,
    const int *csr_rowIdx_zip_l,
    const float *csr_val_l,
    const int *csc_colPtr_r,
    const int *csc_rowIdx_r,
    const int *csc_colIdx_zip_r,
    const float *csc_val_r,
    const int local_row_lb,
    const int remote_col_lb,
    const int local_row_num,
    const int remote_col_num,
    const int dim,
    const int nodesPerPE,
    const int peid,
    const int T)
{
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    int g_warpId = blockIdx.x * blkWarpNum + warpId;
    int offset = blockIdx.y * CF * 32;
    extern __shared__ float tmp[];
    int CrowIdx, r_GPUid, lb, hb, cur_col;
    float res[CF] = {0.0};
    float tmpA = 0.0;
    int CFnum;
    if (g_warpId >= T)
        return;

    if (is_residue)
        CFnum = CEIL_DIV(dim - laneId - offset, 32);

    if (local_row_num <= g_warpId && g_warpId < T)
    {
        int idx;
        if ((peid & 2) == 0)
            idx = g_warpId - local_row_num;
        else
            idx = T - g_warpId - 1;
        idx += remote_col_lb;
        cur_col = csc_colIdx_zip_r[idx];
        r_GPUid = cur_col / nodesPerPE;
        cur_col = cur_col % nodesPerPE;

        nvshmemx_float_get_warp((float *)&comm_buffer[idx * dim],
                                &input[cur_col * dim], dim, r_GPUid);
        lb = csc_colPtr_r[idx];
        hb = csc_colPtr_r[idx + 1];
        // cache the embedding in registers
        if (is_residue)
        {
#pragma unroll
            for (int j = 0; j < CFnum; j++)
                res[j] = comm_buffer[idx * dim + j * 32 + laneId];

            for (; lb < hb; lb++)
            {
                CrowIdx = csc_rowIdx_r[lb];
                tmpA = csc_val_r[lb];
#pragma unroll
                for (int j = 0; j < CFnum; j++)
                    atomicAdd(&output[CrowIdx * dim + offset + j * 32 + laneId], res[j] * tmpA);
            }
        }
        else
        {
#pragma unroll
            for (int j = 0; j < CF; j++)
                res[j] = comm_buffer[idx * dim + j * 32 + laneId];

            for (; lb < hb; lb++)
            {
                CrowIdx = csc_rowIdx_r[lb];
                tmpA = csc_val_r[lb];
#pragma unroll
                for (int j = 0; j < CF; j++)
                    atomicAdd(&output[CrowIdx * dim + offset + j * 32 + laneId], res[j] * tmpA);
            }
        }
        return;
    }

    // ### local computation ###
    if (g_warpId < local_row_num)
    {
        lb = csr_rowPtr_l[g_warpId];
        hb = csr_rowPtr_l[g_warpId + 1];
        if (is_residue)
        {
            for (; lb < hb; lb++)
            {
                cur_col = csr_colIdx_l[lb];
                tmpA = csr_val_l[lb];
#pragma unroll
                for (int i = 0; i < CFnum; i++)
                    res[i] += input[cur_col * dim + offset + i * 32 + laneId] * tmpA;
            }
            CrowIdx = csr_rowIdx_zip_l[g_warpId];
#pragma unroll
            for (int i = 0; i < CFnum; i++)
                atomicAdd(&output[CrowIdx * dim + offset + i * 32 + laneId], res[i]);
        }
        else
        {
            CrowIdx = csr_rowIdx_zip_l[g_warpId];
            for (; lb < hb; lb++)
            {
                cur_col = csr_colIdx_l[lb];
                tmpA = csr_val_l[lb];
#pragma unroll
                for (int i = 0; i < CF; i++)
                    res[i] += input[cur_col * dim + offset + i * 32 + laneId] * tmpA;
            }
#pragma unroll
            for (int i = 0; i < CF; i++)
                atomicAdd(&output[CrowIdx * dim + offset + i * 32 + laneId], res[i]);
        }
        return;
    }
}

template <int CF, bool is_residue>
void ppl0_batch_buffer_SpMM(
    float *output,
    const float *input, // NVSHMEM
    const float *comm_buffer,
    const int *csr_rowPtr_l, // local CSR
    const int *csr_colIdx_l,
    const int *csr_rowIdx_zip_l,
    const float *csr_val_l,
    const int *csc_colPtr_r, // remote CSC
    const int *csc_rowIdx_r,
    const int *csc_colIdx_zip_r,
    const float *csc_val_r,
    const int local_row_lb,
    const int remote_col_lb,
    const int local_row_num,
    const int remote_col_num,
    const int dim,
    const int nodesPerPE,
    const int peid,
    cudaStream_t stream)
{
    constexpr int blkWarpNum = 8;
    int T = local_row_num + remote_col_num;
    int dim_x, dim_y;
    const int shared_mem = blkWarpNum * 32 * CF * sizeof(float);
    dim_y = CEIL_DIV(dim, CF * 32);
    dim_x = CEIL_DIV(T, blkWarpNum);
    dim3 blockDim(blkWarpNum * 32, 1, 1);
    dim3 gridDim(dim_x, dim_y, 1);

    if (T != 0)
        ppl0_batch_buffer_SpMM_kernel<CF, blkWarpNum, is_residue><<<gridDim, blockDim, shared_mem, stream>>>(
            output, input, comm_buffer,
            csr_rowPtr_l, csr_colIdx_l, csr_rowIdx_zip_l, csr_val_l,
            csc_colPtr_r, csc_rowIdx_r, csc_colIdx_zip_r, csc_val_r,
            local_row_lb, remote_col_lb, local_row_num, remote_col_num,
            dim, nodesPerPE, peid, T);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error @ppl0_batch_SpMM: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

template <int CF, int blkWarpNum, bool is_residue>
__global__ void ppl0_batch_SpMM_seq_kernel(
    float *output,
    const float *input, // NVSHMEM
    const int *csr_rowPtr_l,
    const int *csr_colIdx_l,
    const int *csr_rowIdx_zip_l,
    const float *csr_val_l,
    const int *csc_colPtr_r,
    const int *csc_rowIdx_r,
    const int *csc_colIdx_zip_r,
    const float *csc_val_r,
    const int local_row_lb,
    const int remote_col_lb,
    const int local_row_num,
    const int remote_col_num,
    const int dim,
    const int nodesPerPE,
    const int peid,
    const int T)
{
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    int g_warpId = blockIdx.x * blkWarpNum + warpId;
    int offset = blockIdx.y * CF * 32;
    extern __shared__ float tmp[];
    int CrowIdx, r_GPUid, lb, hb, cur_col;
    float tmpA = 0.0;
    int CFnum;
    if (g_warpId >= T)
        return;

    if (is_residue)
        CFnum = CEIL_DIV(dim - laneId - offset, 32);

    if (g_warpId < remote_col_num)
    {
        float res[CF] = {0.0};
        int idx = g_warpId + remote_col_lb;
        cur_col = csc_colIdx_zip_r[idx];
        r_GPUid = cur_col / nodesPerPE;
        cur_col = cur_col % nodesPerPE;

        nvshmemx_float_get_warp((float *)&tmp[warpId * dim],
                                &input[cur_col * dim], dim, r_GPUid);
        // nvshmemx_float_get_warp((float *)&comm_buffer_nv[idx * dim],
        //                         &input[cur_col * dim], dim, r_GPUid);
        lb = csc_colPtr_r[idx];
        hb = csc_colPtr_r[idx + 1];
        // cache the embedding in registers
        if (is_residue)
        {
#pragma unroll
            for (int j = 0; j < CFnum; j++)
                // res[j] = comm_buffer_nv[idx * dim + j * 32 + laneId];
                res[j] = tmp[warpId * dim + j * 32 + laneId];

            for (; lb < hb; lb++)
            {
                CrowIdx = csc_rowIdx_r[lb];
                tmpA = csc_val_r[lb];
#pragma unroll
                for (int j = 0; j < CFnum; j++)
                    atomicAdd(&output[CrowIdx * dim + offset + j * 32 + laneId], res[j] * tmpA);
            }
        }
        else
        {
#pragma unroll
            for (int j = 0; j < CF; j++)
                // res[j] = comm_buffer_nv[idx * dim + j * 32 + laneId];
                res[j] = tmp[warpId * dim + j * 32 + laneId];

            for (; lb < hb; lb++)
            {
                CrowIdx = csc_rowIdx_r[lb];
                tmpA = csc_val_r[lb];
#pragma unroll
                for (int j = 0; j < CF; j++)
                    atomicAdd(&output[CrowIdx * dim + offset + j * 32 + laneId], res[j] * tmpA);
            }
        }
    }

    // ### local computation ###
    if (g_warpId < local_row_num)
    {
        float res[CF] = {0.0};
        lb = csr_rowPtr_l[g_warpId];
        hb = csr_rowPtr_l[g_warpId + 1];
        if (is_residue)
        {
            for (; lb < hb; lb++)
            {
                cur_col = csr_colIdx_l[lb];
                tmpA = csr_val_l[lb];
#pragma unroll
                for (int i = 0; i < CFnum; i++)
                    res[i] += input[cur_col * dim + offset + i * 32 + laneId] * tmpA;
            }
            CrowIdx = csr_rowIdx_zip_l[g_warpId];
#pragma unroll
            for (int i = 0; i < CFnum; i++)
                atomicAdd(&output[CrowIdx * dim + offset + i * 32 + laneId], res[i]);
        }
        else
        {
            CrowIdx = csr_rowIdx_zip_l[g_warpId];
            for (; lb < hb; lb++)
            {
                cur_col = csr_colIdx_l[lb];
                tmpA = csr_val_l[lb];
#pragma unroll
                for (int i = 0; i < CF; i++)
                    res[i] += input[cur_col * dim + offset + i * 32 + laneId] * tmpA;
            }
#pragma unroll
            for (int i = 0; i < CF; i++)
                atomicAdd(&output[CrowIdx * dim + offset + i * 32 + laneId], res[i]);
        }
    }
    return;
}

// sequential execution of ppl0_batch_SpMM
template <int CF, bool is_residue>
void ppl0_batch_SpMM_seq(
    float *output,
    const float *input,      // NVSHMEM
    const int *csr_rowPtr_l, // local CSR
    const int *csr_colIdx_l,
    const int *csr_rowIdx_zip_l,
    const float *csr_val_l,
    const int *csc_colPtr_r, // remote CSC
    const int *csc_rowIdx_r,
    const int *csc_colIdx_zip_r,
    const float *csc_val_r,
    const int local_row_lb,
    const int remote_col_lb,
    const int local_row_num,
    const int remote_col_num,
    const int dim,
    const int nodesPerPE,
    const int peid,
    cudaStream_t stream)
{
    constexpr int blkWarpNum = 8;
    int T = std::max(local_row_num, remote_col_num);
    int dim_x, dim_y;
    const int shared_mem = blkWarpNum * 32 * CF * sizeof(float);
    dim_y = CEIL_DIV(dim, CF * 32);
    dim_x = CEIL_DIV(T, blkWarpNum);
    dim3 blockDim(blkWarpNum * 32, 1, 1);
    dim3 gridDim(dim_x, dim_y, 1);

    if (T != 0)
        ppl0_batch_SpMM_seq_kernel<CF, blkWarpNum, is_residue><<<gridDim, blockDim, shared_mem, stream>>>(
            output, input,
            csr_rowPtr_l, csr_colIdx_l, csr_rowIdx_zip_l, csr_val_l,
            csc_colPtr_r, csc_rowIdx_r, csc_colIdx_zip_r, csc_val_r,
            local_row_lb, remote_col_lb, local_row_num, remote_col_num,
            dim, nodesPerPE, peid, T);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error @ppl0_batch_SpMM: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

template <int CF, int blkWarpNum, bool is_residue>
__global__ void ppl_batch_SpMM_kernel(
    float *output,
    const float *input, // NVSHMEM
    const int *csc_colPtr_r,
    const int *csc_rowIdx_r,
    const int *csc_colIdx_zip_r,
    const float *csc_val_r,
    const int local_col_lb,
    const int local_col_ub,
    const int start_col_offset,
    const int stage,
    const int *idx_ppl,
    const int dim,
    const int nodesPerPE,
    const int group_size,
    const int group_offset,
    const int peid,
    const int T)
{
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    int g_warpId = blockIdx.x * blkWarpNum + warpId;
    int offset = blockIdx.y * CF * 32;
    extern __shared__ float tmp[];
    int CrowIdx, r_GPUid, lb, hb, cur_col;
    float res[CF] = {0.0};
    float tmpA = 0.0;
    int CFnum, idx = g_warpId + start_col_offset;

    if (g_warpId >= T)
        return;

    if (is_residue)
        CFnum = CEIL_DIV(dim - laneId - offset, 32);

    // ### local computation ###
    if (idx >= local_col_lb && idx < local_col_ub)
    {
        cur_col = idx_ppl[g_warpId];
        lb = csc_colPtr_r[idx];
        hb = csc_colPtr_r[idx + 1];

        if (is_residue)
        {
#pragma unroll
            for (int i = 0; i < CFnum; i++)
                res[i] = input[cur_col * dim + offset + i * 32 + laneId];

            for (; lb < hb; lb++)
            {
                CrowIdx = csc_rowIdx_r[lb];
                tmpA = csc_val_r[lb];
#pragma unroll
                for (int j = 0; j < CFnum; j++)
                    atomicAdd(&output[CrowIdx * dim + offset + j * 32 + laneId], res[j] * tmpA);
            }
        }
        else
        {
#pragma unroll
            for (int i = 0; i < CF; i++)
                res[i] = input[cur_col * dim + offset + i * 32 + laneId];

            for (; lb < hb; lb++)
            {
                CrowIdx = csc_rowIdx_r[lb];
                tmpA = csc_val_r[lb];
#pragma unroll
                for (int j = 0; j < CF; j++)
                    atomicAdd(&output[CrowIdx * dim + offset + j * 32 + laneId], res[j] * tmpA);
            }
        }
        return;
    }
    else
    {
        cur_col = csc_colIdx_zip_r[idx];
        r_GPUid = (cur_col / nodesPerPE) % group_size + group_offset;
        cur_col = idx_ppl[g_warpId];

        nvshmemx_float_get_warp((float *)&tmp[warpId * dim],
                                &input[cur_col * dim], dim, r_GPUid);
        lb = csc_colPtr_r[idx];
        hb = csc_colPtr_r[idx + 1];

        if (is_residue)
        {
#pragma unroll
            for (int i = 0; i < CFnum; i++)
                res[i] = tmp[warpId * dim + i * 32 + laneId];

            for (; lb < hb; lb++)
            {
                CrowIdx = csc_rowIdx_r[lb];
                tmpA = csc_val_r[lb];
#pragma unroll
                for (int j = 0; j < CFnum; j++)
                    atomicAdd(&output[CrowIdx * dim + offset + j * 32 + laneId], res[j] * tmpA);
            }
        }
        else
        {
#pragma unroll
            for (int i = 0; i < CF; i++)
                res[i] = tmp[warpId * dim + i * 32 + laneId];

            for (; lb < hb; lb++)
            {
                CrowIdx = csc_rowIdx_r[lb];
                tmpA = csc_val_r[lb];
#pragma unroll
                for (int j = 0; j < CF; j++)
                    atomicAdd(&output[CrowIdx * dim + offset + j * 32 + laneId], res[j] * tmpA);
            }
        }
        return;
    }
}

template <int CF, bool is_residue>
void ppl_batch_SpMM(
    float *output,
    const float *input,      // NVSHMEM
    const int *csc_colPtr_r, // remote CSC
    const int *csc_rowIdx_r,
    const int *csc_colIdx_zip_r,
    const float *csc_val_r,
    const int local_col_lb,
    const int local_col_ub,
    const int remote_col_lb,
    const int remote_col_ub,
    const int stage,
    const int *idx_ppl,
    const int dim,
    const int nodesPerPE,
    const int group_size,
    const int group_offset,
    const int peid,
    cudaStream_t stream)
{
    constexpr int blkWarpNum = 8;
    // [local_col_lb, local_col_ub) is in [remote_col_lb, remote_col_ub)
    int T = remote_col_ub - remote_col_lb;
    int start_col_offset = remote_col_lb;
    int dim_x, dim_y;
    const int shared_mem = blkWarpNum * 32 * CF * sizeof(float);
    dim_y = CEIL_DIV(dim, CF * 32);
    dim_x = CEIL_DIV(T + 1, blkWarpNum);
    dim3 blockDim(blkWarpNum * 32, 1, 1);
    dim3 gridDim(dim_x, dim_y, 1);
    // printf("PE-%d, stage:%d, local_col_lb:%d, local_col_ub:%d, remote_col_lb:%d, remote_col_ub:%d, T:%d\n",
    //        peid, stage, local_col_lb, local_col_ub, remote_col_lb, remote_col_ub, T);
    // printf("PE-%d, stage:%d, dim_x:%d, dim_y:%d, blockDim:%d, gridDim:%d\n",
    //        peid, stage, dim_x, dim_y, blockDim.x, gridDim.x);
    // directly write result to output
    // if (T > 0)
    ppl_batch_SpMM_kernel<CF, blkWarpNum, is_residue><<<gridDim, blockDim, shared_mem, stream>>>(
        output, input,
        csc_colPtr_r, csc_rowIdx_r, csc_colIdx_zip_r, csc_val_r,
        local_col_lb, local_col_ub, start_col_offset, stage, idx_ppl,
        dim, nodesPerPE, group_size, group_offset, peid, T);
    // else
    //     warmup_kernel<<<1024, 32, 0, stream>>>();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error @ppl_batch_SpMM: %s, PE-%d, stage:%d\n", cudaGetErrorString(error), peid, stage);
        exit(-1);
    }
}

template <int CF, bool is_residue>
inline void distSpMM_ppl_batch(const int group_size,
                               float *output,
                               const float *input_nvsh,
                               float **ppl_buffer_nvsh,
                               float *ppl_send_nvsh,
                               int *local_offset_ppl,
                               int *remote_offset_ppl,
                               int *put_offset_ppl,
                               int **put_colIdx_ppl,
                               int **idx_ppl,
                               d_csr_zip *csr_zip_l,                   // local csr_zip
                               d_csc_zip *csc_zip_r,                   // remote csc_zip
                               const int block_lb, const int block_ub, // output embedding range
                               const int dim,
                               const int nodesPerPE,
                               const int nranks, const int mype_node,
                               cudaStream_t stream) // loop of profiling
{
    const int nstages = nranks / group_size;
    const int group_offset = mype_node / group_size * group_size;
    int local_lb, local_ub, remote_lb, remote_ub;

    if (nstages > 1)
    {
        // stage 1: transfer the first stage of cross-node data
        int target_pe = (mype_node - 1 * group_size) % nranks;
        target_pe = (target_pe + nranks) % nranks;
        int nemb = put_offset_ppl[1] - put_offset_ppl[0];
        if (nemb > 0)
            batching_data_ppl<<<nemb, 32, 0, stream>>>(input_nvsh, ppl_send_nvsh, put_colIdx_ppl[0], dim, nodesPerPE, group_size);
        nvshmemx_float_put_nbi_on_stream(ppl_buffer_nvsh[0], ppl_send_nvsh, nemb * dim, target_pe, stream);
        // nvshmemx_float_put_on_stream(ppl_buffer_nvsh[0], ppl_send_nvsh, nemb * dim, target_pe, stream);
        // coalescing_data_ppl<<<nemb, 32, 0, stream>>>(input_nvsh, ppl_send_nvsh, ppl_buffer_nvsh[0], put_colIdx_ppl[0], nemb, dim, nodesPerPE, target_pe);

        // stage 2: compute
        local_lb = local_offset_ppl[0], local_ub = local_offset_ppl[1];
        remote_lb = remote_offset_ppl[0], remote_ub = remote_offset_ppl[1];
        ppl0_batch_SpMM<CF, is_residue>(
            output, input_nvsh,
            csr_zip_l->d_csr_rowPtr, csr_zip_l->d_csr_colIdx, csr_zip_l->d_csr_rowIdx_zip, csr_zip_l->d_csr_val,
            csc_zip_r->d_csc_colPtr, csc_zip_r->d_csc_rowIdx, csc_zip_r->d_csc_colIdx_zip, csc_zip_r->d_csc_val,
            local_lb, remote_lb, local_ub - local_lb, remote_ub - remote_lb,
            dim, nodesPerPE, mype_node, stream);

        // transfer the second stage of cross-node data
        for (int i = 1; i < nstages - 1; i++)
        {
            nvshmemx_barrier_all_on_stream(stream);
            target_pe = (mype_node - (i + 1) * group_size) % nranks;
            target_pe = (target_pe + nranks) % nranks;
            nemb = put_offset_ppl[i * 2 + 1] - put_offset_ppl[i * 2];
            if (nemb > 0)
                batching_data_ppl<<<nemb, 32, 0, stream>>>(input_nvsh, ppl_send_nvsh, put_colIdx_ppl[i], dim, nodesPerPE, group_size);
            nvshmemx_float_put_nbi_on_stream(ppl_buffer_nvsh[i % 2], ppl_send_nvsh, nemb * dim, target_pe, stream);
            // nvshmemx_float_put_on_stream(ppl_buffer_nvsh[i % 2], ppl_send_nvsh, nemb * dim, target_pe, stream);
            // coalescing_data_ppl<<<nemb, 32, 0, stream>>>(input_nvsh, ppl_send_nvsh, ppl_buffer_nvsh[i % 2], put_colIdx_ppl[i], nemb, dim, nodesPerPE, target_pe);

            local_lb = local_offset_ppl[i * 2], local_ub = local_offset_ppl[i * 2 + 1];
            remote_lb = remote_offset_ppl[i * 2], remote_ub = remote_offset_ppl[i * 2 + 1];
            ppl_batch_SpMM<CF, is_residue>(output, ppl_buffer_nvsh[(i - 1) % 2],
                                           csc_zip_r->d_csc_colPtr, csc_zip_r->d_csc_rowIdx, csc_zip_r->d_csc_colIdx_zip, csc_zip_r->d_csc_val,
                                           local_lb, local_ub, remote_lb, remote_ub, i, idx_ppl[i - 1],
                                           dim, nodesPerPE, group_size, group_offset, mype_node, stream);
        }

        // compute with the prepared remote data
        nvshmemx_barrier_all_on_stream(stream);
        local_lb = local_offset_ppl[(nstages - 1) * 2], local_ub = local_offset_ppl[(nstages - 1) * 2 + 1];
        remote_lb = remote_offset_ppl[(nstages - 1) * 2], remote_ub = remote_offset_ppl[(nstages - 1) * 2 + 1];
        ppl_batch_SpMM<CF, is_residue>(output, ppl_buffer_nvsh[(nstages - 2) % 2],
                                       csc_zip_r->d_csc_colPtr, csc_zip_r->d_csc_rowIdx, csc_zip_r->d_csc_colIdx_zip, csc_zip_r->d_csc_val,
                                       local_lb, local_ub, remote_lb, remote_ub, nstages - 1, idx_ppl[nstages - 2],
                                       dim, nodesPerPE, group_size, group_offset, mype_node, stream);
    }
    else
    {
        local_lb = local_offset_ppl[0], local_ub = local_offset_ppl[1];
        remote_lb = remote_offset_ppl[0], remote_ub = remote_offset_ppl[1];
        ppl0_batch_SpMM<CF, is_residue>(
            // ppl0_batch_SpMM_seq<CF, is_residue>(
            output, input_nvsh,
            csr_zip_l->d_csr_rowPtr, csr_zip_l->d_csr_colIdx, csr_zip_l->d_csr_rowIdx_zip, csr_zip_l->d_csr_val,
            csc_zip_r->d_csc_colPtr, csc_zip_r->d_csc_rowIdx, csc_zip_r->d_csc_colIdx_zip, csc_zip_r->d_csc_val,
            local_lb, remote_lb, local_ub - local_lb, remote_ub - remote_lb,
            dim, nodesPerPE, mype_node, stream);
    }
}

void distSpMM_ppl_batch_wrapper(const int group_size,
                                float *output,
                                const float *input_nvsh,
                                float **ppl_buffer_nvsh,
                                float *ppl_send_nvsh,
                                float *comm_buffer_nvsh,
                                int *local_offset_ppl,
                                int *remote_offset_ppl,
                                int *put_offset_ppl,
                                int **put_colIdx_ppl,
                                int **idx_ppl,
                                d_csr_zip *csr_zip_l,
                                d_csc_zip *csc_zip_r,
                                const int block_lb, const int block_ub,
                                const int dim,
                                const int nodesPerPE,
                                const int nranks, const int mype_node,
                                cudaStream_t stream)
{
    const int CF = CEIL_DIV(dim, 32);
    const bool is_residue = dim & 31;
    if (group_size == nranks && nranks > 8)
    {
        int local_lb = local_offset_ppl[0], local_ub = local_offset_ppl[1];
        int remote_lb = remote_offset_ppl[0], remote_ub = remote_offset_ppl[1];
        ppl0_batch_buffer_SpMM<1, false>(
            output, input_nvsh, comm_buffer_nvsh,
            csr_zip_l->d_csr_rowPtr, csr_zip_l->d_csr_colIdx, csr_zip_l->d_csr_rowIdx_zip, csr_zip_l->d_csr_val,
            csc_zip_r->d_csc_colPtr, csc_zip_r->d_csc_rowIdx, csc_zip_r->d_csc_colIdx_zip, csc_zip_r->d_csc_val,
            local_lb, remote_lb, local_ub - local_lb, remote_ub - remote_lb,
            dim, nodesPerPE, mype_node, stream);
        return;
    }

    if (CF == 1)
        if (is_residue)
            distSpMM_ppl_batch<1, true>(group_size, output, input_nvsh, ppl_buffer_nvsh, ppl_send_nvsh,
                                        local_offset_ppl, remote_offset_ppl, put_offset_ppl, put_colIdx_ppl, idx_ppl,
                                        csr_zip_l, csc_zip_r,
                                        block_lb, block_ub, dim, nodesPerPE, nranks, mype_node, stream);
        else
            distSpMM_ppl_batch<1, false>(group_size, output, input_nvsh, ppl_buffer_nvsh, ppl_send_nvsh,
                                         local_offset_ppl, remote_offset_ppl, put_offset_ppl, put_colIdx_ppl, idx_ppl,
                                         csr_zip_l, csc_zip_r,
                                         block_lb, block_ub, dim, nodesPerPE, nranks, mype_node, stream);
    else if (CF == 2)
        if (is_residue)
            distSpMM_ppl_batch<2, true>(group_size, output, input_nvsh, ppl_buffer_nvsh, ppl_send_nvsh,
                                        local_offset_ppl, remote_offset_ppl, put_offset_ppl, put_colIdx_ppl, idx_ppl,
                                        csr_zip_l, csc_zip_r,
                                        block_lb, block_ub, dim, nodesPerPE, nranks, mype_node, stream);
        else
            distSpMM_ppl_batch<2, false>(group_size, output, input_nvsh, ppl_buffer_nvsh, ppl_send_nvsh,
                                         local_offset_ppl, remote_offset_ppl, put_offset_ppl, put_colIdx_ppl, idx_ppl,
                                         csr_zip_l, csc_zip_r,
                                         block_lb, block_ub, dim, nodesPerPE, nranks, mype_node, stream);
    else if (CF == 4)
        if (is_residue)
            distSpMM_ppl_batch<4, true>(group_size, output, input_nvsh, ppl_buffer_nvsh, ppl_send_nvsh,
                                        local_offset_ppl, remote_offset_ppl, put_offset_ppl, put_colIdx_ppl, idx_ppl,
                                        csr_zip_l, csc_zip_r,
                                        block_lb, block_ub, dim, nodesPerPE, nranks, mype_node, stream);
        else
            distSpMM_ppl_batch<4, false>(group_size, output, input_nvsh, ppl_buffer_nvsh, ppl_send_nvsh,
                                            local_offset_ppl, remote_offset_ppl, put_offset_ppl, put_colIdx_ppl, idx_ppl,
                                            csr_zip_l, csc_zip_r,
                                            block_lb, block_ub, dim, nodesPerPE, nranks, mype_node, stream);
}

template <bool validate, bool profiling, bool profile_detail = false>
void run_distSpMM_ppl_batch(const int group_size,
                            float *output,
                            const float *input_nvsh,
                            float **ppl_buffer_nvsh,
                            float *ppl_send_nvsh,
                            float *comm_buffer_nvsh,
                            int *local_offset_ppl,
                            int *remote_offset_ppl,
                            int *put_offset_ppl,
                            int **put_colIdx_ppl,
                            int **idx_ppl,
                            d_csr_zip *csr_zip_l,                   // local csr_zip
                            d_csc_zip *csc_zip_r,                   // remote csc_zip
                            const int block_lb, const int block_ub, // output embedding range
                            const int dim,
                            const int nodesPerPE,
                            const int nranks, const int mype_node,
                            const int *global_rowPtr, // host data
                            const int *global_colIdx,
                            const float *h_in,
                            cudaStream_t stream)
{
    if (mype_node == 0 && (validate || profiling))
        printf("Running ppl_batch SpMM with group_size:%d\n", group_size);
    // TODO
    distSpMM_ppl_batch_wrapper(group_size, output, input_nvsh, ppl_buffer_nvsh, ppl_send_nvsh, comm_buffer_nvsh,
                               local_offset_ppl, remote_offset_ppl, put_offset_ppl, put_colIdx_ppl, idx_ppl,
                               csr_zip_l, csc_zip_r,
                               block_lb, block_ub, dim, nodesPerPE, nranks, mype_node, stream);
    gpuErrchk(cudaStreamSynchronize(stream));
    if (validate)
    {
        float *h_out = (float *)malloc(nodesPerPE * dim * sizeof(float));
        float *h_out_ref = (float *)malloc(nodesPerPE * dim * sizeof(float));
        memset(h_out_ref, 0, static_cast<size_t>(nodesPerPE) * dim * sizeof(float));
        SAG_reference_host<int, float>(block_lb, block_ub, dim, global_rowPtr, global_colIdx, h_in, h_out_ref, mype_node);
        gpuErrchk(cudaMemcpy(h_out, output, nodesPerPE * dim * sizeof(float), cudaMemcpyDeviceToHost));
        check_result(block_ub - block_lb, dim, h_out, h_out_ref, mype_node);
        free(h_out);
        free(h_out_ref);
    }

    if (profiling)
    {
        const int num_profiling = 200;
        for (int i = 0; i < 50; i++)
        {
            warmup_kernel<<<1024, 256>>>();
        }
        double t1, t2;

        MPI_Barrier(MPI_COMM_WORLD);
        t1 = MPI_Wtime();
        for (int i = 0; i < num_profiling; i++)
            distSpMM_ppl_batch_wrapper(group_size, output, input_nvsh, ppl_buffer_nvsh, ppl_send_nvsh, comm_buffer_nvsh,
                                       local_offset_ppl, remote_offset_ppl, put_offset_ppl, put_colIdx_ppl, idx_ppl,
                                       csr_zip_l, csc_zip_r,
                                       block_lb, block_ub, dim, nodesPerPE, nranks, mype_node, stream);
        gpuErrchk(cudaStreamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);
        t2 = MPI_Wtime();
        if (mype_node == 0)
        {
            printf("nodesPerPE:%d\n", nodesPerPE);
            printf("PE-%d MPI time (ms) %.3f\n", mype_node, (t2 - t1) * 1e3 / num_profiling);
        }
    }
}