#pragma once

#include <vector>
#include <algorithm> // for copy() and assign()
#include <iterator>  // for back_inserter

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
using nidType = int;

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__global__ void warmup_kernel()
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}

void fill_certain(float array[], int size, float val)
{
    for (int i = 0; i < size; i++)
        array[i] = val;
}

void fill_seq(float array[], int size, int nodesPerPE, float val)
{
    for (int i = 0; i < size; i++)
        array[i] = i / 32 % nodesPerPE * val;
}

template <typename Index, typename DType>
void SAG_reference_host(
    int lb, int ub, int N, // number of B_columns
    const Index *csr_indptr, const int *csr_indices,
    const DType *B,         // assume row-major
    DType *C_ref, int peid) // assume row-major
{
    int M = ub - lb;
    memset(C_ref, 0, sizeof(DType) * M * N);
    for (int i = 0; lb < ub; i++, lb++)
    {
        Index begin = csr_indptr[lb];
        Index end = csr_indptr[lb + 1];
        // if (i == 0)
        //     printf("peid-%d, begin:%d, end:%d\n", peid, begin, end);
        for (Index p = begin; p < end; p++)
        {
            int k = csr_indices[p];
            // DType val = csr_values[p];
            for (int64_t j = 0; j < N; j++)
            {
                C_ref[i * N + j] += B[k * N + j];
            }
        }
    }
}

template <typename DType>
void check_result(int M, int N, DType *C, DType *C_ref, int PEid)
{
    bool passed = true;
    int err_count = 0;
    for (int64_t i = 0; i < M; i++)
    {
        for (int64_t j = 0; j < N; j++)
        {
            DType c = C[i * N + j];
            DType c_ref = C_ref[i * N + j];
            if (fabs(c - c_ref) > 1e-2 * fabs(c_ref))
            {
                printf(
                    "peid-%d Wrong result: i = %ld, j = %ld, result = %lf, reference = %lf.\n",
                    PEid, i, j, c, c_ref);
                passed = false;
                err_count++;
                if (err_count > 32)
                    exit(-1);
            }
        }
    }
    if (passed)
    {
        printf("PEid-%d passed\n", PEid);
        return;
    }
    // else
    //     exit(-1);
}