#include <iostream>
#include <stdio.h>
#include <ctime>
#include <cmath>
#include <string>
#include <algorithm>
#include <vector>
#include <fstream>
#include <mpi.h>
#include "sparse_data.h"
#include "dist_spmm_ppl_batch.cuh"
#include <nvshmem.h>
#include <nvshmemx.h>
// using namespace std;
#define valiadation 1

int main(int argc, char *argv[])
{
    const char *graph = argv[1];
    int num_GPUs = atoi(argv[2]);   // 2
    int dim = atoi(argv[3]);        // dimension of dense matrix
    int group_size = atoi(argv[4]); //
    int if_split = atoi(argv[5]);   // split nnz for workload balancing in per GPU

    int rank, nranks;
    cudaStream_t stream;
    nvshmemx_init_attr_t attr;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    attr.mpi_comm = &mpi_comm;
    // Set up NVSHMEM device.
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    // int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    int mype_node = nvshmem_my_pe();
    int local_gpu_num = 0;
    cudaGetDeviceCount(&local_gpu_num);
    cudaSetDevice(mype_node % local_gpu_num);
    printf("PE-%d, local_gpu_num: %d, local_gpu_id: %d\n", mype_node, local_gpu_num, mype_node % local_gpu_num);
    cudaStreamCreate(&stream);

    if (mype_node == 0)
        printf("graph_name: %s, num_GPUs: %d, dim: %d, group_size: %d for ppl\n", graph, num_GPUs, dim, group_size);

    std::clock_t c_start_proc = std::clock();

    // local CSC_zip
    std::string graph_name = graph;
    std::string pos = std::to_string(num_GPUs) + "_" + std::to_string(mype_node);
    std::string dir = "../storage/SP/" + graph_name + "/GPU" + std::to_string(num_GPUs) + "/";
    std::vector<int> global_rowPtr, global_colIdx, global_val;
    global_rowPtr = bin2vec<int>("../storage/SP/" + graph_name + "/global_rowPtr.bin");
    global_colIdx = bin2vec<int>("../storage/SP/" + graph_name + "/global_colIdx.bin");

    // data for SpMM
    std::vector<int> csc_colPtr_r, csc_rowIdx_r, csc_colIdx_zip_r;
    std::vector<int> zipcsr_rowPtr_l, zipcsr_colIdx_l, zipcsr_rowIdx_l;
    std::vector<float> val_csc_r, val_csr_l;
    // data for cross-node communication
    int nstages = nranks / group_size;
    std::string dir_ppl = "../storage/SP/" + graph_name + "/ppl" + std::to_string(nstages) + "/";
    // the indices of embedding to in another PE's
    std::vector<std::vector<int>> put_colIdx_ppl(nstages - 1, std::vector<int>());
    // the indices of embedding for the current PE's
    std::vector<std::vector<int>> idx_ppl(nstages - 1, std::vector<int>());

    std::vector<int> put_offset_ppl(nstages * 2, 0);
    // local and remote offset range for SpMM
    std::vector<int> local_offset_ppl(nstages * 2, 0);
    std::vector<int> remote_offset_ppl(nstages * 2, 0);
    int num_nodes = global_rowPtr.size() - 1;
    int nodesPerPE = (num_nodes + nranks - 1) / nranks; // ideal number of nodes per PE
    int block_lb = nodesPerPE * mype_node;
    int block_ub = std::min(num_nodes, block_lb + nodesPerPE);

    // LSC
    auto tmp_csr_rowPtr_l = bin2vec<int>(dir + "csr_rowPtr_l_" + pos + ".bin");
    zipcsr_colIdx_l = bin2vec<int>(dir + "csr_colIdx_l_" + pos + ".bin");
    auto tmp_csr_rowIdx_l = bin2vec<int>(dir + "csr_rowIdx_l_" + pos + ".bin");
    nnz_split(tmp_csr_rowIdx_l, tmp_csr_rowPtr_l, zipcsr_colIdx_l, zipcsr_rowIdx_l, zipcsr_rowPtr_l, mype_node, true);
    val_csr_l = std::vector<float>(zipcsr_colIdx_l.size(), 1.0);

    // RSC
    if (if_split)
    {
        auto tmp_csc_colPtr_r = bin2vec<int>(dir + "csc_colPtr_r_" + pos + ".bin");
        csc_rowIdx_r = bin2vec<int>(dir + "csc_rowIdx_r_" + pos + ".bin");
        auto tmp_csc_colIdx_zip_r = bin2vec<int>(dir + "csc_colIdx_r_" + pos + ".bin");
        nnz_split(tmp_csc_colIdx_zip_r, tmp_csc_colPtr_r, csc_rowIdx_r, csc_colIdx_zip_r, csc_colPtr_r, mype_node, false);
    }
    else
    {
        csc_colPtr_r = bin2vec<int>(dir + "csc_colPtr_r_" + pos + ".bin");
        csc_rowIdx_r = bin2vec<int>(dir + "csc_rowIdx_r_" + pos + ".bin");
        csc_colIdx_zip_r = bin2vec<int>(dir + "csc_colIdx_r_" + pos + ".bin");
    }
    val_csc_r = std::vector<float>(csc_rowIdx_r.size(), 1.0);

    // get the offset for mype_node
    std::vector<int> colIdx_workgroup; 
    if (nstages>1)
        colIdx_workgroup = bin2vec<int>(dir_ppl + "colIdx_" + std::to_string(nranks) + "_" + std::to_string(nstages) + "_" + std::to_string(mype_node / group_size) + ".bin");

    std::vector<int>::iterator remote_end;
    for (int i = 0; i < nstages; i++)
    {
        int next_pos = (mype_node + group_size * i) % nranks;
        int group_id = next_pos / group_size;
        // [local_lb, local_ub) is in [remote_lb, remote_ub)
        int local_lb = next_pos * nodesPerPE;
        int local_ub = std::min(num_nodes, local_lb + nodesPerPE);
        int remote_lb = group_id * group_size * nodesPerPE;
        int remote_ub = std::min(num_nodes, remote_lb + group_size * nodesPerPE);

        if (i == 0)
        {
            // directly get the offset from LSC zipcsr_rowIdx_l
            local_offset_ppl[0] = 0;
            local_offset_ppl[1] = zipcsr_rowIdx_l.size();
        }
        else
        {
            // get the offset from RSC csc_colIdx_zip_r
            local_offset_ppl[i * 2] = std::distance(csc_colIdx_zip_r.begin(),
                                                    std::lower_bound(csc_colIdx_zip_r.begin(), csc_colIdx_zip_r.end(), local_lb));
            remote_end = std::lower_bound(csc_colIdx_zip_r.begin(), csc_colIdx_zip_r.end(), local_ub);
            local_offset_ppl[i * 2 + 1] = std::distance(csc_colIdx_zip_r.begin(), remote_end);
        }

        remote_offset_ppl[i * 2] = std::distance(csc_colIdx_zip_r.begin(),
                                                 std::lower_bound(csc_colIdx_zip_r.begin(), csc_colIdx_zip_r.end(), remote_lb));
        remote_end = std::lower_bound(csc_colIdx_zip_r.begin(), csc_colIdx_zip_r.end(), remote_ub);
        remote_offset_ppl[i * 2 + 1] = std::distance(csc_colIdx_zip_r.begin(), remote_end);

        if (i != 0)
        {
            std::vector<std::vector<int>> workgroup_csc_colIdx(group_size, std::vector<int>());
            std::vector<int> idx_pos(group_size, 0);
            for (int j = 0; j < group_size; j++)
            {
                int src_pe = j + group_id * group_size;
                int lb = src_pe * nodesPerPE;
                int ub = std::min(num_nodes, lb + nodesPerPE);

                lb = std::distance(colIdx_workgroup.begin(), std::lower_bound(colIdx_workgroup.begin(), colIdx_workgroup.end(), lb));
                ub = std::distance(colIdx_workgroup.begin(), std::lower_bound(colIdx_workgroup.begin(), colIdx_workgroup.end(), ub));
                // get workgroup_csc_colIdx with colIdx within [lb, ub)
                workgroup_csc_colIdx[j] = std::vector<int>(colIdx_workgroup.begin() + lb, colIdx_workgroup.begin() + ub);
            }

            for (int j = remote_offset_ppl[i * 2]; j < remote_offset_ppl[i * 2 + 1]; j++)
            {
                int cur_col = csc_colIdx_zip_r[j];
                int r_GPUid = (cur_col / nodesPerPE) % group_size;
                int k = idx_pos[r_GPUid];
                while (k < workgroup_csc_colIdx[r_GPUid].size())
                {
                    if (workgroup_csc_colIdx[r_GPUid][k] == cur_col)
                    {
                        idx_ppl[i - 1].push_back(k);
                        idx_pos[r_GPUid] = k;
                        break;
                    }
                    k++;
                }
            }
        }
    }

    // generate put_colIdx_ppl
    for (int i = 0; i < nstages - 1; i++)
    {
        // for example, PE-0, i=1, target_pe = nranks-1
        int target_pe = (mype_node - (i + 1) * group_size) % nranks;
        target_pe = (target_pe + nranks) % nranks;
        int target_workgroup = target_pe / group_size;
        std::vector<int> colIdx_r_target = bin2vec<int>(dir_ppl + "colIdx_" + std::to_string(nranks) + "_" + std::to_string(nstages) + "_" + std::to_string(target_workgroup) + ".bin");
        int put_lb = mype_node * nodesPerPE;
        int put_ub = std::min(num_nodes, put_lb + nodesPerPE);
        put_lb = std::distance(colIdx_r_target.begin(),
                               std::lower_bound(colIdx_r_target.begin(), colIdx_r_target.end(), put_lb));
        remote_end = std::lower_bound(colIdx_r_target.begin(), colIdx_r_target.end(), put_ub);
        if (remote_end == colIdx_r_target.end())
            put_ub = colIdx_r_target.size();
        else
            put_ub = std::distance(colIdx_r_target.begin(), remote_end);
        put_offset_ppl[i * 2] = put_lb;
        put_offset_ppl[i * 2 + 1] = put_ub;
        put_colIdx_ppl[i] = std::vector<int>(colIdx_r_target.begin() + put_lb,
                                             colIdx_r_target.begin() + put_ub);
    }

    // deive-side sparse matrices data for SpMM
    d_csc_zip *d_csc_zip_r = nullptr;
    d_csr_zip *d_csr_zip_l = nullptr;
    d_csc_zip_r = new d_csc_zip(csc_colPtr_r, csc_rowIdx_r, csc_colIdx_zip_r, val_csc_r);
    d_csr_zip_l = new d_csr_zip(zipcsr_rowPtr_l, zipcsr_colIdx_l, zipcsr_rowIdx_l, val_csr_l);

    // device-side put_colIdx_ppl, which is array of list of int
    int **d_put_colIdx_ppl = new int *[nstages - 1];
    int **d_idx_ppl = new int *[nstages - 1];
    for (int i = 0; i < nstages - 1; i++)
    {
        gpuErrchk(cudaMalloc((void **)&d_put_colIdx_ppl[i], put_colIdx_ppl[i].size() * sizeof(int)));
        gpuErrchk(cudaMemcpy(d_put_colIdx_ppl[i], put_colIdx_ppl[i].data(), put_colIdx_ppl[i].size() * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMalloc((void **)&d_idx_ppl[i], idx_ppl[i].size() * sizeof(int)));
        gpuErrchk(cudaMemcpy(d_idx_ppl[i], idx_ppl[i].data(), idx_ppl[i].size() * sizeof(int), cudaMemcpyHostToDevice));
    }

    std::clock_t c_end_proc = std::clock();
    float preproc_time_elapsed_ms = 1000.0 * (c_end_proc - c_start_proc) / CLOCKS_PER_SEC;
    if (mype_node == 0)
        printf("IO read from bin time: %f ms\n", preproc_time_elapsed_ms);

    // host-side dense matrices for validation
    float *h_in;
    h_in = (float *)malloc(static_cast<size_t>(nodesPerPE * nranks) * dim * sizeof(float)); // full dense matrix
    // fill_seq(h_in, nodesPerPE * nranks * dim, nodesPerPE, 1.0);
    fill_certain(h_in, nodesPerPE * nranks * dim, 1.0);
    // device-side dense matrices
    float *d_out = nullptr, *d_in_nvsh = nullptr, *d_ppl_send_nvsh = nullptr;
    float **d_ppl_buffer_nvsh = new float *[2];
    gpuErrchk(cudaMalloc((void **)&d_out, static_cast<size_t>(nodesPerPE) * dim * sizeof(float)));
    gpuErrchk(cudaMemset(d_out, 0, static_cast<size_t>(nodesPerPE) * dim * sizeof(float)));
    d_in_nvsh = (float *)nvshmem_malloc(static_cast<size_t>(nodesPerPE) * dim * sizeof(float));
    gpuErrchk(cudaMemcpy(d_in_nvsh, h_in + mype_node * nodesPerPE * dim, static_cast<size_t>(nodesPerPE) * dim * sizeof(float), cudaMemcpyHostToDevice));
    float *comm_buffer_nvsh = nullptr;
    if (group_size == nranks)
        comm_buffer_nvsh = (float *)nvshmem_malloc(static_cast<size_t>(nodesPerPE) * dim * sizeof(float));

    if (group_size > 0)
    {
        // buffer for cross-node buffer
        d_ppl_buffer_nvsh[0] = (float *)nvshmem_malloc(static_cast<size_t>(nodesPerPE) * dim * sizeof(float));
        gpuErrchk(cudaMemset(d_ppl_buffer_nvsh[0], 0, static_cast<size_t>(nodesPerPE) * dim * sizeof(float)));
        d_ppl_buffer_nvsh[1] = (float *)nvshmem_malloc(static_cast<size_t>(nodesPerPE) * dim * sizeof(float));
        gpuErrchk(cudaMemset(d_ppl_buffer_nvsh[1], 0, static_cast<size_t>(nodesPerPE) * dim * sizeof(float)));
        d_ppl_send_nvsh = (float *)nvshmem_malloc(static_cast<size_t>(nodesPerPE) * dim * sizeof(float));
        gpuErrchk(cudaMemset(d_ppl_send_nvsh, 0, static_cast<size_t>(nodesPerPE) * dim * sizeof(float)));
    }
    run_distSpMM_ppl_batch<true, true, false>(group_size, d_out, d_in_nvsh, d_ppl_buffer_nvsh, d_ppl_send_nvsh, comm_buffer_nvsh,
                                            local_offset_ppl.data(), remote_offset_ppl.data(), put_offset_ppl.data(), d_put_colIdx_ppl, d_idx_ppl,
                                            d_csr_zip_l, d_csc_zip_r, block_lb, block_ub, dim, nodesPerPE, nranks, mype_node,
                                            global_rowPtr.data(), global_colIdx.data(), h_in, stream);

    // ################## free memory ##################
    delete d_csc_zip_r;
    delete d_csr_zip_l;

    free(h_in);

    gpuErrchk(cudaFree(d_out));

    nvshmem_free(d_in_nvsh);
    nvshmem_free(d_ppl_buffer_nvsh[0]);
    nvshmem_free(d_ppl_buffer_nvsh[1]);

    delete[] d_ppl_buffer_nvsh;
    nvshmem_free(d_ppl_send_nvsh);
    for (int i = 0; i < nstages - 1; i++)
        gpuErrchk(cudaFree(d_put_colIdx_ppl[i]));
    delete[] d_put_colIdx_ppl;

    for (int i = 0; i < nstages - 1; i++)
        gpuErrchk(cudaFree(d_idx_ppl[i]));
    delete[] d_idx_ppl;
    nvshmem_free(comm_buffer_nvsh);
    nvshmem_finalize();
    MPI_Finalize();

    if (mype_node == 0)
        printf("=================Done!=================\n");
    return 0;
}