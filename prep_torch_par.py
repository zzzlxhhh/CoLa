import os
import csv
import time
import argparse
from multiprocessing import Process
import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix, coo_matrix
from scipy import sparse

import threading


def load_mtx(graph_name, mtx_path):
    mtx_file = mtx_path + "/{}.mtx".format(graph_name)
    if not os.path.exists(mtx_path):
        raise ValueError("mtx_file{} does not exist.".format(mtx_file))
    print("Loading mtx file: ", mtx_file)
    A = mmread(mtx_file)
    return A


# generate colIdx for each row pannel and LSC and RSC
def row_partition(COO, nGPU, graph_name, bin_path):
    dir = bin_path + "/{}/GPU{}".format(graph_name, nGPU)
    os.makedirs(dir, exist_ok=True)
    
    origin_row = np.array(COO.row)
    origin_col = np.array(COO.col)
    num_nodes = COO.shape[0]
    nodesPerGPU = (num_nodes + nGPU - 1) // nGPU
    
    for rankId in range(nGPU):
        lb = rankId*nodesPerGPU
        ub = min((rankId+1)*nodesPerGPU, num_nodes)
        mask = (origin_row >= lb) & (origin_row < ub)
        COO_row = origin_row[mask] - lb
        COO_col = origin_col[mask]
        
        local_mask = (rankId*nodesPerGPU <= COO_col) & (COO_col < (rankId+1)*nodesPerGPU)
        local_COO_row = COO_row[local_mask]
        local_COO_col = COO_col[local_mask]
        
        local_CSR = coo_matrix(
            (np.ones(len(local_COO_row)), (local_COO_row, local_COO_col%nodesPerGPU)),
            shape=(ub-lb, nodesPerGPU),
        ).tocsr()
        local_rowIdx = np.unique(local_COO_row)
        
        assert np.all(np.diff(local_rowIdx) > 0)
        local_csr_rowPtr = np.array(local_CSR.indptr, dtype=np.int32)
        local_csr_rowPtr = np.unique(local_csr_rowPtr)
        assert np.all(np.diff(local_csr_rowPtr) > 0)
        local_csr_colIdx = np.array(local_CSR.indices, dtype=np.int32)
        
        local_csr_rowPtr.tofile("{}/csr_rowPtr_l_{}_{}.bin".format(dir, nGPU, rankId))
        local_csr_colIdx.tofile("{}/csr_colIdx_l_{}_{}.bin".format(dir, nGPU, rankId))
        local_rowIdx.tofile("{}/csr_rowIdx_l_{}_{}.bin".format(dir, nGPU, rankId))    
        
        remote_mask = ~local_mask
        remote_COO_row = COO_row[remote_mask]
        remote_COO_col = COO_col[remote_mask]
        remote_CSC = coo_matrix(
            (np.ones(len(remote_COO_row)), (remote_COO_row, remote_COO_col)),
            shape=(ub-lb, num_nodes),
        ).tocsc()
        remote_colIdx = np.array(remote_COO_col, dtype=np.int32)
        remote_colIdx = np.unique(remote_colIdx)
        
        # check if remote_colIdx is ascending
        assert np.all(np.diff(remote_colIdx) > 0)
        remote_csc_colPtr = np.array(remote_CSC.indptr, dtype=np.int32)
        remote_csc_colPtr = np.unique(remote_csc_colPtr)
        assert np.all(np.diff(remote_csc_colPtr) > 0)
        remote_csc_rowIdx = np.array(remote_CSC.indices, dtype=np.int32)
        
        remote_csc_rowIdx.tofile("{}/csc_rowIdx_r_{}_{}.bin".format(dir, nGPU, rankId))
        remote_csc_colPtr.tofile("{}/csc_colPtr_r_{}_{}.bin".format(dir, nGPU, rankId))
        remote_colIdx.tofile("{}/csc_colIdx_r_{}_{}.bin".format(dir, nGPU, rankId))
    
    

def fork_serialize(A, nGPU, graph_name, bin_path):
    print("Processing graph: {} with nGPU:{} ".format(graph_name, nGPU))
        # CSC generation
    row_partition(
        A,
        nGPU=nGPU,
        graph_name=graph_name,
        bin_path=bin_path,
    )

def serialize(A, graph_name, bin_path, nGPU_list):
    # serialize global sparse data
    dir = bin_path + "/{}".format(graph_name)
    os.makedirs(dir, exist_ok=True)
    global_CSR = A.tocsr()
    global_rowPtr = np.array(global_CSR.indptr)
    global_colIdx = np.array(global_CSR.indices)
    global_rowPtr.tofile("{}/global_rowPtr.bin".format(dir))
    global_colIdx.tofile("{}/global_colIdx.bin".format(dir))
    
    # serialize data for ppl
    sub_processes = []
    for nGPU in nGPU_list:
        sub_process = Process(target=fork_serialize, args=(A, nGPU, graph_name, bin_path))
        sub_process.start()
        sub_processes.append(sub_process)
    for sub_p in sub_processes:
        sub_p.join()


def process_graph(graph_name, mtx_path, bin_path, nGPU_list):
    A = load_mtx(graph_name, mtx_path)
    start = time.time()
    serialize(A, graph_name, bin_path, nGPU_list)
    end = time.time()
    print("serialize time:", end - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rabbit", type=str, choices=["False", "True"], default="False")
    dataset = [
        "com-Youtube",
    ]
    mtx_path = "./storage/mtx"
    bin_path = "./storage/SP"
    args = parser.parse_args()
    rabbit = args.rabbit == "True"
    nGPU_list = [2, 4, 8, 16]
    if rabbit:
        mtx_path = mtx_path + "/rabbit"

    processes = []

    start_time = time.time()
    for graph_name in dataset:
        if rabbit:
            graph_name = graph_name + "_rabbit"
        process = Process(target=process_graph, args=(graph_name, mtx_path, bin_path, nGPU_list))
        process.start()
        processes.append(process)
    for p in processes:
        p.join()
    print("Total time:", time.time() - start_time)
