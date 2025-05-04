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
def row_partition(COO, graph_name, nGPU, gs):
    ppl_stage = nGPU//gs
    dir = "./storage/SP/{}/ppl{}".format(graph_name, ppl_stage)
    os.makedirs(dir, exist_ok=True)
    
    origin_row = np.array(COO.row)
    origin_col = np.array(COO.col)
    num_nodes = COO.shape[0]
    
    nodesPerGPU = (num_nodes + nGPU - 1) // nGPU
    stage_lb = []
    stage_ub = []
    for stage in range(ppl_stage):
        stage_lb.append(stage * gs * nodesPerGPU)
        stage_ub.append(min((stage+1)*gs*nodesPerGPU, num_nodes))
    
    # row partitioning
    for stage in range(ppl_stage):
        lb = stage_lb[stage]
        ub = stage_ub[stage]
        mask = (origin_row >= lb) & (origin_row < ub)
        COO_col = origin_col[mask]
        unique_COO_col = np.unique(COO_col)
        # ascending order
        assert np.all(np.diff(unique_COO_col) > 0)
        unique_COO_col.tofile("{}/colIdx_{}_{}_{}.bin".format(dir, nGPU, ppl_stage, stage))

def fork_serialize(A, graph_name, nGPU, gs):
    print("Processing graph: {} with nGPU:{} gs:{} ".format(graph_name, nGPU, gs))
        # CSC generation
    row_partition(
        A,
        graph_name=graph_name,
        nGPU=nGPU,
        gs=gs,
    )

def serialize(A, graph_name, bin_path, nGPU_list, gs_list):

    # serialize data for ppl
    sub_processes = []
    for nGPU in nGPU_list:
        for gs in gs_list:
            if gs < nGPU:
                sub_process = Process(target=fork_serialize, args=(A, graph_name, nGPU, gs))
                sub_process.start()
                sub_processes.append(sub_process)
    for sub_p in sub_processes:
        sub_p.join()


def process_graph(graph_name, mtx_path, bin_path, nGPU_list, gs_list):
    A = load_mtx(graph_name, mtx_path)
    start = time.time()
    serialize(A, graph_name, bin_path, nGPU_list, gs_list)
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
    # number of GPUs
    nGPU_list = [2, 4, 8, 16]
    # group size list
    gs_list = [1, 8]
    
    if rabbit:
        mtx_path = mtx_path + "/rabbit"

    processes = []

    start_time = time.time()
    for graph_name in dataset:
        if rabbit:
            graph_name = graph_name + "_rabbit"
        process = Process(target=process_graph, args=(graph_name, mtx_path, bin_path, nGPU_list, gs_list))
        process.start()
        processes.append(process)
    for p in processes:
        p.join()
    print("Total time:", time.time() - start_time)
