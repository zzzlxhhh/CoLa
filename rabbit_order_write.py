#!/usr/bin/env python3
import glob
import argparse
import os
import time
import torch
import pickle
import rabbit
import numpy as np
import scipy.sparse as sparse
import scipy.io as sio


class graph_input(object):
    def __init__(self, path=None):
        self.load_flag = False
        self.reorder_flag = False
        self.path = path
        self.edge_index = None
        self.num_nodes = 0

    def load(self, load_from_txt=True):
        """
        load the graph from the disk --> CPU memory.
        """
        if self.path == None:
            raise ValueError("Graph path must be assigned first")

        start = time.perf_counter()
        if load_from_txt:
            A = sio.mmread(self.path)
            src_li = A.row.tolist()
            dst_li = A.col.tolist()
            src_idx = torch.IntTensor(src_li)
            dst_idx = torch.IntTensor(dst_li)
            self.edge_index = torch.stack([src_idx, dst_idx], dim=0)
            self.num_nodes = A.shape[0]

        dur = time.perf_counter() - start
        print("Loading graph from mtx source (ms): {:.3f}".format(dur * 1e3))

        self.load_flag = True

    def reorder(self):
        """
        reorder the graph if specified.
        """
        if not self.load_flag:
            raise ValueError("Graph MUST be loaded Before reordering.")

        new_edge_index = rabbit.reorder(self.edge_index)
        self.edge_index = new_edge_index
        self.reorder_flag = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()
    print(args)
    
    dataset = [
        "com-Youtube",
    ]
    mtx_path = "./storage/mtx/"

    os.makedirs(mtx_path + "rabbit", exist_ok=True)
    for graph_name in dataset:
        graph = graph_input(mtx_path + graph_name + ".mtx")
        graph.load(load_from_txt=True)
        graph.reorder()
        d = np.ones(graph.edge_index.size()[1])
        spmatrix = sparse.coo_matrix(
            (d, graph.edge_index), shape=(graph.num_nodes, graph.num_nodes)
        )
        sio.mmwrite(mtx_path + "rabbit/" + graph_name + "_rabbit.mtx", spmatrix)
