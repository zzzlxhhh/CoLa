# CoLa: Towards Communication-efficient Distributed Sparse Matrix-Matrix Multiplication on GPUs
CoLa is a highly communication-efficient distributed SpMM framework. CoLa is accepted by ICS 2025.

# 1. Setup
- Install dependencies including pytorch, NCCL, openmpi and NVSHMEM. Especially for NVSHMEM, please refer to the [official website](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html). For multi-node multi-GPUs with InfiniBand, please make sure to enable InfiniBand GPUDirect Async (IBGDA) transport.
    
    ```
    GCC: 9.3.0 
    CUDA: 11.8 
    NCCL: 2.16.5-1+cuda11.8
    OpenMPI: 4.1.1
    NVSHMEM: 3.1.7
    ```

- Data preprocessing

    - Graph reordering with rabbit reordering ([github page](https://github.com/araij/rabbit_order.git))
    
    ```bash
    # Install the rabbit reordering module 
    python ./rabbit_module/src/setup.py install
    # (Optional) Reorder the graph
    python ./rabbit_order_write.py
    ```

    - Data preparation for CoLa

    ```bash
    # set the nGPU_list in the python scripts to prepare the data for CoLa under different GPU numbers
    python ./prep_torch_ppl.py --rabbit True
    python ./prep_torch_par.py --rabbit True
    ```
    The processed data is in the `storage` folder.

# 2. Build and Run
Our running script is provided in the `slurm_script` folder, where CoLa is running as a slurm job in a HPC system. Make sure to set the correct environment variables in the slurm and cmake script based on your own environment e.g., NVSHMEM_HOME, MPI_HOME, CUDA_HOME, etc.

- compile the code
    ```bash 
    sbatch compile.slurm
    ```
- run the code
    ```bash
    # this script provides an example CoLa under 1node-4GPU test
    # some modification is needed for CoLa under different GPU platforms based on your own environment
    sbatch cola_ppl_batch.slurm
    ```
