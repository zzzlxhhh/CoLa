#!/bin/bash 
#SBATCH -p g078t
#SBATCH -w gpu1
#SBATCH -N 1  
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4  
#SBATCH --gres=gpu:4
#SBATCH --time=120:00  
#SBATCH --output=1node_4gpu_cola_%j.out


APP_PATH="./distSpMM_ppl_batch"
# nvcc --version 
# which nvcc
module purge
module load gcc/9.3.0

echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "task per node: $SLURM_NTASKS_PER_NODE"
echo "Executable path: $APP_PATH"
echo "Task num: $SLURM_NTASKS"
export GDRCOPY_HOME=~/mylib/gdrcopy-2.4.4
export NCCL_HOME=~/mylib/nccl_2.16.5-1+cuda11.8_x86_64
export CPATH=${NCCL_HOME}/include:${GDRCOPY_HOME}/include:~/miniconda3/envs/cola/include:$CPATH
export LIBRARY_PATH=${NCCL_HOME}/lib:${GDRCOPY_HOME}/lib:~/miniconda3/envs/cola/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=${NCCL_HOME}/lib:${GDRCOPY_HOME}/lib:~/miniconda3/envs/cola/lib:$LD_LIBRARY_PATH

export NVSHMEM_SYMMETRIC_SIZE=21474836480
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_USE_GDRCOPY=1

# export NVSHMEM_INFO=true
# export NVSHMEM_DEBUG=TRACE
# export NVSHMEM_DEBUG_SUBSYS=ALL
# export NVSHMEM_DEBUG_FILE=nvshmemdebug

lsmod | grep gdrdrv
lsmod | grep nvidia_peermem

cd ../build

# mpirun -np $SLURM_NTASKS $APP_PATH com-Amazon $SLURM_NTASKS 32 16 1
DATA_LIST=(
    "com-Youtube_rabbit"
    "com-Youtube"
)


for DATA in ${DATA_LIST[@]}
do
    # arglist: num_GPUs, APP_PATH, DATA, num_GPUs, embedding dimension, workgroup size, if_split(if apply nonzero-split on RSC)
    
    # this is an example for 1node-4GPU test, the workgroup size is 4, embedding dimension is 32
    mpirun -np $SLURM_NTASKS $APP_PATH $DATA $SLURM_NTASKS 32 4 1
    
    # this is an example for 2node-16GPU test, the workgroup size is 8, embedding dimension is 32
    # when test on 2nodes, make sure to set the slurm job correctly, e.g., 
    # sbatch -N 2 --ntasks=16 --gres=gpu:8
    # mpirun -np $SLURM_NTASKS $APP_PATH $DATA $SLURM_NTASKS 32 8 1
done

