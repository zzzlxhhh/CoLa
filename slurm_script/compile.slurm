#!/bin/bash 
#SBATCH -p g078t
#SBATCH -w gpu1
#SBATCH -N 1  
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4  
#SBATCH --gres=gpu:4
#SBATCH --time=120:00  
#SBATCH --output=compile_cola_%j.out


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


lsmod | grep gdrdrv
lsmod | grep nvidia_peermem

cd ..
rm -rf build
mkdir build
cd build
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:$LD_LIBRARY_PATH
module load gcc/9.3.0
cmake ..
make -j


