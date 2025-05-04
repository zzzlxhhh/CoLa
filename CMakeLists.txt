cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(MGG LANGUAGES CXX CUDA)


#############################################################################
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 17)

# set($ENV{PATH} "${CMAKE_CURRENT_SOURCE_DIR}/local/openmpi-4.1.1/bin/:$ENV{PATH}")
# set(NVSHMEM_HOME "${CMAKE_CURRENT_SOURCE_DIR}/local/nvshmem_src_2.0.3-0/build")
# set(MPI_HOME "${CMAKE_CURRENT_SOURCE_DIR}/local/openmpi-4.1.1")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda/")
set(MPI_HOME "./mylib/openmpi-4.1.1")
set(CUDA_HOME "/usr/local/cuda/")
set(NVSHMEM_HOME "./mylib/nvshmem-3.1.7")
set(NCCL_HOME "./mylib/nccl_2.16.5-1+cuda11.8_x86_64")

#############################################################################

set(SM_ARCH "sm_80")
set($ENV{NVCC_GENCODE} ${SM_ARCH})
set($ENV{NVSHMEM_USE_GDRCOPY} "1")


#############################################################################
link_directories(${NVSHMEM_HOME}/lib)
link_directories(${MPI_HOME}/lib)
link_directories(/usr/local/lib)
link_directories(./miniconda3/envs/coco/lib)
link_directories(${NCCL_HOME}/lib)

#############################################################################
add_executable(distSpMM_ppl_batch
    src/distSpMM_ppl_batch.cu 
)

target_include_directories(distSpMM_ppl_batch 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    include
)

target_compile_options(distSpMM_ppl_batch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(distSpMM_ppl_batch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(distSpMM_ppl_batch PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(distSpMM_ppl_batch
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
    gomp 
    curand
)

