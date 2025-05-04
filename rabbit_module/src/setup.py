import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# Get the directory containing setup.py
setup_dir = os.path.dirname(os.path.abspath(__file__))
source_file = os.path.join(setup_dir, 'reorder.cpp')

setup(
    name='rabbit',
    ext_modules=[
        CppExtension(
            name='rabbit',
            sources=[source_file], # Use explicit path
            extra_compile_args=['-O3', '-fopenmp', '-mcx16'],
            libraries=["numa", "tcmalloc_minimal"]
         )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


