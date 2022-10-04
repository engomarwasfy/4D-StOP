# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

_ext_src_root = "_ext_src"
_ext_sources = glob.glob(f"{_ext_src_root}/src/*.cpp") + glob.glob(
    f"{_ext_src_root}/src/*.cu"
)

_ext_headers = glob.glob(f"{_ext_src_root}/include/*")

setup(
    name='pointnet2',
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", f"-I{_ext_src_root}/include"],
                "nvcc": ["-O2", f"-I{_ext_src_root}/include"],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
