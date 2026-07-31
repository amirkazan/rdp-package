import platform
import sys

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np


def extra_compile_args():
    machine = platform.machine().lower()
    is_x86 = machine in ("x86_64", "amd64", "x64")
    if sys.platform == "win32":
        args = ["/O2", "/DNDEBUG", "/std:c++17"]
        if is_x86:
            args.append("/arch:AVX2")
        return args

    args = ["-O3", "-DNDEBUG", "-std=c++17"]
    if is_x86:
        # Binary may contain AVX2 opcodes; AdaptiveRdp.h dispatches at runtime.
        args += ["-march=native", "-mavx2", "-mfma"]
    return args


extensions = [
    Extension(
        "rdp",
        sources=["rdp.pyx"],
        language="c++",
        include_dirs=[np.get_include(), "cpp"],
        extra_compile_args=extra_compile_args(),
        libraries=[] if sys.platform == "win32" else ["m"],
    )
]

setup(
    name="rdp-package",
    version="0.2.0",
    description="RDP Cython package + optional rdp_adaptive_cpp (C++/AVX2)",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
    zip_safe=False,
)
