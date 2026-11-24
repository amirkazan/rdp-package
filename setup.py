import os
import platform
import sys

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np


def extra_compile_args():
    machine = platform.machine().lower()
    is_x86 = machine in ("x86_64", "amd64", "x64")
    # Portable wheels / other machines: never use -march=native.
    portable = os.environ.get("CIBUILDWHEEL") == "1" or os.environ.get("RDP_PORTABLE") == "1"

    if sys.platform == "win32":
        args = ["/O2", "/DNDEBUG", "/std:c++17"]
        if is_x86:
            args.append("/arch:AVX2")
        return args

    args = ["-O3", "-DNDEBUG", "-std=c++17"]
    if is_x86:
        # AVX2 opcodes are behind a runtime CPUID check in AdaptiveRdp.h.
        args += ["-mavx2", "-mfma"]
        if not portable:
            args.append("-march=native")
    return args


# Prefer .pyx when present (dev / sdist). Fall back to pre-generated .cpp.
_source = "rdp.pyx" if os.path.exists("rdp.pyx") else "rdp.cpp"

extensions = [
    Extension(
        "rdp",
        sources=[_source],
        language="c++",
        include_dirs=[np.get_include(), "cpp"],
        extra_compile_args=extra_compile_args(),
        libraries=[] if sys.platform == "win32" else ["m"],
    )
]

if _source.endswith(".pyx"):
    ext_modules = cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    )
else:
    ext_modules = extensions

_here = os.path.dirname(os.path.abspath(__file__))
_readme = os.path.join(_here, "README.md")
long_description = ""
if os.path.exists(_readme):
    with open(_readme, encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="rdp-package",
    version="0.3.0",
    description="RDP Cython package + optional rdp_adaptive_cpp (C++/AVX2)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    include_package_data=True,
    zip_safe=False,
)
