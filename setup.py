from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "rdp",
        sources=["rdp.pyx"],
        extra_compile_args=["-O3"],
        libraries=["m"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(extensions),
)
