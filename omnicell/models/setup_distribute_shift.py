from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    name="distribute_shift",  # Note: local module name
    sources=["distribute_shift.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=['-O3'],
)

setup(
    ext_modules=cythonize([ext])
)