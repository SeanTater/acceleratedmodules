#!/usr/bin/env python3
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(
        "simulation.pyx",
        language_level=3
    ),
    include_dirs=[numpy.get_include()]
)