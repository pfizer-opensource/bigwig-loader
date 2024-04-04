import numpy
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

setup(
    name="bigwig_loader",
    ext_modules=cythonize(
        [
            Extension(
                "bigwig_loader.merge_intervals",
                sources=["bigwig_loader/merge_intervals.pyx"],
                include_dirs=[
                    numpy.get_include(),
                ],
                library_dirs=[],
            ),
            Extension(
                "bigwig_loader.subtract_intervals",
                sources=["bigwig_loader/subtract_intervals.pyx"],
                include_dirs=[
                    numpy.get_include(),
                ],
                library_dirs=[],
            ),
        ],
        annotate=True,
    ),
)
