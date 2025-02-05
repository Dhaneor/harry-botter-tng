from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "src.analysis.models.portfolio",
        ["src/analysis/models/portfolio.pyx"],
        include_dirs=[np.get_include()],
        language="c++"
    )
]

setup(
    name="harry-botter-tng",
    ext_modules=cythonize(extensions),
    zip_safe=False,
)