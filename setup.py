from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension(
        "src.analysis.statistics.cython_statistics",
        ["src/analysis/statistics/cython_statistics.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-O3"],
    ),
    Extension(
        "src.analysis.models.market_data_store",
        ["src/analysis/models/market_data_store.pyx"],
        include_dirs=[np.get_include()],
        language="c++"
    ),
    Extension(
        "src.analysis.models.position",
        ["src/analysis/models/position.pyx"],
        include_dirs=[np.get_include()],
        language="c++"
    ),
    Extension(
        "src.analysis.models.portfolio",
        ["src/analysis/models/portfolio.pyx"],
        include_dirs=[np.get_include()],
        language="c++"
    ),
    Extension(
        "src.analysis.backtest.backtest_cy",
        ["src/analysis/backtest/backtest_cy.pyx"],
        include_dirs=[np.get_include()],
        language="c++"
    ),
    Extension(
        "src.analysis.models.struct_bench",
        ["src/analysis/models/struct_bench.pyx"],
        include_dirs=[np.get_include()],
        language="c++"
    ),
]

setup(
    name="harry-botter-tng",
    ext_modules=cythonize(extensions),
    zip_safe=False,
)