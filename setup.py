from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension(
        "analysis.statistics.cython_statistics",
        ["src/analysis/statistics/cython_statistics.pyx"],
        include_dirs=[np.get_include(), "src"],
        language="c++",
        extra_compile_args=["-O3"],
    ),
    Extension(
        "analysis.models.market_data_store",
        ["src/analysis/models/market_data_store.pyx"],
        include_dirs=[np.get_include(), "src"],
        language="c++"
    ),
    Extension(
        "analysis.models.position",
        ["src/analysis/models/position.pyx"],
        include_dirs=[np.get_include(), "src"],
        language="c++"
    ),
    Extension(
        "analysis.models.account",
        ["src/analysis/models/account.pyx"],
        include_dirs=[np.get_include(), "src"],
        language="c++"
    ),
    Extension(
        "analysis.backtest.backtest_cy",
        ["src/analysis/backtest/backtest_cy.pyx"],
        include_dirs=[np.get_include(), "src"],
        language="c++",
        extra_compile_args=["-O3", "-fno-strict-aliasing"],
    ),
    Extension(
        "analysis.backtest.backtest",
        ["src/analysis/backtest/backtest.pyx"],
        include_dirs=[np.get_include(), "src"],
        language="c++",
        extra_compile_args=["-O2", "-fno-strict-aliasing"],
    ),
    Extension(
        "analysis.models.struct_bench",
        ["src/analysis/models/struct_bench.pyx"],
        include_dirs=[np.get_include(), "src"],
        language="c++"
    ),
]

setup(
    name="harry-botter-tng",
    version="2.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(extensions, include_path=["src"]),
    include_dirs=[np.get_include(), "src"],
    zip_safe=False,
    install_requires=[
        'numpy>=1.24,<2.2',
        'numba>=0.61.0',
        'cython'
    ],
)