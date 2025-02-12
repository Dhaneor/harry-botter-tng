import time
from analysis.models.struct_bench import CStructArray, NumpyArray  # Import the compiled Cython module

def benchmark_access(cls, name, rows, cols):
    instance = cls(rows, cols)
    instance.initialize()
    
    start = time.perf_counter()
    instance.access_elements()
    end = time.perf_counter()
    
    print(f"{name}: Accessing elements took {end - start:.6f} seconds")

def benchmark_set(cls, name, rows, cols, value=1.0):
    instance = cls(rows, cols)
    instance.initialize()
    
    start = time.perf_counter()
    instance.set_elements(value)
    end = time.perf_counter()
    
    print(f"{name}: Setting elements took {end - start:.6f} seconds")

def run_benchmarks(rows=1000, cols=1000):
    print(f"Benchmarking {rows}x{cols} arrays:\n")
    
    benchmark_access(CStructArray, "C Struct Array", rows, cols)
    benchmark_access(NumpyArray, "NumPy Array", rows, cols)
    print()
    benchmark_set(CStructArray, "C Struct Array", rows, cols)
    benchmark_set(NumpyArray, "NumPy Array", rows, cols)

if __name__ == "__main__":
    run_benchmarks(1000, 1000)
