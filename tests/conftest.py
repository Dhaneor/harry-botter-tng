import os
import sys

# Add the directory containing your Cython modules to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def pytest_configure(config):
    import pyximport
    pyximport.install()

    print("pyimport installed: OK")