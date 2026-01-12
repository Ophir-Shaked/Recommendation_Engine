import sys                 # Access Python runtime info + executable path (used for pip installs)
import subprocess          # Run shell commands (we use it to run pip)
import importlib           # Import modules dynamically to check if packages exist
import warnings            # Control/disable warning messages for cleaner notebook output

import numpy as np         # Numerical computing (arrays, random, math)
import pandas as pd        # DataFrames for saving/reading CSVs and quick inspection


def try_import(module_name: str) -> bool:
    """
    Check if a Python module can be imported.
    Returns True if import succeeds, otherwise False.
    """
    try:
        importlib.import_module(module_name)  # Attempt import by name (no side effects beyond import)
        return True                           # Success => module is available
    except Exception:
        return False                          # Failure => module is missing or broken


def pip_install(pkgs):
    """
    Best-effort quiet pip install for missing packages.
    NOTE: We do NOT force upgrades of numpy/pandas/sklearn (to avoid breaking Colab env).
    """
    try:
        # Run: python -m pip install -q <pkgs...>
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])
    except Exception as e:
        # If pip fails, we print a warning and continue (so the notebook doesn't crash here)
        print("[WARN] pip install failed:", pkgs, "| err:", repr(e))


# List the packages we rely on in this notebook (benchmark + word2vec + plots).
required = ["datasets", "matplotlib", "gensim", "scikit-learn", "scipy"]  # Dependencies for loading data, models, metrics, and plots

# Determine which packages are missing *without* upgrading anything.
# We try to map pip names to import names:
# - scikit-learn pip name => import name is sklearn
# - others typically match
missing = []  # Will collect pip package names that we need to install
for pkg in required:  # Iterate over required pip packages
    # Convert pip package name to import name when needed
    import_name = "sklearn" if pkg == "scikit-learn" else pkg  # scikit-learn installs as sklearn
    if not try_import(import_name):                            # If import fails => missing
        missing.append(pkg)                                    # Add pip package name to install list

# If anything is missing, install it (quietly).
if missing:
    print("[SETUP] Installing missing packages:", missing)  # Inform the user what will be installed
    pip_install(missing)                                    # Try to install missing packages

# Import libraries AFTER installation step (so imports succeed even on fresh Colab runtime).
import scipy                 # Scientific computing (some metrics/helpers)
import sklearn                # Machine learning library (models, CV, metrics)

import matplotlib             # Plotting library
matplotlib.use("Agg")         # Use non-interactive backend
import matplotlib.pyplot as plt  # pyplot API for creating/saving plots

from datasets import load_dataset     # HuggingFace Datasets loader (e.g., ag_news)
from gensim.models import Word2Vec    # Gensim Word2Vec implementation


def print_versions():
    """
    Print versions for reproducibility.
    Helpful when debugging differences between environments.
    """
    import datasets, gensim           # Import here to read their __version__ safely
    print("---- VERSIONS ----")        # Header
    print("python:", sys.version.split()[0])      # Python version
    print("numpy:", np.__version__)               # Numpy version
    print("pandas:", pd.__version__)              # Pandas version
    print("sklearn:", sklearn.__version__)        # scikit-learn version
    print("scipy:", scipy.__version__)            # SciPy version
    print("datasets:", datasets.__version__)      # HuggingFace datasets version
    print("matplotlib:", matplotlib.__version__)  # Matplotlib version
    print("gensim:", gensim.__version__)          # Gensim version
    print("------------------")          # Footer


warnings.filterwarnings("ignore")  # Suppress warnings globally (keeps output clean; ok for notebooks)
print_versions()                   # Print versions now so your run is documented
