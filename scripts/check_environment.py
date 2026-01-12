"""
Environment Verification Script for HEIMDALL
Checks if all dependencies are correctly installed
"""

import sys
from importlib import import_module

def check_module(module_name, display_name=None):
    """Check if a module is installed"""
    if display_name is None:
        display_name = module_name

    try:
        mod = import_module(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"[OK] {display_name:20s} - Version: {version}")
        return True
    except ImportError:
        print(f"[X]  {display_name:20s} - NOT INSTALLED")
        return False

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n[OK] CUDA available - GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("\n[!] CUDA not available - Using CPU")
            return False
    except:
        return False

def main():
    print("="*60)
    print("ENVIRONMENT CHECK - HEIMDALL")
    print("="*60)

    print(f"\nPython: {sys.version.split()[0]}")

    print("\nMain Dependencies:")
    print("-" * 60)

    modules = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers (HF)'),
        ('gudhi', 'GUDHI (TDA)'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('sklearn', 'Scikit-learn'),
        ('scipy', 'SciPy'),
        ('tqdm', 'tqdm')
    ]

    results = []
    for module, display in modules:
        results.append(check_module(module, display))

    check_cuda()

    print("\n" + "="*60)
    if all(results):
        print("[OK] ENVIRONMENT CONFIGURED CORRECTLY!")
        print("You can run: python tests/test_gpt2.py")
    else:
        print("[X] SOME DEPENDENCIES ARE MISSING")
        print("Run: pip install -r requirements.txt")
    print("="*60)

if __name__ == "__main__":
    main()
