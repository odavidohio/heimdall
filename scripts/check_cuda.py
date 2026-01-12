"""
CUDA Diagnostic Script
Verifies if CUDA is correctly configured for PyTorch
"""

import sys
import subprocess

def print_section(title):
    """Print formatted section"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def check_nvidia_gpu():
    """Check if NVIDIA GPU is installed"""
    print_section("1. CHECKING NVIDIA GPU")

    try:
        result = subprocess.run(['nvidia-smi'],
                              capture_output=True,
                              text=True,
                              timeout=5)

        if result.returncode == 0:
            print("[OK] NVIDIA GPU detected!\n")
            print(result.stdout)
            return True
        else:
            print("[ERROR] nvidia-smi not found or failed")
            return False
    except FileNotFoundError:
        print("[ERROR] nvidia-smi not found")
        print("   This means NVIDIA drivers are not installed.")
        return False
    except Exception as e:
        print(f"[ERROR] Error checking GPU: {e}")
        return False

def check_pytorch_cuda():
    """Check if PyTorch has CUDA support"""
    print_section("2. CHECKING PYTORCH WITH CUDA")

    try:
        import torch
        print(f"[OK] PyTorch installed: version {torch.__version__}")

        cuda_available = torch.cuda.is_available()

        if cuda_available:
            print(f"[OK] CUDA available in PyTorch!")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Number of GPUs: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                print(f"\n   GPU {i}:")
                print(f"     Name: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"     Compute Capability: {props.major}.{props.minor}")
                print(f"     Total VRAM: {props.total_memory / 1024**3:.1f} GB")

            return True
        else:
            print("[ERROR] PyTorch installed, but CUDA is NOT available")
            print("   This means PyTorch was installed without CUDA support")
            return False

    except ImportError:
        print("[ERROR] PyTorch is not installed")
        return False
    except Exception as e:
        print(f"[ERROR] Error checking PyTorch: {e}")
        return False

def check_cuda_version():
    """Check CUDA version installed on the system"""
    print_section("3. CHECKING SYSTEM CUDA VERSION")

    try:
        result = subprocess.run(['nvcc', '--version'],
                              capture_output=True,
                              text=True,
                              timeout=5)

        if result.returncode == 0:
            print("[OK] CUDA Toolkit installed:")
            print(result.stdout)

            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print(f"\n   Detected version: {line.strip()}")
            return True
        else:
            print("[WARNING] nvcc not found (CUDA Toolkit may not be installed)")
            print("   Note: PyTorch can work without local CUDA Toolkit")
            return False

    except FileNotFoundError:
        print("[WARNING] nvcc not found in PATH")
        print("   This is normal if you didn't install the full CUDA Toolkit")
        print("   PyTorch can still work (comes with CUDA runtime)")
        return False
    except Exception as e:
        print(f"[WARNING] Error: {e}")
        return False

def test_cuda_operations():
    """Test CUDA operations"""
    print_section("4. TESTING CUDA OPERATIONS")

    try:
        import torch

        if not torch.cuda.is_available():
            print("[ERROR] CUDA not available, cannot test")
            return False

        print("Testing tensor creation on GPU...")
        x = torch.randn(1000, 1000).cuda()
        print("[OK] Tensor created on GPU")

        print("\nTesting math operation on GPU...")
        y = torch.matmul(x, x)
        print("[OK] Matrix multiplication successful")

        print("\nTesting GPU -> CPU transfer...")
        z = y.cpu()
        print("[OK] Transfer successful")

        del x, y, z
        torch.cuda.empty_cache()

        print("\n[OK] ALL TESTS PASSED!")
        return True

    except Exception as e:
        print(f"[ERROR] Error during test: {e}")
        return False

def provide_recommendations(gpu_ok, pytorch_ok, cuda_ok):
    """Provide recommendations based on results"""
    print_section("DIAGNOSIS AND RECOMMENDATIONS")

    if gpu_ok and pytorch_ok:
        print("\n[SUCCESS] ALL READY!")
        print("   Your system is correctly configured to use CUDA.")
        print("\nNext steps:")
        print("   1. Run: python tests/test_optimized.py")
        print("   2. Mistral 7B will be downloaded automatically (~5GB)")
        print("   3. Analysis will use your GPU automatically")
        return

    if gpu_ok and not pytorch_ok:
        print("\n[WARNING] PYTORCH WITHOUT CUDA SUPPORT")
        print("   You have an NVIDIA GPU, but PyTorch is not using CUDA.")
        print("\nSOLUTION:")
        print("   Run the installation script:")
        print("   -> python scripts/install_pytorch_cuda.py")
        print("\n   OR install manually:")
        print("   -> pip uninstall torch torchvision torchaudio")

        try:
            import torch
            if hasattr(torch.version, 'cuda') and torch.version.cuda:
                cuda_ver = torch.version.cuda.split('.')[0] + torch.version.cuda.split('.')[1]
                print(f"   -> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{cuda_ver}")
            else:
                print("   -> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        except:
            print("   -> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return

    if not gpu_ok:
        print("\n[WARNING] NVIDIA DRIVER NOT DETECTED")
        print("   Could not detect NVIDIA GPU.")
        print("\nSOLUTION:")
        print("   1. Download the latest driver:")
        print("      -> https://www.nvidia.com/Download/index.aspx")
        print("   2. Install the driver")
        print("   3. Restart your computer")
        print("   4. Run this script again")
        return

def main():
    """Main function"""
    print("\n")
    print("=" * 70)
    print("              CUDA DIAGNOSTIC - HEIMDALL")
    print("=" * 70)

    gpu_ok = check_nvidia_gpu()
    pytorch_ok = check_pytorch_cuda()
    cuda_ok = check_cuda_version()

    if pytorch_ok:
        test_cuda_operations()

    provide_recommendations(gpu_ok, pytorch_ok, cuda_ok)

    print("\n" + "="*70)
    print()

if __name__ == "__main__":
    main()
