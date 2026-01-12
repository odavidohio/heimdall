"""
PyTorch with CUDA Installation Script
Reinstalls PyTorch with appropriate CUDA support
"""

import subprocess
import sys

def print_section(title):
    """Print formatted section"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def detect_cuda_version():
    """Detect available CUDA version"""
    print_section("DETECTING CUDA VERSION")

    try:
        result = subprocess.run(['nvidia-smi'],
                              capture_output=True,
                              text=True,
                              timeout=5)

        if result.returncode == 0:
            output = result.stdout
            for line in output.split('\n'):
                if 'CUDA Version' in line:
                    import re
                    match = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', line)
                    if match:
                        major = int(match.group(1))
                        minor = int(match.group(2))
                        print(f"[OK] CUDA {major}.{minor} detected via nvidia-smi")
                        return major, minor

    except Exception as e:
        print(f"[WARNING] Could not detect via nvidia-smi: {e}")

    try:
        import torch
        if hasattr(torch.version, 'cuda') and torch.version.cuda:
            ver = torch.version.cuda.split('.')
            major = int(ver[0])
            minor = int(ver[1])
            print(f"[WARNING] Using current PyTorch version: CUDA {major}.{minor}")
            return major, minor
    except:
        pass

    print("[WARNING] Could not detect automatically")
    return None, None

def choose_cuda_version(detected_major, detected_minor):
    """Allow choosing CUDA version"""
    print_section("CUDA VERSION SELECTION")

    cuda_options = {
        '1': ('11.8', 'cu118', 'Compatible with older GPUs (recommended)'),
        '2': ('12.1', 'cu121', 'For newer GPUs (RTX 40xx, etc)'),
        '3': ('12.4', 'cu124', 'Latest version (experimental)'),
        '4': ('13.1', 'cu131', 'For GPUs with CUDA 13.1 (Experimental)')
    }

    if detected_major and detected_minor:
        print(f"\nDetected version: CUDA {detected_major}.{detected_minor}")

        if detected_major == 11:
            suggested = '1'
        elif detected_major == 12 and detected_minor <= 1:
            suggested = '2'
        elif detected_major == 12 and detected_minor > 1:
            suggested = '3'
        elif detected_major == 13:
            suggested = '4'
        else:
            suggested = '2'

        print(f"Suggestion: Option {suggested} (CUDA {cuda_options[suggested][0]})")
    else:
        suggested = '1'
        print("Could not detect. Suggestion: CUDA 11.8 (most compatible)")

    print("\nAvailable options:")
    for key, (version, code, desc) in cuda_options.items():
        print(f"  {key}. CUDA {version} - {desc}")

    print("\n  0. Cancel installation")

    while True:
        choice = input(f"\nChoice [0-3] (default: {suggested}): ").strip()

        if choice == '0':
            print("Installation cancelled.")
            return None

        if choice == '':
            choice = suggested

        if choice in cuda_options:
            return cuda_options[choice]

        print("[ERROR] Invalid option. Try again.")

def uninstall_current_pytorch():
    """Uninstall current PyTorch"""
    print_section("UNINSTALLING CURRENT PYTORCH")

    packages = ['torch', 'torchvision', 'torchaudio']

    print("Removing existing packages...")
    for package in packages:
        print(f"  Uninstalling {package}...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', package],
                         check=False,
                         capture_output=True)
        except:
            pass

    print("[OK] Previous PyTorch removed")

def install_pytorch_cuda(cuda_version, cuda_code):
    """Install PyTorch with CUDA support"""
    print_section(f"INSTALLING PYTORCH WITH CUDA {cuda_version}")

    index_url = f"https://download.pytorch.org/whl/{cuda_code}"

    print(f"Installing PyTorch {cuda_version}...")
    print("This may take a few minutes (download ~2GB)...\n")

    cmd = [
        sys.executable, '-m', 'pip', 'install',
        'torch', 'torchvision', 'torchaudio',
        '--index-url', index_url
    ]

    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)

        if result.returncode == 0:
            print("\n[OK] PyTorch with CUDA installed successfully!")
            return True
        else:
            print("\n[ERROR] Error during installation")
            return False

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Error during installation: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        return False

def verify_installation():
    """Verify if installation was successful"""
    print_section("VERIFYING INSTALLATION")

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")

            print("\nTesting GPU operation...")
            x = torch.randn(100, 100).cuda()
            y = x @ x
            print("[OK] Test successful!")

            del x, y
            torch.cuda.empty_cache()

            return True
        else:
            print("[ERROR] CUDA still not available")
            return False

    except ImportError:
        print("[ERROR] PyTorch was not installed correctly")
        return False
    except Exception as e:
        print(f"[ERROR] Error during verification: {e}")
        return False

def main():
    """Main function"""
    print("\n")
    print("=" * 70)
    print("          PYTORCH WITH CUDA INSTALLATION - HEIMDALL")
    print("=" * 70)

    major, minor = detect_cuda_version()

    choice = choose_cuda_version(major, minor)

    if choice is None:
        return

    cuda_version, cuda_code, description = choice

    print(f"\nInstalling PyTorch with CUDA {cuda_version}")
    print(f"   {description}")

    confirm = input("\nDo you want to continue? [Y/n]: ").strip().lower()
    if confirm and confirm not in ['y', 'yes', '']:
        print("Installation cancelled.")
        return

    uninstall_current_pytorch()

    success = install_pytorch_cuda(cuda_version, cuda_code)

    if not success:
        print("\n[ERROR] Installation failed.")
        print("Try installing manually:")
        print(f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{cuda_code}")
        return

    import time
    time.sleep(2)

    if verify_installation():
        print("\n" + "="*70)
        print("[OK] INSTALLATION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nNext steps:")
        print("   1. Run: python tests/test_optimized.py")
        print("   2. The script will use your GPU automatically")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("[WARNING] INSTALLATION COMPLETED, BUT VERIFICATION FAILED")
        print("="*70)
        print("\nTry:")
        print("   1. Restart your terminal/IDE")
        print("   2. Run: python scripts/check_cuda.py")
        print("   3. If the problem persists, check NVIDIA drivers")
        print("="*70)

    print()

if __name__ == "__main__":
    main()
