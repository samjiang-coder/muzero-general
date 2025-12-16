import sys
import platform

print("="*30)
print(f"Python Version: {platform.python_version()}")
print(f"Executable: {sys.executable}")
print("="*30)

dependencies = ['nevergrad', 'ray', 'torch', 'numpy', 'gym']
print("Checking dependencies:")
for dep in dependencies:
    try:
        __import__(dep)
        print(f"[OK] {dep}")
    except ImportError as e:
        print(f"[MISSING] {dep}: {e}")
print("="*30)
