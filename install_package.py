"""
Run this once after installing requirements.txt to register the src package.
This allows cross-module imports like `from src.constants import INSTRUCTION`
to work correctly from any script in the project.

Usage:
    python install_package.py
"""
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ".", "--quiet"])
print("OK: Package installed in editable mode. All src imports will now work.")
