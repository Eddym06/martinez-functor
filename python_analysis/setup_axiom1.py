# setup_axiom1.py — Full environment bootstrap
import subprocess, sys

packages = [
    "pysr",           # Symbolic regression engine
    "sympy",          # Computer algebra system
    "numpy",
    "torch",          # For GPU GEMM validation
    "scipy",
    "einops",         # Tensor operations
]

print("Iniciando la construcción del entorno de AXIOM-1...")
for pkg in packages:
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)
        print(f"[OK] {pkg} instalado.")
    except Exception as e:
        print(f"[WARN] Error instalando {pkg}: {e}")

print("Environment ready. AXIOM-1 initialized.")
